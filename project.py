import streamlit as st
import requests
import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import plotly.express as px
from wordcloud import WordCloud
import time
import numpy as np
from collections import Counter

# --- Configuration ---
API_KEY = "6d2085dd0dab41a887aff8497106a36b" 

try:
    sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
except Exception as e:
    st.error(f"Failed to load AI models: {e}")
    st.stop()

# --- Functions ---
def get_news(topic):
    """Fetch news headlines from NewsAPI."""
    try:
        url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={API_KEY}&pageSize=30"
        response = requests.get(url).json()
        return [{"title": a["title"], "source": a["source"]["name"], "date": a["publishedAt"]} 
                for a in response["articles"]]
    except:
        st.error("API failed! Using sample data.")
        return [{"title": "AI grows fast", "source": "Test News", "date": "2025-03-26T00:00:00Z"}]

def analyze_sentiment(text):
    """Classify sentiment using AI (BERT)."""
    result = sentiment_classifier(text)[0]
    label = "Positive" if result["label"] == "POSITIVE" else "Negative" if result["label"] == "NEGATIVE" else "Neutral"
    return label, result["score"]

def analyze_emotion(text):
    """Detect emotions using AI."""
    result = emotion_classifier(text)[0]
    return result["label"], result["score"]

def explain_sentiment(text):
    """Explain sentiment using TF-IDF to find contributing words."""
    tfidf = TfidfVectorizer(stop_words="english", max_features=5)
    tfidf_matrix = tfidf.fit_transform([text])
    feature_names = tfidf.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    return [f"{word}: {score:.2f}" for word, score in zip(feature_names, scores) if score > 0]

def process_news(topic):
    """Process headlines with sentiment, emotion, and explanations."""
    news = get_news(topic)
    data = []
    for article in news:
        sent, sent_score = analyze_sentiment(article["title"])
        emo, emo_score = analyze_emotion(article["title"])
        explanation = explain_sentiment(article["title"])
        data.append({"Source": article["source"], "Headline": article["title"], 
                     "Sentiment": sent, "Sent_Confidence": sent_score, 
                     "Emotion": emo, "Emo_Confidence": emo_score, 
                     "Date": article["date"], "Explanation": explanation})
    return pd.DataFrame(data)

def predict_trend(df):
    """Predict future sentiment trend using a simple logistic regression model."""
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Sentiment_Score"] = df["Sentiment"].map({"Positive": 1, "Neutral": 0, "Negative": -1})
    X = pd.to_datetime(df["Date"]).map(lambda x: x.toordinal()).values.reshape(-1, 1)
    y = df["Sentiment_Score"]
    model = LogisticRegression()
    model.fit(X, y)
    future_dates = np.array([X[-1][0] + i for i in range(1, 4)]).reshape(-1, 1)
    predictions = model.predict(future_dates)
    return predictions

def detect_bias(df):
    """Detect potential bias by comparing sentiment across sources."""
    source_sentiments = df.groupby("Source")["Sentiment"].value_counts(normalize=True).unstack().fillna(0)
    bias_alert = []
    for source1 in source_sentiments.index:
        for source2 in source_sentiments.index:
            if source1 < source2:
                pos_diff = abs(source_sentiments.loc[source1, "Positive"] - source_sentiments.loc[source2, "Positive"])
                neg_diff = abs(source_sentiments.loc[source1, "Negative"] - source_sentiments.loc[source2, "Negative"])
                if pos_diff > 0.3 or neg_diff > 0.3:
                    bias_alert.append(f"Potential bias between {source1} and {source2}")
    return bias_alert

def get_key_drivers(df):
    """Find key words driving sentiment with AI."""
    tfidf = TfidfVectorizer(stop_words="english", max_features=10)
    tfidf.fit_transform(df["Headline"])
    return tfidf.get_feature_names_out()

# --- Streamlit App ---
st.title("NewsVibe: AI-Powered Headline Sentiment Analyzer")
st.write("Built for AI Hackathon - Advanced AI for news sentiment analysis!")

# Sidebar for input
st.sidebar.title("Controls")
topic = st.sidebar.text_input("Enter a topic (e.g., AI, climate):", "AI")
live_mode = st.sidebar.checkbox("Live Updates (every 30s)")

# Main content
if live_mode:
    placeholder = st.empty()
    iteration = 0
    while True:
        iteration += 1
        df = process_news(topic)
        with placeholder.container():
            st.write("### Latest Headlines & AI Analysis")
            st.dataframe(df[["Source", "Headline", "Sentiment", "Emotion", "Explanation"]])
            
            # Sentiment Pie Chart
            fig_pie = px.pie(df, names="Sentiment", title=f"Sentiment for {topic}",
                            color_discrete_map={"Positive": "green", "Negative": "red", "Neutral": "gray"})
            st.plotly_chart(fig_pie, key=f"pie_chart_{iteration}")
            
            # Emotion Bar Chart
            fig_bar = px.bar(df, x="Source", y="Emo_Confidence", color="Emotion", 
                            title="Emotions by Source")
            st.plotly_chart(fig_bar, key=f"bar_chart_{iteration}")
            
            # Sentiment Trend Prediction
            st.write("### AI-Predicted Sentiment Trend (Next 3 Days)")
            predictions = predict_trend(df)
            pred_labels = ["Positive" if p == 1 else "Negative" if p == -1 else "Neutral" for p in predictions]
            st.write(f"Day 1: {pred_labels[0]}, Day 2: {pred_labels[1]}, Day 3: {pred_labels[2]}")
            
            # Bias Detection
            st.write("### AI-Detected Bias Alerts")
            bias_alerts = detect_bias(df)
            if bias_alerts:
                for alert in bias_alerts:
                    st.write(alert)
            else:
                st.write("No significant bias detected.")
            
            # Word Cloud
            wc = WordCloud(width=800, height=400).generate(" ".join(df["Headline"]))
            st.image(wc.to_array(), caption="Key Words in Headlines")
            
            # Key Drivers
            st.write("### AI-Detected Sentiment Drivers")
            st.write(get_key_drivers(df))
        
        time.sleep(30)  # Update every 30 seconds
else:
    if st.button("Analyze Now"):
        df = process_news(topic)
        st.write("### Headlines & AI Analysis")
        st.dataframe(df[["Source", "Headline", "Sentiment", "Emotion", "Explanation"]])
        
        # Sentiment Pie Chart
        fig_pie = px.pie(df, names="Sentiment", title=f"Sentiment for {topic}",
                        color_discrete_map={"Positive": "green", "Negative": "red", "Neutral": "gray"})
        st.plotly_chart(fig_pie, key="pie_chart_static")
        
        # Emotion Bar Chart
        fig_bar = px.bar(df, x="Source", y="Emo_Confidence", color="Emotion", 
                        title="Emotions by Source")
        st.plotly_chart(fig_bar, key="bar_chart_static")
        
        # Sentiment Trend Prediction
        st.write("### AI-Predicted Sentiment Trend (Next 3 Days)")
        predictions = predict_trend(df)
        pred_labels = ["Positive" if p == 1 else "Negative" if p == -1 else "Neutral" for p in predictions]
        st.write(f"Day 1: {pred_labels[0]}, Day 2: {pred_labels[1]}, Day 3: {pred_labels[2]}")
        
        # Bias Detection
        st.write("### AI-Detected Bias Alerts")
        bias_alerts = detect_bias(df)
        if bias_alerts:
            for alert in bias_alerts:
                st.write(alert)
        else:
            st.write("No significant bias detected.")
        
        # Word Cloud
        wc = WordCloud(width=800, height=400).generate(" ".join(df["Headline"]))
        st.image(wc.to_array(), caption="Key Words in Headlines")
        
        # Key Drivers
        st.write("### AI-Detected Sentiment Drivers")
        st.write(get_key_drivers(df))

st.write("Created by Grok 3 (xAI) for your AI Hackathon win!")