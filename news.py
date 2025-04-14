import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
from wordcloud import WordCloud
import joblib
import os
from transformers import pipeline
import torch
from gtts import gTTS
import io
import base64
import requests
from PIL import Image
from io import BytesIO
import json
from datetime import datetime
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Set page config as the first Streamlit command
st.set_page_config(page_title="NewsVibe Pro", layout="wide")

# Download NLTK data (run once during app initialization)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configuration
MODEL_PATH = "news_sentiment_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"
AUDIO_FILE = "temp_recording.wav"
HISTORY_FILE = "news_history.json"
NEWS_API_KEY = "6d2085dd0dab41a887aff8497106a36b"  # Your NewsAPI key

# Initialize emotion classifier
try:
    device = 0 if torch.cuda.is_available() else -1
    emotion_classifier = pipeline(
        "text-classification", 
        model="j-hartmann/emotion-english-distilroberta-base",
        device=device
    )
except Exception as e:
    st.error(f"Failed to load emotion model: {e}")
    st.stop()

# Define your datasets configuration
DATASETS = {
    "data/news_sentiment.csv": {
        "text_col": "Headline",
        "sentiment_map": {
            "positive": 1,
            "neutral": 0,
            "negative": -1
        }
    }
}

# Model Loading and Training Functions
def load_saved_model():
    """Load saved model and vectorizer if they exist"""
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    return None, None

def save_model(model, vectorizer):
    """Save model and vectorizer to disk"""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

def train_model():
    """Train a new sentiment analysis model"""
    dfs = []
    
    for filepath, config in DATASETS.items():
        df = load_dataset(filepath, config)
        if df is None:
            st.warning(f"Couldn't load {filepath}")
            continue
            
        if "sample_size" in config:
            df = df.sample(min(config["sample_size"], len(df)), random_state=42)
        
        if "synthetic_labels" in config and config["synthetic_labels"]:
            text_col = config["text_col"]
            df["Sentiment_Score"] = generate_sentiment_labels(df[text_col])
            df = df.rename(columns={text_col: "Headline"})
        else:
            df["Sentiment"] = df["Sentiment"].str.lower()
            df["Sentiment_Score"] = df["Sentiment"].map(config["sentiment_map"])
        
        dfs.append(df[["Headline", "Sentiment_Score"]])
    
    if not dfs:
        st.error("No datasets could be loaded")
        st.stop()
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.dropna()
    combined_df = combined_df[combined_df["Headline"].str.strip().astype(bool)]

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=2000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9
    )
    
    X = tfidf.fit_transform(combined_df["Headline"])
    y = combined_df["Sentiment_Score"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='liblinear'
    )
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    st.success(f"Model trained on {len(combined_df)} headlines (Accuracy: {accuracy:.2f})")
    
    save_model(model, tfidf)
    return model, tfidf

# Data Loading Functions
def load_dataset(filepath, config):
    """Load dataset with multiple encoding attempts"""
    encodings = ["utf-8", "latin-1", "iso-8859-1", "windows-1252"]
    for encoding in encodings:
        try:
            if "columns" in config:
                df = pd.read_csv(filepath, header=None, names=config["columns"], encoding=encoding)
                
            else:
                df = pd.read_csv(filepath, encoding=encoding)
            return df
        except Exception:
            continue
    return None

def generate_sentiment_labels(texts):
    """Generate synthetic sentiment labels using emotion classifier"""
    batch_size = 100
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results.extend(emotion_classifier(batch.tolist()))
    return [1 if res['label'] in ['joy'] else 
            -1 if res['label'] in ['anger', 'fear', 'sadness'] else 
            0 for res in results]

# History Functions
def load_history():
    """Load analysis history from file"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
                # Ensure all entries have required fields
                for entry in history:
                    if 'timestamp' not in entry:
                        entry['timestamp'] = datetime.now().isoformat()
                    if 'sentiment' not in entry:
                        entry['sentiment'] = {'label': 'N/A', 'confidence': 0}
                    if 'emotion' not in entry:
                        entry['emotion'] = {'label': 'N/A', 'confidence': 0}
                return history
    except Exception as e:
        st.error(f"Failed to load history: {e}")
    return []

def save_to_history(entry):
    """Save analysis to history file"""
    try:
        # Ensure all required fields are present
        required_fields = ['text', 'timestamp', 'sentiment', 'emotion']
        for field in required_fields:
            if field not in entry:
                if field == 'timestamp':
                    entry[field] = datetime.now().isoformat()
                elif field == 'sentiment':
                    entry[field] = {'label': 'N/A', 'confidence': 0}
                elif field == 'emotion':
                    entry[field] = {'label': 'N/A', 'confidence': 0}
                elif field == 'text':
                    entry[field] = 'No text provided'
        
        history = load_history()
        history.append(entry)
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f)
    except Exception as e:
        st.error(f"Failed to save history: {e}")

# Audio Functions
def text_to_speech(text, lang='en'):
    """Convert text to speech and return audio data with error handling"""
    try:
        tts = gTTS(text=text, lang=lang)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.warning(f"Text-to-speech failed: {e}. Voice output disabled.")
        return None

def autoplay_audio(audio_bytes):
    """Autoplay audio in Streamlit"""
    if audio_bytes:
        audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
        audio_html = f"""
        <audio autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        st.components.v1.html(audio_html, height=0)

def record_audio(duration=5, sample_rate=44100):
    """Record audio from microphone"""
    st.info(f"Recording for {duration} seconds... Speak now!")
    try:
        recording = sd.rec(int(duration * sample_rate), 
                         samplerate=sample_rate, 
                         channels=1,
                         dtype='float32')
        sd.wait()
        sf.write(AUDIO_FILE, recording, sample_rate)
        return AUDIO_FILE
    except Exception as e:
        st.error(f"Recording failed: {e}")
        return None

def transcribe_audio(audio_file):
    """Transcribe audio to text using SpeechRecognition"""
    try:
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
            text = r.recognize_google(audio)
            return text
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None

# Analysis Functions
def analyze_sentiment(text, model, tfidf):
    """Analyze text sentiment using trained model"""
    X = tfidf.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    prob = max(proba)
    label = "Positive" if pred == 1 else "Negative" if pred == -1 else "Neutral"
    return label, prob, proba

def analyze_emotion(text):
    """Analyze text emotion using transformer model"""
    result = emotion_classifier(text)[0]
    return result["label"].capitalize(), result["score"]

def generate_wordcloud(text):
    """Generate word cloud from text"""
    wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)
    return wc.to_array()

def get_similar_news(current_text, api_key, tfidf, num_recommendations=3):
    """Fetch similar current news from NewsAPI based on input text"""
    try:
        # Step 1: Extract keywords from input text
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(current_text.lower())
        # Prioritize nouns and key terms, limit to 3 keywords
        keywords = [word for word in tokens if word.isalnum() and word not in stop_words][:3]
        if not keywords:
            st.warning("No meaningful keywords extracted from input text.")
            return []
        
        # Step 2: Construct queries (main and fallback)
        main_query = " ".join(keywords)
        # Add synonyms for broader search
        synonyms = {
            "earthquake": "tremor OR aftershock",
            "panic": "flee OR rush",
            "myanmar": "burma"
        }
        expanded_query = " ".join(synonyms.get(word, word) for word in keywords)
        fallback_query = " ".join(keywords[:2]) if len(keywords) > 2 else main_query
        
        # Step 3: Query NewsAPI /everything endpoint
        base_url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": api_key,
            "q": expanded_query,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 50,  # Increased to fetch more candidates
            "from": (datetime.now() - pd.Timedelta(days=7)).isoformat()  # Last 7 days
        }
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        articles = []
        if data["status"] == "ok" and data["totalResults"] > 0:
            articles = data["articles"]
        
        # Step 4: Fallback query if no results
        if not articles:
            params["q"] = fallback_query
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            if data["status"] == "ok" and data["totalResults"] > 0:
                articles = data["articles"]
        
        if not articles:
            st.warning(f"No news found for query: {expanded_query}")
            return []
        
        # Step 5: Process and filter articles
        input_vector = tfidf.transform([current_text])
        filtered_articles = []
        
        # Debug: Log article titles and similarities
        st.write("Articles retrieved:")
        
        for article in articles:
            if article["title"] and article["title"] != "[Removed]":
                article_vector = tfidf.transform([article["title"]])
                similarity = cosine_similarity(input_vector, article_vector)[0][0]
                
                # Log for debugging
                
                
                # Skip zero-vector articles, but log them
                
                
                # Exclude near-identical titles (text-based check for safety)
                
                
                # Lowered threshold for broader matches
                if similarity > 0.01:
                    filtered_articles.append({
                        "title": article["title"],
                        "description": article["description"] or "No description available",
                        "url": article["url"],
                        "source": article["source"]["name"],
                        "publishedAt": article["publishedAt"],
                        "similarity": similarity
                    })
        
        # Step 6: Sort by similarity and limit to num_recommendations
        filtered_articles = sorted(filtered_articles, key=lambda x: x["similarity"], reverse=True)
        
        # Debug: Log query details and results
        
        
        if not filtered_articles:
            st.info(f"No sufficiently similar news found for: {main_query}")
            return []
        
        return filtered_articles[:num_recommendations]
    
    except Exception as e:
        st.error(f"Error fetching similar news: {str(e)}")
        return []

# News Functions
def get_trending_news(api_key, country=None, category="general", page_size=6):
    """Fetch real trending news from NewsAPI"""
    base_url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": api_key,
        "pageSize": page_size,
        "category": category
    }
    
    # Only add country parameter if specified
    if country:
        params["country"] = country
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "ok" and data["totalResults"] > 0:
            articles = []
            for article in data["articles"]:
                if article["title"] and article["title"] != "[Removed]":
                    articles.append({
                        "title": article["title"],
                        "description": article["description"] or "No description available",
                        "urlToImage": article["urlToImage"],
                        "url": article["url"],
                        "source": article["source"]["name"],
                        "publishedAt": article["publishedAt"]
                    })
            return articles
        else:
            st.warning(f"No articles found for category: {category}")
            return []
    except Exception as e:
        st.error(f"Failed to fetch news: {e}")
        return []

def load_image_from_url(url):
    """Load image from URL with error handling"""
    try:
        if not url:
            raise ValueError("Empty URL")
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        return Image.new('RGB', (300, 200), color=(73, 109, 137))

# Display Functions
def display_analysis(text, model, tfidf):
    """Display analysis results for a given text"""
    with st.spinner("Analyzing..."):
        # Create analysis entry for history
        analysis_entry = {
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "sentiment": {
                "label": "N/A",
                "confidence": 0,
                "probabilities": [0, 0, 0]
            },
            "emotion": {
                "label": "N/A",
                "confidence": 0
            }
        }
        
        # Display headline
        st.subheader("📝 Analyzing:")
        st.info(f'"{text}"')
        
        # Sentiment analysis
        try:
            sent, sent_conf, sent_proba = analyze_sentiment(text, model, tfidf)
            analysis_entry["sentiment"] = {
                "label": sent,
                "confidence": sent_conf,
                "probabilities": [float(p) for p in sent_proba]
            }
        except Exception as e:
            st.error(f"Sentiment analysis failed: {e}")
        
        # Emotion analysis
        try:
            emo, emo_conf = analyze_emotion(text)
            analysis_entry["emotion"] = {
                "label": emo,
                "confidence": emo_conf
            }
        except Exception as e:
            st.error(f"Emotion analysis failed: {e}")
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sentiment Analysis")
            st.metric("Prediction", analysis_entry["sentiment"]["label"], 
                     f"{analysis_entry['sentiment']['confidence']:.0%} confidence")
            fig_pie = px.pie(
                names=[analysis_entry["sentiment"]["label"]],
                title="Sentiment Distribution",
                color_discrete_sequence=[
                    "green" if analysis_entry["sentiment"]["label"] == "Positive" else 
                    "red" if analysis_entry["sentiment"]["label"] == "Negative" else 
                    "gray"
                ]
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("🧠 Emotion Detection")
            st.metric("Prediction", analysis_entry["emotion"]["label"], 
                     f"{analysis_entry['emotion']['confidence']:.0%} confidence")
            fig_bar = px.bar(
                x=[analysis_entry["emotion"]["label"]],
                y=[analysis_entry["emotion"]["confidence"]],
                title="Emotion Confidence",
                color=[analysis_entry["emotion"]["label"]],
                color_discrete_map={
                    'Joy': 'gold',
                    'Anger': 'red',
                    'Fear': 'purple',
                    'Sadness': 'blue',
                    'Surprise': 'orange',
                    'Neutral': 'gray'
                }
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Word cloud
        st.subheader("🔍 Key Terms")
        try:
            st.image(generate_wordcloud(text), caption="Word Cloud")
        except Exception as e:
            st.error(f"Failed to generate word cloud: {e}")
        
        # Recommendations from NewsAPI
        st.subheader("📰 Similar Current News")
        similar_news = get_similar_news(text, NEWS_API_KEY, tfidf)
        if similar_news:
            for news in similar_news:
                with st.expander(f"{news['title'][:50]}..."):
                    st.write(f"**Source**: {news['source']}")
                    st.write(f"**Published**: {news['publishedAt'][:10]}")
                    st.write(f"**Description**: {news['description']}")
                    st.markdown(f"[Read more]({news['url']})")
                    st.write(f"**Similarity**: {news['similarity']:.2%}")
        else:
            st.info("No similar news articles found. Try a different headline or check back later.")
        
        # Voice output
        st.subheader("🎧 Voice Output")
        voice_output = f"""
        The headline is: {text}.
        Sentiment analysis shows it's {analysis_entry['sentiment']['label']} with {analysis_entry['sentiment']['confidence']:.0%} confidence.
        The detected emotion is {analysis_entry['emotion']['label']} with {analysis_entry['emotion']['confidence']:.0%} confidence.
        """
        audio_bytes = text_to_speech(voice_output)
        if audio_bytes:
            st.audio(audio_bytes, format='audio/mp3')
            autoplay_audio(audio_bytes)
        
        # Save to history
        save_to_history(analysis_entry)

def history_page():
    """Display analysis history"""
    st.title("📜 Analysis History")
    
    history = load_history()
    if not history:
        st.info("No analysis history yet.")
        return
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        sentiment_filter = st.selectbox("Filter by sentiment", 
                                     ["All", "Positive", "Neutral", "Negative"])
    with col2:
        sort_order = st.selectbox("Sort by", 
                               ["Newest first", "Oldest first"])
    
    # Apply filters
    filtered_history = history
    if sentiment_filter != "All":
        filtered_history = [h for h in history 
                          if isinstance(h.get('sentiment', {}), dict) and 
                          h.get('sentiment', {}).get('label') == sentiment_filter]
    
    # Sort with error handling
    try:
        if sort_order == "Newest first":
            filtered_history = sorted(filtered_history, 
                                  key=lambda x: x.get('timestamp', ''), 
                                  reverse=True)
        else:
            filtered_history = sorted(filtered_history, 
                                  key=lambda x: x.get('timestamp', ''))
    except Exception as e:
        st.error(f"Error sorting history: {e}")
        filtered_history = history
    
    # Create a container for the history entries
    history_container = st.container()
    
    # Display history
    for i, entry in enumerate(filtered_history):
        with history_container.expander(f"{entry.get('text', 'No text')[:50]}..."):
            col1, col2 = st.columns(2)
            with col1:
                sentiment = entry.get('sentiment', {})
                if isinstance(sentiment, str):
                    st.write(f"**Sentiment:** {sentiment} (N/A)")
                else:
                    st.write(f"**Sentiment:** {sentiment.get('label', 'N/A')} ({sentiment.get('confidence', 0):.0%})")
            
            with col2:
                emotion = entry.get('emotion', {})
                if isinstance(emotion, str):
                    st.write(f"**Emotion:** {emotion} (N/A)")
                else:
                    st.write(f"**Emotion:** {emotion.get('label', 'N/A')} ({emotion.get('confidence', 0):.0%})")
            
            st.write(f"**Analyzed on:** {entry.get('timestamp', 'Unknown date')[:19]}")
            
            # Create a unique key for each button using UUID
            import uuid
            button_key = f"reanalyze_{uuid.uuid4()}"
            
            if st.button("Re-analyze", key=button_key):
                # Store the text to analyze in session state
                st.session_state['text_to_analyze'] = entry.get('text', '')
                # Switch to main page
                st.session_state.current_page = "main"
                # Rerun the app
                st.rerun()

# Main App Function
def main():
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "main"
    
    # Model loading - do this first since both pages need it
    model, tfidf = load_saved_model()
    if model is None:
        with st.spinner("Training model for the first time (this may take a few minutes)..."):
            model, tfidf = train_model()

    # Sidebar
    with st.sidebar:
        st.title("📰 NewsVibe Pro")
        st.markdown("---")
        
        # Page navigation
        page = st.radio("Navigate", ["Home", "History"], key="nav_radio")
        if page == "History":
            st.session_state.current_page = "history"
        else:
            st.session_state.current_page = "main"
        
        # Category selection in sidebar
        if st.session_state.current_page == "main":
            category = st.selectbox("Select News Category", [
                "general", "business", "entertainment", 
                "health", "science", "sports", "technology"
            ], key="sidebar_category")
            if st.button("Refresh News", key="sidebar_refresh_btn"):
                st.rerun()

    # Navigation logic
    if st.session_state.current_page == "history":
        history_page()
        return
    
    # Main Page Content
    st.title("📰 NewsVibe Pro: AI Headline Analyzer")
    
    # Check for text to analyze from history
    if 'text_to_analyze' in st.session_state:
        display_analysis(st.session_state.text_to_analyze, model, tfidf)
        del st.session_state.text_to_analyze
    
    # Trending News Section
    st.header("📰 Trending News")
    
    # Get trending news with loading indicator
    with st.spinner(f"Loading {category} news..."):
        trending_news = get_trending_news(NEWS_API_KEY, category=category, page_size=10)
    
    # Display news cards with additional inputs
    if trending_news:
        col1, col2 = st.columns(2)
        for i, article in enumerate(trending_news[:5]):  # First 5 in col1
            with col1:
                with st.expander(f"{article['title']}"):
                    img = load_image_from_url(article['urlToImage'])
                    st.image(img, use_column_width=True)
                    
                    st.write(article['description'])
                    st.caption(f"Source: {article['source']} | {article['publishedAt'][:10]}")
                    
                    if st.button(
                        "Analyze This Headline", 
                        key=f"analyze_{i}_{article['publishedAt']}"
                    ):
                        display_analysis(article["title"], model, tfidf)
        
        for i, article in enumerate(trending_news[5:10]):  # Next 5 in col2
            with col2:
                with st.expander(f"{article['title']}"):
                    img = load_image_from_url(article['urlToImage'])
                    st.image(img, use_column_width=True)
                    
                    st.write(article['description'])
                    st.caption(f"Source: {article['source']} | {article['publishedAt'][:10]}")
                    
                    if st.button(
                        "Analyze This Headline", 
                        key=f"analyze_{i+5}_{article['publishedAt']}"
                    ):
                        display_analysis(article["title"], model, tfidf)
    
    # Unified Input Section
    st.header("🔍 Analyze Your News")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["✏️ Text Input", "🎤 Voice Input"])
    
    with tab1:
        text_input = st.text_area("Enter news headline:", height=100, key="text_input_area")
        if st.button("Analyze Text", key="analyze_text_btn"):
            if text_input.strip():
                display_analysis(text_input, model, tfidf)
            else:
                st.error("Please enter a headline")
    
    with tab2:
        if st.button("Record Speech", key="voice_rec_btn"):
            audio_file = record_audio()
            if audio_file:
                transcribed_text = transcribe_audio(audio_file)
                if transcribed_text:
                    st.session_state.voice_input = transcribed_text
        
        if 'voice_input' in st.session_state:
            st.text_area("Transcribed Text:", value=st.session_state.voice_input, height=100, key="voice_input_area")
            if st.button("Analyze Voice Input", key="analyze_voice_btn"):
                if st.session_state.voice_input.strip():
                    display_analysis(st.session_state.voice_input, model, tfidf)
                else:
                    st.error("No voice input detected")

if __name__ == "__main__":
    main()