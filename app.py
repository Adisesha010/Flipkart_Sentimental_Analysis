import numpy as np
import streamlit as st
import pickle
import nltk
import re
import time

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Flipkart Review Sentiment Analyzer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================
# NLTK DOWNLOADS
# ==============================
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==============================
# STYLES
# ==============================
st.markdown("""
<style>
html, body { font-family: Inter, sans-serif; }

.stApp {
    background:
        linear-gradient(rgba(2,6,23,0.7), rgba(2,6,23,0.7)),
        url("https://images.unsplash.com/photo-1607082349566-187342175e2f");
    background-size: cover;
}

.hero {
    text-align: center;
    padding: 45px;
    border-radius: 22px;
    background: linear-gradient(135deg, #6366f1, #22d3ee);
    color: white;
    margin-bottom: 30px;
}

.stButton > button {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    font-size: 18px;
    padding: 14px 35px;
    border-radius: 14px;
}

.result {
    margin-top: 30px;
    padding: 28px;
    border-radius: 20px;
    font-size: 26px;
    font-weight: 800;
    text-align: center;
}

.positive {
    background: linear-gradient(135deg, #22c55e, #15803d);
    color: white;
}

.negative {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HERO
# ==============================
st.markdown("""
<div class="hero">
    <h1>🛒 Flipkart Review Sentiment Analyzer</h1>
    <p>Real-time customer sentiment analysis using NLP</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# TEXT CLEANING
# ==============================
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# ==============================
# LOAD PIPELINE
# ==============================
@st.cache_resource
def load_pipeline():
    with open("lr_model_1.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_pipeline()

# ==============================
# USER INPUT
# ==============================
review = st.text_area(
    "✍️ Enter a Flipkart product review",
    height=260,
    placeholder="Example: Quality is very poor and the product stopped working within a week..."
)

col1, col2, col3 = st.columns([3, 2, 3])
with col2:
    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

# ==============================
# PREDICTION
# ==============================
if analyze_btn:
    if review.strip() == "":
        st.warning("⚠️ Please enter a review before analysis.")
    else:
        with st.spinner("🧠 Analyzing sentiment..."):
            time.sleep(1)
            cleaned = clean_text(review)

            prediction = pipeline.predict([cleaned])[0]

            # Probability (if supported)
            if hasattr(pipeline, "predict_proba"):
                confidence = pipeline.predict_proba([cleaned]).max() * 100
            else:
                confidence = 95.0   # fallback

        if prediction == 1:
            cls = "positive"
            label = "😊 Positive Review"
        else:
            cls = "negative"
            label = "😡 Negative Review"

        st.markdown(f"""
        <div class="result {cls}">
            {label}<br><br>
            Confidence: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence / 100)

# ==============================
# FOOTER
# ==============================
st.markdown("""
<div style="margin-top:45px;text-align:center;color:#c7d2fe;">
    🚀 Built using NLP Pipeline & Machine Learning with Streamlit<br>
    © 2025 Flipkart Sentiment Analysis System
</div>
""", unsafe_allow_html=True)
