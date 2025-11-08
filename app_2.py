# app.py — Lahari Reddy | Compact, Dark-Mode Adaptive (500x300 px visuals)

import kagglehub
import os
import pandas as pd
import string
import nltk
import joblib
import warnings
import numpy as np
import io
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud
import tensorflow as tf
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# --- Summarization Utils (unchanged) ---
try:
    from summarization_utils import clean_text as clean_text_util, extractive_reduce, abstractive_summarize_text
except ImportError:
    st.error("❌ Could not find summarization_utils.py.")
    st.stop()

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

MODEL_DIR = 'models'
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
MAX_FEATURES = 5000
RANDOM_STATE = 42

# --- Streamlit Config ---
st.set_page_config(
    page_title="Text Insight Studio | Lahari Reddy",
    layout="wide",
    page_icon="💬"
)

# --- Visual theme (keeps previous look but we'll adapt plots to streamlit theme) ---
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #FDEFF9 0%, #ECF4FF 50%, #E8F9F0 100%);
    font-family: 'Poppins', sans-serif;
}
div.block-container {
    padding-top: 1.6rem;
    background-color: rgba(255, 255, 255, 0.94);
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.06);
}
h1, h2, h3 {
    color: #4B0082;
    font-weight: 600;
}
.stButton>button {
    background: linear-gradient(90deg, #6C63FF, #00BFA6);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5em 1.0em;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Data Loading & Preprocess
# -------------------------
@st.cache_data(show_spinner="📦 Loading dataset & models...")
def load_and_preprocess_data():
    try:
        path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
        df = pd.read_csv(os.path.join(path, 'train.csv'), encoding='latin-1')
    except Exception:
        return None, None, None, None, None, None, None

    df.dropna(subset=['text', 'selected_text'], inplace=True)

    for pkg in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_local(t):
        t = str(t).lower().translate(str.maketrans('', '', string.punctuation))
        w = [lemmatizer.lemmatize(x) for x in t.split() if x not in stop_words]
        return ' '.join(w)

    df['cleaned_text'] = df['text'].apply(clean_local)
    vec = TfidfVectorizer(max_features=MAX_FEATURES)
    vec.fit(df['cleaned_text'])
    X = vec.transform(df['cleaned_text'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, df['sentiment'], test_size=0.2, random_state=RANDOM_STATE, stratify=df['sentiment']
    )
    y_train_num = pd.Series(y_train).map(sentiment_mapping).astype(int)
    return df, vec, X_train, y_train_num, X, X_test, y_test

# -------------------------
# Train (cache-safe)
# -------------------------
@st.cache_resource
def train_and_save_models(_X_train, _y_train_num, _vec):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(_X_train.toarray(), _y_train_num)
    joblib.dump(clf, os.path.join(MODEL_DIR, 'rf_sentiment.pkl'))
    joblib.dump(_vec, os.path.join(MODEL_DIR, 'tfidf.pkl'))
    return clf, _vec

# -------------------------
# Theme detection helper
# -------------------------
def is_streamlit_dark():
    """Return True if Streamlit theme is dark; fallback False."""
    try:
        base = st.get_option("theme.base")
        return base == "dark"
    except Exception:
        # if option not present, try reading runtime theme color - fallback light
        return False

# -------------------------
# Utility functions
# -------------------------
def analyze_sentiment(text, vec, clf):
    clean_t = clean_text_util(text)
    X = vec.transform([clean_t]).toarray()
    probs = clf.predict_proba(X)[0]
    results = {reverse_sentiment_mapping[c]: float(p) for c, p in zip(clf.classes_, probs)}
    top = reverse_sentiment_mapping[clf.classes_[np.argmax(probs)]]
    return results, top

def generate_wc_image(text, dark_mode=False):
    """Return PIL Image of WordCloud sized 500x300 pixels."""
    clean_t = clean_text_util(text)
    if not clean_t:
        # Empty white/black image based on mode
        bg = "black" if dark_mode else "white"
        im = Image.new("RGB", (500, 300), color=bg)
        return im

    # WordCloud with exact pixel dimensions
    wc = WordCloud(width=500, height=300,
                   background_color="black" if dark_mode else "white",
                   colormap="plasma" if dark_mode else "viridis",
                   max_words=150).generate(clean_t)

    img = wc.to_image()  # PIL image at 500x300
    return img

def plot_compact_bar(sentiment_dict, dark_mode=False):
    """
    Create a compact bar chart sized 500x300 px (figsize 5x3 at dpi=100).
    Returns the matplotlib Figure.
    """
    labels = list(sentiment_dict.keys())
    vals = [sentiment_dict[k] for k in labels]

    # Theme-aware colors
    if dark_mode:
        bg = "#0b0f14"
        text_color = "white"
        bar_colors = ['#FF6B6B', '#FFD166', '#06D6A0']  # vivid on dark
    else:
        bg = "white"
        text_color = "#222222"
        bar_colors = ['#F87171', '#FACC15', '#34D399']

    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)  # 500x300 px
    bars = ax.bar(labels, vals,
