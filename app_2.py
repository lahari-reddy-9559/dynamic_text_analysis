# app.py — Lahari Reddy | Light Mode + Animated Gradient + Compact 500x300 Visuals

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

# --- Summarization utilities ---
try:
    from summarization_utils import clean_text as clean_text_util, extractive_reduce, abstractive_summarize_text
except ImportError:
    st.error("❌ Missing summarization_utils.py.")
    st.stop()

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

MODEL_DIR = 'models'
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
MAX_FEATURES = 5000
RANDOM_STATE = 42

st.set_page_config(page_title="Text Insight Studio | Lahari Reddy", layout="wide", page_icon="💬")

# ===============================
# 🌈 Light Mode + Gradient Styling
# ===============================
st.markdown(
    """
    <style>
    @keyframes softGradient {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }

    .stApp {
      background: linear-gradient(135deg, #e8f0ff, #f9e6ff, #e6fff2, #fffbe6);
      background-size: 300% 300%;
      animation: softGradient 20s ease infinite;
      color: #1f2937;
      font-family: "Poppins", sans-serif;
    }

    .block-container {
      background: rgba(255,255,255,0.88);
      border-radius: 16px;
      padding: 26px 30px;
      box-shadow: 0 6px 26px rgba(70,80,90,0.08);
    }

    h1, h2, h3 {
      background: linear-gradient(90deg, #6366f1, #10b981, #06b6d4, #f59e0b);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: 700;
      animation: hueShift 12s linear infinite;
    }

    @keyframes hueShift {
      0% { filter: hue-rotate(0deg); }
      100% { filter: hue-rotate(360deg); }
    }

    .stButton>button {
      background: linear-gradient(90deg, #4f46e5, #10b981, #06b6d4);
      background-size: 200% auto;
      color: white;
      border: none;
      border-radius: 10px;
      padding: 8px 14px;
      font-weight: 600;
      box-shadow: 0 4px 14px rgba(0,0,0,0.1);
      transition: all 0.4s ease;
    }

    .stButton>button:hover {
      background-position: right center;
      transform: translateY(-2px);
    }

    .stDownloadButton>button {
      background: linear-gradient(90deg,#22c55e,#06b6d4,#818cf8);
      background-size: 200% auto;
      color: white;
      border-radius: 10px;
      padding: 8px 14px;
      font-weight: 600;
      border: none;
      transition: all 0.4s ease;
    }

    .stDownloadButton>button:hover {
      background-position: right center;
      transform: translateY(-2px);
    }

    textarea[role="textbox"], .stFileUploader {
      border-radius: 10px !important;
      border: 1px solid rgba(150,150,150,0.3);
      background: rgba(255,255,255,0.7);
    }

    .stAlert {
      border-radius: 10px;
      background: rgba(236,253,245,0.9);
      color: #1f2937 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Data Loading & Preprocess
# -------------------------
@st.cache_data(show_spinner="📦 Loading dataset...")
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
# Model Training
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
# Utility Functions
# -------------------------
def analyze_sentiment(text, vec, clf):
    clean_t = clean_text_util(text)
    X = vec.transform([clean_t]).toarray()
    probs = clf.predict_proba(X)[0]
    results = {reverse_sentiment_mapping[c]: float(p) for c, p in zip(clf.classes_, probs)}
    top = reverse_sentiment_mapping[clf.classes_[np.argmax(probs)]]
    return results, top

def generate_wc_image_light(text):
    clean_t = clean_text_util(text)
    if not clean_t:
        return Image.new("RGB", (500, 300), color="#f8fafc")

    wc = WordCloud(
        width=500, height=300,
        background_color="#f8fafc",
        colormap="coolwarm",
        max_words=150
    ).generate(clean_t)
    return wc.to_image()

def plot_compact_bar_light(sentiment_dict):
    labels = list(sentiment_dict.keys())
    vals = [sentiment_dict[k] for k in labels]
    colors = ['#6366f1', '#10b981', '#06b6d4']

    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    ax.bar(labels, vals, color=colors[:len(labels)], width=0.4, edgecolor='#d1d5db')
    ax.set_ylim(0, 1.05)
    ax.set_title("Sentiment Confidence", fontsize=10, color="#1f2937")
    ax.set_ylabel("Probability", fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig

# -------------------------
# Model Initialization
# -------------------------
if 'clf' not in st.session_state:
    df, vec, X_train, y_train_num, _, _, _ = load_and_preprocess_data()
    if df is None:
        st.error("❌ Could not load dataset.")
        st.stop()
    clf, tfidf = train_and_save_models(X_train, y_train_num, vec)
    st.session_state.clf = clf
    st.session_state.vec = tfidf

clf = st.session_state.clf
vec = st.session_state.vec

# -------------------------
# UI
# -------------------------
st.title("💬 Text Insight Studio")
st.caption("Developed by **Lahari Reddy** — Light mode, pastel gradients, compact 500×300 visuals ✨")

text_input = st.text_area("📝 Enter Text:", placeholder="Type or paste your text here...", height=160)
uploaded = st.file_uploader("📄 Or upload a text file (.txt):", type=["txt"])
if uploaded:
    text_input = uploaded.read().decode("utf-8", errors="ignore")

st.markdown("---")

cols = st.columns(4)
choice = None
buttons = [("🧠 Sentiment Analysis", "sentiment"),
           ("✂️ Extractive Summary", "extractive"),
           ("🪶 Abstractive Summary", "abstractive"),
           ("☁️ Word Cloud", "wordcloud")]
for (label, val), col in zip(buttons, cols):
    with col:
        if st.button(label):
            choice = val

st.markdown("---")

# -------------------------
# Main Logic
# -------------------------
if text_input and text_input.strip():
    if choice == "sentiment":
        st.subheader("🧠 Sentiment Analysis")
        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        st.success(f"Predicted Sentiment: **{top_sent.upper()}**")

        fig = plot_compact_bar_light(sentiment_probs)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.pyplot(fig, use_container_width=False)

    elif choice == "extractive":
        st.subheader("✂️ Extractive Summary")
        st.info(extractive_reduce(text_input))

    elif choice == "abstractive":
        st.subheader("🪶 Abstractive Summary")
        try:
            st.info(abstractive_summarize_text(text_input))
        except Exception as e:
            st.error(f"Error: {e}")

    elif choice == "wordcloud":
        st.subheader("☁️ Word Cloud Visualization")
        wc_img = generate_wc_image_light(text_input)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(wc_img, use_column_width=False, width=500)

    if st.button("📥 Download Compact Report (PDF)"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = [
            Paragraph("<b>Text Insight Studio - Compact Report</b>", styles["Title"]),
            Spacer(1, 8),
            Paragraph("Original Text:", styles["Heading2"]),
            Paragraph(text_input[:1200] + ("..." if len(text_input) > 1200 else ""), styles["Normal"]),
            Spacer(1, 8)
        ]
        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        elements.append(Paragraph("Predicted Sentiment:", styles["Heading2"]))
        elements.append(Paragraph(top_sent.upper(), styles["Normal"]))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Extractive Summary:", styles["Heading2"]))
        elements.append(Paragraph(extractive_reduce(text_input), styles["Normal"]))
        elements.append(Spacer(1, 6))
        try:
            elements.append(Paragraph("Abstractive Summary:", styles["Heading2"]))
            elements.append(Paragraph(abstractive_summarize_text(text_input), styles["Normal"]))
        except Exception:
            pass
        wc_img = generate_wc_image_light(text_input)
        img_path = "wordcloud_500x300.png"
        wc_img.save(img_path)
        elements.append(RLImage(img_path, width=5.0*inch, height=3.0*inch))
        doc.build(elements)
        st.download_button("⬇️ Save Compact PDF Report",
                           data=buffer.getvalue(),
                           file_name="Text_Insight_Compact_Report.pdf",
                           mime="application/pdf")
else:
    st.info("💡 Enter text above or upload a file to start analysis.")
