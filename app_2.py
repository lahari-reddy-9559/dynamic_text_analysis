# app.py — Lahari Reddy | Neutral Palette + Animated Background + Compact 500x300 visuals

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

# --- Summarization utilities (unchanged) ---
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

# --- Streamlit page config ---
st.set_page_config(page_title="Text Insight Studio | Lahari Reddy", layout="wide", page_icon="💬")

# --- Neutral animated background + neutral card styles (same in light/dark) ---
st.markdown(
    """
    <style>
    /* Animated neutral gradient background (subtle) */
    @keyframes subtleGradient {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }

    .stApp {
      background: linear-gradient(120deg, #eef2f7 0%, #f6fbff 35%, #eef6f4 65%, #f7f7fb 100%);
      background-size: 300% 300%;
      animation: subtleGradient 18s ease infinite;
      color: #222831;
      font-family: "Poppins", sans-serif;
    }

    /* Card container inside Streamlit */
    .block-container {
      background: rgba(255,255,255,0.88); 
      border-radius: 14px;
      padding: 22px 26px;
      box-shadow: 0 6px 24px rgba(34,40,49,0.06);
    }

    h1, h2, h3 { color: #2b2f36; font-weight:600; }
    .stButton>button {
      background: linear-gradient(90deg, #708090, #88bdb6);
      color: #ffffff;
      border-radius: 8px;
      border: none;
      padding: 8px 14px;
      font-weight:600;
    }
    /* Small subtle hover */
    .stButton>button:hover { transform: translateY(-1px); transition: .18s ease; }

    /* Smaller file uploader + textarea aesthetic */
    textarea[role="textbox"] { border-radius: 10px !important; }
    .stFileUploader { border-radius: 10px !important; }

    /* Make the download button visually clear */
    .stDownloadButton>button { background: linear-gradient(90deg,#6b7280,#4b9ea9); color: white; border-radius: 8px; padding:8px 12px; }

    </style>
    """,
    unsafe_allow_html=True,
)

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
# Utility functions
# -------------------------
def analyze_sentiment(text, vec, clf):
    clean_t = clean_text_util(text)
    X = vec.transform([clean_t]).toarray()
    probs = clf.predict_proba(X)[0]
    results = {reverse_sentiment_mapping[c]: float(p) for c, p in zip(clf.classes_, probs)}
    top = reverse_sentiment_mapping[clf.classes_[np.argmax(probs)]]
    return results, top

def generate_wc_image_neutral(text):
    """Return PIL Image of WordCloud sized 500x300 pixels using neutral palette."""
    clean_t = clean_text_util(text)
    if not clean_t:
        return Image.new("RGB", (500, 300), color="#f4f6f8")  # neutral light background

    wc = WordCloud(
        width=500,
        height=300,
        background_color="#f4f6f8",  # neutral light grayish
        colormap="cividis",          # neutral, readable palette
        max_words=150,
        prefer_horizontal=0.9
    ).generate(clean_t)
    return wc.to_image()

def plot_compact_bar_neutral(sentiment_dict):
    """
    Create a compact neutral bar chart sized 500x300 px (figsize 5x3 at dpi=100).
    Returns the matplotlib Figure.
    """
    labels = list(sentiment_dict.keys())
    vals = [sentiment_dict[k] for k in labels]

    # Neutral color set with good contrast on both themes
    bar_colors = ['#6b7280', '#9aa5a6', '#4b9ea9']  # slate, muted sea, teal-ish

    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)  # 500x300 px
    bars = ax.bar(labels, vals, color=bar_colors[:len(labels)], width=0.35, edgecolor='#d1d5db')

    # Appearance
    ax.set_ylim(0, 1.05)
    ax.set_title("Sentiment Confidence", fontsize=10, color="#1f2937", pad=6)
    ax.set_ylabel("Probability", color="#374151", fontsize=9)
    ax.set_xlabel("")
    ax.grid(axis='y', linestyle='--', alpha=0.25)

    # Neutral background matching the app card look
    fig.patch.set_facecolor("#f9fafb")
    ax.set_facecolor("#f9fafb")

    # Ticks and labels
    ax.tick_params(colors="#374151", which='both')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.setp(ax.get_xticklabels(), fontsize=9, color="#374151")
    plt.setp(ax.get_yticklabels(), fontsize=8, color="#374151")

    plt.tight_layout()
    return fig

# -------------------------
# Load / initialize model
# -------------------------
if 'clf' not in st.session_state:
    df, vec, X_train, y_train_num, _, _, _ = load_and_preprocess_data()
    if df is None:
        st.error("Setup failed: could not load dataset. Check Kaggle config and dataset availability.")
        st.stop()
    clf, tfidf = train_and_save_models(X_train, y_train_num, vec)
    st.session_state.clf = clf
    st.session_state.vec = tfidf

clf = st.session_state.clf
vec = st.session_state.vec

# -------------------------
# UI Header
# -------------------------
st.title("💬 Text Insight Studio")
st.caption("Developed by **Lahari Reddy** — Neutral palette, compact visuals, subtle animated background ✨")

# Input area
text_input = st.text_area("📝 Enter Text:", placeholder="Paste or type text to analyze...", height=160)
uploaded = st.file_uploader("📄 Or upload a text file (.txt):", type=["txt"])
if uploaded:
    text_input = uploaded.read().decode("utf-8", errors="ignore")

st.markdown("---")

# Buttons row
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
# Main Logic: center visuals
# -------------------------
if text_input and text_input.strip():
    if choice == "sentiment":
        st.subheader("🧠 Sentiment Analysis")
        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        st.success(f"Predicted Sentiment: **{top_sent.upper()}**")

        # Create neutral bar figure
        fig = plot_compact_bar_neutral(sentiment_probs)

        # Center the plot using columns (left spacer, center, right spacer)
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
            st.error(f"Abstractive summarization error: {e}")

    elif choice == "wordcloud":
        st.subheader("☁️ Word Cloud Visualization")
        wc_img = generate_wc_image_neutral(text_input)

        # Center the image using columns and show at exact size 500x300 px
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(wc_img, use_column_width=False, width=500)

    # PDF generation (compact visuals)
    if st.button("📥 Download Full Report (PDF)"):
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

        # Sentiment
        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        elements.append(Paragraph("Predicted Sentiment:", styles["Heading2"]))
        elements.append(Paragraph(str(top_sent).upper(), styles["Normal"]))
        elements.append(Spacer(1, 6))

        # Extractive summary
        elements.append(Paragraph("Extractive Summary:", styles["Heading2"]))
        elements.append(Paragraph(extractive_reduce(text_input), styles["Normal"]))
        elements.append(Spacer(1, 6))

        # Abstractive summary (if available)
        try:
            elements.append(Paragraph("Abstractive Summary:", styles["Heading2"]))
            elements.append(Paragraph(abstractive_summarize_text(text_input), styles["Normal"]))
            elements.append(Spacer(1, 6))
        except Exception:
            pass

        # Wordcloud image for PDF (500x300)
        wc_img = generate_wc_image_neutral(text_input)
        img_path = "wordcloud_neutral_500x300.png"
        wc_img.save(img_path)
        elements.append(RLImage(img_path, width=5.0*inch, height=3.0*inch))
        elements.append(Spacer(1, 8))

        doc.build(elements)
        # Offer download
        st.download_button("⬇️ Save Compact PDF Report",
                           data=buffer.getvalue(),
                           file_name="Text_Insight_Compact_Report.pdf",
                           mime="application/pdf")
else:
    st.info("💡 Enter text above or upload a file to start analysis.")
