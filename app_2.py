# app.py — Lahari Reddy | Compact, Dark/Light Theme Toggle (500x300 px visuals)

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

# --- Summarization Utils ---
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

# --- Theme Toggle ---
st.sidebar.header("🎨 Theme Settings")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

# --- Base Styles ---
if dark_mode:
    st.markdown("""
    <style>
    body { background: linear-gradient(135deg, #0B0F14 0%, #1A1E25 100%); color: #FFFFFF; font-family: 'Poppins', sans-serif; }
    div.block-container {
        background-color: rgba(22, 27, 34, 0.92);
        border-radius: 14px;
        box-shadow: 0px 4px 20px rgba(255,255,255,0.05);
        padding: 24px;
    }
    h1, h2, h3, label, .stTextInput, .stMarkdown { color: #E6E6E6; }
    .stButton>button {
        background: linear-gradient(90deg, #6366F1, #10B981);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.5em 1.0em;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    body { background: linear-gradient(135deg, #FDEFF9 0%, #ECF4FF 50%, #E8F9F0 100%); font-family: 'Poppins', sans-serif; }
    div.block-container {
        background-color: rgba(255, 255, 255, 0.94);
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.06);
    }
    h1, h2, h3 { color: #4B0082; }
    .stButton>button {
        background: linear-gradient(90deg, #6C63FF, #00BFA6);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.5em 1.0em;
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

# --- Train Model ---
@st.cache_resource
def train_and_save_models(_X_train, _y_train_num, _vec):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(_X_train.toarray(), _y_train_num)
    joblib.dump(clf, os.path.join(MODEL_DIR, 'rf_sentiment.pkl'))
    joblib.dump(_vec, os.path.join(MODEL_DIR, 'tfidf.pkl'))
    return clf, _vec

# --- Utility Functions ---
def analyze_sentiment(text, vec, clf):
    clean_t = clean_text_util(text)
    X = vec.transform([clean_t]).toarray()
    probs = clf.predict_proba(X)[0]
    results = {reverse_sentiment_mapping[c]: float(p) for c, p in zip(clf.classes_, probs)}
    top = reverse_sentiment_mapping[clf.classes_[np.argmax(probs)]]
    return results, top

def generate_wc_image(text, dark_mode=False):
    clean_t = clean_text_util(text)
    bg = "black" if dark_mode else "white"
    cmap = "plasma" if dark_mode else "viridis"
    wc = WordCloud(width=500, height=300, background_color=bg, colormap=cmap, max_words=150)
    wc.generate(clean_t or "text insight")
    return wc.to_image()

def plot_compact_bar(sentiment_dict, dark_mode=False):
    labels = list(sentiment_dict.keys())
    vals = [sentiment_dict[k] for k in labels]

    if dark_mode:
        bg = "#0b0f14"
        text_color = "white"
        bar_colors = ['#FF6B6B', '#FFD166', '#06D6A0']
    else:
        bg = "white"
        text_color = "#222222"
        bar_colors = ['#F87171', '#FACC15', '#34D399']

    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    ax.bar(labels, vals, color=bar_colors[:len(labels)], width=0.35, edgecolor='gray')
    ax.set_ylim(0, 1.05)
    ax.set_title("Sentiment Confidence", fontsize=10, color=text_color, pad=6)
    ax.set_ylabel("Probability", color=text_color, fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.tick_params(colors=text_color)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig

# --- Load / Initialize ---
if 'clf' not in st.session_state:
    df, vec, X_train, y_train_num, _, _, _ = load_and_preprocess_data()
    if df is None:
        st.error("Setup failed: could not load dataset.")
        st.stop()
    clf, tfidf = train_and_save_models(X_train, y_train_num, vec)
    st.session_state.clf, st.session_state.vec = clf, tfidf
clf, vec = st.session_state.clf, st.session_state.vec

# --- UI ---
st.title("💬 Text Insight Studio")
st.caption("Developed by **Lahari Reddy** — Compact visuals, dark/light toggle ✨")

text_input = st.text_area("📝 Enter Text:", placeholder="Paste or type text to analyze...", height=160)
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

# --- Logic & Visualization ---
if text_input and text_input.strip():
    if choice == "sentiment":
        st.subheader("🧠 Sentiment Analysis")
        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        st.success(f"Predicted Sentiment: **{top_sent.upper()}**")

        fig = plot_compact_bar(sentiment_probs, dark_mode=dark_mode)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.pyplot(fig)

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
        wc_img = generate_wc_image(text_input, dark_mode)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(wc_img, width=500, use_column_width=False)

    if st.button("📥 Download Full Report (PDF)"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = [
            Paragraph("<b>Text Insight Studio - Compact Report</b>", styles["Title"]),
            Spacer(1, 8),
            Paragraph("Original Text:", styles["Heading2"]),
            Paragraph(text_input[:1000] + ("..." if len(text_input) > 1000 else ""), styles["Normal"]),
            Spacer(1, 8)
        ]
        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        elements.append(Paragraph("Predicted Sentiment:", styles["Heading2"]))
        elements.append(Paragraph(str(top_sent).upper(), styles["Normal"]))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Extractive Summary:", styles["Heading2"]))
        elements.append(Paragraph(extractive_reduce(text_input), styles["Normal"]))
        try:
            elements.append(Spacer(1, 6))
            elements.append(Paragraph("Abstractive Summary:", styles["Heading2"]))
            elements.append(Paragraph(abstractive_summarize_text(text_input), styles["Normal"]))
        except Exception:
            pass
        wc_img = generate_wc_image(text_input, dark_mode)
        img_path = "wordcloud_500x300.png"
        wc_img.save(img_path)
        elements.append(Spacer(1, 6))
        elements.append(RLImage(img_path, width=5.0*inch, height=3.0*inch))
        doc.build(elements)
        st.download_button("⬇️ Save Compact PDF Report",
                           data=buffer.getvalue(),
                           file_name="Text_Insight_Compact_Report.pdf",
                           mime="application/pdf")
else:
    st.info("💡 Enter text or upload a file to start analysis.")
