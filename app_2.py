# app.py — Lahari Reddy | Compact & Aesthetic Version 🌈✨

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

# --- 🌸 Compact Colorful Theme ---
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #FDEFF9 0%, #ECF4FF 50%, #E8F9F0 100%);
    font-family: 'Poppins', sans-serif;
}
div.block-container {
    padding-top: 2rem;
    background-color: rgba(255, 255, 255, 0.92);
    border-radius: 18px;
    padding: 25px 30px;
    box-shadow: 0px 4px 25px rgba(0,0,0,0.08);
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
    font-weight: bold;
    padding: 0.55em 1.1em;
    transition: 0.25s ease-in-out;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #00BFA6, #6C63FF);
}
.result-box {
    background: rgba(255, 255, 255, 0.85);
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data(show_spinner="📦 Loading dataset & models...")
def load_and_preprocess_data():
    try:
        path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
        df = pd.read_csv(os.path.join(path, 'train.csv'), encoding='latin-1')
    except Exception:
        return None, None, None, None, None, None, None

    df.dropna(subset=['text', 'selected_text'], inplace=True)
    for pkg in ['punkt', 'stopwords', 'wordnet']:
        try: nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
        except LookupError: nltk.download(pkg, quiet=True)

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

@st.cache_resource
def train_and_save_models(X_train, y_train_num, vec):
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train.toarray(), y_train_num)
    joblib.dump(clf, os.path.join(MODEL_DIR, 'rf_sentiment.pkl'))
    joblib.dump(vec, os.path.join(MODEL_DIR, 'tfidf.pkl'))
    return clf, vec

def analyze_sentiment(text, vec, clf):
    clean_t = clean_text_util(text)
    X = vec.transform([clean_t]).toarray()
    probs = clf.predict_proba(X)[0]
    results = {reverse_sentiment_mapping[c]: float(p) for c, p in zip(clf.classes_, probs)}
    top = reverse_sentiment_mapping[clf.classes_[np.argmax(probs)]]
    return results, top

def generate_wc(text):
    clean_t = clean_text_util(text)
    freq = pd.Series(clean_t.split()).value_counts().to_dict()
    wc = WordCloud(width=350, height=180, background_color='white', colormap='cool').generate_from_frequencies(freq)
    img_io = io.BytesIO(); wc.to_image().save(img_io, 'PNG'); img_io.seek(0)
    return Image.open(img_io)

# --- Model Load ---
if 'clf' not in st.session_state:
    df, vec, X_train, y_train_num, _, _, _ = load_and_preprocess_data()
    clf, tfidf = train_and_save_models(X_train, y_train_num, vec)
    st.session_state.clf, st.session_state.vec = clf, tfidf
clf, vec = st.session_state.clf, st.session_state.vec

# --- UI Header ---
st.title("💬 Text Insight Studio")
st.caption("Developed by **Lahari Reddy** 🌸 | Sentiment, Summaries & Visual Insights — Compact & Beautiful ✨")

# --- Input Area ---
text_input = st.text_area("📝 Enter Text:", placeholder="Type or paste your text here...", height=160)
uploaded = st.file_uploader("📄 Or upload a text file:", type=["txt"])
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

# --- Main Logic ---
if text_input.strip():
    if choice == "sentiment":
        st.subheader("🧠 Sentiment Analysis")
        data, top = analyze_sentiment(text_input, vec, clf)
        st.success(f"Predicted Sentiment: **{top.upper()}**")

        # Compact Bar Graph (smaller width + height)
        df_s = pd.DataFrame({"Sentiment": list(data.keys()), "Probability": list(data.values())})
        color_map = {'negative': '#F87171', 'neutral': '#FACC15', 'positive': '#34D399'}
        fig, ax = plt.subplots(figsize=(2.2,1.2))
        ax.bar(df_s["Sentiment"], df_s["Probability"],
               color=[color_map[s] for s in df_s["Sentiment"]],
               width=0.3, edgecolor='gray')
        ax.set_ylim(0, 1.05)
        ax.set_title("Sentiment Confidence Levels", fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
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
        wc_img = generate_wc(text_input)
        st.image(wc_img, caption="Compact Word Cloud", use_column_width=False, width=350)

    # --- PDF Download ---
    if st.button("📥 Download Full Report (PDF)"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = [
            Paragraph("<b>Text Insight Studio - Compact Report</b>", styles["Title"]),
            Spacer(1, 12),
            Paragraph("Original Text:", styles["Heading2"]),
            Paragraph(text_input[:1000] + ("..." if len(text_input) > 1000 else ""), styles["Normal"]),
            Spacer(1, 10)
        ]

        data, top = analyze_sentiment(text_input, vec, clf)
        elements.append(Paragraph("Predicted Sentiment:", styles["Heading2"]))
        elements.append(Paragraph(str(top).upper(), styles["Normal"]))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph("Extractive Summary:", styles["Heading2"]))
        elements.append(Paragraph(extractive_reduce(text_input), styles["Normal"]))

        try:
            elements.append(Spacer(1, 8))
            elements.append(Paragraph("Abstractive Summary:", styles["Heading2"]))
            elements.append(Paragraph(abstractive_summarize_text(text_input), styles["Normal"]))
        except:
            pass

        img_path = "wordcloud_small.png"
        generate_wc(text_input).save(img_path)
        elements.append(Spacer(1, 8))
        elements.append(RLImage(img_path, width=3.8*inch, height=2.2*inch))
        doc.build(elements)

        st.download_button("⬇️ Save Compact PDF Report",
                           data=buffer.getvalue(),
                           file_name="Text_Insight_Compact_Report.pdf",
                           mime="application/pdf")

else:
    st.info("💡 Enter text or upload a file to start analyzing.")

