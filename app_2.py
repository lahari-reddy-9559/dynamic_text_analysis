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

# --- Import Summarization Utilities ---
try:
    from summarization_utils import clean_text as clean_text_util, extractive_reduce, abstractive_summarize_text
except ImportError:
    st.error("❌ Could not find summarization_utils.py. Please ensure it's in the same directory.")
    st.stop()

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

MODEL_DIR = 'models'
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
MAX_FEATURES = 5000
RANDOM_STATE = 42

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Text Insight Studio | Lahari Reddy",
    layout="wide",
    page_icon="💬"
)

# --- Custom Theme ---
st.markdown("""
<style>
:root {
    --primary-color: #6C63FF;
    --accent-color: #00BFA6;
    --background-color: #F7F9FB;
    --secondary-background-color: #FFFFFF;
    --text-color: #2D2D2D;
}
body { background-color: var(--background-color); color: var(--text-color); }
h1, h2, h3, h4 { color: var(--primary-color); font-family: 'Poppins', sans-serif; }
.stButton>button {
    background: var(--primary-color);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.6em 1.5em;
    font-weight: bold;
}
.stButton>button:hover { background: var(--accent-color); transition: 0.3s; }
.result-box {
    background: var(--secondary-background-color);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 3px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# --- Data Preprocessing ---
@st.cache_data(show_spinner="Loading data and preparing models...")
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

@st.cache_resource(show_spinner=False)
def train_and_save_models(_X_train, _y_train_numeric, _tfidf_vectorizer):
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(_X_train.toarray(), _y_train_numeric)
    joblib.dump(clf, os.path.join(MODEL_DIR, 'rf_sentiment.pkl'))
    joblib.dump(_tfidf_vectorizer, os.path.join(MODEL_DIR, 'tfidf.pkl'))
    return clf, _tfidf_vectorizer

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
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq)
    img_io = io.BytesIO(); wc.to_image().save(img_io, 'PNG'); img_io.seek(0)
    return Image.open(img_io)

# --- Load model (once) ---
if 'clf' not in st.session_state:
    df, vec, X_train, y_train_num, _, _, _ = load_and_preprocess_data()
    clf, tfidf = train_and_save_models(X_train, y_train_num, vec)
    st.session_state.clf, st.session_state.vec = clf, tfidf
clf, vec = st.session_state.clf, st.session_state.vec

# --- UI Header ---
st.title("💬 Text Insight Studio")
st.caption("Developed by **Lahari Reddy** | A professional-grade text analysis suite for sentiment, summarization, and visualization")

# --- Input Section ---
text_input = st.text_area("📝 Enter Text:", placeholder="Type or paste text to analyze...", height=200)
uploaded = st.file_uploader("📄 Or upload a text file:", type=["txt"])
if uploaded: text_input = uploaded.read().decode("utf-8", errors="ignore")

# --- Button Controls ---
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
choice = None
with col1:
    if st.button("🧠 Sentiment Analysis"): choice = "sentiment"
with col2:
    if st.button("✂️ Extractive Summary"): choice = "extractive"
with col3:
    if st.button("🪶 Abstractive Summary"): choice = "abstractive"
with col4:
    if st.button("☁️ Word Cloud"): choice = "wordcloud"

st.markdown("---")

# --- Main Logic ---
if text_input.strip():
    if choice == "sentiment":
        st.subheader("🧠 Sentiment Analysis")
        data, top = analyze_sentiment(text_input, vec, clf)
        st.success(f"Predicted Sentiment: **{top.upper()}**")

        # --- Compact Bar Chart (Improved Design) ---
        df_s = pd.DataFrame({"Sentiment": list(data.keys()), "Probability": list(data.values())})
        color_map = {'negative': '#EF5350', 'neutral': '#FFD54F', 'positive': '#66BB6A'}
        fig, ax = plt.subplots(figsize=(5, 3))  # smaller, compact
        bars = ax.bar(df_s["Sentiment"], df_s["Probability"],
                      color=[color_map[s] for s in df_s["Sentiment"]],
                      width=0.4, edgecolor='gray')

        ax.set_ylim(0, 1.05)
        ax.set_title("Sentiment Confidence Levels", fontsize=10, pad=10)
        ax.set_xlabel("Sentiment", fontsize=8)
        ax.set_ylabel("Probability", fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # Remove unnecessary borders for cleaner look
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
            st.error(f"Model Error: {e}")

    elif choice == "wordcloud":
        st.subheader("☁️ Word Cloud Visualization")
        wc_img = generate_wc(text_input)
        st.image(wc_img, caption="Word Frequency Cloud", use_column_width=True)

    # --- PDF Download Section ---
    if st.button("📥 Download Full Report (PDF)"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = [
            Paragraph("<b>Text Insight Studio - Analysis Report</b>", styles["Title"]),
            Spacer(1, 12),
            Paragraph("Original Text:", styles["Heading2"]),
            Paragraph(text_input[:1000] + ("..." if len(text_input) > 1000 else ""), styles["Normal"]),
            Spacer(1, 12)
        ]

        # Sentiment
        data, top = analyze_sentiment(text_input, vec, clf)
        elements.append(Paragraph("Predicted Sentiment:", styles["Heading2"]))
        elements.append(Paragraph(str(top).upper(), styles["Normal"]))
        elements.append(Spacer(1, 12))

        # Extractive summary
        elements.append(Paragraph("Extractive Summary:", styles["Heading2"]))
        elements.append(Paragraph(extractive_reduce(text_input), styles["Normal"]))

        # Abstractive summary (optional)
        try:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Abstractive Summary:", styles["Heading2"]))
            elements.append(Paragraph(abstractive_summarize_text(text_input), styles["Normal"]))
        except:
            pass

        # Word cloud
        img_path = "wordcloud_temp.png"
        generate_wc(text_input).save(img_path)
        elements.append(Spacer(1, 12))
        elements.append(RLImage(img_path, width=5*inch, height=3*inch))

        doc.build(elements)
        st.download_button("⬇️ Save PDF Report", data=buffer.getvalue(),
                           file_name="Text_Insight_Report.pdf",
                           mime="application/pdf")

else:
    st.info("Enter text above or upload a file to start analysis.")
