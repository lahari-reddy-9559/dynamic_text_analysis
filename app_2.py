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
import math
import matplotlib.pyplot as plt
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud
import tensorflow as tf
from datetime import datetime

# --- Import Summarization Utilities ---
# Assumes summarization_utils.py is present in the same directory
try:
    from summarization_utils import clean_text as clean_text_util, extractive_reduce, abstractive_summarize_text
except ImportError:
    st.error("Setup failed: Could not find summarization_utils.py. Please ensure it is in the same directory.")
    st.stop() 

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# --- Configuration for ML Backend ---
MODEL_DIR = 'models'
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
MAX_FEATURES = 5000
RANDOM_STATE = 42
RANDOM_FOREST_MODEL_NAME = 'random_forest_sentiment_classifier.pkl'

# --- 2. ML Backend Functions (SILENT EXECUTION) ---

@st.cache_data(show_spinner="Preparing data and models (This may take a moment)...")
def load_and_preprocess_data():
    # Uses Kaggle Data, TF-IDF, and Lemmatization
    try:
        # NOTE: If running outside a secure environment, KaggleHub might require authentication setup.
        path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
        file_path = os.path.join(path, 'train.csv')
        df = pd.read_csv(file_path, encoding='latin-1')
    except Exception as e:
        # Fallback for local testing if Kaggle fails
        return None, None, None, None, None, None, None

    df.dropna(subset=['text', 'selected_text'], inplace=True)

    for package in ['punkt', 'stopwords', 'wordnet']:
        try: nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except LookupError: nltk.download(package, quiet=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_text_local(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    df['cleaned_text'] = df['text'].apply(clean_text_local)
    
    tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    tfidf_vectorizer.fit(df['cleaned_text'])
    tfidf_matrix = tfidf_vectorizer.transform(df['cleaned_text'])
    
    X_train, X_test, y_train_str, y_test_str = train_test_split(
        tfidf_matrix, df['sentiment'], test_size=0.2, random_state=RANDOM_STATE, stratify=df['sentiment']
    )
    
    y_train_numeric = pd.Series(y_train_str).map(sentiment_mapping).astype(int)

    return df, tfidf_vectorizer, X_train, y_train_numeric, tfidf_matrix, X_test, y_test_str

@st.cache_resource(show_spinner=False)
def train_and_save_models(_X_train, _y_train_numeric, _tfidf_vectorizer):
    # Trains the RandomForestClassifier model (NEW MODEL)
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Initialize and train RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    
    # RandomForestClassifier requires dense input for training
    X_train_dense = _X_train.toarray()
    clf.fit(X_train_dense, _y_train_numeric)
    
    joblib.dump(_tfidf_vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(clf, os.path.join(MODEL_DIR, RANDOM_FOREST_MODEL_NAME))
    
    return clf, _tfidf_vectorizer

# --- 3. Sentiment Prediction Function (Using ML Model) ---
def analyze_sentiment_and_get_data(text, vectorizer, classifier):
    cleaned_text = clean_text_util(text)
    text_vec = vectorizer.transform([cleaned_text])
    
    # RandomForestClassifier requires dense input for prediction
    text_vec_dense = text_vec.toarray()
    
    probabilities = classifier.predict_proba(text_vec_dense)[0]
    
    # RandomForestClassifier classes might not be sorted 0, 1, 2, so we rely on classifier.classes_
    sentiment_data = {}
    for i, label_id in enumerate(classifier.classes_):
        label_name = reverse_sentiment_mapping.get(label_id, f'Unknown_{label_id}')
        sentiment_data[label_name] = float(probabilities[i])
        
    top_sentiment_label = reverse_sentiment_mapping.get(classifier.classes_[np.argmax(probabilities)])
    
    return sentiment_data, top_sentiment_label

# --- 4. Word Cloud Function ---
def generate_wc_image(text):
    cleaned_text = clean_text_util(text)
    if not cleaned_text:
        return Image.new('RGB', (800, 400), color = 'white')

    word_freq = pd.Series(cleaned_text.split()).value_counts().to_dict()

    wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(word_freq)
    img_io = io.BytesIO()
    wc.to_image().save(img_io, 'PNG')
    img_io.seek(0)
    return Image.open(img_io)

# --- 5. Report Generation Function ---
def generate_report(text, sentiment_probs, top_sentiment, extractive_sum, abstractive_sum):
    """Generates a Markdown report string for download."""
    report_content = io.StringIO()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_content.write(f"# Text Analysis Report\n")
    report_content.write(f"**Date:** {now}\n")
    report_content.write(f"**Total Characters:** {len(text)}\n\n")
    report_content.write("---\n")
    
    # 1. Sentiment
    report_content.write("## 1. Sentiment Analysis\n")
    report_content.write(f"**Predicted Overall Sentiment:** {top_sentiment.upper()}\n")
    report_content.write("| Sentiment | Probability |\n")
    report_content.write("| :--- | :--- |\n")
    for sent, prob in sorted(sentiment_probs.items(), key=lambda item: item[1], reverse=True):
        report_content.write(f"| {sent.capitalize()} | {prob:.4f} |\n")
    report_content.write("\n")
    
    # 2. Summaries
    report_content.write("## 2. Text Summarization\n")
    report_content.write("### Extractive Summary (Key Sentences)\n")
    report_content.write(f"> {extractive_sum}\n\n")
    
    report_content.write("### Abstractive Summary (AI Paraphrase)\n")
    report_content.write(f"> {abstractive_sum}\n\n")
    
    # 3. Original Text
    report_content.write("---\n")
    report_content.write("## 3. Original Input Text\n")
    report_content.write("```\n")
    report_content.write(text)
    report_content.write("\n```\n")

    return report_content.getvalue().encode('utf-8')


# --- 6. Streamlit Application Execution (UI Focus) ---
st.set_page_config(layout="centered", page_title="Professional Text Analyzer")

# --- Initialize Session State for Results ---
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
    st.session_state.input_text = ""
    st.session_state.sentiment_probs = None
    st.session_state.top_sentiment = None
    st.session_state.extractive_sum = None
    st.session_state.abstractive_sum = None
    st.session_state.wc_image = None
    st.session_state.show_sentiment = False
    st.session_state.show_extractive = False
    st.session_state.show_abstractive = False
    st.session_state.show_wordcloud = False

# --- UI Styling (New Theme: Dark Blue & Gold/Teal) ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F4F7F9; /* Light background */
    }
    h1 {
        color: #1A374D; /* Dark Blue */
        text-align: center;
        padding-bottom: 10px;
        border-bottom: 2px solid #FFD700; /* Gold accent */
    }
    .result-box { 
        padding: 20px; 
        border-radius: 8px; 
        background: white; 
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); 
        margin-top: 15px;
        border-left: 5px solid #1A374D;
    }
    /* Primary button style (Run Analysis) */
    .stButton>button {
        background-color: #1A374D; /* Dark Blue */ 
        color: white; 
        font-weight: bold;
        border-radius: 8px;
        border: 2px solid #FFD700; /* Gold outline */
        padding: 10px 20px;
        box-shadow: 0 4px #0F202B; /* Darker shadow for depth */
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #2D5E7E; 
        box-shadow: 0 2px #0F202B;
        top: 2px;
    }
    /* Secondary buttons (Show/Hide results) */
    .stDownloadButton>button, .stFormSubmitButton>button, .st-emotion-cache-1c7v8eh .st-emotion-cache-1c7v8eh .st-emotion-cache-1c7v8eh button {
        background-color: #3399A8 !important; /* Teal */
        color: white !important;
        border: 1px solid #1A374D;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 6px;
        border: 1px solid #D3D3D3;
        padding: 10px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Initial Backend Setup (Silent) ---
# Use st.session_state to hold the model and vectorizer for persistence
if 'clf' not in st.session_state or 'tfidf_vectorizer' not in st.session_state:
    df, tfidf_vectorizer_init, X_train, y_train_numeric, _, _, _ = load_and_preprocess_data()

    if df is None:
        st.error("Setup failed: Could not load ML data. Check your dependencies or data source.")
        st.stop()

    clf, tfidf_vectorizer = train_and_save_models(_X_train=X_train, _y_train_numeric=y_train_numeric, _tfidf_vectorizer=tfidf_vectorizer_init)
    st.session_state.clf = clf
    st.session_state.tfidf_vectorizer = tfidf_vectorizer
else:
    clf = st.session_state.clf
    tfidf_vectorizer = st.session_state.tfidf_vectorizer


# --- Main Streamlit Interface ---
st.title("Professional Text Analysis Dashboard")
st.markdown("### 1. Input & Execution")

# --- Input Area ---
with st.form(key='analysis_form'):
    col_a, col_b = st.columns([3,1])
    
    with col_a:
        text_input = st.text_area(
            "Paste Text Here:", 
            height=200, 
            placeholder="Enter the text you wish to analyze...",
            value=st.session_state.input_text
        )

    with col_b:
        uploaded = st.file_uploader("Or Upload File (.txt):", type=["txt"])
        
        if uploaded is not None:
            try:
                file_text = uploaded.read().decode("utf-8", errors='ignore')
                text_input = file_text
                st.session_state.input_text = file_text
                st.toast("File loaded. Ready to analyze.", icon='📄')
            except Exception as e:
                st.error(f"File Read Error: {e}")
        
    run = st.form_submit_button("▶️ RUN FULL ANALYSIS", type="primary", use_container_width=True)

# --- ANALYSIS EXECUTION ---
if run:
    if not text_input or not text_input.strip():
        st.error("🚨 Please provide text to analyze.")
        st.session_state.analysis_run = False
    else:
        st.session_state.input_text = text_input
        st.session_state.show_sentiment = True # Show first result by default
        
        with st.spinner("Processing all analyses..."):
            # 1. Sentiment Analysis
            sentiment_probs, top_sentiment = analyze_sentiment_and_get_data(text_input, tfidf_vectorizer, clf)
            st.session_state.sentiment_probs = sentiment_probs
            st.session_state.top_sentiment = top_sentiment
            
            # 2. Extractive Summary
            st.session_state.extractive_sum = extractive_reduce(text_input)

            # 3. Abstractive Summary
            try:
                st.session_state.abstractive_sum = abstractive_summarize_text(text_input, model_name="t5-small")
            except Exception as e:
                error_text = str(e)
                if "Keras 3" in error_text or "tf-keras" in error_text or "No module named 'tf_keras'" in error_text:
                     st.session_state.abstractive_sum = f"Abstractive model failed: Dependency error detected. Run 'pip install tf-keras' to resolve the Keras 3/Transformers compatibility issue. Full Error: {error_text}"
                else:
                    st.session_state.abstractive_sum = f"Abstractive model failed: {error_text}"
            
            # 4. Word Cloud
            st.session_state.wc_image = generate_wc_image(text_input)
            
        st.session_state.analysis_run = True
        st.success("✅ Analysis Complete! Use the buttons below to view results.")

# --- RESULTS DISPLAY ---
if st.session_state.analysis_run:
    st.markdown("---")
    st.markdown("### 2. View Results & Report")
    
    # 2.1 Display Control Buttons
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
    
    # Sentiment Button
    if col1.button("📊 Sentiment", help="Show/Hide Sentiment Analysis"):
        st.session_state.show_sentiment = not st.session_state.show_sentiment
        
    # Extractive Button
    if col2.button("✂️ Extractive Sum.", help="Show/Hide Extractive Summary"):
        st.session_state.show_extractive = not st.session_state.show_extractive
        
    # Abstractive Button
    if col3.button("🧠 Abstractive Sum.", help="Show/Hide Abstractive Summary"):
        st.session_state.show_abstractive = not st.session_state.show_abstractive
        
    # Word Cloud Button
    if col4.button("☁️ Word Cloud", help="Show/Hide Word Cloud Visualization"):
        st.session_state.show_wordcloud = not st.session_state.show_wordcloud
        
    # PDF Download Button
    report_data = generate_report(
        st.session_state.input_text,
        st.session_state.sentiment_probs,
        st.session_state.top_sentiment,
        st.session_state.extractive_sum,
        st.session_state.abstractive_sum
    )
    col5.download_button(
        label="⬇️ Download Report (PDF/MD)",
        data=report_data,
        file_name="text_analysis_report.md",
        mime="text/markdown",
        help="Download the full analysis as a Markdown file, which can be printed to PDF."
    )
        
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    
    # 2.2 Individual Result Rendering Sections
    
    # --- Sentiment Analysis ---
    if st.session_state.show_sentiment:
        st.subheader("📊 Sentiment Analysis Results") 
        sentiment_probs = st.session_state.sentiment_probs
        top_sentiment = st.session_state.top_sentiment
        
        st.markdown(f"#### Predicted Overall Sentiment: **{top_sentiment.upper()}**")
        
        sentiment_df = pd.DataFrame({
            'Sentiment': list(sentiment_probs.keys()),
            'Probability': list(sentiment_probs.values())
        })
        color_map = {'negative': '#EF5350', 'neutral': '#FFEE58', 'positive': '#66BB6A'}
        sentiment_df['Color'] = sentiment_df['Sentiment'].map(color_map)

        # Plot the bar chart
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.bar(
            sentiment_df['Sentiment'], 
            sentiment_df['Probability'], 
            color=sentiment_df['Color'].tolist(),
            width=0.45
        )
        ax.set_title('Sentiment Probability Distribution', fontsize=10)
        ax.set_ylim(0, 1.05)
        st.pyplot(fig)
        
    
    # --- Extractive Summary ---
    if st.session_state.show_extractive:
        if st.session_state.show_sentiment: st.markdown("---")
        st.subheader("✂️ Extractive Summary (Key Sentences)")
        st.info(st.session_state.extractive_sum)

    # --- Abstractive Summary ---
    if st.session_state.show_abstractive:
        if st.session_state.show_sentiment or st.session_state.show_extractive: st.markdown("---")
        st.subheader("🧠 Abstractive Summary (AI Paraphrase)")
        if "Abstractive model failed" in st.session_state.abstractive_sum:
            st.error(st.session_state.abstractive_sum)
        else:
            st.info(st.session_state.abstractive_sum)

    # --- Word Cloud ---
    if st.session_state.show_wordcloud:
        if st.session_state.show_sentiment or st.session_state.show_extractive or st.session_state.show_abstractive: st.markdown("---")
        st.subheader("☁️ Word Cloud Visualization")
        st.image(st.session_state.wc_image, caption='Word Frequency Cloud (Processed Text)', use_column_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
