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
# Extractive reduce now calls the abstractive model with short length
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

# --- 2. ML Backend Functions (SILENT EXECUTION) ---

@st.cache_data(show_spinner="Preparing data and models (This may take a moment)...")
def load_and_preprocess_data():
    try:
        path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
        file_path = os.path.join(path, 'train.csv')
        df = pd.read_csv(file_path, encoding='latin-1')
    except Exception as e:
        st.error(f"Data Load Error: Failed to load data from KaggleHub. ({e})")
        return None, None, None, None, None, None, None

    # Handle NLTK setup in the data loading function
    for package in ['punkt', 'stopwords', 'wordnet']:
        try: nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except LookupError: nltk.download(package, quiet=True)
    
    df.dropna(subset=['text', 'selected_text'], inplace=True)

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
    try:
        if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        X_train_dense = _X_train.toarray()
        clf.fit(X_train_dense, _y_train_numeric)
        
        joblib.dump(_tfidf_vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
        joblib.dump(clf, os.path.join(MODEL_DIR, 'random_forest_sentiment_classifier.pkl'))
        
        return clf, _tfidf_vectorizer
    except Exception as e:
        st.error(f"Model Training Error: Failed to train or save model. ({e})")
        return None, None

# --- 3. Sentiment Prediction Function (Using ML Model) ---
def analyze_sentiment_and_get_data(text, vectorizer, classifier):
    try:
        cleaned_text = clean_text_util(text)
        if not cleaned_text:
             return {"Error": 1.0}, "SENTIMENT FAILED: Input text is empty after cleaning."
             
        text_vec = vectorizer.transform([cleaned_text])
        text_vec_dense = text_vec.toarray()
        probabilities = classifier.predict_proba(text_vec_dense)[0]
        
        sentiment_data = {}
        for i, label_id in enumerate(classifier.classes_):
            label_name = reverse_sentiment_mapping.get(label_id, f'Unknown_{label_id}')
            sentiment_data[label_name] = float(probabilities[i])
            
        top_sentiment_label = reverse_sentiment_mapping.get(classifier.classes_[np.argmax(probabilities)])
        
        return sentiment_data, top_sentiment_label
    except Exception as e:
        return {"Error": 1.0}, f"SENTIMENT FAILED: {str(e)[:50]}..."

# --- 4. Word Cloud Function ---
def generate_wc_image(text):
    try:
        cleaned_text = clean_text_util(text)
        if not cleaned_text:
            return Image.new('RGB', (800, 400), color = '#1E1E1E')

        word_freq = pd.Series(cleaned_text.split()).value_counts().to_dict()

        # WordCloud config for dark theme
        wc = WordCloud(width=800, height=400, background_color='#1E1E1E', colormap='Wistia').generate_from_frequencies(word_freq)
        img_io = io.BytesIO()
        wc.to_image().save(img_io, 'PNG')
        img_io.seek(0)
        return Image.open(img_io)
    except Exception as e:
        st.error(f"WordCloud failed: {e}")
        return Image.new('RGB', (800, 400), color = 'red')


# --- 5. Report Generation Function ---
def generate_report(text, sentiment_probs, top_sentiment, extractive_sum, abstractive_sum):
    """Generates a Markdown report string for download."""
    report_content = io.StringIO()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_content.write(f"# Professional Text Analysis Report\n")
    report_content.write(f"**Date:** {now}\n")
    report_content.write(f"**Total Characters:** {len(text)}\n\n")
    report_content.write("---\n")
    
    # 1. Sentiment
    report_content.write("## 1. Sentiment Analysis\n")
    report_content.write(f"**Predicted Overall Sentiment:** **{top_sentiment.upper()}**\n")
    report_content.write("| Sentiment | Probability |\n")
    report_content.write("| :--- | :--- |\n")
    for sent, prob in sorted(sentiment_probs.items(), key=lambda item: item[1], reverse=True):
        report_content.write(f"| {sent.capitalize()} | {prob:.4f} |\n")
    report_content.write("\n")
    
    # 2. Summaries
    report_content.write("## 2. Text Summarization\n")
    report_content.write("### Key Insights (Short Abstractive Reduction)\n")
    report_content.write(f"> {extractive_sum}\n\n")
    
    report_content.write("### Detailed Abstract (Abstractive Summary)\n")
    report_content.write(f"> {abstractive_sum}\n\n")
    
    # 3. Word Cloud 
    report_content.write("## 3. Word Cloud Visualization\n")
    report_content.write("*(Visualization is available in the web application and is based on word frequency.)*\n\n")

    # 4. Original Text
    report_content.write("---\n")
    report_content.write("## 4. Original Input Text\n")
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
    st.session_state.active_view = None # Stores the key of the currently active view ('sentiment', 'wordcloud', etc.)

# --- UI Styling (High Contrast Dark Theme) ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #121212; /* Dark background */
        color: #E0E0E0; /* Light text */
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown, .css-q8sbsg p {
        color: #E0E0E0 !important; /* Ensure all text is light */
    }
    .result-box { 
        padding: 20px; 
        border-radius: 8px; 
        background: #1E1E1E; /* Slightly lighter dark for contrast */
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4); 
        margin-top: 20px;
        border: 2px solid #00BFFF; /* Deep sky blue accent */
    }
    /* Primary button style (Run Analysis) */
    .stButton>button {
        background-color: #00BFFF; /* Deep Sky Blue */ 
        color: black; 
        font-weight: bold;
        border-radius: 6px;
        border: none;
        padding: 10px 20px;
        box-shadow: 0 4px #0077A0; 
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #009ACD; 
        box-shadow: 0 2px #0077A0;
    }
    /* Button Bar (Toggle Buttons) - Targets the Streamlit button container */
    .st-emotion-cache-1r6500u button { 
        background-color: #333333 !important; /* Dark Grey for Toggles */
        color: #00BFFF !important; /* Blue text */
        border: 1px solid #00BFFF;
        box-shadow: none !important;
        font-weight: normal;
        margin: 5px 2px;
    }
    /* Streamlit's st.info box (for summaries) */
    div[data-testid="stText"] {
        color: #00BFFF !important; 
        background-color: #1E1E1E !important;
        border-radius: 6px;
        padding: 15px;
        border: 1px solid #333333;
    }
    /* Fix Streamlit's info background */
    div[data-testid="stAlert"] > div {
        background-color: #1E1E1E !important;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Initial Backend Setup (Silent) ---
if 'clf' not in st.session_state or 'tfidf_vectorizer' not in st.session_state:
    df, tfidf_vectorizer_init, X_train, y_train_numeric, _, _, _ = load_and_preprocess_data()

    if df is None:
        st.stop() 

    clf, tfidf_vectorizer = train_and_save_models(_X_train=X_train, _y_train_numeric=y_train_numeric, _tfidf_vectorizer=tfidf_vectorizer_init)
    
    if clf is None or tfidf_vectorizer is None:
        st.stop() 
        
    st.session_state.clf = clf
    st.session_state.tfidf_vectorizer = tfidf_vectorizer
else:
    clf = st.session_state.clf
    tfidf_vectorizer = st.session_state.tfidf_vectorizer


# --- Main Streamlit Interface ---
st.title("Professional Text Analysis Dashboard")

# --- Input Area ---
st.markdown("### 1. Input Text")
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
        
    run = st.form_submit_button("🚀 RUN FULL ANALYSIS", type="primary", use_container_width=True)

# --- ANALYSIS EXECUTION ---
if run:
    if not text_input or not text_input.strip():
        st.error("🚨 Please provide text to analyze.")
        st.session_state.analysis_run = False
    else:
        st.session_state.input_text = text_input
        # KEY FIX 1: Ensure no view is active upon starting the analysis
        st.session_state.active_view = None
        
        with st.spinner("Processing all analyses..."):
            # 1. Sentiment Analysis
            sentiment_probs, top_sentiment = analyze_sentiment_and_get_data(text_input, tfidf_vectorizer, clf)
            st.session_state.sentiment_probs = sentiment_probs
            st.session_state.top_sentiment = top_sentiment
            
            # 2. Key Insights (Transformer-based reduction)
            try:
                st.session_state.extractive_sum = extractive_reduce(text_input)
            except Exception as e:
                st.session_state.extractive_sum = f"KEY INSIGHTS FAILED: {str(e)}"
            
            # 3. Abstractive Summary
            try:
                st.session_state.abstractive_sum = abstractive_summarize_text(text_input, model_name="t5-small")
            except Exception as e:
                st.session_state.abstractive_sum = f"ABSTRACTIVE SUMMARY FAILED: {str(e)}"
            
            # 4. Word Cloud
            st.session_state.wc_image = generate_wc_image(text_input)
            
        st.session_state.analysis_run = True
        st.success("✅ Analysis Complete! Use the buttons below to view results.")

# --- Results Toggle Logic ---
def set_active_view(view_key):
    """Sets the active view key, toggling off if already active."""
    if st.session_state.active_view == view_key:
        st.session_state.active_view = None
    else:
        st.session_state.active_view = view_key
        
# --- RESULTS DISPLAY ---
if st.session_state.analysis_run:
    st.markdown("### 2. Analysis Report")
    
    # 2.1 Display Control Buttons
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1.2])
    
    # Sentiment Button
    if col1.button("📊 Sentiment", key="btn_sent"):
        set_active_view('sentiment')
        
    # Extractive Button
    if col2.button("🔑 Key Insights", key="btn_ext"):
        set_active_view('extractive')
        
    # Abstractive Button
    if col3.button("🧠 Abstractive", key="btn_abs"):
        set_active_view('abstractive')
        
    # Word Cloud Button
    if col4.button("☁️ Word Cloud", key="btn_wc"):
        set_active_view('wordcloud')
        
    # PDF Download Button
    report_data = generate_report(
        st.session_state.input_text,
        st.session_state.sentiment_probs,
        st.session_state.top_sentiment,
        st.session_state.extractive_sum,
        st.session_state.abstractive_sum,
    )
    col5.download_button(
        label="⬇️ Download Report (MD)",
        data=report_data,
        file_name="text_analysis_report.md",
        mime="text/markdown",
        help="Download the full analysis as a Markdown file (can be printed to PDF)."
    )
        
    # 2.2 Individual Result Rendering Sections
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    
    active_view = st.session_state.active_view

    # --- Sentiment Analysis ---
    if active_view == 'sentiment':
        st.subheader("📊 Sentiment Analysis Results") 
        sentiment_probs = st.session_state.sentiment_probs
        top_sentiment = st.session_state.top_sentiment
        
        st.markdown(f"#### Predicted Overall Sentiment: **{top_sentiment.upper()}**")
        
        sentiment_df = pd.DataFrame({
            'Sentiment': list(sentiment_probs.keys()),
            'Probability': list(sentiment_probs.values())
        })
        color_map = {'negative': '#EF5350', 'neutral': '#FFEE58', 'positive': '#66BB6A', 'Error': '#FF4500'}
        sentiment_df['Color'] = sentiment_df['Sentiment'].map(color_map)

        # Plot the bar chart
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.bar(
            sentiment_df['Sentiment'], 
            sentiment_df['Probability'], 
            color=sentiment_df['Color'].tolist(),
            width=0.6 
        )
        ax.set_title('Sentiment Probability Distribution', color='#E0E0E0', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_facecolor('#1E1E1E')
        fig.patch.set_facecolor('#1E1E1E')
        ax.tick_params(axis='x', colors='#E0E0E0')
        ax.tick_params(axis='y', colors='#E0E0E0')
        ax.spines['bottom'].set_color('#E0E0E0')
        ax.spines['top'].set_color('#1E1E1E')
        ax.spines['left'].set_color('#E0E0E0')
        ax.spines['right'].set_color('#1E1E1E')
        st.pyplot(fig)
    
    # --- Key Insights (Short Abstractive Reduction) ---
    elif active_view == 'extractive':
        st.subheader("🔑 Key Insights (Short Abstractive Reduction)")
        if st.session_state.extractive_sum.startswith("KEY INSIGHTS FAILED") or "Transformer model failed" in st.session_state.extractive_sum:
            st.error(st.session_state.extractive_sum)
        else:
            st.info(st.session_state.extractive_sum)

    # --- Abstractive Summary ---
    elif active_view == 'abstractive':
        st.subheader("🧠 Detailed Abstract (Abstractive Summary)")
        if st.session_state.abstractive_sum.startswith("ABSTRACTIVE SUMMARY FAILED") or "Transformer model failed" in st.session_state.abstractive_sum:
            st.error(st.session_state.abstractive_sum)
        else:
            st.info(st.session_state.abstractive_sum)

    # --- Word Cloud ---
    elif active_view == 'wordcloud':
        st.subheader("☁️ Word Cloud Visualization")
        st.image(st.session_state.wc_image, caption='Word Frequency Cloud (Processed Text)', use_column_width=True)
        
    else:
         st.info("Click a toggle button above to display the analysis results.")

    st.markdown('</div>', unsafe_allow_html=True)
