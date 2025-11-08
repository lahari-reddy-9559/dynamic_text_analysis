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

# --- Import Summarization Utilities (Requires summarization_utils.py) ---
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
    # Uses Kaggle Data, TF-IDF, and Lemmatization
    try:
        path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
        file_path = os.path.join(path, 'train.csv')
        df = pd.read_csv(file_path, encoding='latin-1')
    except Exception as e:
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
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    X_train_dense = _X_train.toarray()
    clf.fit(X_train_dense, _y_train_numeric)
    
    joblib.dump(_tfidf_vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(clf, os.path.join(MODEL_DIR, 'random_forest_sentiment_classifier.pkl'))
    
    return clf, _tfidf_vectorizer

# --- 3. Sentiment Prediction Function (Using ML Model) ---
def analyze_sentiment_and_get_data(text, vectorizer, classifier):
    cleaned_text = clean_text_util(text)
    text_vec = vectorizer.transform([cleaned_text])
    text_vec_dense = text_vec.toarray()
    probabilities = classifier.predict_proba(text_vec_dense)[0]
    
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


# --- 5. Streamlit Application Execution (UI Focus) ---
st.set_page_config(layout="centered", page_title="Advanced Text Analyzer")

# --- UI Styling (Aesthetic Neutral) ---
st.markdown(
    """
    <style>
    /* Aesthetic, Neutral Theme Configuration */
    /* Adjust these colors for a different neutral palette */
    :root {
        --primary-color: #6A6E73; /* Dark Grey/Slate for accent */
        --secondary-background-color: #F8F8F8; /* Light Grey background for containers */
        --background-color: #FFFFFF; /* White main background */
        --text-color: #333333; /* Dark text */
        --border-color: #DDDDDD; /* Light border */
    }

    /* Apply custom theme variables */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    .main .block-container {
        padding-top: 2.5rem; 
        padding-bottom: 2.5rem;
    }
    
    /* Result 'Card' Style - Soft, rounded, and subtle */
    .result-card { 
        padding: 20px; 
        border-radius: 10px; 
        background-color: var(--secondary-background-color); 
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05); /* Soft shadow */
        margin-bottom: 25px;
        border: 1px solid var(--border-color);
    }
    
    /* Input Area Box Style - Clean separation */
    .input-box {
        padding: 25px;
        border-radius: 10px;
        background-color: #FFFFFF; /* Pure white input section */
        border: 1px solid var(--border-color);
        margin-bottom: 30px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    /* Primary Button Style - Full width for main action, with neutral accent */
    .stButton>button {
        background-color: var(--primary-color); 
        color: white; 
        font-weight: 600;
        border-radius: 6px;
        border: none;
        padding: 12px 20px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #5A5E63; /* Slightly darker on hover */
    }
    
    /* Secondary/Action Buttons - Lighter look for view/download */
    .stDownloadButton button, .stButton button[kind="secondary"] {
        background-color: #EFEFEF !important;
        color: var(--primary-color) !important;
        border: 1px solid #CCCCCC !important;
        box-shadow: none;
        padding: 8px 15px;
    }

    h1 {
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }

    /* Subheadings/Section titles - Clean, understated border */
    h3 {
        margin-top: 0;
        padding-bottom: 5px;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 15px;
    }
    
    </style>
    """, unsafe_allow_html=True
)

# --- Session State Management (Crucial for the "View" buttons) ---
if 'results_data' not in st.session_state:
    st.session_state.results_data = {
        'run_complete': False,
        'sentiment_probs': None,
        'top_sentiment': None,
        'extractive_sum': None,
        'abstractive_sum': "Run analysis first.",
        'wordcloud_image': None,
    }

# --- Initial Backend Setup (Silent) ---
if 'clf' not in st.session_state or 'tfidf_vectorizer' not in st.session_state:
    df, tfidf_vectorizer_init, X_train, y_train_numeric, _, _, _ = load_and_preprocess_data()

    if df is None:
        st.error("Setup failed: Could not load ML data. Check your Kaggle API setup or data file path.")
        st.stop()

    clf, tfidf_vectorizer = train_and_save_models(_X_train=X_train, _y_train_numeric=y_train_numeric, _tfidf_vectorizer=tfidf_vectorizer_init)
    st.session_state.clf = clf
    st.session_state.tfidf_vectorizer = tfidf_vectorizer
else:
    clf = st.session_state.clf
    tfidf_vectorizer = st.session_state.tfidf_vectorizer


# --- Main Streamlit Interface ---
st.title("Advanced Text Analysis Platform")
st.caption("A tool developed by Lahari Reddy. Input data, analyze, and view the integrated ML and NLP results.")

st.markdown("---")

# --- 1. Input Section ---
st.header("1. Data Input")

st.markdown('<div class="input-box">', unsafe_allow_html=True)
with st.form(key='analysis_form'):
    col_a, col_b = st.columns([3,1])
    
    if 'default_text_input' not in st.session_state: st.session_state.default_text_input = ""

    with col_a:
        text_input = st.text_area(
            "Text Payload:", 
            height=200, 
            placeholder="Paste text for analysis here...",
            value=st.session_state.default_text_input
        )

    with col_b:
        st.markdown("##### Load Option")
        uploaded = st.file_uploader("Upload .txt File:", type=["txt"])
        
        if uploaded is not None:
            try:
                file_text = uploaded.read().decode("utf-8", errors='ignore')
                text_input = file_text
                st.session_state.default_text_input = file_text
                st.toast("File loaded successfully.", icon='✅')
            except Exception as e:
                st.error(f"File Read Error: {e}")
        
    st.markdown("---")
    # This is the single button that triggers the full pipeline
    run = st.form_submit_button("▶️ Run Complete Analysis", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# --- 2. Analysis Execution ---
if run:
    if not text_input or not text_input.strip():
        st.session_state.results_data['run_complete'] = False
        st.warning("Please input text or upload a file to run the analysis.")
    else:
        st.session_state.default_text_input = text_input 
        
        # --- EXECUTE ALL ANALYSES AND STORE IN SESSION STATE ---
        with st.spinner("Processing data through ML/NLP pipeline..."):
            
            # Sentiment
            sentiment_probs, top_sentiment = analyze_sentiment_and_get_data(text_input, tfidf_vectorizer, clf)
            
            # Summarization
            extractive_sum = extractive_reduce(text_input)
            try:
                abstractive_sum = abstractive_summarize_text(text_input, model_name="t5-small")
            except Exception as e:
                error_text = str(e)
                if "Keras 3" in error_text or "tf-keras" in error_text:
                    abstractive_sum = "Abstractive Model Failed: Dependency error (tf-keras/Keras 3 issue)."
                else:
                    abstractive_sum = f"Abstractive Model Failed: {error_text[:60]}..."

            # Word Cloud
            wc_image = generate_wc_image(text_input)

            # Store results
            st.session_state.results_data.update({
                'run_complete': True,
                'sentiment_probs': sentiment_probs,
                'top_sentiment': top_sentiment,
                'extractive_sum': extractive_sum,
                'abstractive_sum': abstractive_sum,
                'wordcloud_image': wc_image,
            })
            
        st.success("✅ Analysis Complete! See Section 2 for results.")
        st.balloons()


# --- 3. Results Section (Action-Driven) ---
st.header("2. Analysis Results")
st.caption("Click the buttons below to view the detailed output for each analysis.")

# Use columns for action buttons
col1, col2, col3, col4 = st.columns(4)

if st.session_state.results_data['run_complete']:
    
    # --- Action Buttons ---
    sentiment_button = col1.button("View Sentiment 📈", key="view_sent")
    summary_button = col2.button("View Summaries ✂️", key="view_sum")
    wordcloud_button = col3.button("View Word Cloud 🖼️", key="view_wc")
    
    # Placeholder for Download Button
    download_button = col4.button("Download Report 📄", key="download_report")
    
    # --- Action Logic ---
    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    if sentiment_button:
        st.subheader("Sentiment Classification (Random Forest)")
        sentiment_probs = st.session_state.results_data['sentiment_probs']
        top_sentiment = st.session_state.results_data['top_sentiment']

        st.markdown(f"**Primary Classification:** <span style='font-size: 1.2em; font-weight: bold;'>{top_sentiment.upper()}</span>", unsafe_allow_html=True)
        
        # Bar Chart Visualization
        sentiment_df = pd.DataFrame({
            'Sentiment': list(sentiment_probs.keys()),
            'Probability': list(sentiment_probs.values())
        })
        order = ['positive', 'neutral', 'negative']
        sentiment_df = sentiment_df.set_index('Sentiment').reindex(order).reset_index()

        color_map = {'negative': '#EF5350', 'neutral': '#FFEE58', 'positive': '#66BB6A'}
        sentiment_df['Color'] = sentiment_df['Sentiment'].map(color_map)

        fig, ax = plt.subplots(figsize=(6, 3.5)) 
        ax.bar(
            sentiment_df['Sentiment'], 
            sentiment_df['Probability'], 
            color=sentiment_df['Color'].tolist(),
            width=0.5
        )
        ax.set_title('Probability Distribution', fontsize=12, color=st.get_option("theme.text_color"))
        ax.set_ylim(0, 1.05)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        st.pyplot(fig)
        
    elif summary_button:
        st.subheader("Text Summarization Results")
        col_ext, col_abs = st.columns(2)
        
        with col_ext:
            st.markdown("##### Extractive Summary (Key Sentence Extraction)")
            st.code(st.session_state.results_data['extractive_sum'], language='text')

        with col_abs:
            st.markdown("##### Abstractive Summary (T5 Model Output)") 
            st.code(st.session_state.results_data['abstractive_sum'], language='text')

    elif wordcloud_button:
        st.subheader("Keyword Visualization (Word Cloud)")
        st.image(st.session_state.results_data['wordcloud_image'], caption='Term Frequency Map', use_column_width=True)

    elif download_button:
        st.subheader("Report Download Placeholder")
        st.info("The 'Download Report' functionality would be implemented here, generating a PDF or structured JSON/CSV file from the stored session state data.")
        st.code("def generate_pdf_report(data):\n    # Implementation using libraries like FPDF or ReportLab\n    pass", language="python")

    else:
        st.info("Click a button above to display the analysis output.")
        
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Run the analysis pipeline in Section 1 to generate results.")
