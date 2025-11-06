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
from sklearn.linear_model import SGDClassifier
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
    # Trains the SGDClassifier model
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    clf = SGDClassifier(loss='log_loss', penalty='l2', random_state=RANDOM_STATE, learning_rate='adaptive', eta0=0.01)
    classes = np.unique(_y_train_numeric)
    
    for epoch in range(50):
        perm = np.random.permutation(_X_train.shape[0])
        X_shuf, y_shuf = _X_train[perm], _y_train_numeric.iloc[perm]
        clf.partial_fit(X_shuf, y_shuf, classes=classes)
        
    joblib.dump(_tfidf_vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(clf, os.path.join(MODEL_DIR, 'sgd_sentiment_classifier.pkl'))
    
    return clf, _tfidf_vectorizer

# --- 3. Sentiment Prediction Function (Using ML Model) ---
def analyze_sentiment_and_get_data(text, vectorizer, classifier):
    cleaned_text = clean_text_util(text)
    text_vec = vectorizer.transform([cleaned_text])
    probabilities = classifier.predict_proba(text_vec)[0]

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
st.set_page_config(layout="centered", page_title="Text Analysis Dashboard")

# --- UI Styling (For clean results box and primary button) ---
st.markdown(
    """
    <style>
    .result-box { 
        padding: 18px; 
        border-radius: 6px; 
        background: var(--secondary-background-color); 
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); 
        margin-bottom: 20px;
        border: 1px solid var(--primary-color);
    }
    .stButton>button {
        background-color: var(--primary-color); 
        color: white; 
        font-weight: bold;
        border-radius: 4px;
        border: none;
        padding: 10px 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True
)


# --- Initial Backend Setup (Silent) ---
df, tfidf_vectorizer_init, X_train, y_train_numeric, _, _, _ = load_and_preprocess_data()

if df is None:
    st.error("Setup failed: Could not load ML data. Check your Kaggle API setup or data file path.")
    st.stop()

clf, tfidf_vectorizer = train_and_save_models(_X_train=X_train, _y_train_numeric=y_train_numeric, _tfidf_vectorizer=tfidf_vectorizer_init)


# --- Main Streamlit Interface ---
st.title("Welcome to Text Analysis world 😶‍🌫️")
st.info("Input text and click 'Run Full Analysis' for sentiment, summarization, and word cloud visualization.")

# --- Input Area (Single Button & UI Structure) ---
with st.form(key='analysis_form'):
    st.header("1. Text Input")

    col_a, col_b = st.columns([3,1])
    
    if 'default_text_input' not in st.session_state: st.session_state.default_text_input = ""

    with col_a:
        text_input = st.text_area(
            "Paste Text Here:", 
            height=260, 
            placeholder="Enter the text you wish to analyze...",
            value=st.session_state.default_text_input
        )

    with col_b:
        uploaded = st.file_uploader("Or Upload File (.txt):", type=["txt"])
        
        if uploaded is not None:
            try:
                file_text = uploaded.read().decode("utf-8", errors='ignore')
                text_input = file_text
                st.session_state.default_text_input = file_text
                st.toast("File loaded. Ready to analyze.", icon='📄')
            except Exception as e:
                st.error(f"File Read Error: {e}")
        
    # ONLY ONE BUTTON
    run = st.form_submit_button("Run Full Analysis", type="primary", use_container_width=True)

# --- RESULTS DISPLAY ---
if run:
    if not text_input or not text_input.strip():
        st.error("🚨 Please provide text to analyze.")
    else:
        st.session_state.default_text_input = text_input 
        
        st.markdown("---")
        st.header("2. Analysis Results")
        st.markdown('<div class="result-box">', unsafe_allow_html=True)

        
        # --- Sentiment Analysis (ML Model + Custom Bar Styling) ---
        st.subheader("Sentiment Analysis") 
        
        with st.spinner("Running Inference on Trained Model..."):
            sentiment_probs, top_sentiment = analyze_sentiment_and_get_data(text_input, tfidf_vectorizer, clf)
            
            st.markdown(f"#### Predicted Overall Sentiment: **{top_sentiment.upper()}**")
            
            # Bar Chart Styling (Medium size, Thinner bars)
            sentiment_df = pd.DataFrame({
                'Sentiment': list(sentiment_probs.keys()),
                'Probability': list(sentiment_probs.values())
            })
            color_map = {'negative': '#EF5350', 'neutral': '#FFEE58', 'positive': '#66BB6A'}
            sentiment_df['Color'] = sentiment_df['Sentiment'].map(color_map)

            fig, ax = plt.subplots(figsize=(6, 4)) # Medium Size
            
            bars = ax.bar(
                sentiment_df['Sentiment'], 
                sentiment_df['Probability'], 
                color=sentiment_df['Color'].tolist(),
                width=0.45  # Controlled bar width
            )
            ax.set_title('Sentiment Probability Distribution')
            ax.set_ylim(0, 1.05)
            st.pyplot(fig)


        # --- Summarization ---
        st.markdown("---")
        st.subheader("Text Summarization")
        col_ext, col_abs = st.columns(2)
        
        with col_ext:
            st.markdown("##### ✂️ Extractive Summary (Key Sentences)")
            with st.spinner("Generating Extractive Summary..."):
                extractive_sum = extractive_reduce(text_input)
                st.info(extractive_sum)

        with col_abs:
            st.markdown("##### 🧠 Abstractive Summary (AI Paraphrase)") # No model name
            with st.spinner("Generating Abstractive Summary..."):
                try:
                    abstractive_sum = abstractive_summarize_text(text_input, model_name="t5-small")
                    st.info(abstractive_sum)
                except Exception as e:
                    error_text = str(e)
                    
                    # --- GUARANTEED ERROR MESSAGE ---
                    if "Keras 3" in error_text or "tf-keras" in error_text or "No module named 'tf_keras'" in error_text:
                        st.error("Abstractive model failed: Dependency error detected. Run pip install tf-keras to resolve the Keras 3/Transformers compatibility issue.")
                    else:
                        st.error(f"Abstractive model failed: {error_text}")

            
        # --- Word Cloud ---
        st.markdown("---")
        st.subheader("Word Cloud Visualization")
        with st.spinner("Generating Word Cloud..."):
            wc_image = generate_wc_image(text_input)
            st.image(wc_image, caption='Word Frequency Cloud (Processed Text)', use_column_width=True)
            
        # Topic Modeling Insights (LDA) REMOVED
        st.markdown('</div>', unsafe_allow_html=True)

