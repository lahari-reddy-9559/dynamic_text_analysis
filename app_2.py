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
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# --- Import Utilities ---
try:
    # This line MUST be present for summarization to work
    from summarization_utils import clean_text as clean_text_util, extractive_reduce, abstractive_summarize_text
except ImportError:
    st.error("Could not find summarization_utils.py. Please ensure it is in the same directory.")
    st.stop() # Stop execution if utils file is missing

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# --- Configuration ---
MODEL_DIR = 'models'
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
MAX_FEATURES = 5000
RANDOM_STATE = 42

# --- 1. Data Download, Preprocessing, and Model Training (Hidden) ---
@st.cache_data
def load_and_preprocess_data():
    try:
        # Download dataset from Kaggle Hub
        path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
        file_path = os.path.join(path, 'train.csv')
        df = pd.read_csv(file_path, encoding='latin-1')
    except Exception as e:
        return None, None, None, None, None, None, None

    df.dropna(subset=['text', 'selected_text'], inplace=True)

    # NLTK Downloads Check
    for package in ['punkt', 'stopwords', 'wordnet']:
        try: nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except LookupError: nltk.download(package, quiet=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_text_local(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    df['cleaned_text'] = df['text'].apply(clean_text_local)
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    tfidf_vectorizer.fit(df['cleaned_text'])
    tfidf_matrix = tfidf_vectorizer.transform(df['cleaned_text'])
    
    # Split data
    X_train, X_test, y_train_str, y_test_str = train_test_split(
        tfidf_matrix, df['sentiment'], test_size=0.2, random_state=RANDOM_STATE, stratify=df['sentiment']
    )
    
    y_train_numeric = pd.Series(y_train_str).map(sentiment_mapping).astype(int)

    return df, tfidf_vectorizer, X_train, y_train_numeric, tfidf_matrix, X_test, y_test_str

@st.cache_resource
def train_and_save_models(_X_train, y_train_numeric, _tfidf_vectorizer):
    # This function is completely silent
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # --- Train SGD Classifier ---
    clf = SGDClassifier(loss='log_loss', penalty='l2', random_state=RANDOM_STATE, learning_rate='adaptive', eta0=0.01)
    classes = np.unique(y_train_numeric)
    
    for epoch in range(50):
        perm = np.random.permutation(_X_train.shape[0])
        X_shuf, y_shuf = _X_train[perm], y_train_numeric.iloc[perm]
        clf.partial_fit(X_shuf, y_shuf, classes=classes)
        
    # --- Save Artifacts ---
    joblib.dump(_tfidf_vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(clf, os.path.join(MODEL_DIR, 'sgd_sentiment_classifier.pkl'))
    joblib.dump(sentiment_mapping, os.path.join(MODEL_DIR, 'sentiment_mapping.pkl'))
    
    return clf, _tfidf_vectorizer

# --- 2. UI Helper Functions (analyze_sentiment_and_get_data and generate_wc_image unchanged) ---
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

# --- 3. Streamlit Application Execution ---
st.set_page_config(layout="wide", page_title="Sentiment & Analysis Pipeline")

# --- Initial Setup Run (Silent) ---
df, tfidf_vectorizer_init, X_train, y_train_numeric, tfidf_matrix, X_test, y_test_str = load_and_preprocess_data()

if df is None:
    st.error("Setup failed: Could not load data. Check your Kaggle API setup or data file path.")
    st.stop()

# Execute training and saving.
clf, tfidf_vectorizer = train_and_save_models(_X_train=X_train, y_train_numeric=y_train_numeric, _tfidf_vectorizer=tfidf_vectorizer_init)

# --- Main Streamlit Interface ---
st.title("Complete Text Analysis Dashboard 📊")

# --- Input Area ---
st.header("Text Input")
input_text = st.text_area("Paste Text Here:", height=200, key="text_input")
uploaded_file = st.file_uploader("Or Upload Text File (.txt):", type=['txt'])

if uploaded_file is not None:
    file_contents = uploaded_file.read().decode("utf-8")
    input_text = file_contents
    st.info(f"File '{uploaded_file.name}' loaded.")

# --- Single Button Logic ---
if st.button("Run Full Analysis", type="primary", use_container_width=True):
    if not input_text:
        st.error("Please provide text or upload a file to analyze.")
    else:
        st.header("--- Analysis Results ---")
        
        # --- Sentiment Analysis ---
        st.subheader("Sentiment Analysis") 
        
        with st.spinner("Running Inference..."):
            sentiment_probs, top_sentiment = analyze_sentiment_and_get_data(input_text, tfidf_vectorizer, clf)
            
            st.markdown(f"#### Predicted Overall Sentiment: **{top_sentiment.upper()}**")
            
            # Bar Chart Styling (Medium size, Thinner bars)
            sentiment_df = pd.DataFrame({
                'Sentiment': list(sentiment_probs.keys()),
                'Probability': list(sentiment_probs.values())
            })
            color_map = {'negative': '#EF5350', 'neutral': '#FFEE58', 'positive': '#66BB6A'}
            sentiment_df['Color'] = sentiment_df['Sentiment'].map(color_map)

            # Medium size fig, thinner bars (width=0.45)
            fig, ax = plt.subplots(figsize=(6, 4)) 
            
            bars = ax.bar(
                sentiment_df['Sentiment'], 
                sentiment_df['Probability'], 
                color=sentiment_df['Color'].tolist(),
                width=0.45  
            )
            ax.set_title('Sentiment Probability Distribution')
            ax.set_ylim(0, 1.05)
            st.pyplot(fig)

        # --- Summarization ---
        st.subheader("Text Summarization")
        col_ext, col_abs = st.columns(2)
        
        with col_ext:
            st.
