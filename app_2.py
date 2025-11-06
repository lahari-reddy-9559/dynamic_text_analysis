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
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf # Keep import for dependency list completeness

# --- Import Utilities ---
try:
    from summarization_utils import clean_text as clean_text_util, extractive_reduce, abstractive_summarize_text
except ImportError:
    st.error("Could not find summarization_utils.py. Please ensure it is in the same directory.")

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# --- Configuration ---
MODEL_DIR = 'models'
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
MAX_FEATURES = 5000
RANDOM_STATE = 42

# --- 1. Data Download, Preprocessing, and Model Training ---
@st.cache_data
def load_and_preprocess_data():
    st.text("Step 1/3: Downloading Data and Setting up NLTK...")
    try:
        # Download dataset from Kaggle Hub
        path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
        file_path = os.path.join(path, 'train.csv')
        df = pd.read_csv(file_path, encoding='latin-1')
    except Exception as e:
        st.error(f"Data download failed. Ensure Kaggle API credentials are set up. Error: {e}")
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
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])
    
    # Split data
    X_train, X_test, y_train_str, y_test_str = train_test_split(
        tfidf_matrix, df['sentiment'], test_size=0.2, random_state=RANDOM_STATE, stratify=df['sentiment']
    )
    
    y_train_numeric = pd.Series(y_train_str).map(sentiment_mapping).astype(int)

    return df, tfidf_vectorizer, X_train, y_train_numeric, tfidf_matrix, X_test, y_test_str

@st.cache_resource
# FIX: Renamed _X_train and _tfidf_vectorizer to avoid hashing issues with sparse matrix and Scikit-learn object
def train_and_save_models(_X_train, y_train_numeric, _tfidf_vectorizer):
    st.text("Step 2/3: Training Models and Saving Artifacts...")
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # --- Train SGD Classifier ---
    clf = SGDClassifier(loss='log_loss', penalty='l2', random_state=RANDOM_STATE, learning_rate='adaptive', eta0=0.01)
    classes = np.unique(y_train_numeric)
    
    # Epoch training loop
    for epoch in range(50):
        perm = np.random.permutation(_X_train.shape[0])
        X_shuf, y_shuf = _X_train[perm], y_train_numeric.iloc[perm]
        clf.partial_fit(X_shuf, y_shuf, classes=classes)
        
    # --- Save Artifacts ---
    joblib.dump(_tfidf_vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(clf, os.path.join(MODEL_DIR, 'sgd_sentiment_classifier.pkl'))
    joblib.dump(sentiment_mapping, os.path.join(MODEL_DIR, 'sentiment_mapping.pkl'))
    
    st.success("Models trained and saved to 'models/'.")
    
    return clf, _tfidf_vectorizer

# --- 2. UI Helper Functions ---

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
st.set_page_config(layout="wide", page_title="Full Deployment Pipeline")

# --- Initial Setup Run ---
df, tfidf_vectorizer_init, X_train, y_train_numeric, tfidf_matrix, X_test, y_test_str = load_and_preprocess_data()

if df is None:
    st.stop()

# Execute training and saving. Note the arguments passed match the corrected function signature.
clf, tfidf_vectorizer = train_and_save_models(_X_train=X_train, y_train_numeric=y_train_numeric, _tfidf_vectorizer=tfidf_vectorizer)

# --- Main Streamlit Interface ---
st.text("Step 3/3: Launching User Interface...")
st.title("Complete Sentiment Analysis Dashboard 📊")

# --- Input Area ---
st.header("Text Input")
input_text = st.text_area("Paste Text Here:", height=200, key="text_input")
uploaded_file = st.file_uploader("Or Upload Text File (.txt):", type=['txt'])

if uploaded_file is not None:
    file_contents = uploaded_file.read().decode("utf-8")
    input_text = file_contents
    st.info(f"File '{uploaded_file.name}' loaded.")

if st.button("Run Full Analysis", type="primary", use_container_width=True) and input_text:
    
    st.header("--- Analysis Results ---")
    
    # --- Sentiment Analysis ---
    st.subheader("1. Sentiment Analysis (SGD Classifier)")
    
    with st.spinner("Running Inference..."):
        sentiment_probs, top_sentiment = analyze_sentiment_and_get_data(input_text, tfidf_vectorizer, clf)
        
        st.markdown(f"#### Predicted Overall Sentiment: **{top_sentiment.upper()}**")
        
        # Bar Chart
        sentiment_df = pd.DataFrame({
            'Sentiment': list(sentiment_probs.keys()),
            'Probability': list(sentiment_probs.values())
        })
        color_map = {'negative': '#EF5350', 'neutral': '#FFEE58', 'positive': '#66BB6A'}
        sentiment_df['Color'] = sentiment_df['Sentiment'].map(color_map)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Sentiment', y='Probability', data=sentiment_df, palette=sentiment_df['Color'].tolist(), ax=ax)
        ax.set_title('Sentiment Probability Distribution')
        ax.set_ylim(0, 1.05)
        st.pyplot(fig)

    # --- Summarization ---
    st.subheader("2. Text Summarization")
    col_ext, col_abs = st.columns(2)
    
    with col_ext:
        st.markdown("**Extractive Summary**")
        with st.spinner("Generating Extractive Summary..."):
            extractive_sum = extractive_reduce(input_text)
            st.info(extractive_sum)

    with col_abs:
        st.markdown("**Abstractive Summary (T5-Small)**")
        with st.spinner("Generating Abstractive Summary..."):
            abstractive_sum = abstractive_summarize_text(input_text, model_name="t5-small")
            st.info(abstractive_sum)
            
    # --- Word Cloud ---
    st.subheader("3. Word Cloud Visualization")
    with st.spinner("Generating Word Cloud..."):
        wc_image = generate_wc_image(input_text)
        st.image(wc_image, caption='Word Frequency Cloud (Processed Text)', use_column_width=True)

    # --- Topic Modeling Placeholder ---
    st.subheader("4. Topic Modeling Insights (LDA)")
    st.warning("""
    Topic-level visualization is disabled because the LDA model and related data were not explicitly trained/saved in this consolidated script.
    """)

elif st.button("Run Full Analysis", use_container_width=True) and not input_text:
    st.error("Please provide text or upload a file to analyze.")
