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
import base64 

# --- Import Summarization Utilities (Requires summarization_utils.py) ---
try:
    from summarization_utils import clean_text as clean_text_util, extractive_reduce, abstractive_summarize_text
except ImportError:
    st.error("Setup failed: Could not find summarization_utils.py. Please ensure it is in the same directory.")
    st.stop() 

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# --- Configuration ---
MODEL_DIR = 'models'
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
MAX_FEATURES = 5000
RANDOM_STATE = 42

# --- UI Helper Functions ---

def get_insights(top_sentiment: str, sentiment_probs: dict) -> str:
    """Generates detailed, conversational insights based on model output."""
    
    # Sort for detailed view
    sorted_probs = sorted(sentiment_probs.items(), key=lambda item: item[1], reverse=True)
    
    insights = f"**Overall Assessment:** The text's primary emotional tone is **{top_sentiment.upper()}**.\n\n"
    
    if sorted_probs[0][1] > 0.65:
        insights += f"**Focus:** With a probability of {sorted_probs[0][1]:.2f}, this sentiment is strongly dominant, suggesting a clear and unambiguous focus in the text.\n"
    elif sorted_probs[0][1] > 0.4:
        insights += f"**Nuance:** The sentiment is mixed, but the **{sorted_probs[0][0].upper()}** label is the strongest, indicating the text may contain competing ideas or emotions.\n"

    # Add details on secondary sentiment
    if len(sorted_probs) > 1 and sorted_probs[1][1] > 0.2:
        insights += f"**Secondary Emotion:** Note the significant presence of **{sorted_probs[1][0]}** sentiment ({sorted_probs[1][1]:.2f} probability), which might temper the overall message.\n"

    return insights

def create_download_link(data, filename, text):
    """Generates a downloadable file link using base64 encoding."""
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/markdown;base64,{b64}" download="{filename}">{text}</a>'
    return href

def generate_markdown_report(text_input, sentiment_probs, top_sentiment, extractive_sum, abstractive_sum):
    """Compiles all analysis results into a structured Markdown report."""
    report = f"""# 📊 Full Text Analysis Report
    
**Date Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📝 1. Input Text
> {text_input}

---

## 📈 2. Sentiment Analysis

The analysis was performed using a **RandomForest Classifier** trained on the Kaggle sentiment dataset.

### Overall Predicted Sentiment: **{top_sentiment.upper()}**

### Probability Distribution:
| Sentiment | Probability |
| :--- | :--- |
"""
    for sentiment, prob in sentiment_probs.items():
        report += f"| {sentiment.capitalize()} | {prob:.4f} |\n"
        
    report += "\n---\n\n## 💡 3. Key Insights (Textual Observations)\n\n"
    report += get_insights(top_sentiment, sentiment_probs)
    
    report += "\n---\n\n## 📑 4. Summarization Results\n\n"
    
    report += "### Extractive Summary (Key Sentences):\n"
    report += f"> {extractive_sum}\n\n"
    
    report += "### Abstractive Summary (AI Paraphrase):\n"
    # Ensure the error message is included if the model failed
    if "Abstractive model failed:" in abstractive_sum:
        report += f"**Model Status:** FAILED. Dependency Issue.\n"
        report += f"**Message:** {abstractive_sum.replace('Abstractive model failed: ', '')}\n"
    else:
        report += f"> {abstractive_sum}\n"
        
    report += "\n\n---"
    return report

# --- 2. ML Backend Functions ---

@st.cache_data(show_spinner="Preparing data and models (This may take a moment)...")
def load_and_preprocess_data():
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
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
        return ' '.join(words)

    df['cleaned_text'] = df['text'].apply(clean_text_local)
    tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    tfidf_vectorizer.fit(df['cleaned_text'])
    tfidf_matrix = tfidf_vectorizer.transform(df['cleaned_text'])
    
    X_train, X_test, y_train_str, y_test_str = train_test_split(
        tfidf_matrix, df['sentiment'], test_size=0.2, random_state=RANDOM_STATE, stratify=df['sentiment']
    )
    
    # FIX APPLIED HERE: Map the already split y_train_str to numeric values, avoiding the X_train.index error.
    y_train_numeric = y_train_str.map(sentiment_mapping).astype(int)
    
    return df, tfidf_vectorizer, X_train, y_train_numeric, tfidf_matrix, X_test, y_test_str

@st.cache_resource(show_spinner=False)
def train_and_save_models(_X_train, _y_train_numeric, _tfidf_vectorizer):
    st.info("🌳 Starting RandomForest Model Training... This runs once.")
    X_train_dense = _X_train.toarray()
    
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train_dense, _y_train_numeric)
    
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    joblib.dump(_tfidf_vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(clf, os.path.join(MODEL_DIR, 'random_forest_sentiment_classifier.pkl'))
    
    st.success("✅ Machine Learning Models Loaded and Ready!")
    return clf, _tfidf_vectorizer

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

# --- 3. Streamlit Application Execution (UI Focus) ---

st.set_page_config(layout="wide", page_title="AI Text Analysis Suite")

# --- Custom App Styling (Enhanced) ---
st.markdown(
    """
    <style>
    .reportview-container .main { background-color: #f0f2f6; }
    .stApp { background-color: #f0f2f6; }
    .main-header { font-size: 2.5em; font-weight: bold; color: #4F8BF9; text-align: center; margin-bottom: 20px;}
    .result-container { padding: 25px; border-radius: 12px; background: white; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); }
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white; 
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 12px 28px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover { background-color: #45a049; transform: translateY(-2px); }
    .metric-box { border: 1px solid #4F8BF9; border-radius: 6px; padding: 10px; margin-bottom: 10px; text-align: center;}
    </style>
    """, unsafe_allow_html=True
)
st.markdown('<div class="main-header">AI Text Analysis Suite 🧠✨</div>', unsafe_allow_html=True)


# --- Initial Backend Setup (Silent, uses cache) ---
with st.spinner("⏳ Initializing ML Environment..."):
    df, tfidf_vectorizer_init, X_train, y_train_numeric, _, _, _ = load_and_preprocess_data()
    if df is None:
        st.error("Setup failed: Could not load ML data.")
        st.stop()
    
    # Store model and vectorizer in session state to avoid re-initializing on UI interaction
    if 'clf' not in st.session_state or 'tfidf_vectorizer' not in st.session_state:
        clf, tfidf_vectorizer = train_and_save_models(_X_train=X_train, _y_train_numeric=y_train_numeric, _tfidf_vectorizer=tfidf_vectorizer_init)
        st.session_state.clf = clf
        st.session_state.tfidf_vectorizer = tfidf_vectorizer
    else:
        clf = st.session_state.clf
        tfidf_vectorizer = st.session_state.tfidf_vectorizer


# --- Input and Control Area ---
with st.container():
    st.header("1. Input Document 📄")
    col_input, col_file = st.columns([3, 1])
    
    if 'text_input' not in st.session_state: st.session_state.text_input = ""

    with col_input:
        text_input = st.text_area(
            "Paste Text Here:", 
            height=200, 
            placeholder="Enter the text you wish to analyze...",
            value=st.session_state.text_input
        )
        st.session_state.text_input = text_input

    with col_file:
        uploaded = st.file_uploader("Or Upload File (.txt):", type=["txt"])
        if uploaded is not None:
            try:
                file_text = uploaded.read().decode("utf-8", errors='ignore')
                st.session_state.text_input = file_text
                st.success("File loaded! Ready for analysis.")
                st.rerun()
            except Exception as e:
                st.error(f"File Read Error: {e}")

# --- Analysis Trigger Button ---
if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
    if not st.session_state.text_input or not st.session_state.text_input.strip():
        st.error("🚨 Please paste or upload text before running the analysis.")
    else:
        st.session_state.run_analysis = True
        st.session_state.results = {}
        st.toast("Analysis starting...", icon='🔍')

# --- RESULTS DISPLAY ---
if st.session_state.get('run_analysis', False) and st.session_state.text_input:
    
    # --- Execute Analysis Core Logic ---
    with st.spinner("Processing sentiment and summarization..."):
        
        # 1. Sentiment Analysis
        sentiment_probs, top_sentiment = analyze_sentiment_and_get_data(st.session_state.text_input, tfidf_vectorizer, clf)
        
        # 2. Extractive Summary
        extractive_sum = extractive_reduce(st.session_state.text_input)
        
        # 3. Abstractive Summary (with Keras Error Check)
        abstractive_sum = "Abstractive model failed: Dependency error detected. Run pip install tf-keras to resolve the Keras 3/Transformers compatibility issue."
        try:
            abstractive_sum_raw = abstractive_summarize_text(st.session_state.text_input, model_name="t5-small")
            abstractive_sum = abstractive_sum_raw
        except Exception as e:
            error_text = str(e)
            if "Keras 3" in error_text or "tf-keras" in error_text or "No module named 'tf_keras'" in error_text:
                abstractive_sum = "Abstractive model failed: Dependency error detected. Run pip install tf-keras to resolve the Keras 3/Transformers compatibility issue."
            else:
                abstractive_sum = f"Abstractive model failed: {error_text}"
                
        # 4. Word Cloud Generation
        wc_image = generate_wc_image(st.session_state.text_input)
        
        # Store results for display and report generation
        st.session_state.results = {
            'sentiment_probs': sentiment_probs,
            'top_sentiment': top_sentiment,
            'extractive_sum': extractive_sum,
            'abstractive_sum': abstractive_sum,
            'wc_image': wc_image
        }
        
    st.success("Analysis Complete! View the results below. 🎉")
    
    
    # --- Dynamic Results Display using Tabs ---
    
    tab1, tab2, tab3 = st.tabs(["💡 Overview & Insights", "📊 Sentiment Details", "☁️ Word Cloud"])
    results = st.session_state.results
    
    # TAB 1: Overview & Insights
    with tab1:
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.subheader("Key Findings Summary 🔍")
        
        col_metric_1, col_metric_2 = st.columns(2)
        with col_metric_1:
            st.markdown(f'<div class="metric-box">Primary Sentiment: <br>**{results["top_sentiment"].upper()}**</div>', unsafe_allow_html=True)
        with col_metric_2:
            st.markdown(f'<div class="metric-box">Word Count: <br>**{len(st.session_state.text_input.split())}** words</div>', unsafe_allow_html=True)

        st.markdown("---")
        
        st.subheader("Key Insights (Conversational) 🗣️")
        st.markdown(get_insights(results['top_sentiment'], results['sentiment_probs']))
        
        st.markdown("---")
        
        st.subheader("Summaries")
        st.markdown("##### ✂️ Extractive Summary")
        st.info(results['extractive_sum'])
        
        st.markdown("##### 🧠 Abstractive Summary")
        if "Abstractive model failed:" in results['abstractive_sum']:
            st.error(results['abstractive_sum'])
        else:
            st.success(results['abstractive_sum'])
        
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 2: Sentiment Details
    with tab2:
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.subheader("Full Sentiment Distribution 📉")
        
        st.markdown(f"#### Overall Predicted Sentiment: **{results['top_sentiment'].upper()}**")
        
        # Bar Chart Styling (Smaller Size as requested)
        sentiment_df = pd.DataFrame({
            'Sentiment': list(results['sentiment_probs'].keys()),
            'Probability': list(results['sentiment_probs'].values())
        })
        color_map = {'negative': '#EF5350', 'neutral': '#FFEE58', 'positive': '#66BB6A'}
        sentiment_df['Color'] = sentiment_df['Sentiment'].map(color_map)

        fig, ax = plt.subplots(figsize=(5, 3)) # SMALLER SIZE
        
        ax.bar(
            sentiment_df['Sentiment'], 
            sentiment_df['Probability'], 
            color=sentiment_df['Color'].tolist(),
            width=0.45
        )
        ax.set_title('Sentiment Probability Distribution')
        ax.set_ylim(0, 1.05)
        st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)


    # TAB 3: Word Cloud
    with tab3:
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.subheader("Word Frequency Visualization 🖼️")
        st.image(results['wc_image'], caption='Word Frequency Cloud (Processed Text)', use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    
    # --- Download Report Section ---
    st.markdown("---")
    st.subheader("3. Generate & Download Report 📥")
    
    # Compile the report data
    report_content = generate_markdown_report(
        st.session_state.text_input, 
        results['sentiment_probs'], 
        results['top_sentiment'], 
        results['extractive_sum'], 
        results['abstractive_sum']
    )
    
    # Use Streamlit's built-in download button
    st.download_button(
        label="Download Full Analysis Report (.md)",
        data=report_content,
        file_name="analysis_report.md",
        mime="text/markdown",
        help="Click to download a detailed Markdown report containing all analysis results."
    )
