import re
import math
import heapq
import string
from typing import List

# Attempt to import NLTK and Hugging Face components
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import nltk
except ImportError:
    stopwords = None
    WordNetLemmatizer = object
    nltk = None

try:
    from transformers import pipeline, AutoTokenizer
    import torch
except ImportError:
    pipeline = None
    AutoTokenizer = None
    torch = None

# --- Text Preprocessing Setup ---
if nltk:
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer() if WordNetLemmatizer is not object else None
stop_words = set(stopwords.words('english')) if stopwords else set()

def clean_text(text: str) -> str:
    """Cleans text for TF-IDF vectorization."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    if lemmatizer:
        words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# --- Extractive Summarization Functions ---
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "without",
    "to", "from", "by", "for", "of", "on", "in", "at", "is", "are", "was", "were",
    "this", "that", "these", "those", "it", "its", "be", "as", "which", "not",
}

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents or [text.strip()]

def word_tokens(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"\w+", text) if w.lower() not in STOPWORDS]

def extractive_reduce(text: str, ratio: float = 0.3, min_sentences: int = 1, max_sentences: int = 8) -> str:
    """Generates an extractive summary based on word frequency."""
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return text

    freq = {}
    for sent in sentences:
        for w in word_tokens(sent):
            freq[w] = freq.get(w, 0) + 1

    scores = []
    for i, sent in enumerate(sentences):
        s = sum(freq.get(w, 0) for w in word_tokens(sent))
        scores.append((s, i, sent))

    keep = max(min_sentences, min(max_sentences, math.ceil(len(sentences) * ratio)))
    top = heapq.nlargest(keep, scores, key=lambda x: (x[0], -x[1]))
    top_sorted = sorted(top, key=lambda x: x[1])
    reduced = " ".join([s for (_score, _i, s) in top_sorted])
    return reduced

# --- Abstractive Summarization Functions ---
_abstractive_pipeline = None

def make_abstractive_pipeline(model_name: str = "t5-small"):
    global _abstractive_pipeline
    if _abstractive_pipeline is None:
        if pipeline is None:
            raise RuntimeError("transformers/torch not installed or imported correctly.")
        device = 0 if torch and torch.cuda.is_available() else -1
        _abstractive_pipeline = pipeline("summarization", model=model_name, tokenizer=model_name, device=device)
    return _abstractive_pipeline

def trim_for_model(text: str, model_name: str, fraction_of_model_max: float = 0.9) -> str:
    if AutoTokenizer is None:
        return text
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_max = getattr(tokenizer, "model_max_length", 512) or 512
    budget = max(64, int(model_max * fraction_of_model_max))

    sentences = split_sentences(text)
    
    trimmed_sents = []
    current_tokens = 0
    for sent in sentences:
        ids = tokenizer.encode(sent, add_special_tokens=False, truncation=False)
        if current_tokens + len(ids) < budget:
            trimmed_sents.append(sent)
            current_tokens += len(ids)
        else:
            break

    return " ".join(trimmed_sents)

def abstractive_summarize_text(text: str, model_name: str = "t5-small",
                               max_length: int = 120, min_length: int = 20) -> str:
    """Generates an abstractive summary using a Hugging Face model."""
    if pipeline is None:
        return "Abstractive summarization unavailable (Check requirements.txt: transformers, torch)."

    reduced = extractive_reduce(text, ratio=0.25, min_sentences=1, max_sentences=8)
    trimmed = trim_for_model(reduced, model_name)
    
    try:
        summarizer = make_abstractive_pipeline(model_name)
        out = summarizer(trimmed, max_length=max_length, min_length=min_length, do_sample=False)
        if isinstance(out, list) and out:
            return out[0].get("summary_text", "").strip()
        return str(out)
    except Exception as e:
        return f"Abstractive model failed: {e}"