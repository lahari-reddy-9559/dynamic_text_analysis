import re
import math
import heapq
import nltk
import streamlit as st
from nltk.stem import WordNetLemmatizer
from typing import List

# --- Setup ---
try:
    # Ensure NLTK data is downloaded if not present (important for environment setup)
    for package in ['punkt', 'stopwords', 'wordnet']:
        try: nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except LookupError: nltk.download(package, quiet=True)
        
    lemmatizer = WordNetLemmatizer()
    STOPWORDS = set(nltk.corpus.stopwords.words("english"))
except Exception as e:
    lemmatizer = None
    STOPWORDS = set()
    print(f"NLTK setup failed: {e}")

# Global check for Transformers availability
_TRANSFORMERS_AVAILABLE = False

def try_enable_transformers():
    """Checks for transformers and returns a brief error message about missing dependencies."""
    global _TRANSFORMERS_AVAILABLE
    if _TRANSFORMERS_AVAILABLE: return True, None
    try:
        from transformers import pipeline, AutoTokenizer
        import torch
        _TRANSFORMERS_AVAILABLE = True
        return True, None
    except Exception as e:
        _TRANSFORMERS_AVAILABLE = False
        err_str = str(e)
        
        # Check for specific dependency errors related to ML models
        if "No module named 'transformers'" in err_str:
             return False, "Transformer model failed: Please install the 'transformers' library."
        if "Keras 3" in err_str or "tf-keras" in err_str:
             return False, "Transformer model failed: Dependency error detected (Keras 3/tf-keras). Run 'pip install tf-keras'."
        return False, f"Transformer model failed: Unknown error. ({err_str[:40]}...)"


# --- Core NLP Utilities ---
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents or [text.strip()]

def word_tokens(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"\w+", text)]

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    t = text.lower()
    t = t.translate(str.maketrans("", "", r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""))
    # Ensure lemmatizer is available
    toks = [lemmatizer.lemmatize(w) for w in t.split() if w and w not in STOPWORDS] if lemmatizer else t.split()
    return " ".join(toks)

# Helper function for frequency reduction (used internally for pre-reduction before model)
def _frequency_reduce(text: str, ratio: float = 0.3, min_sentences: int = 1, max_sentences: int = 6) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= 1: return text
    freq = {}
    for sent in sentences:
        for w in word_tokens(sent): freq[w] = freq.get(w, 0) + 1
    scores = []
    for i, sent in enumerate(sentences):
        s = sum(freq.get(w, 0) for w in word_tokens(sent))
        scores.append((s, i, sent))
    keep = max(min_sentences, min(max_sentences, math.ceil(len(sentences) * ratio)))
    top = heapq.nlargest(keep, scores, key=lambda x: (x[0], -x[1]))
    top_sorted = sorted(top, key=lambda x: x[1])
    return " ".join([s for (_score, _i, s) in top_sorted])


# --- NEW: Context detection for technical lists ---
def _is_technical_list(text: str) -> bool:
    """Checks if the text looks like a short, non-prose list (like requirements.txt)."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # Check if text is short overall but has multiple distinct lines
    if len(lines) > 2 and len(text) < 300: 
        # Check if most lines look like single items/dependencies (no sentence-ending punctuation)
        non_sentence_lines = sum(1 for line in lines if not re.search(r'[.!?]$', line))
        # If more than 70% of lines don't end in punctuation, assume it's a list.
        return non_sentence_lines / len(lines) > 0.7 
    return False


@st.cache_resource(show_spinner=False)
def make_abstractive_pipeline(model_name: str = "t5-small"):
    avail, err = try_enable_transformers()
    if not avail: 
        raise RuntimeError(err or "models not available")
    
    from transformers import pipeline
    import torch as _torch
    device = 0 if _torch.cuda.is_available() else -1
    return pipeline("summarization", model=model_name, tokenizer=model_name, device=device)

def trim_for_model(text: str, model_name: str, fraction_of_model_max: float = 0.9) -> str:
    avail, err = try_enable_transformers()
    if not avail: raise RuntimeError(err or "models not available")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_max = getattr(tokenizer, "model_max_length", 512) or 1024
    budget = max(64, int(model_max * fraction_of_model_max))
    sentences = split_sentences(text)
    if not sentences: return text
    
    def token_count(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False, truncation=False))
    
    joined = " ".join(sentences)
    if token_count(joined) <= budget: return joined
    
    trimmed_sents = []
    current_tokens = 0
    for sent in sentences:
        sent_tokens = token_count(sent)
        if current_tokens + sent_tokens + 2 <= budget:
            trimmed_sents.append(sent)
            current_tokens += sent_tokens
        elif current_tokens == 0:
            ids = tokenizer.encode(sent, add_special_tokens=False)[:budget]
            return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return " ".join(trimmed_sents)


def abstractive_summarize_text(text: str, model_name: str = "t5-small", max_length: int = 150, min_length: int = 30, use_reduced: bool = True) -> str:
    """
    Summarizes text using a transformer model. Increased max_length and min_length 
    to provide a more detailed abstract. Now includes contextual prompting for lists.
    """
    avail, err = try_enable_transformers()
    if not avail: raise RuntimeError(err or "models not available")
    
    reduced = _frequency_reduce(text, ratio=0.25, min_sentences=1, max_sentences=6) if use_reduced else text
    initial_text = reduced # Text to be passed to the model
    
    # --- NEW LOGIC: Contextual Prompting for Technical Lists ---
    prompt_prefix = ""
    if _is_technical_list(text):
        if max_length < 100:
             # Short prompt for Key Insights
             prompt_prefix = "The following is a list of software requirements or dependencies. Describe the nature and function of this list concisely: "
        else:
             # Longer prompt for Detailed Abstract
             prompt_prefix = "The following is a list of software requirements or dependencies. Provide a professional analysis of the project context and potential application based on these modules: "

    trimmed = trim_for_model(prompt_prefix + initial_text, model_name)
    
    if not trimmed:
        return "Input text was too short or failed preprocessing."
        
    try:
        summarizer = make_abstractive_pipeline(model_name)
        out = summarizer(trimmed, max_length=max_length, min_length=min_length, do_sample=False)
        if isinstance(out, list) and out:
            return out[0].get("summary_text", "").strip()
        return str(out)
    except Exception as e:
        raise e

# --- Transformer-based "Extractive" (Key Insights) function using the same model ---
def extractive_reduce(text: str, model_name: str = "t5-small") -> str:
    """
    Generates a very short, abstractive summary (Key Insights) using the transformer.
    Uses abstractive_summarize_text with short length parameters.
    """
    try:
        # Optimized parameters for very concise output
        return abstractive_summarize_text(
            text=text, 
            model_name=model_name, 
            max_length=80,
            min_length=25,
            use_reduced=True
        )
    except Exception as e:
        raise e
