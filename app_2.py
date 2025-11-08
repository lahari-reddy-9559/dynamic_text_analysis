import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import pandas as pd

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Text Insight Studio",
    page_icon="💬",
    layout="wide",
)

# ================================
# AUTO-DETECT THEME (DARK/LIGHT)
# ================================
theme_base = st.get_option("theme.base")
is_dark = theme_base == "dark"

# ================================
# CUSTOM STYLING - Animated gradients, theme adaptive
# ================================
light_bg = """
linear-gradient(135deg, #c7e8f3, #f5e6ff, #dfffe2, #fdf5f1)
"""
dark_bg = """
linear-gradient(135deg, #1c1f26, #232730, #2d3140, #3a3f4e)
"""

st.markdown(
    f"""
    <style>
    @keyframes dreamyFlow {{
      0% {{ background-position: 0% 50%; }}
      50% {{ background-position: 100% 50%; }}
      100% {{ background-position: 0% 50%; }}
    }}
    @keyframes hueRotate {{
      0% {{ filter: hue-rotate(0deg); }}
      100% {{ filter: hue-rotate(360deg); }}
    }}

    .stApp {{
      background: {dark_bg if is_dark else light_bg};
      background-size: 300% 300%;
      animation: dreamyFlow 16s ease infinite;
      color: {'#e5e7eb' if is_dark else '#283747'};
      font-family: "Poppins", sans-serif;
    }}

    .block-container {{
      background: rgba(255,255,255,0.08) if {is_dark} else rgba(255,255,255,0.83);
      border-radius: 16px;
      padding: 26px 30px;
      box-shadow: 0 6px 28px rgba(52,61,73,0.08);
      backdrop-filter: blur(6px);
      transition: all 0.4s ease;
    }}

    h1, h2, h3 {{
      background: linear-gradient(90deg, #667eea, #8fd3f4, #6ee7b7, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: hueRotate 8s linear infinite;
      background-size: 400% 400%;
      font-weight: 700;
    }}

    .stButton>button {{
      background: linear-gradient(90deg, #7dd3fc, #a5b4fc, #f9a8d4);
      background-size: 200% auto;
      color: {'#f3f4f6' if is_dark else '#2f3e46'};
      border: none;
      border-radius: 10px;
      font-weight: 600;
      padding: 0.6em 1.2em;
      box-shadow: 0 4px 14px rgba(0,0,0,0.08);
      transition: 0.4s ease;
    }}
    .stButton>button:hover {{
      background-position: right center;
      transform: translateY(-2px);
    }}

    textarea[role="textbox"], .stFileUploader {{
      border-radius: 10px !important;
      background: rgba(255,255,255,0.08) if {is_dark} else rgba(255,255,255,0.65);
      color: {'#e5e7eb' if is_dark else '#374151'} !important;
      border: 1px solid rgba(150,150,150,0.2);
      box-shadow: inset 0 2px 4px rgba(0,0,0,0.04);
    }}

    .stDownloadButton>button {{
      background: linear-gradient(90deg, #93c5fd, #c084fc, #fda4af);
      background-size: 200% auto;
      color: {'#f3f4f6' if is_dark else '#1f2937'};
      border-radius: 10px;
      border: none;
      padding: 0.6em 1.2em;
      font-weight: 600;
      transition: 0.4s ease;
    }}
    .stDownloadButton>button:hover {{
      background-position: right center;
      transform: translateY(-2px);
    }}

    .stAlert {{
      border-radius: 10px;
      background: linear-gradient(135deg, rgba(217, 236, 255, 0.9), rgba(230, 255, 250, 0.9));
      color: #1e293b !important;
    }}

    div[data-testid="stMarkdownContainer"]:hover {{
      transform: translateY(-2px);
      transition: 0.3s ease;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ================================
# APP HEADER
# ================================
st.title("💬 Text Insight Studio")
st.caption("Developed by Lahari Reddy — Compact visuals, auto theme, dark/light adaptive ✨")

# ================================
# TEXT INPUT AREA
# ================================
st.subheader("📝 Enter Text:")
user_text = st.text_area("Paste or type text to analyze...", height=180, label_visibility="collapsed")

uploaded_file = st.file_uploader("📂 Or upload a text file (.txt):", type=["txt"])
if uploaded_file:
    user_text = uploaded_file.read().decode("utf-8")

if not user_text.strip():
    st.info("Please enter or upload some text to analyze 💡")
    st.stop()

# ================================
# TEXT ANALYSIS
# ================================
blob = TextBlob(user_text)
sentences = blob.sentences

# Sentiment analysis summary
data = {"Sentence": [str(s) for s in sentences],
        "Polarity": [s.sentiment.polarity for s in sentences],
        "Subjectivity": [s.sentiment.subjectivity for s in sentences]}
df = pd.DataFrame(data)

avg_polarity = df["Polarity"].mean()
avg_subjectivity = df["Subjectivity"].mean()

# ================================
# CHARTS SECTION
# ================================
st.markdown("---")
st.subheader("📊 Sentiment Analysis Overview")

col1, col2 = st.columns(2)

# --- Compact bar chart (500x300 px) ---
with col1:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(df.index, df["Polarity"], color=("#6ee7b7" if not is_dark else "#93c5fd"))
    ax.set_title("Sentence Polarity", color="#e5e7eb" if is_dark else "#1f2937")
    ax.set_xlabel("Sentence Index")
    ax.set_ylabel("Polarity")
    ax.grid(alpha=0.3)
    fig.patch.set_facecolor("#111827" if is_dark else "#ffffff")
    st.pyplot(fig, use_container_width=False)

# --- Word cloud (500x300 px) ---
with col2:
    wc = WordCloud(width=500, height=300, background_color=("black" if is_dark else "white"),
                   colormap="viridis").generate(user_text)
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.imshow(wc, interpolation="bilinear")
    ax2.axis("off")
    st.pyplot(fig2, use_container_width=False)

# ================================
# SUMMARY METRICS
# ================================
st.markdown("---")
st.subheader("📈 Summary Insights")
col3, col4 = st.columns(2)
col3.metric("Average Polarity", f"{avg_polarity:.2f}")
col4.metric("Average Subjectivity", f"{avg_subjectivity:.2f}")

# ================================
# EXPORT RESULTS
# ================================
st.download_button(
    label="💾 Download Results (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="text_sentiment_analysis.csv",
    mime="text/csv"
)
