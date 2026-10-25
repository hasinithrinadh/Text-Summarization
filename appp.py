import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="wide")

MODELS = {
    "DistilBART (fast, good for news)": "sshleifer/distilbart-cnn-12-6",
    "BART Large CNN (higher quality, slower)": "facebook/bart-large-cnn",
    "T5 Small (lightweight)": "t5-small",
}


@st.cache_resource(show_spinner="Loading model... (first run downloads it)")
def load_summarizer(model_name: str):
    return pipeline("summarization", model=model_name)


def chunk_text(text: str, max_words: int = 600):
    """Split long text into word-based chunks the model can handle."""
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i : i + max_words])


def summarize(text: str, summarizer, min_len: int, max_len: int) -> str:
    parts = []
    for chunk in chunk_text(text):
        out = summarizer(
            chunk,
            max_length=max_len,
            min_length=min(min_len, max_len - 5),
            do_sample=False,
            truncation=True,
        )
        parts.append(out[0]["summary_text"])
    return " ".join(parts)


st.title("📝 Text Summarizer")
st.caption("Built with Hugging Face Transformers + Streamlit")

with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox("Model", list(MODELS.keys()))
    min_len = st.slider("Min summary length (tokens)", 10, 100, 30)
    max_len = st.slider("Max summary length (tokens)", 50, 300, 130)

input_mode = st.radio("Input", ["Paste text", "Upload .txt file"], horizontal=True)

text = ""
if input_mode == "Paste text":
    text = st.text_area("Enter the text to summarize", height=300)
else:
    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded:
        text = uploaded.read().decode("utf-8", errors="ignore")
        st.text_area("Preview", text, height=200, disabled=True)

if st.button("Summarize", type="primary"):
    if len(text.split()) < 30:
        st.warning("Please provide at least ~30 words of text.")
    else:
        summarizer = load_summarizer(MODELS[model_label])
        with st.spinner("Summarizing..."):
            summary = summarize(text, summarizer, min_len, max_len)

        orig_words, sum_words = len(text.split()), len(summary.split())
        st.subheader("Summary")
        st.success(summary)

        c1, c2, c3 = st.columns(3)
        c1.metric("Original words", orig_words)
        c2.metric("Summary words", sum_words)
        c3.metric("Compression", f"{100 - (sum_words / orig_words * 100):.0f}%")

        st.download_button("Download summary", summary, file_name="summary.txt")