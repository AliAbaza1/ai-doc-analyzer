import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from modules.spacy_nlp import process_text
from modules.llm_processor import summarize_text, answer_question
from modules.summarizer import extractive_summary
from modules.visualizer import plot_entity_distribution, plot_sentiment_timeline

# Page configuration
st.set_page_config(
    page_title="📊 AI Document Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Header
st.title("📄 AI-Powered Document Analyzer")
st.caption("Built with Generative AI + NLP | Developed by @Ali")
st.markdown("---")

# Upload or paste text
st.sidebar.header("📤 Upload or Paste Document")
uploaded_file = st.sidebar.file_uploader("Upload a text file (TXT, MD):", type=["txt", "md"])
text_input = st.sidebar.text_area("Or paste your text below:", height=200, placeholder="Enter or paste your text here...")

# Read text
text = uploaded_file.read().decode("utf-8") if uploaded_file else text_input.strip() if text_input else None

if not text:
    st.info("👈 Please upload a file or paste some text in the sidebar to get started.")
    st.stop()

# Basic document stats
word_count = len(text.split())
char_count = len(text)
reading_time = max(1, word_count // 200)

with st.expander("📄 Document Overview", expanded=True):
    st.write(f"**Word Count:** {word_count}")
    st.write(f"**Character Count:** {char_count}")
    st.write(f"**Estimated Reading Time:** {reading_time} minute(s)")

st.markdown("---")

# Processing
with st.spinner("⚙️ Analyzing document with AI..."):
    try:
        processed = process_text(text)
        extractive = extractive_summary(text)
        abstractive = summarize_text(text)
    except Exception as e:
        st.error(f"❌ Error during processing: {e}")
        st.stop()

# Tabs for different outputs
tabs = st.tabs(["📝 Summaries", "🏷️ Entities", "📈 Sentiment", "❓ Q&A"])

# Summaries Tab
with tabs[0]:
    st.subheader("📝 Text Summaries")
    st.markdown("**🔹 Extractive Summary**")
    st.success(extractive if extractive else "No extractive summary available.")
    st.markdown("**🔹 Abstractive Summary (AI-Generated)**")
    st.info(abstractive if abstractive else "No abstractive summary available.")

# Entities Tab
with tabs[1]:
    st.subheader("🏷️ Named Entity Recognition")
    if processed.get("entities"):
        st.plotly_chart(plot_entity_distribution(processed["entities"]), use_container_width=True)
    else:
        st.warning("No named entities found in the text.")

# Sentiment Tab
with tabs[2]:
    st.subheader("📈 Sentiment Timeline")
    if processed.get("sentiment"):
        st.plotly_chart(plot_sentiment_timeline([processed["sentiment"]]), use_container_width=True)
    else:
        st.warning("No sentiment data available.")

# Q&A Tab
with tabs[3]:
    st.subheader("❓ Ask AI About Your Document")
    question = st.text_input("Ask a question related to the document:")
    if question:
        with st.spinner("🤖 Generating answer..."):
            try:
                answer = answer_question(text, question)
                st.success(f"💬 Answer: {answer}" if answer else "⚠️ No answer generated.")
            except Exception as e:
                st.error(f"Failed to answer: {e}")
