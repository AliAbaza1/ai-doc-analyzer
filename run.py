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

# App header
st.title("📄 AI-Powered Document Analyzer")
st.markdown("Upload a text file, paste content, or explore advanced text analysis using **Generative AI** and **Natural Language Processing (NLP)**.")

# File upload and text input
st.markdown("### 📤 Upload or Paste Your Document")

uploaded_file = st.file_uploader("Upload a text file (TXT, MD):", type=["txt", "md"])
text_input = st.text_area("Or paste your text here:", height=250, placeholder="Enter or paste your text here for analysis...")

# Combine inputs
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
elif text_input:
    text = text_input
else:
    text = None

# Display document stats
if text:
    word_count = len(text.split())
    char_count = len(text)

    with st.expander("📊 Document Statistics", expanded=True):
        st.write(f"**Word Count:** {word_count}")
        st.write(f"**Character Count:** {char_count}")
        reading_time = max(1, word_count // 200)
        st.write(f"**Estimated Reading Time:** {reading_time} min")

    st.markdown("---")

    # Process text
    with st.spinner("🧠 Processing your document..."):
        processed = process_text(text)
        extractive = extractive_summary(text)
        abstractive = summarize_text(text)

    # Summaries
    st.subheader("📝 Summaries")

    st.write("**Extractive Summary:**")
    st.success(extractive if extractive else "No extractive summary available.")

    st.write("**Abstractive Summary (AI-generated):**")
    st.info(abstractive if abstractive else "No abstractive summary available.")

    # Entities
    st.subheader("🏷️ Named Entity Recognition")
    if processed.get("entities"):
        st.plotly_chart(plot_entity_distribution(processed["entities"]), use_container_width=True)
    else:
        st.warning("No entities found.")

    # Sentiment
    st.subheader("📈 Sentiment Analysis")
    if processed.get("sentiment"):
        st.plotly_chart(plot_sentiment_timeline([processed["sentiment"]]), use_container_width=True)
    else:
        st.warning("No sentiment data available.")

    # Q&A Section
    st.subheader("❓ Ask a Question About the Document")
    question = st.text_input("Type your question here:")

    if question:
        with st.spinner("🤖 Generating answer..."):
            answer = answer_question(text, question)
        if answer:
            st.success(f"💬 Answer: {answer}")
        else:
            st.error("Could not generate an answer. Please try a different question.")
else:
    st.info("Please upload a file or paste text to get started.")
