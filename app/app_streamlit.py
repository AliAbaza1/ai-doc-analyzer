import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st

from modules.llm_processor import DEFAULT_MODEL, KEY_INFO_FIELDS, answer_question, extract_key_info, summarize_text
from modules.loader import SUPPORTED_TYPES, read_document
from modules.spacy_nlp import extract_entities, overall_sentiment, sentence_sentiment
from modules.summarizer import extractive_summary
from modules.visualizer import plot_entity_distribution, plot_sentiment_timeline

st.set_page_config(page_title="AI Document Analyzer", page_icon="📄", layout="wide")


# Cached so that typing a question does not re-run (and re-pay for) the other AI calls.
@st.cache_data(show_spinner=False)
def ai_summary(text: str) -> str:
    return summarize_text(text)


@st.cache_data(show_spinner=False)
def ai_key_info(text: str) -> dict:
    return extract_key_info(text)


@st.cache_data(show_spinner=False)
def nlp_results(text: str) -> dict:
    return {
        "entities": extract_entities(text),
        "timeline": sentence_sentiment(text),
        "overall": overall_sentiment(text),
        "extractive": extractive_summary(text),
    }


st.title("📄 AI Document Analyzer")
st.caption(f"Summaries, key facts and Q&A with Claude ({DEFAULT_MODEL}) · entities and sentiment with spaCy")

st.sidebar.header("Document")
uploaded = st.sidebar.file_uploader("Upload a file", type=SUPPORTED_TYPES)
pasted = st.sidebar.text_area("…or paste text", height=220)

text = None
if uploaded:
    try:
        text = read_document(uploaded.name, uploaded.getvalue())
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()
elif pasted.strip():
    text = pasted.strip()

if not text:
    st.info("👈 Upload a TXT, MD, PDF or DOCX file, or paste text in the sidebar.")
    st.stop()

words = len(text.split())
c1, c2, c3 = st.columns(3)
c1.metric("Words", f"{words:,}")
c2.metric("Characters", f"{len(text):,}")
c3.metric("Reading time", f"{max(1, round(words / 200))} min")

tab_summary, tab_facts, tab_qa, tab_entities, tab_sentiment = st.tabs(
    ["📝 Summary", "🔎 Key facts", "❓ Ask", "🏷️ Entities", "📈 Sentiment"]
)

with tab_summary:
    with st.spinner("Summarizing…"):
        try:
            st.markdown(ai_summary(text))
        except Exception as exc:
            st.error(f"AI summary failed: {exc}")
    with st.expander("Extractive summary (key sentences taken directly from the text)"):
        st.write(nlp_results(text)["extractive"])

with tab_facts:
    with st.spinner("Extracting key facts…"):
        try:
            facts = ai_key_info(text)
            rows = [{"type": k.replace("_", " ").title(), "value": v} for k in KEY_INFO_FIELDS for v in facts[k]]
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button("Download as CSV", df.to_csv(index=False).encode("utf-8-sig"),
                                   "key_facts.csv", "text/csv")
            else:
                st.warning("No key facts found.")
        except Exception as exc:
            st.error(f"Extraction failed: {exc}")

with tab_qa:
    question = st.text_input("Ask a question about the document")
    if question:
        with st.spinner("Thinking…"):
            try:
                st.success(answer_question(text, question))
            except Exception as exc:
                st.error(f"Could not answer: {exc}")

with tab_entities:
    try:
        entities = nlp_results(text)["entities"]
        if entities:
            st.plotly_chart(plot_entity_distribution(entities), use_container_width=True)
            st.dataframe(pd.DataFrame(entities).value_counts().reset_index(name="mentions"),
                         use_container_width=True, hide_index=True)
        else:
            st.warning("No named entities found (entity recognition works on English text).")
    except Exception as exc:
        st.error(str(exc))

with tab_sentiment:
    results = nlp_results(text)
    st.metric("Overall polarity", results["overall"]["polarity"],
              help="−1 is very negative, +1 is very positive (English text)")
    if len(results["timeline"]) > 1:
        st.plotly_chart(plot_sentiment_timeline(results["timeline"]), use_container_width=True)
