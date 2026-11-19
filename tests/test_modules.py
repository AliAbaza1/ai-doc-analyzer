import io

import pytest
from docx import Document

from modules import llm_processor
from modules.loader import read_document
from modules.spacy_nlp import extract_entities, sentence_sentiment
from modules.summarizer import extractive_summary, split_sentences
from modules.visualizer import plot_entity_distribution, plot_sentiment_timeline

SAMPLE = open("samples/service_agreement.txt", encoding="utf-8").read()


def test_read_txt_and_docx():
    assert read_document("a.txt", "مرحبا".encode("utf-8")) == "مرحبا"
    doc = Document()
    doc.add_paragraph("Hello from Word")
    buf = io.BytesIO()
    doc.save(buf)
    assert read_document("a.docx", buf.getvalue()) == "Hello from Word"
    with pytest.raises(ValueError):
        read_document("a.exe", b"x")


def test_sentences_and_extractive_summary_keep_order():
    assert split_sentences("One. Two? ثلاثة؟ Four!") == ["One.", "Two?", "ثلاثة؟", "Four!"]
    summary = extractive_summary(SAMPLE, num_sentences=2)
    sentences = split_sentences(SAMPLE)
    picked = [s for s in sentences if s in summary]
    assert len(picked) == 2 and sentences.index(picked[0]) < sentences.index(picked[1])


def test_entities_and_charts():
    entities = extract_entities(SAMPLE)
    labels = {e["label"] for e in entities}
    assert {"ORG", "DATE", "PERSON"} <= labels
    assert plot_entity_distribution(entities) is not None
    rows = sentence_sentiment(SAMPLE)
    assert rows[0]["index"] == 1 and plot_sentiment_timeline(rows) is not None


def test_parse_json_object_tolerates_code_fences():
    raw = '```json\n{"people": ["Sarah Ahmed"], "amounts": ["USD 4,800 total fee"], "extra": [1]}\n```'
    data = llm_processor.parse_json_object(raw)
    assert data["people"] == ["Sarah Ahmed"] and data["dates"] == [] and "extra" not in data


class FakeMessages:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = type("B", (), {"type": "text", "text": self.reply})()
        return type("R", (), {"content": [block]})()


def test_claude_calls_are_well_formed(monkeypatch):
    fake = FakeMessages('{"people": ["James Carter"]}')
    monkeypatch.setattr(llm_processor, "_client", type("C", (), {"messages": fake})())
    assert llm_processor.extract_key_info(SAMPLE)["people"] == ["James Carter"]
    call = fake.calls[0]
    assert call["model"] == llm_processor.DEFAULT_MODEL and call["temperature"] == 0
    assert "<document>" in call["messages"][0]["content"]


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.setattr(llm_processor, "_client", None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm_processor.summarize_text("hi")
