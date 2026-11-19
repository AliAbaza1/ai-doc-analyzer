"""Classic NLP: named entities (spaCy) and sentence-level sentiment (TextBlob). English documents."""

from __future__ import annotations

from functools import lru_cache

from textblob import TextBlob

from modules.summarizer import split_sentences


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError("spaCy model missing. Run: python -m spacy download en_core_web_sm") from exc


def extract_entities(text: str) -> list[dict]:
    doc = _nlp()(" ".join(text.split()))
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]


def sentence_sentiment(text: str) -> list[dict]:
    rows = []
    for i, sentence in enumerate(split_sentences(text), start=1):
        sentiment = TextBlob(sentence).sentiment
        rows.append({"index": i, "sentence": sentence, "polarity": round(sentiment.polarity, 3)})
    return rows


def overall_sentiment(text: str) -> dict:
    sentiment = TextBlob(text).sentiment
    return {"polarity": round(sentiment.polarity, 3), "subjectivity": round(sentiment.subjectivity, 3)}
