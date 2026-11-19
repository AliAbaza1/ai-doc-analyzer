"""Extractive summary: picks the sentences most similar to the rest of the document (TF-IDF)."""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SENTENCE_END = re.compile(r"(?<=[.!?؟。])\s+|\n{2,}")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if len(s.strip()) > 1]


def extractive_summary(text: str, num_sentences: int = 3) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
    vectors = TfidfVectorizer().fit_transform(sentences)
    scores = cosine_similarity(vectors).sum(axis=1)
    top = sorted(np.argsort(-scores)[:num_sentences])  # keep the original order
    return " ".join(sentences[i] for i in top)
