from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def extractive_summary(text: str, num_sentences: int = 3) -> str:
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(vectors)

    sentence_scores = sim_matrix.sum(axis=1)
    ranked_sentences = [sentences[i] for i in np.argsort(-sentence_scores)[:num_sentences]]
    return " ".join(ranked_sentences)
