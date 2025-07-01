import spacy
from spacy import displacy
from typing import List, Dict
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")

def clean_text(text: str) -> str:
    return text.strip().replace("\n", " ").replace("\t", " ")

def extract_entities(text: str) -> List[Dict]:
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

def analyze_sentiment(text: str) -> Dict:
    blob = TextBlob(text)
    return {
        "polarity": round(blob.sentiment.polarity, 3),
        "subjectivity": round(blob.sentiment.subjectivity, 3)
    }

def process_text(text: str) -> Dict:
    cleaned = clean_text(text)
    return {
        "entities": extract_entities(cleaned),
        "sentiment": analyze_sentiment(cleaned)
    }

