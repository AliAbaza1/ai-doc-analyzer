"""Plotly charts for the Streamlit app."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def plot_entity_distribution(entities: list[dict]):
    if not entities:
        return None
    counts = pd.DataFrame(entities)["label"].value_counts().rename_axis("label").reset_index(name="count")
    return px.bar(counts, x="label", y="count", labels={"label": "Entity type", "count": "Count"},
                  title="Named entities by type")


def plot_sentiment_timeline(rows: list[dict]):
    if not rows:
        return None
    fig = px.line(pd.DataFrame(rows), x="index", y="polarity", markers=True, hover_data=["sentence"],
                  labels={"index": "Sentence", "polarity": "Sentiment (−1 negative … +1 positive)"},
                  title="Sentiment through the document")
    fig.add_hline(y=0, line_dash="dot")
    return fig
