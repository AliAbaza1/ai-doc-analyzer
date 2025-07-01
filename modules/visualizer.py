import plotly.express as px
import pandas as pd

def plot_entity_distribution(entities):
    """
    يرسم توزيع الكيانات المسماة المستخرجة من النص.
    
    :param entities: قائمة من الكيانات [{'entity': 'Apple', 'label': 'ORG'}, ...]
    """
    if not entities:
        return None

    df = pd.DataFrame(entities)
    fig = px.bar(df['label'].value_counts().reset_index(),
                 x='index', y='label',
                 labels={'index': 'Entity Label', 'label': 'Count'},
                 title='Named Entity Distribution')
    return fig

def plot_sentiment_timeline(sentiments):
    """
    يرسم تغير المشاعر مع مرور الوقت.
    
    :param sentiments: قائمة من القيم {'sentence': str, 'polarity': float, 'index': int}
    """
    if not sentiments:
        return None

    df = pd.DataFrame(sentiments)
    fig = px.line(df, x='index', y='polarity', markers=True,
                  labels={'index': 'Sentence Number', 'polarity': 'Sentiment Polarity'},
                  title='Sentiment Timeline')
    return fig
