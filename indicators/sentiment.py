# indicators/sentiment.py
import pandas as pd

def calculate_sentiment(data, overbought=70, oversold=30):
    data['Sentiment'] = 0
    data.loc[data['RSI'] > overbought, 'Sentiment'] = -1  # Overbought (negative sentiment)
    data.loc[data['RSI'] < oversold, 'Sentiment'] = 1     # Oversold (positive sentiment)
    return data