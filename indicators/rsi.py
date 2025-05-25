# indicators/rsi.py
import pandas as pd

def calculate_rsi(data, window=14, overbought=70, oversold=30):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI_Overbought'] = overbought
    data['RSI_Oversold'] = oversold
    return data