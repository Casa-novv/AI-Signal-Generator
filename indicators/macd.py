# indicators/macd.py
import pandas as pd

def calculate_macd(data, fast=12, slow=26):
    data['MACD'] = data['close'].ewm(span=fast, adjust=False).mean() - data['close'].ewm(span=slow, adjust=False).mean()
    return data