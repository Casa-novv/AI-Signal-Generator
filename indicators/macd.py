# indicators/macd.py
import pandas as pd

def calculate_macd(data, fast=12, slow=26, signal=9):
    data['MACD'] = data['close'].ewm(span=fast, adjust=False).mean() - data['close'].ewm(span=slow, adjust=False).mean()
    data['MACD_Signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    return data