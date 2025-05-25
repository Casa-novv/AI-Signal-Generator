# indicators/sma.py
import pandas as pd

def calculate_sma(data, window=10):
    data['SMA'] = data['close'].rolling(window=window).mean()
    return data