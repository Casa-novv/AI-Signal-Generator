# indicators/sma.py
import pandas as pd

def calculate_sma(data, windows=[10, 20, 50]):
    for window in windows:
        data[f'SMA_{window}'] = data['close'].rolling(window=window).mean()
    return data