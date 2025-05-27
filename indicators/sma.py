# indicators/sma.py
import talib
import pandas as pd
import numpy as np

def calculate_sma(data, windows=[10, 20, 50, 200]):
    """Calculate Simple Moving Average using TA-Lib 0.6.3"""
    try:
        for window in windows:
            # TA-Lib 0.6.3 optimized syntax
            data[f'SMA_{window}'] = talib.SMA(data['close'].values, timeperiod=window)
        return data
    except Exception as e:
        print(f"Error calculating SMA with TA-Lib 0.6.3: {e}")
        return data