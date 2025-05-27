# indicators/macd.py
import talib
import pandas as pd
import numpy as np

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD using TA-Lib 0.6.3"""
    try:
        # TA-Lib 0.6.3 MACD calculation
        macd, macd_signal, macd_histogram = talib.MACD(
            data['close'].values, 
            fastperiod=fast, 
            slowperiod=slow, 
            signalperiod=signal
        )
        
        data['MACD'] = macd
        data['MACD_Signal'] = macd_signal
        data['MACD_Histogram'] = macd_histogram
        
        # Generate MACD signals
        data['MACD_Signal_Line'] = 0
        data.loc[data['MACD'] > data['MACD_Signal'], 'MACD_Signal_Line'] = 1
        data.loc[data['MACD'] < data['MACD_Signal'], 'MACD_Signal_Line'] = -1
        
        return data
    except Exception as e:
        print(f"Error calculating MACD with TA-Lib 0.6.3: {e}")
        return data
