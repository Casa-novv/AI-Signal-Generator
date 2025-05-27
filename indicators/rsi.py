# indicators/rsi.py
import talib
import pandas as pd
import numpy as np

def calculate_rsi(data, window=14, overbought=70, oversold=30):
    """Calculate RSI using TA-Lib 0.6.3"""
    try:
        # TA-Lib 0.6.3 RSI calculation
        data['RSI'] = talib.RSI(data['close'].values, timeperiod=window)
        
        # Add signal levels
        data['RSI_Overbought'] = overbought
        data['RSI_Oversold'] = oversold
        
        # Generate RSI signals
        data['RSI_Signal'] = 0
        data.loc[data['RSI'] < oversold, 'RSI_Signal'] = 1  # Buy signal
        data.loc[data['RSI'] > overbought, 'RSI_Signal'] = -1  # Sell signal
        
        return data
    except Exception as e:
        print(f"Error calculating RSI with TA-Lib 0.6.3: {e}")
        return data
