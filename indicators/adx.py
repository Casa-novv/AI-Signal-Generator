# indicators/adx.py
import talib
import pandas as pd
import numpy as np

def calculate_adx(data, window=14):
    """Calculate ADX using TA-Lib 0.6.3"""
    try:
        # TA-Lib 0.6.3 ADX calculation
        data['ADX'] = talib.ADX(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=window
        )
        
        # Calculate +DI and -DI for additional analysis
        data['PLUS_DI'] = talib.PLUS_DI(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=window
        )
        
        data['MINUS_DI'] = talib.MINUS_DI(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=window
        )
        
        # ADX trend strength signals
        data['ADX_Trend'] = 'Weak'
        data.loc[data['ADX'] > 25, 'ADX_Trend'] = 'Strong'
        data.loc[data['ADX'] > 50, 'ADX_Trend'] = 'Very Strong'
        
        return data
    except Exception as e:
        print(f"Error calculating ADX with TA-Lib 0.6.3: {e}")
        return data
