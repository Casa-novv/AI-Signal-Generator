# indicators/adx.py
import pandas as pd

def calculate_adx(data, window=14):
    # Calculate True Range
    data['High-Low'] = data['high'] - data['low']
    data['High-PrevClose'] = abs(data['high'] - data['close'].shift(1))
    data['Low-PrevClose'] = abs(data['low'] - data['close'].shift(1))
    data['TR'] = data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    
    # CalculateDirectionalMovement
    data['+DM'] = ((data['high'] - data['high'].shift(1)) > (data['low'].shift(1) - data['low'])) * (data['high'] - data['high'].shift(1))
    data['+DM'] = data['+DM'].where(data['+DM'] > 0, 0)
    data['-DM'] = ((data['low'].shift(1) - data['low']) > (data['high'] - data['high'].shift(1))) * (data['low'].shift(1) - data['low'])
    data['-DM'] = data['-DM'].where(data['-DM'] > 0, 0)
    
    # Calculate Smoothed Directional Movement
    data['+DM_MA'] = data['+DM'].rolling(window=window).sum()
    data['-DM_MA'] = data['-DM'].rolling(window=window).sum()
    
    # Calculate Directional Index
    data['+DI'] = (data['+DM_MA'] / data['TR'].rolling(window=window).sum()) * 100
    data['-DI'] = (data['-DM_MA'] / data['TR'].rolling(window=window).sum()) * 100
    
    # Calculate Average Directional Index
    data['ADX'] = ((data['+DI'] - data['-DI']).abs() / (data['+DI'] + data['-DI'])) * 100
    data['ADX'] = data['ADX'].rolling(window=window).mean()
    
    return data