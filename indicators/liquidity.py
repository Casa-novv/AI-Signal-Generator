# indicators/liquidity.py
import pandas as pd

def calculate_liquidity_zones(data, lookback=20):
    data['HighZone'] = data['high'].rolling(window=lookback).max()
    data['LowZone'] = data['low'].rolling(window=lookback).min()
    return data