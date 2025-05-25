# data/data_loader.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

def fetch_historical_data(symbol, timeframe, num_candles):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    return df

def fetch_real_time_data(symbol, timeframe, num_candles):
    return fetch_historical_data(symbol, timeframe, num_candles)