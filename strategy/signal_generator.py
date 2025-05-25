# strategy/signal_generator.py
import pandas as pd

def generate_signals(data):
    data['Signal'] = 0
    # Example signal generation logic
    data.loc[(data['SMA'] > data['close']) & (data['RSI'] < 30), 'Signal'] = 1  # Buy signal
    data.loc[(data['SMA'] < data['close']) & (data['RSI'] > 70), 'Signal'] = -1  # Sell signal
    return data
