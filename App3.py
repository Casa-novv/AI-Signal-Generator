import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import MetaTrader5 as mt5

# Initialize MT5
if not mt5.initialize():
    st.error("MT5 initialization failed")
    st.stop()

# Load Data
from_time = datetime.now() - timedelta(days=10)
rates = mt5.copy_rates_from("EURUSD", mt5.TIMEFRAME_H1, from_time, 1000)
data = pd.DataFrame(rates)
data['datetime'] = pd.to_datetime(data['time'], unit='s')

# Calculate Indicators
data['SMA10'] = data['close'].rolling(window=10).mean()
data['SMA20'] = data['close'].rolling(window=20).mean()
data['RSI'] = data['close'].rolling(window=14).apply(lambda x: 100 - (100 / (1 + x.diff().mean() / x.diff().abs().mean())))
data['MACD'] = data['close'].ewm(span=12, adjust=False).mean() - data['close'].ewm(span=26, adjust=False).mean()

# Create Plotly Chart
fig = go.Figure()

# Candlesticks
fig.add_trace(go.Candlestick(x=data['datetime'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name='Candlesticks'))

# SMA Lines
fig.add_trace(go.Scatter(x=data['datetime'], y=data['SMA10'], line=dict(color='blue'), name='SMA10'))
fig.add_trace(go.Scatter(x=data['datetime'], y=data['SMA20'], line=dict(color='orange'), name='SMA20'))

# RSI
fig.add_trace(go.Scatter(x=data['datetime'], y=data['RSI'], line=dict(color='green'), name='RSI'))

# MACD
fig.add_trace(go.Scatter(x=data['datetime'], y=data['MACD'], line=dict(color='red'), name='MACD'))

# Layout
fig.update_layout(title='Forex Trading Dashboard', xaxis_title='Time', yaxis_title='Price', template='plotly_dark')

# Display Chart
st.plotly_chart(fig, use_container_width=True)

# Shutdown MT5
mt5.shutdown()
