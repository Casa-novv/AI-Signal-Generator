# app.py
import streamlit as st
import pandas as pd
from config import settings
from utils import mt5_initializer
from data import data_loader
from indicators import sma, rsi, macd, liquidity
from strategy import signal_generator
from visualization import chart

# Initialize MT5
if not mt5_initializer.initialize_mt5():
    st.error("MT5 initialization failed")
    st.stop()

st.title("AI Forex Signal Generator")

# User inputs
symbol = st.selectbox("Select Symbol", settings.SYMBOLS)
timeframe = settings.TIMEFRAMES["H1"]
num_candles = 1000

# Fetch data
data = data_loader.fetch_real_time_data(symbol, timeframe, num_candles)

# Calculate indicators
data = sma.calculate_sma(data, settings.SMA_WINDOW)
data = rsi.calculate_rsi(data, settings.RSI_WINDOW)
data = macd.calculate_macd(data, settings.MACD_FAST, settings.MACD_SLOW)
data = liquidity.calculate_liquidity_zones(data, settings.LIQUIDITY_LOOKBACK)

# Generate signals
data = signal_generator.generate_signals(data)

# Create and display chart
fig = chart.create_chart(data, symbol)
st.plotly_chart(fig)

# Shutdown MT5
mt5_initializer.shutdown_mt5()