# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from config import settings
from utils import mt5_initializer
from data import data_loader
from indicators import sma, rsi, macd, liquidity, order_blocks
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
data = sma.calculate_sma(data, windows=settings.SMA_WINDOWS)
data = rsi.calculate_rsi(data, window=settings.RSI_WINDOW, overbought=settings.RSI_OVERBOUGHT, oversold=settings.RSI_OVERSOLD)
data = macd.calculate_macd(data, fast=settings.MACD_FAST, slow=settings.MACD_SLOW, signal=settings.MACD_SIGNAL)
data = liquidity.calculate_liquidity_zones(data, settings.LIQUIDITY_LOOKBACK)
data = order_blocks.calculate_order_blocks(data, lookback=settings.ORDER_BLOCKS_LOOKBACK, threshold=settings.ORDER_BLOCKS_THRESHOLD)

# Generate signals
data = signal_generator.generate_signals(data)

# Create and display chart
fig = chart.create_chart(data, symbol)
st.plotly_chart(fig)

# Shutdown MT5
mt5_initializer.shutdown_mt5()