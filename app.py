# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from config import settings
from utils import mt5_initializer
from data import data_loader
from indicators import sma, rsi, macd, liquidity, order_blocks, adx, sentiment
from strategy import signal_generator, ml_model
from visualization import chart

# Initialize MT5
if not mt5_initializer.initialize_mt5():
    st.error("MT5 initialization failed")
    st.stop()

st.title("AI Forex Signal Generator")

# User inputs
symbol = st.selectbox("Select Symbol", settings.SYMBOLS)
timeframe = mt5_initializer.TIMEFRAMES["H1"]  # Use the timeframe from mt5_initializer
num_candles = 1000

# Fetch data
data = data_loader.fetch_real_time_data(symbol, timeframe, num_candles)

# Calculate indicators
data = sma.calculate_sma(data, windows=settings.SMA_WINDOWS)
data = rsi.calculate_rsi(data, window=settings.RSI_WINDOW, overbought=settings.RSI_OVERBOUGHT, oversold=settings.RSI_OVERSOLD)
data = macd.calculate_macd(data, fast=settings.MACD_FAST, slow=settings.MACD_SLOW, signal=settings.MACD_SIGNAL)
data = liquidity.calculate_liquidity_zones(data, settings.LIQUIDITY_LOOKBACK)
data = order_blocks.calculate_order_blocks(data, lookback=settings.ORDER_BLOCKS_LOOKBACK, threshold=settings.ORDER_BLOCKS_THRESHOLD)
data = adx.calculate_adx(data, window=settings.ADX_WINDOW)
data = sentiment.calculate_sentiment(data, overbought=settings.RSI_OVERBOUGHT, oversold=settings.RSI_OVERSOLD)

# Generate signals
data = signal_generator.generate_signals(data)

# Train or load ML model
try:
    model = joblib.load('ml_model.pkl')
except:
    model = ml_model.train_model(data)

# Predict signals using ML model
data = ml_model.predict_signal(data, model)

# Generate combined signals
data = signal_generator.generate_combined_signals(data)

# Create and display chart
fig = chart.create_chart(data, symbol)
st.plotly_chart(fig)

# Display signals
st.subheader("Traditional Signals")
st.write(data[['datetime', 'Final_Signal']].tail())

st.subheader("ML Signals")
st.write(data[['datetime', 'ML_Signal']].tail())

st.subheader("Combined Signals")
st.write(data[['datetime', 'Combined_Signal']].tail())

# Shutdown MT5
mt5_initializer.shutdown_mt5()