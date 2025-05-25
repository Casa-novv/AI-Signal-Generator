# config/settings.py

# Forex symbols and timeframes
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
TIMEFRAMES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "H1": 60,
    "H4": 240,
    "D1": 1440
}

# MT5 initialization status
MT5_INITIALIZED = False

# Lookback periods for indicators
SMA_WINDOW = 10
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
LIQUIDITY_LOOKBACK = 20