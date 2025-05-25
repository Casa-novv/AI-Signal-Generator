# utils/mt5_initializer.py
import MetaTrader5 as mt5

# Forex timeframes
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1
}

def initialize_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return False
    print("MT5 initialized successfully")
    return True

def shutdown_mt5():
    mt5.shutdown()
    print("MT5 connection shut down")