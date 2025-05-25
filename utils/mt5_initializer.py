# utils/mt5_initializer.py
import MetaTrader5 as mt5

def initialize_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return False
    print("MT5 initialized successfully")
    return True

def shutdown_mt5():
    mt5.shutdown()
    print("MT5 connection shut down")