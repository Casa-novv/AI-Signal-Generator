import talib
import numpy as np
import pandas as pd

print("🧪 Testing TA-Lib 0.6.3...")
print("✅ TA-Lib version:", talib.__version__)

# Create sample OHLCV data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')

# Generate realistic price data
base_price = 100
price_changes = np.random.randn(100) * 2
prices = base_price + np.cumsum(price_changes)

data = pd.DataFrame({
    'time': dates,
    'open': prices,
    'high': prices + np.abs(np.random.randn(100)),
    'low': prices - np.abs(np.random.randn(100)),
    'close': prices + np.random.randn(100) * 0.5,
    'volume': np.random.randint(1000, 10000, 100)
})

print(f"✅ Sample data created: {len(data)} rows")

# Test TA-Lib functions
try:
    # Test SMA
    sma_20 = talib.SMA(data['close'].values, timeperiod=20)
    print("✅ SMA-20 calculation: PASSED")
    
    # Test RSI
    rsi = talib.RSI(data['close'].values, timeperiod=14)
    print("✅ RSI calculation: PASSED")
    
    # Test MACD
    macd, macd_signal, macd_hist = talib.MACD(data['close'].values)
    print("✅ MACD calculation: PASSED")
    
    # Test ADX
    adx = talib.ADX(data['high'].values, data['low'].values, data['close'].values)
    print("✅ ADX calculation: PASSED")
    
    # Test Bollinger Bands
    bb_upper, bb_middle, bb_lower = talib.BBANDS(data['close'].values)
    print("✅ Bollinger Bands calculation: PASSED")
    
    # Test Stochastic
    slowk, slowd = talib.STOCH(data['high'].values, data['low'].values, data['close'].values)
    print("✅ Stochastic calculation: PASSED")
    
    print(f"\n🎉 All TA-Lib functions working perfectly!")
    print(f"📊 Available TA-Lib functions: {len(talib.get_functions())}")
    print("🚀 Ready to run your AI Forex Signal Generator!")
    
except Exception as e:
    print(f"❌ Error testing TA-Lib functions: {e}")
