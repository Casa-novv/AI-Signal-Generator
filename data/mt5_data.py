import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def initialize_mt5():
    """Initialize MT5 connection"""
    try:
        if not mt5.initialize():
            logger.warning("MT5 initialization failed - MT5 may not be installed")
            return False
        
        account_info = mt5.account_info()
        if account_info is None:
            logger.warning("Failed to get account info - MT5 may not be logged in")
            return False
        
        logger.info(f"MT5 initialized successfully. Account: {account_info.login}")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing MT5: {e}")
        return False

def get_mt5_data(symbol="EURUSD", timeframe="H1", count=1000):
    """Fetch data from MT5"""
    try:
        # Try to initialize MT5
        if not initialize_mt5():
            logger.warning("MT5 not available, returning sample data")
            return get_sample_data(symbol, count)
        
        # Map timeframe string to MT5 constant
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        
        mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
        
        # Get rates
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No data received for {symbol}, returning sample data")
            return get_sample_data(symbol, count)
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Rename columns to match expected format
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        logger.info(f"Successfully fetched {len(df)} records for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching MT5 data: {e}")
        return get_sample_data(symbol, count)
    
    finally:
        # Always try to shutdown MT5 connection
        try:
            mt5.shutdown()
        except:
            pass

def get_sample_data(symbol="EURUSD", count=1000):
    """Generate sample forex data when MT5 is not available"""
    try:
        import numpy as np
        
        logger.info(f"Generating sample data for {symbol}")
        
        # Generate realistic forex data
        np.random.seed(42)  # For reproducible results
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=count)
        dates = pd.date_range(start=start_date, end=end_date, freq='H')[:count]
        
        # Generate price data with realistic forex movements
        base_price = 1.1000 if symbol == "EURUSD" else 1.0000
        
        # Generate returns with some autocorrelation (more realistic)
        returns = np.random.normal(0, 0.0005, count)  # Small forex-like volatility
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]  # Add some momentum
        
        # Calculate prices
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC data
        data = []
        for i in range(count):
            open_price = prices[i]
            
            # Generate realistic high/low around the price
            volatility = abs(returns[i]) * 2
            high_price = open_price + np.random.exponential(volatility)
            low_price = open_price - np.random.exponential(volatility)
            
            # Close price with some bias towards the direction
            close_bias = returns[i] * 0.5
            close_price = open_price + close_bias + np.random.normal(0, volatility * 0.5)
            
            # Ensure OHLC logic is maintained
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate volume
            volume = np.random.randint(100, 1000)
            
            data.append({
                'time': dates[i],
                'open': round(open_price, 5),
                'high': round(high_price, 5),
                'low': round(low_price, 5),
                'close': round(close_price, 5),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)
        
        logger.info(f"Generated {len(df)} sample records for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        # Return minimal data structure
        return pd.DataFrame({
            'open': [1.1000],
            'high': [1.1010],
            'low': [1.0990],
            'close': [1.1005],
            'volume': [500]
        }, index=[datetime.now()])

def get_available_symbols():
    """Get available symbols from MT5 or return common forex pairs"""
    try:
        if initialize_mt5():
            symbols = mt5.symbols_get()
            if symbols:
                symbol_names = [s.name for s in symbols if 'USD' in s.name or 'EUR' in s.name]
                mt5.shutdown()
                return sorted(symbol_names[:20])  # Return top 20
        
        # Return common forex pairs if MT5 not available
        return [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
            "NZDUSD", "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "EURAUD",
            "EURCHF", "AUDNZD", "NZDJPY", "GBPAUD", "GBPCAD", "EURNZD",
            "AUDCAD", "GBPCHF"
        ]
        
    except Exception as e:
        logger.error(f"Error getting symbols: {e}")
        return ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]

def test_mt5_connection():
    """Test MT5 connection and return status"""
    try:
        if initialize_mt5():
            account_info = mt5.account_info()
            mt5.shutdown()
            return {
                'connected': True,
                'account': account_info.login if account_info else 'Unknown',
                'message': 'MT5 connected successfully'
            }
        else:
            return {
                'connected': False,
                'account': None,
                'message': 'MT5 not available - using sample data'
            }
    except Exception as e:
        return {
            'connected': False,
            'account': None,
            'message': f'MT5 error: {str(e)}'
        }
