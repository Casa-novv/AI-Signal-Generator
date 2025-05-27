# data/data_loader.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
import os
import json

# Configure logging
logger = logging.getLogger(__name__)

# Try to import MT5, but make it optional
try:
    from utils import mt5_initializer
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 not available. Only CSV data source will work.")

def fetch_real_time_data(symbol: str, timeframe: int, count: int = 1000) -> Optional[pd.DataFrame]:
    """
    Fetch real-time data from MT5
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        timeframe: MT5 timeframe constant
        count: Number of bars to fetch
    
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    if not MT5_AVAILABLE:
        logger.error("MT5 not available for real-time data")
        return None
    
    try:
        # Initialize MT5 if not already done
        if not mt5_initializer.check_connection():
            if not mt5_initializer.initialize_mt5():
                logger.error("Failed to initialize MT5")
                return None
        
        # Fetch market data
        data = mt5_initializer.get_market_data(symbol, timeframe, count)
        
        if data is None or data.empty:
            logger.error(f"No data received for {symbol}")
            return None
        
        # Ensure required columns exist and rename if necessary
        required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        
        # Check if data has the required structure
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Data missing required columns. Available: {list(data.columns)}")
            return None
        
        # Rename tick_volume to volume if needed
        if 'tick_volume' in data.columns and 'volume' not in data.columns:
            data = data.rename(columns={'tick_volume': 'volume'})
        
        # Ensure time column is datetime
        if 'time' in data.columns:
            data['time'] = pd.to_datetime(data['time'])
        
        # Sort by time and reset index
        data = data.sort_values('time').reset_index(drop=True)
        
        logger.info(f"Successfully fetched {len(data)} bars for {symbol}")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching real-time data for {symbol}: {e}")
        return None

def fetch_historical_data(symbol: str, timeframe: int, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """
    Fetch historical data from MT5 for a specific date range
    
    Args:
        symbol: Trading symbol
        timeframe: MT5 timeframe constant
        start_date: Start date for data
        end_date: End date for data
    
    Returns:
        DataFrame with historical OHLCV data or None if failed
    """
    if not MT5_AVAILABLE:
        logger.error("MT5 not available for historical data")
        return None
    
    try:
        # Initialize MT5 if not already done
        if not mt5_initializer.check_connection():
            if not mt5_initializer.initialize_mt5():
                logger.error("Failed to initialize MT5")
                return None
        
        # Fetch historical data
        data = mt5_initializer.get_market_data_range(symbol, timeframe, start_date, end_date)
        
        if data is None or data.empty:
            logger.error(f"No historical data received for {symbol}")
            return None
        
        # Process data similar to real-time data
        if 'tick_volume' in data.columns and 'volume' not in data.columns:
            data = data.rename(columns={'tick_volume': 'volume'})
        
        if 'time' in data.columns:
            data['time'] = pd.to_datetime(data['time'])
        
        data = data.sort_values('time').reset_index(drop=True)
        
        logger.info(f"Successfully fetched {len(data)} historical bars for {symbol}")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return None

def load_csv_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load data from CSV file
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"CSV file not found: {file_path}")
            return None
        
        # Try to read CSV with different separators and encodings
        separators = [',', ';', '\t']
        encodings = ['utf-8', 'latin-1', 'cp1252']
        data = None
        
        for encoding in encodings:
            for sep in separators:
                try:
                    data = pd.read_csv(file_path, sep=sep, encoding=encoding)
                    if len(data.columns) > 1 and len(data) > 0:  # Successfully parsed
                        logger.info(f"Successfully read CSV with separator '{sep}' and encoding '{encoding}'")
                        break
                except Exception as e:
                    continue
            if data is not None and len(data.columns) > 1:
                break
        
        if data is None or data.empty:
            logger.error(f"Failed to read CSV file: {file_path}")
            return None
        
        # Standardize column names
        data.columns = data.columns.str.lower().str.strip()
        
        # Map common column name variations
        column_mapping = {
            'datetime': 'time',
            'date': 'time',
            'timestamp': 'time',
            'time_stamp': 'time',
            'dt': 'time',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vol': 'volume',
            'tick_volume': 'volume',
            'tickvol': 'volume',
            'real_volume': 'volume'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in data.columns:
                data = data.rename(columns={old_name: new_name})
        
        # Ensure required columns exist
        required_columns = ['time', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"CSV missing required columns: {missing_columns}")
            logger.info(f"Available columns: {list(data.columns)}")
            return None
        
        # Convert time column to datetime
        try:
            # Try different datetime formats
            datetime_formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M',
                '%Y-%m-%d',
                '%d.%m.%Y %H:%M:%S',
                '%d.%m.%Y %H:%M',
                '%d.%m.%Y',
                '%m/%d/%Y %H:%M:%S',
                '%m/%d/%Y %H:%M',
                '%m/%d/%Y'
            ]
            
            parsed = False
            for fmt in datetime_formats:
                try:
                    data['time'] = pd.to_datetime(data['time'], format=fmt)
                    parsed = True
                    break
                except:
                    continue
            
            if not parsed:
                data['time'] = pd.to_datetime(data['time'], infer_datetime_format=True)
                
        except Exception as e:
            logger.error(f"Failed to parse time column: {e}")
            return None
        
        # Convert price columns to numeric
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Add volume column if missing
        if 'volume' not in data.columns:
            data['volume'] = np.random.randint(1000, 10000, len(data))
            logger.info("Added synthetic volume data")
        else:
            data['volume'] = pd.to_numeric(data['volume'], errors='coerce')
            data['volume'] = data['volume'].fillna(1000)
        
        # Remove rows with NaN values in price columns
        data = data.dropna(subset=price_columns)
        
        # Sort by time
        data = data.sort_values('time').reset_index(drop=True)
        
        # Validate OHLC data
        data = validate_ohlc_data(data)
        
        logger.info(f"Successfully loaded {len(data)} rows from CSV")
        return data
        
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        return None

def validate_ohlc_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and fix OHLC data consistency
    
    Args:
        data: DataFrame with OHLC data
    
    Returns:
        Validated DataFrame
    """
    try:
        original_length = len(data)
        
        # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
        for i in range(len(data)):
            open_price = data.iloc[i]['open']
            close_price = data.iloc[i]['close']
            high_price = data.iloc[i]['high']
            low_price = data.iloc[i]['low']
            
            # Fix high price
            min_high = max(open_price, close_price)
            if high_price < min_high:
                data.iloc[i, data.columns.get_loc('high')] = min_high
            
            # Fix low price
            max_low = min(open_price, close_price)
            if low_price > max_low:
                data.iloc[i, data.columns.get_loc('low')] = max_low
        
        # Remove any rows with invalid data
        data = data.dropna(subset=['open', 'high', 'low', 'close'])
        data = data[data['open'] > 0]
        data = data[data['high'] > 0]
        data = data[data['low'] > 0]
        data = data[data['close'] > 0]
        
        # Remove duplicate timestamps
        data = data.drop_duplicates(subset=['time'], keep='last')
        
        # Reset index
        data = data.reset_index(drop=True)
        
        if len(data) != original_length:
            logger.info(f"Data validation: {original_length - len(data)} invalid rows removed")
        
        return data
        
    except Exception as e:
        logger.error(f"Error validating OHLC data: {e}")
        return data

def generate_sample_data(symbol: str = "EURUSD", days: int = 30, timeframe_minutes: int = 60) -> pd.DataFrame:
    """
    Generate sample forex data for testing
    
    Args:
        symbol: Trading symbol
        days: Number of days of data
        timeframe_minutes: Timeframe in minutes
    
    Returns:
        DataFrame with sample OHLCV data
    """
    try:
        # Calculate number of bars
        bars_per_day = 24 * 60 // timeframe_minutes
        total_bars = days * bars_per_day
        
        # Generate time series
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        time_index = pd.date_range(start=start_time, end=end_time, periods=total_bars)
        
        # Generate price data with realistic forex movements
        base_prices = {
            "EURUSD": 1.1000,
            "GBPUSD": 1.3000,
            "USDJPY": 110.00,
            "USDCHF": 0.9200,
            "AUDUSD": 0.7500,
            "USDCAD": 1.2500,
            "NZDUSD": 0.7000
        }
        
        base_price = base_prices.get(symbol, 1.0000)
        
        # Generate random walk for close prices
        volatility = 0.001 if symbol != "USDJPY" else 0.1  # JPY pairs have different volatility
        returns = np.random.normal(0, volatility, total_bars)
        returns[0] = 0  # First return is 0
        
        # Add some trend and mean reversion
        trend = np.sin(np.linspace(0, 4*np.pi, total_bars)) * 0.0005
        returns += trend
        
        close_prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC data
        data = []
        for i in range(total_bars):
            close = close_prices[i]
            
            # Generate open price (close of previous bar or base price for first bar)
            if i == 0:
                open_price = base_price
            else:
                open_price = close_prices[i-1]
            
            # Generate high and low with some randomness
            bar_volatility = abs(close - open_price) + np.random.uniform(0.0001, 0.0020)
            if symbol == "USDJPY":
                bar_volatility *= 100  # Adjust for JPY pairs
            
            high = max(open_price, close) + np.random.uniform(0, bar_volatility)
            low = min(open_price, close) - np.random.uniform(0, bar_volatility)
            
            # Ensure low is not negative
            low = max(low, base_price * 0.5)
            
            # Generate volume with some correlation to volatility
            base_volume = 5000
            volatility_factor = bar_volatility / volatility if volatility > 0 else 1
            volume = int(base_volume * (1 + volatility_factor) * np.random.uniform(0.5, 2.0))
            
            data.append({
                'time': time_index[i],
                'open': round(open_price, 5 if symbol != "USDJPY" else 3),
                'high': round(high, 5 if symbol != "USDJPY" else 3),
                'low': round(low, 5 if symbol != "USDJPY" else 3),
                'close': round(close, 5 if symbol != "USDJPY" else 3),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} sample bars for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        return pd.DataFrame()

def get_available_csv_files(data_directory: str = "data/csv") -> List[str]:
    """
    Get list of available CSV files
    
    Args:
        data_directory: Directory containing CSV files
    
    Returns:
        List of CSV file paths
    """
    try:
        if not os.path.exists(data_directory):
            os.makedirs(data_directory, exist_ok=True)
            logger.info(f"Created data directory: {data_directory}")
            return []
        
        csv_files = []
        for file in os.listdir(data_directory):
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(data_directory, file))
        
        # Sort files by modification time (newest first)
        csv_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        return csv_files
        
    except Exception as e:
        logger.error(f"Error getting CSV files: {e}")
        return []

def save_data_to_csv(data: pd.DataFrame, file_path: str) -> bool:
    """
    Save DataFrame to CSV file
    
    Args:
        data: DataFrame to save
        file_path: Output file path
    
    Returns:
        bool: True if successful
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Save to CSV with proper formatting
        data.to_csv(file_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
        logger.info(f"Data saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")
        return False

def get_data_info(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Get information about the dataset
    
    Args:
        data: DataFrame to analyze
    
    Returns:
        Dictionary with data information
    """
    try:
        if data.empty:
            return {"error": "No data available"}
        
        # Calculate duration
        duration = data['time'].max() - data['time'].min()
        
        # Calculate price statistics
        price_stats = {
            "min": float(data['low'].min()),
            "max": float(data['high'].max()),
            "current": float(data['close'].iloc[-1]),
            "first": float(data['open'].iloc[0]),
            "change": float(data['close'].iloc[-1] - data['open'].iloc[0]),
            "change_percent": float((data['close'].iloc[-1] - data['open'].iloc[0]) / data['open'].iloc[0] * 100)
        }
        
        # Calculate volume statistics if available
        volume_stats = {}
        if 'volume' in data.columns:
            volume_stats = {
                "min": int(data['volume'].min()),
                "max": int(data['volume'].max()),
                "avg": float(data['volume'].mean()),
                "total": int(data['volume'].sum())
            }
        
        # Detect timeframe
        if len(data) > 1:
            time_diff = data['time'].iloc[1] - data['time'].iloc[0]
            timeframe_minutes = time_diff.total_seconds() / 60
        else:
            timeframe_minutes = 0
        
        info = {
            "total_bars": len(data),
            "start_time": data['time'].min().strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": data['time'].max().strftime('%Y-%m-%d %H:%M:%S'),
            "duration_days": duration.days,
            "duration_hours": duration.total_seconds() / 3600,
            "timeframe_minutes": timeframe_minutes,
            "columns": list(data.columns),
            "price_range": price_stats,
            "volume_stats": volume_stats,
            "data_quality": analyze_data_quality(data)
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting data info: {e}")
        return {"error": str(e)}

def analyze_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze data quality and detect issues
    
    Args:
        data: DataFrame to analyze
    
    Returns:
        Dictionary with data quality metrics
    """
    try:
        quality_info = {
            "missing_values": {},
            "duplicate_timestamps": 0,
            "invalid_ohlc": 0,
            "gaps_detected": 0,
            "outliers_detected": 0,
            "overall_score": 100
        }
        
        # Check for missing values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                missing = data[col].isna().sum()
                quality_info["missing_values"][col] = missing
                if missing > 0:
                    quality_info["overall_score"] -= min(20, missing / len(data) * 100)
        
        # Check for duplicate timestamps
        duplicates = data['time'].duplicated().sum()
        quality_info["duplicate_timestamps"] = duplicates
        if duplicates > 0:
            quality_info["overall_score"] -= min(15, duplicates / len(data) * 100)
        
        # Check for invalid OHLC relationships
        invalid_ohlc = 0
        for i in range(len(data)):
            try:
                o, h, l, c = data.iloc[i][['open', 'high', 'low', 'close']]
                if h < max(o, c) or l > min(o, c) or h < l:
                    invalid_ohlc += 1
            except:
                invalid_ohlc += 1
        
        quality_info["invalid_ohlc"] = invalid_ohlc
        if invalid_ohlc > 0:
            quality_info["overall_score"] -= min(25, invalid_ohlc / len(data) * 100)
        
        # Check for time gaps (if timeframe is consistent)
        if len(data) > 2:
            time_diffs = data['time'].diff().dropna()
            if len(time_diffs) > 0:
                expected_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
                gaps = (time_diffs > expected_diff * 1.5).sum()
                quality_info["gaps_detected"] = gaps
                if gaps > 0:
                    quality_info["overall_score"] -= min(10, gaps / len(data) * 100)
        
        # Check for price outliers using IQR method
        price_changes = data['close'].pct_change().dropna()
        if len(price_changes) > 0:
            Q1 = price_changes.quantile(0.25)
            Q3 = price_changes.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((price_changes < (Q1 - 3 * IQR)) | (price_changes > (Q3 + 3 * IQR))).sum()
            quality_info["outliers_detected"] = outliers
            if outliers > 0:
                quality_info["overall_score"] -= min(10, outliers / len(data) * 100)
        
        # Ensure score doesn't go below 0
        quality_info["overall_score"] = max(0, quality_info["overall_score"])
        
        return quality_info
        
    except Exception as e:
        logger.error(f"Error analyzing data quality: {e}")
        return {"error": str(e)}

def resample_data(data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """
    Resample data to a different timeframe
    
    Args:
        data: DataFrame with OHLCV data
        target_timeframe: Target timeframe ('1T', '5T', '15T', '30T', '1H', '4H', '1D')
    
    Returns:
        Resampled DataFrame
    """
    try:
        if data.empty:
            return data
        
        # Set time as index for resampling
        data_indexed = data.set_index('time')
        
        # Define aggregation rules
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Only include columns that exist in the data
        available_agg_rules = {k: v for k, v in agg_rules.items() if k in data_indexed.columns}
        
        # Resample data
        resampled = data_indexed.resample(target_timeframe).agg(available_agg_rules)
        
        # Remove rows with NaN values (incomplete periods)
        resampled = resampled.dropna()
        
        # Reset index to get time back as a column
        resampled = resampled.reset_index()
        
        logger.info(f"Resampled data from {len(data)} to {len(resampled)} bars")
        return resampled
        
    except Exception as e:
        logger.error(f"Error resampling data: {e}")
        return data

def filter_data_by_date(data: pd.DataFrame, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
    """
    Filter data by date range
    
    Args:
        data: DataFrame with time column
        start_date: Start date for filtering
        end_date: End date for filtering
    
    Returns:
        Filtered DataFrame
    """
    try:
        if data.empty:
            return data
        
        filtered_data = data.copy()
        
        if start_date:
            filtered_data = filtered_data[filtered_data['time'] >= start_date]
        
        if end_date:
            filtered_data = filtered_data[filtered_data['time'] <= end_date]
        
        filtered_data = filtered_data.reset_index(drop=True)
        
        logger.info(f"Filtered data: {len(filtered_data)} bars remaining")
        return filtered_data
        
    except Exception as e:
        logger.error(f"Error filtering data by date: {e}")
        return data

def export_data_formats(data: pd.DataFrame, base_filename: str, formats: List[str] = None) -> Dict[str, bool]:
    """
    Export data to multiple formats
    
    Args:
        data: DataFrame to export
        base_filename: Base filename without extension
        formats: List of formats to export ('csv', 'json', 'excel', 'parquet')
    
    Returns:
        Dictionary with format: success status
    """
    if formats is None:
        formats = ['csv', 'json']
    
    results = {}
    
    try:
        # Ensure export directory exists
        export_dir = "data/exports"
        os.makedirs(export_dir, exist_ok=True)
        
        for fmt in formats:
            try:
                filepath = os.path.join(export_dir, f"{base_filename}.{fmt}")
                
                if fmt == 'csv':
                    data.to_csv(filepath, index=False, date_format='%Y-%m-%d %H:%M:%S')
                elif fmt == 'json':
                    # Convert datetime to string for JSON serialization
                    data_json = data.copy()
                    data_json['time'] = data_json['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    data_json.to_json(filepath, orient='records', indent=2)
                elif fmt == 'excel':
                    data.to_excel(filepath, index=False)
                elif fmt == 'parquet':
                    data.to_parquet(filepath, index=False)
                else:
                    results[fmt] = False
                    continue
                
                results[fmt] = True
                logger.info(f"Exported data to {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to export to {fmt}: {e}")
                results[fmt] = False
        
        return results
        
    except Exception as e:
        logger.error(f"Error in export_data_formats: {e}")
        return {fmt: False for fmt in formats}

def load_data_from_api(symbol: str, timeframe: str, source: str = "yahoo") -> Optional[pd.DataFrame]:
    """
    Load data from external APIs (fallback when MT5 is not available)
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe string
        source: Data source ('yahoo', 'alpha_vantage')
    
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    try:
        if source == "yahoo":
            try:
                import yfinance as yf
                
                # Convert symbol format for Yahoo Finance
                if symbol == "EURUSD":
                    yahoo_symbol = "EURUSD=X"
                elif symbol == "GBPUSD":
                    yahoo_symbol = "GBPUSD=X"
                elif symbol == "USDJPY":
                    yahoo_symbol = "USDJPY=X"
                else:
                    yahoo_symbol = f"{symbol}=X"
                
                # Download data
                ticker = yf.Ticker(yahoo_symbol)
                
                # Determine period based on timeframe
                if timeframe in ['1m', '5m', '15m', '30m']:
                    period = "7d"
                    interval = timeframe
                elif timeframe in ['1h', '4h']:
                    period = "60d"
                    interval = timeframe
                else:
                    period = "1y"
                    interval = "1d"
                
                data = ticker.history(period=period, interval=interval)
                
                if data.empty:
                    logger.error(f"No data received from Yahoo Finance for {symbol}")
                    return None
                
                # Rename columns to match our format
                data = data.reset_index()
                data.columns = data.columns.str.lower()
                
                column_mapping = {
                    'datetime': 'time',
                    'adj close': 'adj_close'
                }
                
                for old_name, new_name in column_mapping.items():
                    if old_name in data.columns:
                        data = data.rename(columns={old_name: new_name})
                
                # Keep only required columns
                required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
                data = data[required_cols]
                
                logger.info(f"Successfully loaded {len(data)} bars from Yahoo Finance")
                return data
                
            except ImportError:
                logger.error("yfinance not installed. Install with: pip install yfinance")
                return None
            except Exception as e:
                logger.error(f"Error loading data from Yahoo Finance: {e}")
                return None
        
        else:
            logger.error(f"Unsupported data source: {source}")
            return None
            
    except Exception as e:
        logger.error(f"Error in load_data_from_api: {e}")
        return None

def create_sample_csv_files():
    """
    Create sample CSV files for testing
    """
    try:
        # Create data directory
        csv_dir = "data/csv"
        os.makedirs(csv_dir, exist_ok=True)
        
        # Define sample symbols and their configurations
        sample_configs = [
            {"symbol": "EURUSD", "days": 30, "timeframe": 60},
            {"symbol": "GBPUSD", "days": 30, "timeframe": 60},
            {"symbol": "USDJPY", "days": 30, "timeframe": 60},
            {"symbol": "EURUSD", "days": 7, "timeframe": 15},
            {"symbol": "GBPUSD", "days": 7, "timeframe": 15}
        ]
        
        created_files = []
        
        for config in sample_configs:
            symbol = config["symbol"]
            days = config["days"]
            timeframe = config["timeframe"]
            
            # Generate sample data
            data = generate_sample_data(symbol, days, timeframe)
            
            if not data.empty:
                # Create filename
                timeframe_str = f"{timeframe}M" if timeframe < 60 else f"{timeframe//60}H"
                filename = f"{symbol}_{timeframe_str}_{days}D.csv"
                filepath = os.path.join(csv_dir, filename)
                
                # Save to CSV
                if save_data_to_csv(data, filepath):
                    created_files.append(filename)
                    logger.info(f"Created sample file: {filename}")
        
        logger.info(f"Created {len(created_files)} sample CSV files")
        return created_files
        
    except Exception as e:
        logger.error(f"Error creating sample CSV files: {e}")
        return []

def detect_csv_format(file_path: str) -> Dict[str, Any]:
    """
    Detect CSV format and structure
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        Dictionary with format information
    """
    try:
        format_info = {
            "separator": ",",
            "encoding": "utf-8",
            "has_header": True,
            "datetime_column": None,
            "datetime_format": None,
            "columns": [],
            "sample_data": []
        }
        
        # Try different separators
        separators = [',', ';', '\t', '|']
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        best_result = None
        max_columns = 0
        
        for encoding in encodings:
            for sep in separators:
                try:
                    # Read first few rows to detect format
                    sample = pd.read_csv(file_path, sep=sep, encoding=encoding, nrows=5)
                    
                    if len(sample.columns) > max_columns and len(sample) > 0:
                        max_columns = len(sample.columns)
                        best_result = {
                            "separator": sep,
                            "encoding": encoding,
                            "columns": list(sample.columns),
                            "sample_data": sample.head(3).to_dict('records')
                        }
                        
                        # Try to detect datetime column
                        for col in sample.columns:
                            col_lower = col.lower().strip()
                            if any(keyword in col_lower for keyword in ['time', 'date', 'datetime', 'timestamp']):
                                format_info["datetime_column"] = col
                                
                                # Try to detect datetime format
                                sample_value = str(sample[col].iloc[0])
                                datetime_formats = [
                                    '%Y-%m-%d %H:%M:%S',
                                    '%Y-%m-%d %H:%M',
                                    '%Y-%m-%d',
                                    '%d.%m.%Y %H:%M:%S',
                                    '%d.%m.%Y %H:%M',
                                    '%d.%m.%Y',
                                    '%m/%d/%Y %H:%M:%S',
                                    '%m/%d/%Y %H:%M',
                                    '%m/%d/%Y'
                                ]
                                
                                for fmt in datetime_formats:
                                    try:
                                        datetime.strptime(sample_value, fmt)
                                        format_info["datetime_format"] = fmt
                                        break
                                    except:
                                        continue
                                break
                        
                except Exception:
                    continue
        
        if best_result:
            format_info.update(best_result)
        
        return format_info
        
    except Exception as e:
        logger.error(f"Error detecting CSV format: {e}")
        return format_info

def validate_data_completeness(data: pd.DataFrame, expected_timeframe_minutes: int = None) -> Dict[str, Any]:
    """
    Validate data completeness and detect missing periods
    
    Args:
        data: DataFrame with time series data
        expected_timeframe_minutes: Expected timeframe in minutes
    
    Returns:
        Dictionary with completeness analysis
    """
    try:
        if data.empty or len(data) < 2:
            return {"error": "Insufficient data for analysis"}
        
        # Sort data by time
        data_sorted = data.sort_values('time').reset_index(drop=True)
        
        # Calculate time differences
        time_diffs = data_sorted['time'].diff().dropna()
        
        # Detect most common time difference (expected timeframe)
        if expected_timeframe_minutes:
            expected_diff = timedelta(minutes=expected_timeframe_minutes)
        else:
            # Use mode or median of time differences
            if len(time_diffs) > 0:
                expected_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
            else:
                expected_diff = timedelta(hours=1)  # Default to 1 hour
        
        # Find gaps
        tolerance = expected_diff * 1.5  # Allow 50% tolerance
        gaps = time_diffs[time_diffs > tolerance]
        
        # Calculate expected vs actual data points
        total_duration = data_sorted['time'].max() - data_sorted['time'].min()
        expected_points = int(total_duration / expected_diff) + 1
        actual_points = len(data_sorted)
        completeness_ratio = actual_points / expected_points if expected_points > 0 else 1.0
        
        # Identify missing periods
        missing_periods = []
        if len(gaps) > 0:
            gap_indices = time_diffs[time_diffs > tolerance].index
            for idx in gap_indices:
                start_time = data_sorted.iloc[idx-1]['time']
                end_time = data_sorted.iloc[idx]['time']
                missing_periods.append({
                    "start": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "end": end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "duration_hours": (end_time - start_time).total_seconds() / 3600
                })
        
        analysis = {
            "total_points": actual_points,
            "expected_points": expected_points,
            "completeness_ratio": round(completeness_ratio, 4),
            "completeness_percentage": round(completeness_ratio * 100, 2),
            "gaps_found": len(gaps),
            "missing_periods": missing_periods,
            "expected_timeframe_minutes": expected_diff.total_seconds() / 60,
            "data_quality_score": min(100, completeness_ratio * 100)
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error validating data completeness: {e}")
        return {"error": str(e)}

def merge_data_sources(data_list: List[pd.DataFrame], method: str = "concat") -> pd.DataFrame:
    """
    Merge multiple data sources
    
    Args:
        data_list: List of DataFrames to merge
        method: Merge method ('concat', 'outer_join', 'inner_join')
    
    Returns:
        Merged DataFrame
    """
    try:
        if not data_list or len(data_list) == 0:
            return pd.DataFrame()
        
        if len(data_list) == 1:
            return data_list[0].copy()
        
        # Filter out empty DataFrames
        valid_data = [df for df in data_list if not df.empty]
        
        if not valid_data:
            return pd.DataFrame()
        
        if method == "concat":
            # Simple concatenation
            merged = pd.concat(valid_data, ignore_index=True)
            merged = merged.sort_values('time').reset_index(drop=True)
            
            # Remove duplicates based on time
            merged = merged.drop_duplicates(subset=['time'], keep='last')
            
        elif method == "outer_join":
            # Outer join on time column
            merged = valid_data[0].copy()
            for df in valid_data[1:]:
                merged = pd.merge(merged, df, on='time', how='outer', suffixes=('', '_dup'))
                
                # Handle duplicate columns by taking non-null values
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    dup_col = f"{col}_dup"
                    if dup_col in merged.columns:
                        merged[col] = merged[col].fillna(merged[dup_col])
                        merged = merged.drop(columns=[dup_col])
            
            merged = merged.sort_values('time').reset_index(drop=True)
            
        elif method == "inner_join":
            # Inner join on time column
            merged = valid_data[0].copy()
            for df in valid_data[1:]:
                merged = pd.merge(merged, df, on='time', how='inner', suffixes=('', '_dup'))
                
                # Handle duplicate columns
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    dup_col = f"{col}_dup"
                    if dup_col in merged.columns:
                        merged = merged.drop(columns=[dup_col])
            
            merged = merged.sort_values('time').reset_index(drop=True)
        
        else:
            logger.error(f"Unknown merge method: {method}")
            return pd.concat(valid_data, ignore_index=True).sort_values('time').reset_index(drop=True)
        
        logger.info(f"Merged {len(valid_data)} data sources into {len(merged)} records")
        return merged
        
    except Exception as e:
        logger.error(f"Error merging data sources: {e}")
        return pd.DataFrame()

def get_symbol_info(symbol: str) -> Dict[str, Any]:
    """
    Get information about a trading symbol
    
    Args:
        symbol: Trading symbol
    
    Returns:
        Dictionary with symbol information
    """
    try:
        # Predefined symbol information
        symbol_info = {
            "EURUSD": {
                "name": "Euro vs US Dollar",
                "type": "Major Currency Pair",
                "pip_value": 0.0001,
                "typical_spread": 1.5,
                "session_times": "24/5",
                "volatility": "Medium"
            },
            "GBPUSD": {
                "name": "British Pound vs US Dollar",
                "type": "Major Currency Pair",
                "pip_value": 0.0001,
                "typical_spread": 2.0,
                "session_times": "24/5",
                "volatility": "High"
            },
            "USDJPY": {
                "name": "US Dollar vs Japanese Yen",
                "type": "Major Currency Pair",
                "pip_value": 0.01,
                "typical_spread": 1.5,
                "session_times": "24/5",
                "volatility": "Medium"
            },
            "USDCHF": {
                "name": "US Dollar vs Swiss Franc",
                "type": "Major Currency Pair",
                "pip_value": 0.0001,
                "typical_spread": 2.5,
                "session_times": "24/5",
                "volatility": "Medium"
            },
            "AUDUSD": {
                "name": "Australian Dollar vs US Dollar",
                "type": "Major Currency Pair",
                "pip_value": 0.0001,
                "typical_spread": 2.0,
                "session_times": "24/5",
                "volatility": "Medium-High"
            },
            "USDCAD": {
                "name": "US Dollar vs Canadian Dollar",
                "type": "Major Currency Pair",
                "pip_value": 0.0001,
                "typical_spread": 2.5,
                "session_times": "24/5",
                "volatility": "Medium"
            },
            "NZDUSD": {
                "name": "New Zealand Dollar vs US Dollar",
                "type": "Major Currency Pair",
                "pip_value": 0.0001,
                "typical_spread": 3.0,
                "session_times": "24/5",
                "volatility": "High"
            }
        }
        
        return symbol_info.get(symbol, {
            "name": symbol,
            "type": "Unknown",
            "pip_value": 0.0001,
            "typical_spread": 2.0,
            "session_times": "24/5",
            "volatility": "Unknown"
        })
        
    except Exception as e:
        logger.error(f"Error getting symbol info: {e}")
        return {"error": str(e)}

# Export functions and constants
__all__ = [
    'fetch_real_time_data',
    'fetch_historical_data',
    'load_csv_data',
    'validate_ohlc_data',
    'generate_sample_data',
    'get_available_csv_files',
    'save_data_to_csv',
    'get_data_info',
    'analyze_data_quality',
    'resample_data',
    'filter_data_by_date',
    'export_data_formats',
    'load_data_from_api',
    'create_sample_csv_files',
    'detect_csv_format',
    'validate_data_completeness',
    'merge_data_sources',
    'get_symbol_info',
    'MT5_AVAILABLE'
]

# Initialize sample data on first import if no CSV files exist
def _initialize_sample_data():
    """Initialize sample data if no CSV files exist"""
    try:
        csv_files = get_available_csv_files()
        if not csv_files:
            logger.info("No CSV files found. Creating sample data...")
            create_sample_csv_files()
    except Exception as e:
        logger.error(f"Error initializing sample data: {e}")

# Auto-initialize sample data
_initialize_sample_data()
