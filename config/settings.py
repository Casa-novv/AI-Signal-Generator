# config/settings.py
import os
from typing import List, Dict, Any

# Application Settings
APP_NAME = "AI Forex Signal Generator"
APP_VERSION = "1.0.0"
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"

# Data Settings
DEFAULT_SYMBOL = "EURUSD"
DEFAULT_TIMEFRAME = "H1"
DEFAULT_BARS_COUNT = 1000

# Supported symbols
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", 
    "AUDUSD", "USDCAD", "NZDUSD", "EURJPY",
    "GBPJPY", "EURGBP", "AUDCAD", "AUDCHF"
]

# Timeframe mappings for MT5
MT5_TIMEFRAMES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 16408,
    "W1": 32769,
    "MN1": 49153
}

# Technical Indicator Settings
SMA_WINDOWS = [10, 20, 50, 200]
RSI_WINDOW = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

ADX_WINDOW = 14
ADX_THRESHOLD = 25

# Order Blocks Settings
ORDER_BLOCKS_LOOKBACK = 50
ORDER_BLOCKS_THRESHOLD = 0.01

# Liquidity Settings
LIQUIDITY_LOOKBACK = 20
LIQUIDITY_VOLUME_THRESHOLD = 1.5

# Signal Generation Settings
SIGNAL_WEIGHTS = {
    'sma': 0.25,
    'rsi': 0.20,
    'macd': 0.25,
    'order_blocks': 0.20,
    'liquidity': 0.05,
    'adx': 0.05
}

SIGNAL_THRESHOLD = 0.3
MIN_SIGNAL_STRENGTH = 0.1

# Risk Management Settings
DEFAULT_STOP_LOSS_MULTIPLIER = 2.0
DEFAULT_TAKE_PROFIT_MULTIPLIER = 3.0
ATR_PERIOD = 14
MAX_RISK_PERCENT = 2.0

# File Paths
DATA_DIR = "data"
CSV_DIR = os.path.join(DATA_DIR, "csv")
EXPORTS_DIR = os.path.join(DATA_DIR, "exports")
LOGS_DIR = "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, CSV_DIR, EXPORTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG' if DEBUG_MODE else 'INFO',
            'formatter': 'detailed',
            'class': 'logging.FileHandler',
            'filename': os.path.join(LOGS_DIR, 'app.log'),
            'mode': 'a',
        }
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG' if DEBUG_MODE else 'INFO',
            'propagate': False
        }
    }
}

# Chart Settings
CHART_CONFIG = {
    'theme': 'plotly_dark',
    'height': 600,
    'show_volume': True,
    'show_indicators': True,
    'candlestick_colors': {
        'increasing': '#00ff88',
        'decreasing': '#ff4444'
    },
    'indicator_colors': {
        'sma_10': '#ffaa00',
        'sma_20': '#00aaff',
        'sma_50': '#aa00ff',
        'rsi': '#ff6600',
        'macd': '#0066ff',
        'signal': '#ff0066',
        'histogram': '#666666'
    }
}

# Performance Monitoring
PERFORMANCE_CONFIG = {
    'enable_monitoring': True,
    'log_slow_operations': True,
    'slow_operation_threshold': 1.0,  # seconds
    'memory_monitoring': True,
    'profile_functions': DEBUG_MODE
}

# API Settings (for future use)
API_CONFIG = {
    'rate_limit': 100,  # requests per minute
    'timeout': 30,  # seconds
    'retry_attempts': 3,
    'retry_delay': 1  # seconds
}

# Database Settings (for future use)
DATABASE_CONFIG = {
    'type': 'sqlite',
    'path': os.path.join(DATA_DIR, 'trading_data.db'),
    'backup_enabled': True,
    'backup_interval': 24  # hours
}

# Notification Settings (for future use)
NOTIFICATION_CONFIG = {
    'email_enabled': False,
    'telegram_enabled': False,
    'discord_enabled': False,
    'signal_threshold': 0.7  # Only notify for strong signals
}

# Machine Learning Settings (for future use)
ML_CONFIG = {
    'model_type': 'lstm',
    'training_data_size': 10000,
    'validation_split': 0.2,
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'model_save_path': os.path.join(DATA_DIR, 'models'),
    'retrain_interval': 168  # hours (1 week)
}

# Backtesting Settings
BACKTEST_CONFIG = {
    'initial_balance': 10000,
    'position_size_percent': 1.0,
    'commission': 0.0001,  # 1 pip
    'slippage': 0.0001,    # 1 pip
    'max_positions': 1,
    'enable_compounding': True
}

# Data Quality Settings
DATA_QUALITY_CONFIG = {
    'min_data_points': 100,
    'max_gap_tolerance': 5,  # missing periods
    'price_change_threshold': 0.1,  # 10% max change between bars
    'volume_outlier_threshold': 5.0,  # standard deviations
    'auto_clean_data': True
}

# Export Settings
EXPORT_CONFIG = {
    'default_format': 'csv',
    'include_indicators': True,
    'include_signals': True,
    'compress_files': False,
    'max_file_size_mb': 100
}

# Security Settings
SECURITY_CONFIG = {
    'enable_rate_limiting': True,
    'max_file_upload_size_mb': 50,
    'allowed_file_extensions': ['.csv', '.xlsx', '.json'],
    'sanitize_inputs': True
}

# Feature Flags
FEATURE_FLAGS = {
    'enable_mt5': True,
    'enable_csv_upload': True,
    'enable_sample_data': True,
    'enable_export': True,
    'enable_backtesting': False,  # Future feature
    'enable_ml_predictions': False,  # Future feature
    'enable_notifications': False,  # Future feature
    'enable_portfolio_management': False  # Future feature
}

# Environment-specific overrides
if os.getenv('ENVIRONMENT') == 'production':
    DEBUG_MODE = False
    LOGGING_CONFIG['handlers']['default']['level'] = 'WARNING'
    PERFORMANCE_CONFIG['profile_functions'] = False
elif os.getenv('ENVIRONMENT') == 'development':
    DEBUG_MODE = True
    FEATURE_FLAGS['enable_ml_predictions'] = True
    FEATURE_FLAGS['enable_backtesting'] = True

# Validation functions
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required directories
    for directory in [DATA_DIR, CSV_DIR, EXPORTS_DIR, LOGS_DIR]:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory {directory}: {e}")
    
    # Validate numeric settings
    if RSI_WINDOW <= 0:
        errors.append("RSI_WINDOW must be positive")
    
    if not (0 <= RSI_OVERBOUGHT <= 100):
        errors.append("RSI_OVERBOUGHT must be between 0 and 100")
    
    if not (0 <= RSI_OVERSOLD <= 100):
        errors.append("RSI_OVERSOLD must be between 0 and 100")
    
    if RSI_OVERSOLD >= RSI_OVERBOUGHT:
        errors.append("RSI_OVERSOLD must be less than RSI_OVERBOUGHT")
    
    # Validate signal weights
    total_weight = sum(SIGNAL_WEIGHTS.values())
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"Signal weights sum to {total_weight}, should be 1.0")
    
    # Validate file paths
    if not os.access(DATA_DIR, os.W_OK):
        errors.append(f"Data directory {DATA_DIR} is not writable")
    
    return errors

def get_config_summary():
    """Get a summary of current configuration"""
    return {
        'app_name': APP_NAME,
        'version': APP_VERSION,
        'debug_mode': DEBUG_MODE,
        'supported_symbols': len(SYMBOLS),
        'supported_timeframes': len(MT5_TIMEFRAMES),
        'indicator_count': len(['SMA', 'RSI', 'MACD', 'ADX', 'Order Blocks', 'Liquidity']),
        'data_directory': DATA_DIR,
        'features_enabled': sum(FEATURE_FLAGS.values()),
        'total_features': len(FEATURE_FLAGS)
    }

# Initialize configuration validation
_config_errors = validate_config()
if _config_errors:
    import warnings
    for error in _config_errors:
        warnings.warn(f"Configuration error: {error}")

# Export all settings
__all__ = [
    'APP_NAME', 'APP_VERSION', 'DEBUG_MODE',
    'DEFAULT_SYMBOL', 'DEFAULT_TIMEFRAME', 'DEFAULT_BARS_COUNT',
    'SYMBOLS', 'MT5_TIMEFRAMES',
    'SMA_WINDOWS', 'RSI_WINDOW', 'RSI_OVERBOUGHT', 'RSI_OVERSOLD',
    'MACD_FAST', 'MACD_SLOW', 'MACD_SIGNAL',
    'ADX_WINDOW', 'ADX_THRESHOLD',
    'ORDER_BLOCKS_LOOKBACK', 'ORDER_BLOCKS_THRESHOLD',
    'LIQUIDITY_LOOKBACK', 'LIQUIDITY_VOLUME_THRESHOLD',
    'SIGNAL_WEIGHTS', 'SIGNAL_THRESHOLD', 'MIN_SIGNAL_STRENGTH',
    'DEFAULT_STOP_LOSS_MULTIPLIER', 'DEFAULT_TAKE_PROFIT_MULTIPLIER',
    'ATR_PERIOD', 'MAX_RISK_PERCENT',
    'DATA_DIR', 'CSV_DIR', 'EXPORTS_DIR', 'LOGS_DIR',
    'LOGGING_CONFIG', 'CHART_CONFIG', 'PERFORMANCE_CONFIG',
    'API_CONFIG', 'DATABASE_CONFIG', 'NOTIFICATION_CONFIG',
    'ML_CONFIG', 'BACKTEST_CONFIG', 'DATA_QUALITY_CONFIG',
    'EXPORT_CONFIG', 'SECURITY_CONFIG', 'FEATURE_FLAGS',
    'validate_config', 'get_config_summary'
]