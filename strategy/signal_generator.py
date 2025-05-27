import pandas as pd
import numpy as np
import logging
from utils.performance_monitor import monitor_performance

# Configure logging
logger = logging.getLogger(__name__)

@monitor_performance
def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive trading signals based on multiple indicators"""
    if data.empty:
        return data

    data = data.copy()

    # Initialize signal and risk-management columns
    data['signal'] = 0
    data['signal_strength'] = 0.0
    data['signal_reason'] = ''
    data['entry_price'] = np.nan
    data['stop_loss'] = np.nan
    data['take_profit'] = np.nan

    try:
        # Generate individual indicator signals
        data = generate_sma_signals(data)
        data = generate_rsi_signals(data)
        data = generate_macd_signals(data)
        data = generate_order_block_signals(data)
        data = generate_liquidity_signals(data)
        data = generate_adx_signals(data)

        # Combine all signals into one
        data = combine_signals(data)

        # Calculate risk levels (entry, stop, take)
        data = calculate_risk_levels(data)

        logger.info(f"Generated signals for {len(data)} rows")
    except Exception as e:
        logger.error(f"Error generating signals: {e}")

    return data

def generate_sma_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Generate signals based on SMA crossovers"""
    data['sma_signal'] = 0
    data['sma_strength'] = 0.0

    try:
        # find columns named "SMA_<period>"
        sma_cols = [c for c in data.columns if c.startswith('SMA_')]
        if len(sma_cols) < 2:
            return data

        # choose the shortest and longest SMA
        sma_short = min(sma_cols, key=lambda c: int(c.split('_')[1]))
        sma_long = max(sma_cols, key=lambda c: int(c.split('_')[1]))

        diff = data[sma_short] - data[sma_long]
        prev_diff = diff.shift(1)

        golden = (diff > 0) & (prev_diff <= 0)
        death = (diff < 0) & (prev_diff >= 0)

        data.loc[golden, 'sma_signal'] = 1
        data.loc[death, 'sma_signal'] = -1

        # strength = normalized distance between SMAs
        data['sma_strength'] = (diff.abs() / data['close']).clip(0, 1)
    except Exception as e:
        logger.error(f"SMA signal error: {e}")

    return data

def generate_rsi_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Generate signals based on RSI levels"""
    data['rsi_signal'] = 0
    data['rsi_strength'] = 0.0

    try:
        if 'RSI' not in data:
            return data

        oversold = data['RSI'] < 30
        overbought = data['RSI'] > 70

        data.loc[oversold, 'rsi_signal'] = 1
        data.loc[overbought, 'rsi_signal'] = -1

        # strength based on deviation from 50
        data['rsi_strength'] = (data['RSI'].sub(50).abs() / 50).clip(0,1)

        # cap extremes
        data.loc[data['RSI'] < 20, 'rsi_strength'] = 1.0
        data.loc[data['RSI'] > 80, 'rsi_strength'] = 1.0
    except Exception as e:
        logger.error(f"RSI signal error: {e}")

    return data

def generate_macd_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Generate signals based on MACD crossovers"""
    data['macd_signal'] = 0
    data['macd_strength'] = 0.0

    try:
        if not {'MACD', 'MACD_Signal'}.issubset(data.columns):
            return data

        macd = data['MACD']
        signal = data['MACD_Signal']
        prev_macd = macd.shift(1)
        prev_signal = signal.shift(1)

        up = (macd > signal) & (prev_macd <= prev_signal)
        down = (macd < signal) & (prev_macd >= prev_signal)

        data.loc[up, 'macd_signal'] = 1
        data.loc[down, 'macd_signal'] = -1

        if 'MACD_Histogram' in data:
            hist = data['MACD_Histogram'].abs()
        else:
            hist = (macd - signal).abs()

        data['macd_strength'] = (hist / data['close']).clip(0,1)
    except Exception as e:
        logger.error(f"MACD signal error: {e}")

    return data

def generate_order_block_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Generate signals based on order block touches"""
    data['ob_signal'] = 0
    data['ob_strength'] = 0.0

    try:
        if not {'Order_Block','Order_Block_High','Order_Block_Low'}.issubset(data.columns):
            return data

        for i in range(len(data)):
            price = data.iat[i, data.columns.get_loc('close')]
            window = data.iloc[max(0,i-50):i]
            blocks = window[window['Order_Block']]

            for _, blk in blocks.iterrows():
                high = blk['Order_Block_High']
                low = blk['Order_Block_Low']
                if pd.isna(high) or pd.isna(low):
                    continue

                # support touch
                if abs(price - low)/price < 0.001:
                    data.iat[i, data.columns.get_loc('ob_signal')] = 1
                    data.iat[i, data.columns.get_loc('ob_strength')] = 0.5
                # resistance touch
                elif abs(price - high)/price < 0.001:
                    data.iat[i, data.columns.get_loc('ob_signal')] = -1
                    data.iat[i, data.columns.get_loc('ob_strength')] = 0.5
    except Exception as e:
        logger.error(f"Order block error: {e}")

    return data

def generate_liquidity_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Generate signals based on liquidity zones"""
    data['liq_signal'] = 0
    data['liq_strength'] = 0.0

    try:
        if not {'Liquidity_High','Liquidity_Low'}.issubset(data.columns):
            return data

        zones = data.dropna(subset=['Liquidity_High','Liquidity_Low'])
        for i in range(len(data)):
            price = data.iat[i, data.columns.get_loc('close')]
            recent = zones.iloc[max(0,i-20):i]

            for _, z in recent.iterrows():
                high = z['Liquidity_High']
                low = z['Liquidity_Low']
                if abs(price - low)/price < 0.002:
                    data.iat[i, data.columns.get_loc('liq_signal')] = 1
                    data.iat[i, data.columns.get_loc('liq_strength')] = 0.3
                elif abs(price - high)/price < 0.002:
                    data.iat[i, data.columns.get_loc('liq_signal')] = -1
                    data.iat[i, data.columns.get_loc('liq_strength')] = 0.3
    except Exception as e:
        logger.error(f"Liquidity signal error: {e}")

    return data

def generate_adx_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Generate signals based on ADX strength"""
    data['adx_signal'] = 0
    data['adx_strength'] = 0.0

    try:
        if 'ADX' not in data:
            return data

        strong = data['ADX'] > 25
        if 'RSI' in data:
            up = strong & (data['RSI'] > 50)
            dn = strong & (data['RSI'] < 50)
            data.loc[up, 'adx_signal'] = 1
            data.loc[dn, 'adx_signal'] = -1

        data['adx_strength'] = (data['ADX']/100).clip(0,1)
    except Exception as e:
        logger.error(f"ADX signal error: {e}")

    return data

def combine_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Combine individual signals into a final signal"""
    try:
        weights = {
            'sma_signal': 0.25,
            'rsi_signal': 0.20,
            'macd_signal': 0.25,
            'ob_signal': 0.20,
            'liq_signal': 0.05,
            'adx_signal': 0.05
        }
        data['weighted_signal'] = 0.0
        data['total_strength'] = 0.0

        for sig, w in weights.items():
            str_col = sig.replace('_signal', '_strength')
            if sig in data and str_col in data:
                data['weighted_signal'] += data[sig] * data[str_col] * w
                data['total_strength']    += data[str_col] * w

        thresh = 0.3
        data.loc[data['weighted_signal'] >  thresh, 'signal'] =  1
        data.loc[data['weighted_signal'] < -thresh, 'signal'] = -1
        data['signal_strength'] = data['weighted_signal'].abs()

        data = add_signal_reasons(data)
    except Exception as e:
        logger.error(f"Combine signals error: {e}")

    return data

def add_signal_reasons(data: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable reasons for each signal"""
    try:
        data['signal_reason'] = ''
        for i in range(len(data)):
            if data.iat[i, data.columns.get_loc('signal')] == 0:
                continue

            parts = []
            if data.iat[i, data.columns.get_loc('sma_signal')] != 0:
                parts.append('SMA crossover')
            if data.iat[i, data.columns.get_loc('rsi_signal')] != 0:
                r = data.iat[i, data.columns.get_loc('RSI')]
                parts.append('RSI overbought' if r>70 else 'RSI oversold')
            if data.iat[i, data.columns.get_loc('macd_signal')] != 0:
                parts.append('MACD crossover')
            if data.iat[i, data.columns.get_loc('ob_signal')] != 0:
                parts.append('Order block touch')
            if data.iat[i, data.columns.get_loc('liq_signal')] != 0:
                parts.append('Liquidity zone')
            if data.iat[i, data.columns.get_loc('adx_signal')] != 0:
                parts.append('ADX trend')

            data.iat[i, data.columns.get_loc('signal_reason')] = ', '.join(parts)
    except Exception as e:
        logger.error(f"Signal reason error: {e}")
    return data

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the Average True Range (ATR)"""
    try:
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close  = (data['low']  - data['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr
    except Exception as e:
        logger.error(f"ATR calculation error: {e}")
        return pd.Series(np.nan, index=data.index)
def calculate_risk_levels(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate stop loss and take profit levels based on ATR multiples"""
    try:
        # Ensure ATR is present
        if 'ATR' not in data.columns:
            data['ATR'] = calculate_atr(data, period=14)

        sl_mult = 1.5
        tp_mult = 3.0

        for i in range(len(data)):
            sig = data.iat[i, data.columns.get_loc('signal')]
            if sig == 0 or np.isnan(data.iat[i, data.columns.get_loc('ATR')]):
                continue

            entry = data.iat[i, data.columns.get_loc('close')]
            atr   = data.iat[i, data.columns.get_loc('ATR')]

            if sig > 0:
                stop = entry - sl_mult * atr
                take = entry + tp_mult * atr
            else:
                stop = entry + sl_mult * atr
                take = entry - tp_mult * atr

            data.iat[i, data.columns.get_loc('entry_price')] = entry
            data.iat[i, data.columns.get_loc('stop_loss')]   = stop
            data.iat[i, data.columns.get_loc('take_profit')] = take

    except Exception as e:
        logger.error(f"Risk level computation error: {e}")
    return data
def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    try:
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
        
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return pd.Series(index=data.index, dtype=float)

def calculate_signal_confidence(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate confidence score for signals"""
    try:
        data['signal_confidence'] = 0.0
        
        # Base confidence from signal strength
        data['signal_confidence'] = data['signal_strength']
        
        # Boost confidence for multiple confirming indicators
        confirming_indicators = 0
        
        # Check SMA trend confirmation
        if 'SMA_10' in data.columns and 'SMA_20' in data.columns:
            sma_bullish = data['SMA_10'] > data['SMA_20']
            sma_bearish = data['SMA_10'] < data['SMA_20']
            
            buy_sma_confirm = (data['signal'] == 1) & sma_bullish
            sell_sma_confirm = (data['signal'] == -1) & sma_bearish
            
            data.loc[buy_sma_confirm | sell_sma_confirm, 'signal_confidence'] += 0.1
            confirming_indicators += 1
        
        # Check RSI confirmation
        if 'RSI' in data.columns:
            rsi_bullish = data['RSI'] < 50  # RSI supporting upward move
            rsi_bearish = data['RSI'] > 50  # RSI supporting downward move
            
            buy_rsi_confirm = (data['signal'] == 1) & rsi_bullish
            sell_rsi_confirm = (data['signal'] == -1) & rsi_bearish
            
            data.loc[buy_rsi_confirm | sell_rsi_confirm, 'signal_confidence'] += 0.1
            confirming_indicators += 1
        
        # Check MACD confirmation
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            macd_bullish = data['MACD'] > data['MACD_Signal']
            macd_bearish = data['MACD'] < data['MACD_Signal']
            
            buy_macd_confirm = (data['signal'] == 1) & macd_bullish
            sell_macd_confirm = (data['signal'] == -1) & macd_bearish
            
            data.loc[buy_macd_confirm | sell_macd_confirm, 'signal_confidence'] += 0.1
            confirming_indicators += 1
        
        # Check ADX trend strength
        if 'ADX' in data.columns:
            strong_trend = data['ADX'] > 25
            data.loc[strong_trend & (data['signal'] != 0), 'signal_confidence'] += 0.15
        
        # Normalize confidence score
        data['signal_confidence'] = np.clip(data['signal_confidence'], 0, 1)
        
        # Filter weak signals based on confidence
        min_confidence = 0.3
        weak_signals = (data['signal'] != 0) & (data['signal_confidence'] < min_confidence)
        data.loc[weak_signals, 'signal'] = 0
        
    except Exception as e:
        logger.error(f"Error calculating signal confidence: {e}")
    
    return data

def detect_market_regime(data: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """Detect market regime (trending vs ranging)"""
    try:
        data['market_regime'] = 'unknown'
        data['regime_strength'] = 0.0
        
        for i in range(lookback, len(data)):
            window_data = data.iloc[i-lookback:i]
            
            # Calculate price volatility
            returns = window_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Calculate trend strength using linear regression
            x = np.arange(len(window_data))
            y = window_data['close'].values
            
            if len(y) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                # Determine regime based on R-squared and volatility
                r_squared = r_value ** 2
                
                if r_squared > 0.7 and abs(slope) > volatility:
                    if slope > 0:
                        regime = 'uptrend'
                    else:
                        regime = 'downtrend'
                    strength = r_squared
                elif volatility < window_data['close'].mean() * 0.01:  # Low volatility
                    regime = 'ranging'
                    strength = 1 - r_squared
                else:
                    regime = 'choppy'
                    strength = volatility
                
                data.iloc[i, data.columns.get_loc('market_regime')] = regime
                data.iloc[i, data.columns.get_loc('regime_strength')] = min(strength, 1.0)
        
    except Exception as e:
        logger.error(f"Error detecting market regime: {e}")
    
    return data

def apply_regime_filters(data: pd.DataFrame) -> pd.DataFrame:
    """Apply regime-based filters to signals"""
    try:
        # Reduce signal strength in choppy markets
        choppy_market = data['market_regime'] == 'choppy'
        data.loc[choppy_market, 'signal_strength'] *= 0.5
        
        # Boost trend-following signals in trending markets
        trending_market = data['market_regime'].isin(['uptrend', 'downtrend'])
        trend_signals = trending_market & (data['signal'] != 0)
        
        # Check if signal aligns with trend
        uptrend_buy = (data['market_regime'] == 'uptrend') & (data['signal'] == 1)
        downtrend_sell = (data['market_regime'] == 'downtrend') & (data['signal'] == -1)
        
        aligned_signals = uptrend_buy | downtrend_sell
        data.loc[aligned_signals, 'signal_strength'] *= 1.2
        
        # Reduce counter-trend signals
        counter_trend = trend_signals & ~aligned_signals
        data.loc[counter_trend, 'signal_strength'] *= 0.3
        
        # Filter out very weak signals
        weak_signals = (data['signal'] != 0) & (data['signal_strength'] < 0.1)
        data.loc[weak_signals, 'signal'] = 0
        
    except Exception as e:
        logger.error(f"Error applying regime filters: {e}")
    
    return data

def calculate_position_sizing(data: pd.DataFrame, account_balance: float = 10000, 
                            risk_percent: float = 2.0) -> pd.DataFrame:
    """Calculate position sizing based on risk management"""
    try:
        data['position_size'] = 0.0
        data['risk_amount'] = 0.0
        
        # Calculate risk amount per trade
        max_risk_amount = account_balance * (risk_percent / 100)
        
        signal_rows = data['signal'] != 0
        
        for idx in data[signal_rows].index:
            entry_price = data.loc[idx, 'entry_price']
            stop_loss = data.loc[idx, 'stop_loss']
            
            if pd.notna(entry_price) and pd.notna(stop_loss):
                # Calculate risk per unit
                risk_per_unit = abs(entry_price - stop_loss)
                
                if risk_per_unit > 0:
                    # Calculate position size
                    position_size = max_risk_amount / risk_per_unit
                    
                    # Apply maximum position size limits
                    max_position_value = account_balance * 0.1  # Max 10% of account
                    max_units = max_position_value / entry_price
                    
                    position_size = min(position_size, max_units)
                    
                    data.loc[idx, 'position_size'] = position_size
                    data.loc[idx, 'risk_amount'] = position_size * risk_per_unit
        
    except Exception as e:
        logger.error(f"Error calculating position sizing: {e}")
    
    return data

def add_signal_metadata(data: pd.DataFrame) -> pd.DataFrame:
    """Add metadata to signals for better analysis"""
    try:
        data['signal_id'] = ''
        data['signal_timestamp'] = pd.NaT
        data['signal_duration'] = 0
        
        signal_counter = 0
        current_signal = 0
        signal_start_idx = None
        
        for i, row in data.iterrows():
            if row['signal'] != 0 and row['signal'] != current_signal:
                # New signal detected
                signal_counter += 1
                current_signal = row['signal']
                signal_start_idx = i
                
                signal_type = 'BUY' if current_signal == 1 else 'SELL'
                signal_id = f"{signal_type}_{signal_counter:04d}"
                
                data.loc[i, 'signal_id'] = signal_id
                data.loc[i, 'signal_timestamp'] = row['time']
                
            elif row['signal'] == 0 and current_signal != 0:
                # Signal ended
                if signal_start_idx is not None:
                    duration = i - signal_start_idx
                    data.loc[signal_start_idx, 'signal_duration'] = duration
                
                current_signal = 0
                signal_start_idx = None
        
    except Exception as e:
        logger.error(f"Error adding signal metadata: {e}")
    
    return data

from typing import Any

def generate_signal_summary(data: pd.DataFrame) -> dict[str, Any]:
    """Generate summary statistics for signals"""
    try:
        signals_data = data[data['signal'] != 0].copy()
        
        if signals_data.empty:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'avg_strength': 0,
                'avg_confidence': 0,
                'strongest_signal': None,
                'latest_signal': None
            }
        
        summary = {
            'total_signals': len(signals_data),
            'buy_signals': len(signals_data[signals_data['signal'] == 1]),
            'sell_signals': len(signals_data[signals_data['signal'] == -1]),
            'avg_strength': signals_data['signal_strength'].mean(),
            'avg_confidence': signals_data.get('signal_confidence', pd.Series([0])).mean(),
            'date_range': {
                'start': signals_data['time'].min(),
                'end': signals_data['time'].max()
            }
        }
        
        # Find strongest signal
        strongest_idx = signals_data['signal_strength'].idxmax()
        summary['strongest_signal'] = {
            'time': signals_data.loc[strongest_idx, 'time'],
            'type': 'BUY' if signals_data.loc[strongest_idx, 'signal'] == 1 else 'SELL',
            'strength': signals_data.loc[strongest_idx, 'signal_strength'],
            'price': signals_data.loc[strongest_idx, 'close'],
            'reason': signals_data.loc[strongest_idx, 'signal_reason']
        }
        
        # Latest signal
        latest_idx = signals_data.index[-1]
        summary['latest_signal'] = {
            'time': signals_data.loc[latest_idx, 'time'],
            'type': 'BUY' if signals_data.loc[latest_idx, 'signal'] == 1 else 'SELL',
            'strength': signals_data.loc[latest_idx, 'signal_strength'],
            'price': signals_data.loc[latest_idx, 'close'],
            'reason': signals_data.loc[latest_idx, 'signal_reason']
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating signal summary: {e}")
        return {'error': str(e)}# Enhanced main function with all features
@monitor_performance
def generate_comprehensive_signals(data: pd.DataFrame, config: dict[str, Any] = None) -> dict[str, Any]:
    """
    Generate comprehensive trading signals with full analysis
    
    Args:
        data: DataFrame with OHLCV data and indicators
        config: Optional configuration dictionary
    
    Returns:
        Dictionary containing processed data and analysis results
    """
    try:
        if data.empty:
            return {'error': 'Empty dataset provided'}
        
        # Use default config if none provided
        if config is None:
            config = {
                'account_balance': 10000,
                'risk_percent': 2.0,
                'enable_regime_detection': True,
                'enable_confidence_scoring': True,
                'enable_position_sizing': True
            }
        
        # Generate basic signals
        processed_data = generate_signals(data.copy())
        
        # Add regime detection if enabled
        if config.get('enable_regime_detection', True):
            processed_data = detect_market_regime(processed_data)
            processed_data = apply_regime_filters(processed_data)
        
        # Add confidence scoring if enabled
        if config.get('enable_confidence_scoring', True):
            processed_data = calculate_signal_confidence(processed_data)
        
        # Add position sizing if enabled
        if config.get('enable_position_sizing', True):
            processed_data = calculate_position_sizing(
                processed_data,
                config.get('account_balance', 10000),
                config.get('risk_percent', 2.0)
            )
        
        # Add signal metadata
        processed_data = add_signal_metadata(processed_data)
        
        # Generate summary
        signal_summary = generate_signal_summary(processed_data)
        
        return {
            'data': processed_data,
            'summary': signal_summary,
            'config': config,
            'processing_timestamp': pd.Timestamp.now(),
            'data_quality': {
                'total_bars': len(processed_data),
                'signal_bars': len(processed_data[processed_data['signal'] != 0]),
                'signal_rate': len(processed_data[processed_data['signal'] != 0]) / len(processed_data) * 100
            }
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive signal generation: {e}")
        return {'error': str(e)}
# Import scipy.stats for regime detection
try:
    from scipy import stats
except ImportError:
    logger.warning("scipy not available, regime detection will be limited")
    # Fallback implementation
    class MockStats:
        @staticmethod
        def linregress(x, y):
            # Simple linear regression fallback
            n = len(x)
            if n < 2:
                return 0, 0, 0, 1, 0
            
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            
            if denominator == 0:
                return 0, y_mean, 0, 1, 0
            
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            
            # Calculate correlation coefficient
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            
            if ss_tot == 0:
                r_value = 1 if ss_res == 0 else 0
            else:
                r_value = np.sqrt(1 - (ss_res / ss_tot))
            
            return slope, intercept, r_value, 0.05, 0
    
    stats = MockStats()

# Export all functions
__all__ = [
    'generate_signals',
    'generate_comprehensive_signals',
    'generate_sma_signals',
    'generate_rsi_signals', 
    'generate_macd_signals',
    'generate_order_block_signals',
    'generate_liquidity_signals',
    'generate_adx_signals',
    'combine_signals',
    'add_signal_reasons',
    'calculate_risk_levels',
    'calculate_atr',
    'calculate_signal_confidence',
    'detect_market_regime',
    'apply_regime_filters',
    'calculate_position_sizing',
    'add_signal_metadata',
    'generate_signal_summary'
]