# indicators/order_blocks.py
import pandas as pd
import numpy as np
from typing import Tuple, Optional

def calculate_order_blocks(data: pd.DataFrame, lookback: int = 50, 
                          threshold: float = 0.01, min_volume_factor: float = 1.5) -> pd.DataFrame:
    """
    Enhanced order block detection with volume analysis and strength scoring
    """
    if data.empty or len(data) < lookback:
        return data
    
    data = data.copy()
    data['Order_Block'] = False
    data['Order_Block_High'] = None
    data['Order_Block_Low'] = None
    data['Order_Block_Type'] = None
    data['Order_Block_Strength'] = 0.0
    data['Order_Block_Age'] = 0
    
    # Calculate average volume for comparison (if volume column exists)
    avg_volume = None
    if 'volume' in data.columns:
        avg_volume = data['volume'].rolling(window=lookback).mean()
    
    for i in range(lookback, len(data)):
        window_data = data.iloc[i-lookback:i]
        
        # Find significant price levels
        high_price = window_data['high'].max()
        low_price = window_data['low'].min()
        range_size = high_price - low_price
        current_price = data.iloc[i]['close']
        
        if range_size > threshold:
            # Enhanced validation
            strength_score = calculate_block_strength(window_data, high_price, low_price, avg_volume)
            
            if strength_score > 0.5:  # Minimum strength threshold
                data.loc[data.index[i], 'Order_Block_High'] = high_price
                data.loc[data.index[i], 'Order_Block_Low'] = low_price
                data.loc[data.index[i], 'Order_Block'] = True
                data.loc[data.index[i], 'Order_Block_Strength'] = strength_score
                
                # Determine block type
                mid_point = (high_price + low_price) / 2
                data.loc[data.index[i], 'Order_Block_Type'] = 'bullish' if current_price > mid_point else 'bearish'
    
    # Update block ages
    data = update_block_ages(data)
    
    return data

def calculate_block_strength(window_data: pd.DataFrame, high: float, low: float, 
                           avg_volume: Optional[pd.Series]) -> float:
    """Calculate strength score for order block"""
    strength = 0.0
    
    # Price rejection count
    touches_high = count_price_touches(window_data, high, tolerance=0.001)
    touches_low = count_price_touches(window_data, low, tolerance=0.001)
    strength += min(max(touches_high, touches_low) * 0.2, 0.4)
    
    # Volume analysis (if available)
    if avg_volume is not None and not avg_volume.empty and len(avg_volume) > 0:
        try:
            avg_vol_value = avg_volume.iloc[-1] if not pd.isna(avg_volume.iloc[-1]) else window_data['volume'].mean()
            if avg_vol_value > 0:
                high_volume_bars = (window_data['volume'] > avg_vol_value * 1.5).sum()
                strength += min(high_volume_bars * 0.1, 0.3)
        except:
            pass  # Skip volume analysis if there's an error
    
    # Range significance
    price_range = high - low
    avg_range = (window_data['high'] - window_data['low']).mean()
    if avg_range > 0:
        range_ratio = price_range / avg_range
        strength += min(range_ratio * 0.1, 0.3)
    
    return min(strength, 1.0)

def count_price_touches(data: pd.DataFrame, level: float, tolerance: float = 0.001) -> int:
    """Count price touches at specific level"""
    touches = 0
    for _, row in data.iterrows():
        if (abs(row['high'] - level) <= tolerance or 
            abs(row['low'] - level) <= tolerance or
            abs(row['close'] - level) <= tolerance):
            touches += 1
    return touches

def update_block_ages(data: pd.DataFrame) -> pd.DataFrame:
    """Update age of existing order blocks"""
    active_blocks = []
    
    for i in range(len(data)):
        if data.iloc[i]['Order_Block']:
            active_blocks.append(i)
        
        # Age existing blocks
        for block_idx in active_blocks[:]:
            age = i - block_idx
            data.loc[data.index[block_idx], 'Order_Block_Age'] = age
            
            # Remove old blocks (optional)
            if age > 100:  # Remove blocks older than 100 periods
                active_blocks.remove(block_idx)
                data.loc[data.index[block_idx], 'Order_Block'] = False
    
    return data

def get_active_order_blocks(data: pd.DataFrame, max_age: int = 50) -> pd.DataFrame:
    """Get currently active order blocks"""
    if 'Order_Block' not in data.columns:
        return pd.DataFrame()
    
    active_blocks = data[
        (data['Order_Block'] == True) & 
        (data['Order_Block_Age'] <= max_age)
    ].copy()
    
    return active_blocks

def is_price_near_order_block(current_price: float, data: pd.DataFrame, tolerance: float = 0.002) -> dict:
    """Check if current price is near any order block"""
    active_blocks = get_active_order_blocks(data)
    
    result = {
        'near_block': False,
        'block_type': None,
        'distance': None,
        'strength': None
    }
    
    for _, block in active_blocks.iterrows():
        high = block['Order_Block_High']
        low = block['Order_Block_Low']
        
        # Check if price is within tolerance of block levels
        if (abs(current_price - high) / current_price <= tolerance or
            abs(current_price - low) / current_price <= tolerance):
            
            result['near_block'] = True
            result['block_type'] = block['Order_Block_Type']
            result['distance'] = min(abs(current_price - high), abs(current_price - low))
            result['strength'] = block['Order_Block_Strength']
            break
    
    return result
