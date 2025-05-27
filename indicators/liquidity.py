import pandas as pd
import numpy as np

def calculate_liquidity(data, lookback=20, volume_threshold=1.5):
    """Calculate liquidity zones based on volume and price action"""
    try:
        data = data.copy()
        
        # Initialize liquidity columns
        data['Liquidity_High'] = np.nan
        data['Liquidity_Low'] = np.nan
        data['Liquidity_Zone'] = False
        data['Liquidity_Strength'] = 0.0
        
        if 'volume' not in data.columns:
            # If no volume data, use price-based liquidity detection
            return calculate_price_based_liquidity(data, lookback)
        
        # Calculate volume-based liquidity zones
        data['Volume_MA'] = data['volume'].rolling(window=lookback).mean()
        data['Volume_Ratio'] = data['volume'] / data['Volume_MA']
        
        # High volume areas indicate potential liquidity zones
        high_volume_mask = data['Volume_Ratio'] > volume_threshold
        
        for i in range(lookback, len(data)):
            if high_volume_mask.iloc[i]:
                # Look for price levels with high volume
                current_high = data['high'].iloc[i]
                current_low = data['low'].iloc[i]
                current_volume = data['volume'].iloc[i]
                
                # Calculate liquidity strength based on volume
                volume_strength = min(data['Volume_Ratio'].iloc[i] / volume_threshold, 3.0) / 3.0
                
                # Set liquidity levels
                data.iloc[i, data.columns.get_loc('Liquidity_High')] = current_high
                data.iloc[i, data.columns.get_loc('Liquidity_Low')] = current_low
                data.iloc[i, data.columns.get_loc('Liquidity_Zone')] = True
                data.iloc[i, data.columns.get_loc('Liquidity_Strength')] = volume_strength
        
        # Clean up temporary columns
        data.drop(['Volume_MA', 'Volume_Ratio'], axis=1, inplace=True, errors='ignore')
        
        return data
        
    except Exception as e:
        print(f"Error calculating liquidity zones: {e}")
        return data

def calculate_price_based_liquidity(data, lookback=20):
    """Calculate liquidity zones based on price action when volume is not available"""
    try:
        data = data.copy()
        
        # Initialize columns
        data['Liquidity_High'] = np.nan
        data['Liquidity_Low'] = np.nan
        data['Liquidity_Zone'] = False
        data['Liquidity_Strength'] = 0.0
        
        # Calculate price volatility
        data['Price_Range'] = data['high'] - data['low']
        data['Range_MA'] = data['Price_Range'].rolling(window=lookback).mean()
        data['Range_Ratio'] = data['Price_Range'] / data['Range_MA']
        
        # Look for areas with high price activity (wide ranges)
        high_activity_threshold = 1.5
        high_activity_mask = data['Range_Ratio'] > high_activity_threshold
        
        for i in range(lookback, len(data)):
            if high_activity_mask.iloc[i]:
                # Areas with high price ranges often indicate liquidity
                current_high = data['high'].iloc[i]
                current_low = data['low'].iloc[i]
                
                # Calculate strength based on range ratio
                range_strength = min(data['Range_Ratio'].iloc[i] / high_activity_threshold, 2.0) / 2.0
                
                data.iloc[i, data.columns.get_loc('Liquidity_High')] = current_high
                data.iloc[i, data.columns.get_loc('Liquidity_Low')] = current_low
                data.iloc[i, data.columns.get_loc('Liquidity_Zone')] = True
                data.iloc[i, data.columns.get_loc('Liquidity_Strength')] = range_strength
        
        # Also detect support/resistance levels as liquidity zones
        data = detect_support_resistance_liquidity(data, lookback)
        
        # Clean up temporary columns
        data.drop(['Price_Range', 'Range_MA', 'Range_Ratio'], axis=1, inplace=True, errors='ignore')
        
        return data
        
    except Exception as e:
        print(f"Error in price-based liquidity calculation: {e}")
        return data

def detect_support_resistance_liquidity(data, lookback=20):
    """Detect support and resistance levels as liquidity zones"""
    try:
        # Find local highs and lows
        data['Local_High'] = data['high'].rolling(window=lookback, center=True).max() == data['high']
        data['Local_Low'] = data['low'].rolling(window=lookback, center=True).min() == data['low']
        
        # Mark significant support/resistance as liquidity zones
        for i in range(lookback, len(data) - lookback):
            if data['Local_High'].iloc[i]:
                # Check if this high was tested multiple times
                price_level = data['high'].iloc[i]
                tolerance = price_level * 0.001  # 0.1% tolerance
                
                # Count how many times price came near this level
                nearby_touches = 0
                for j in range(max(0, i-50), min(len(data), i+50)):
                    if abs(data['high'].iloc[j] - price_level) <= tolerance:
                        nearby_touches += 1
                
                if nearby_touches >= 3:  # If tested 3+ times, it's a liquidity zone
                    data.iloc[i, data.columns.get_loc('Liquidity_High')] = price_level
                    data.iloc[i, data.columns.get_loc('Liquidity_Zone')] = True
                    data.iloc[i, data.columns.get_loc('Liquidity_Strength')] = min(nearby_touches / 5.0, 1.0)
            
            if data['Local_Low'].iloc[i]:
                # Check if this low was tested multiple times
                price_level = data['low'].iloc[i]
                tolerance = price_level * 0.001  # 0.1% tolerance
                
                # Count how many times price came near this level
                nearby_touches = 0
                for j in range(max(0, i-50), min(len(data), i+50)):
                    if abs(data['low'].iloc[j] - price_level) <= tolerance:
                        nearby_touches += 1
                
                if nearby_touches >= 3:  # If tested 3+ times, it's a liquidity zone
                    data.iloc[i, data.columns.get_loc('Liquidity_Low')] = price_level
                    data.iloc[i, data.columns.get_loc('Liquidity_Zone')] = True
                    data.iloc[i, data.columns.get_loc('Liquidity_Strength')] = min(nearby_touches / 5.0, 1.0)
        
        # Clean up temporary columns
        data.drop(['Local_High', 'Local_Low'], axis=1, inplace=True, errors='ignore')
        
        return data
        
    except Exception as e:
        print(f"Error detecting support/resistance liquidity: {e}")
        return data

def get_active_liquidity_zones(data, current_price, lookback=50):
    """Get currently active liquidity zones near current price"""
    try:
        if data.empty:
            return []
        
        # Get recent liquidity zones
        recent_data = data.tail(lookback)
        liquidity_zones = recent_data[recent_data['Liquidity_Zone'] == True]
        
        if liquidity_zones.empty:
            return []
        
        active_zones = []
        price_tolerance = current_price * 0.02  # 2% tolerance
        
        for idx, zone in liquidity_zones.iterrows():
            zone_high = zone.get('Liquidity_High', np.nan)
            zone_low = zone.get('Liquidity_Low', np.nan)
            strength = zone.get('Liquidity_Strength', 0.5)
            
            # Check if current price is near this liquidity zone
            near_high = not pd.isna(zone_high) and abs(current_price - zone_high) <= price_tolerance
            near_low = not pd.isna(zone_low) and abs(current_price - zone_low) <= price_tolerance
            
            if near_high or near_low:
                active_zones.append({
                    'timestamp': idx,
                    'high': zone_high,
                    'low': zone_low,
                    'strength': strength,
                    'type': 'resistance' if near_high else 'support',
                    'distance': min(
                        abs(current_price - zone_high) if not pd.isna(zone_high) else float('inf'),
                        abs(current_price - zone_low) if not pd.isna(zone_low) else float('inf')
                    )
                })
        
        # Sort by distance to current price
        active_zones.sort(key=lambda x: x['distance'])
        
        return active_zones[:5]  # Return top 5 closest zones
        
    except Exception as e:
        print(f"Error getting active liquidity zones: {e}")
        return []

