import pandas as pd
import numpy as np
from typing import Optional, List, Tuple

def validate_ohlcv_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate OHLCV data structure and quality"""
    errors = []
    
    # Check if data is empty
    if data.empty:
        errors.append("Data is empty")
        return False, errors
    
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return False, errors
    
    # Check data quality
    try:
        # Check for invalid OHLC relationships
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.any():
            errors.append(f"Invalid OHLC relationships found in {invalid_ohlc.sum()} rows")
        
        # Check for missing values
        null_counts = data[required_columns].isnull().sum()
        if null_counts.any():
            errors.append(f"Missing values found: {null_counts.to_dict()}")
        
        # Check for extreme values
        for col in required_columns:
            if (data[col] <= 0).any():
                errors.append(f"Non-positive values found in {col}")
        
        # Check for duplicate timestamps if time column exists
        if 'time' in data.columns:
            duplicates = data['time'].duplicated().sum()
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate timestamps")
        
    except Exception as e:
        errors.append(f"Error during validation: {str(e)}")
    
    return len(errors) == 0, errors

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for analysis"""
    if data.empty:
        return data
    
    data = data.copy()
    
    try:
        # Remove duplicates
        initial_length = len(data)
        data = data.drop_duplicates()
        if len(data) < initial_length:
            print(f"Removed {initial_length - len(data)} duplicate rows")
        
        # Sort by time if time column exists
        if 'time' in data.columns:
            data = data.sort_values('time').reset_index(drop=True)
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        # Forward fill missing values (conservative approach)
        data[numeric_columns] = data[numeric_columns].fillna(method='ffill')
        
        # Backward fill any remaining NaN values at the beginning
        data[numeric_columns] = data[numeric_columns].fillna(method='bfill')
        
        # Remove any remaining NaN rows
        initial_length = len(data)
        data = data.dropna(subset=['open', 'high', 'low', 'close'])
        if len(data) < initial_length:
            print(f"Removed {initial_length - len(data)} rows with missing OHLC data")
        
        # Fix invalid OHLC relationships
        data = fix_ohlc_relationships(data)
        
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
    
    return data

def fix_ohlc_relationships(data: pd.DataFrame) -> pd.DataFrame:
    """Fix invalid OHLC relationships"""
    data = data.copy()
    
    try:
        # Ensure high is the maximum of open, high, low, close
        data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
        
        # Ensure low is the minimum of open, high, low, close
        data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
        
    except Exception as e:
        print(f"Error fixing OHLC relationships: {str(e)}")
    
    return data

def detect_outliers(data: pd.DataFrame, column: str = 'close', method: str = 'iqr') -> pd.Series:
    """Detect outliers in price data"""
    if column not in data.columns:
        return pd.Series(dtype=bool)
    
    try:
        if method == 'iqr':
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            outliers = z_scores > 3
        
        else:
            outliers = pd.Series([False] * len(data), index=data.index)
        
        return outliers
    
    except Exception as e:
        print(f"Error detecting outliers: {str(e)}")
        return pd.Series([False] * len(data), index=data.index)

def get_data_quality_report(data: pd.DataFrame) -> dict:
    """Generate a comprehensive data quality report"""
    report = {
        'total_rows': len(data),
        'date_range': None,
        'missing_values': {},
        'outliers': {},
        'data_types': {},
        'validation_passed': False,
        'errors': []
    }
    
    try:
        # Basic info
        if 'time' in data.columns:
            report['date_range'] = {
                'start': data['time'].min(),
                'end': data['time'].max()
            }
        
        # Missing values
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                report['missing_values'][col] = {
                    'count': missing_count,
                    'percentage': (missing_count / len(data)) * 100
                }
        
        # Outliers for numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            outliers = detect_outliers(data, col)
            outlier_count = outliers.sum()
            if outlier_count > 0:
                report['outliers'][col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(data)) * 100
                }
        
        # Data types
        report['data_types'] = data.dtypes.to_dict()
        
        # Validation
        is_valid, errors = validate_ohlcv_data(data)
        report['validation_passed'] = is_valid
        report['errors'] = errors
        
    except Exception as e:
        report['errors'].append(f"Error generating report: {str(e)}")
    
    return report

def ensure_required_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure data has required columns with proper names"""
    data = data.copy()
    
    # Common column name mappings
    column_mappings = {
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'Time': 'time',
        'Datetime': 'time',
        'Date': 'time'
    }
    
    # Rename columns if needed
    for old_name, new_name in column_mappings.items():
        if old_name in data.columns and new_name not in data.columns:
            data = data.rename(columns={old_name: new_name})
    
    return data
