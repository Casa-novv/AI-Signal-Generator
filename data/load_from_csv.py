# data/load_from_csv.py
import pandas as pd
import streamlit as st
from data import data_loader
import io
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def load_from_csv(uploaded_file):
    """
    Load data from uploaded CSV file in Streamlit
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        DataFrame with processed data or None if failed
    """
    try:
        if uploaded_file is not None:
            # Read the uploaded file
            bytes_data = uploaded_file.read()
            
            # Try to decode with different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    # Convert bytes to string
                    string_data = bytes_data.decode(encoding)
                    
                    # Create StringIO object
                    string_io = io.StringIO(string_data)
                    
                    # Try different separators
                    separators = [',', ';', '\t']
                    
                    for sep in separators:
                        try:
                            string_io.seek(0)  # Reset position
                            data = pd.read_csv(string_io, sep=sep)
                            
                            if len(data.columns) > 1 and len(data) > 0:
                                # Successfully parsed, now process the data
                                processed_data = process_csv_data(data)
                                
                                if processed_data is not None:
                                    logger.info(f"Successfully loaded CSV with {len(processed_data)} rows")
                                    return processed_data
                                    
                        except Exception as e:
                            continue
                            
                except UnicodeDecodeError:
                    continue
            
            # If we get here, all attempts failed
            st.error("Failed to parse CSV file. Please check the file format.")
            return None
            
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        st.error(f"Error loading CSV file: {e}")
        return None

def process_csv_data(data):
    """
    Process raw CSV data into standardized format
    
    Args:
        data: Raw DataFrame from CSV
    
    Returns:
        Processed DataFrame or None if failed
    """
    try:
        # Make a copy to avoid modifying original
        processed = data.copy()
        
        # Normalize column names
        processed.columns = processed.columns.str.lower().str.strip()
        
        # Common column mappings
        column_mappings = {
            'datetime': 'time',
            'timestamp': 'time',
            'date': 'time',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vol': 'volume',
            'adj close': 'adj_close',
            'adj_close': 'adj_close'
        }
        
        # Apply column mappings
        for old_name, new_name in column_mappings.items():
            if old_name in processed.columns:
                processed = processed.rename(columns={old_name: new_name})
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in processed.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.info("Required columns: open, high, low, close")
            st.info(f"Available columns: {list(processed.columns)}")
            return None
        
        # Handle time column
        if 'time' not in processed.columns:
            # Try to find a time-like column
            time_candidates = []
            for col in processed.columns:
                if any(keyword in col for keyword in ['time', 'date', 'datetime', 'timestamp']):
                    time_candidates.append(col)
            
            if time_candidates:
                processed['time'] = processed[time_candidates[0]]
            else:
                # Create a synthetic time column
                st.warning("No time column found. Creating synthetic timestamps.")
                processed['time'] = pd.date_range(
                    start='2024-01-01', 
                    periods=len(processed), 
                    freq='H'
                )
        
        # Convert time column to datetime
        try:
            processed['time'] = pd.to_datetime(processed['time'])
        except Exception as e:
            st.error(f"Error parsing time column: {e}")
            return None
        
        # Ensure numeric columns are numeric
        numeric_columns = ['open', 'high', 'low', 'close']
        if 'volume' in processed.columns:
            numeric_columns.append('volume')
        
        for col in numeric_columns:
            try:
                processed[col] = pd.to_numeric(processed[col], errors='coerce')
            except Exception as e:
                st.warning(f"Error converting {col} to numeric: {e}")
        
        # Remove rows with NaN values in critical columns
        processed = processed.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Add volume column if missing
        if 'volume' not in processed.columns:
            processed['volume'] = 1000  # Default volume
        
        # Sort by time
        processed = processed.sort_values('time').reset_index(drop=True)
        
        # Validate OHLC data
        processed = data_loader.validate_ohlc_data(processed)
        
        if processed.empty:
            st.error("No valid data remaining after processing")
            return None
        
        # Keep only required columns in correct order
        final_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        processed = processed[final_columns]
        
        return processed
        
    except Exception as e:
        logger.error(f"Error processing CSV data: {e}")
        st.error(f"Error processing CSV data: {e}")
        return None

def preview_csv_data(uploaded_file, max_rows=5):
    """
    Preview CSV data before processing
    
    Args:
        uploaded_file: Streamlit uploaded file object
        max_rows: Maximum rows to preview
    
    Returns:
        Dictionary with preview information
    """
    try:
        if uploaded_file is not None:
            # Read first few lines for preview
            bytes_data = uploaded_file.read()
            string_data = bytes_data.decode('utf-8')
            string_io = io.StringIO(string_data)
            
            # Try to read with comma separator first
            preview_data = pd.read_csv(string_io, nrows=max_rows)
            
            preview_info = {
                'columns': list(preview_data.columns),
                'shape': preview_data.shape,
                'sample_data': preview_data.to_dict('records'),
                'dtypes': preview_data.dtypes.to_dict()
            }
            
            return preview_info
            
    except Exception as e:
        logger.error(f"Error previewing CSV: {e}")
        return None

def validate_csv_format(data):
    """
    Validate CSV format and provide feedback
    
    Args:
        data: DataFrame to validate
    
    Returns:
        Dictionary with validation results
    """
    try:
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = []
        
        for col in required_cols:
            if col not in data.columns:
                # Check for similar column names
                similar_cols = [c for c in data.columns if col.lower() in c.lower()]
                if similar_cols:
                    validation['suggestions'].append(f"Column '{similar_cols[0]}' might be '{col}'")
                else:
                    missing_cols.append(col)
        
        if missing_cols:
            validation['is_valid'] = False
            validation['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Check for time column
        time_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['time', 'date', 'datetime', 'timestamp'])]
        
        if not time_cols:
            validation['warnings'].append("No time column detected. Synthetic timestamps will be created.")
        
        # Check data types
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                try:
                    pd.to_numeric(data[col], errors='raise')
                except:
                    validation['warnings'].append(f"Column '{col}' contains non-numeric values")
        
        # Check for sufficient data
        if len(data) < 10:
            validation['warnings'].append("Dataset is very small (less than 10 rows)")
        
        return validation
        
    except Exception as e:
        logger.error(f"Error validating CSV format: {e}")
        return {'is_valid': False, 'errors': [str(e)], 'warnings': [], 'suggestions': []}

def suggest_column_mapping(data):
    """
    Suggest column mappings for CSV data
    
    Args:
        data: DataFrame with original column names
    
    Returns:
        Dictionary with suggested mappings
    """
    try:
        suggestions = {}
        
        # Common patterns for different columns
        patterns = {
            'time': ['time', 'date', 'datetime', 'timestamp', 'dt'],
            'open': ['open', 'o', 'opening', 'start'],
            'high': ['high', 'h', 'max', 'maximum'],
            'low': ['low', 'l', 'min', 'minimum'],
            'close': ['close', 'c', 'closing', 'end', 'last'],
            'volume': ['volume', 'vol', 'v', 'quantity', 'qty']
        }
        
        for target_col, pattern_list in patterns.items():
            for col in data.columns:
                col_lower = col.lower().strip()
                if any(pattern in col_lower for pattern in pattern_list):
                    suggestions[col] = target_col
                    break
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Error suggesting column mapping: {e}")
        return {}

def create_csv_template():
    """
    Create a CSV template for users to download
    
    Returns:
        CSV string template
    """
    try:
        # Create sample data
        sample_data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=10, freq='H'),
            'open': [1.1000, 1.1005, 1.1010, 1.1008, 1.1012, 1.1015, 1.1018, 1.1020, 1.1022, 1.1025],
            'high': [1.1008, 1.1012, 1.1015, 1.1015, 1.1018, 1.1022, 1.1025, 1.1028, 1.1030, 1.1032],
            'low': [1.0998, 1.1002, 1.1005, 1.1005, 1.1008, 1.1012, 1.1015, 1.1018, 1.1020, 1.1022],
            'close': [1.1005, 1.1010, 1.1008, 1.1012, 1.1015, 1.1018, 1.1020, 1.1022, 1.1025, 1.1028],
            'volume': [1000, 1200, 800, 1500, 900, 1100, 1300, 1000, 1400, 1200]
        })
        
        return sample_data.to_csv(index=False)
        
    except Exception as e:
        logger.error(f"Error creating CSV template: {e}")
        return None

# Export functions
__all__ = [
    'load_from_csv',
    'process_csv_data',
    'preview_csv_data',
    'validate_csv_format',
    'suggest_column_mapping',
    'create_csv_template'
]
