# data/data_loader.py
import pandas as pd

def load_from_csv(file_path):
    # Read the CSV file without parsing dates
    data = pd.read_csv(file_path, delimiter='\t')  # Use the appropriate delimiter
    
    # Ensure the necessary columns exist
    required_columns = ['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>', '<VOL>', '<SPREAD>']
    if not all(column in data.columns for column in required_columns):
        raise ValueError("CSV file is missing required columns.")
    
    # Combine 'DATE' and 'TIME' into a single 'datetime' column
    data['datetime'] = pd.to_datetime(data['<DATE>'].astype(str) + ' ' + data['<TIME>'].astype(str))
    
    # Drop the original 'DATE' and 'TIME' columns if they are no longer needed
    data.drop(['<DATE>', '<TIME>'], axis=1, inplace=True)
    
    # Rename columns to match the rest of your code
    data.rename(columns={
        '<OPEN>': 'open',
        '<HIGH>': 'high',
        '<LOW>': 'low',
        '<CLOSE>': 'close',
        '<TICKVOL>': 'tick_volume',
        '<VOL>': 'volume',
        '<SPREAD>': 'spread'
    }, inplace=True)
    
    return data