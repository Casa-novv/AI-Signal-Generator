# indicators/order_blocks.py
import pandas as pd

def calculate_order_blocks(data, lookback=50, threshold=0.01):
    data['Order_Block'] = False
    data['Order_Block_High'] = None
    data['Order_Block_Low'] = None
    
    for i in range(lookback, len(data)):
        high = data['high'][i-lookback:i].max()
        low = data['low'][i-lookback:i].min()
        range_size = high - low
        
        if range_size > threshold:
            data.loc[i, 'Order_Block_High'] = high
            data.loc[i, 'Order_Block_Low'] = low
            data.loc[i, 'Order_Block'] = True
            
    return data