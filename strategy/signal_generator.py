# strategy/signal_generator.py
import pandas as pd

def generate_signals(data):
    data['Final_Signal'] = 0
    
    for i in range(len(data)):
        # Check buy conditions
        sma_buy_condition = data['SMA_10'].iloc[i] > data['SMA_20'].iloc[i] and data['SMA_20'].iloc[i] > data['SMA_50'].iloc[i]
        rsi_buy_condition = data['RSI'].iloc[i] < data['RSI_Oversold'].iloc[i]
        macd_buy_condition = data['MACD_Hist'].iloc[i] > 0
        order_block_buy_condition = data['Order_Block'].iloc[i] and data['close'].iloc[i] < data['Order_Block_Low'].iloc[i]
        
        # Check sell conditions
        sma_sell_condition = data['SMA_10'].iloc[i] < data['SMA_20'].iloc[i] and data['SMA_20'].iloc[i] < data['SMA_50'].iloc[i]
        rsi_sell_condition = data['RSI'].iloc[i] > data['RSI_Overbought'].iloc[i]
        macd_sell_condition = data['MACD_Hist'].iloc[i] < 0
        order_block_sell_condition = data['Order_Block'].iloc[i] and data['close'].iloc[i] > data['Order_Block_High'].iloc[i]
        
        # Generate buy signal only if all buy conditions are met
        if sma_buy_condition and rsi_buy_condition and macd_buy_condition and order_block_buy_condition:
            data.loc[i, 'Final_Signal'] = 1  # Buy signal
        # Generate sell signal only if all sell conditions are met
        elif sma_sell_condition and rsi_sell_condition and macd_sell_condition and order_block_sell_condition:
            data.loc[i, 'Final_Signal'] = -1  # Sell signal
            
    return data