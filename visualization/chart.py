# visualization/chart.py
import plotly.graph_objects as go

def create_chart(data, symbol):
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(x=data['datetime'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name='Candlesticks'))
    
    # SMA lines
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['SMA_10'], line=dict(color='blue'), name='SMA 10'))
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['SMA_20'], line=dict(color='orange'), name='SMA 20'))
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['SMA_50'], line=dict(color='green'), name='SMA 50'))
    
    # RSI
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['RSI'], line=dict(color='purple'), name='RSI'))
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['RSI_Overbought'], line=dict(color='red', dash='dash'), name='RSI Overbought'))
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['RSI_Oversold'], line=dict(color='green', dash='dash'), name='RSI Oversold'))
    
    # MACD
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['MACD'], line=dict(color='black'), name='MACD'))
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['MACD_Signal'], line=dict(color='gray'), name='MACD Signal'))
    
    # ADX
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['ADX'], line=dict(color='brown'), name='ADX'))
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['+DI'], line=dict(color='green'), name='+DI'))
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['-DI'], line=dict(color='red'), name='-DI'))
    
    # Order Blocks
    fig.add_trace(go.Scatter(x=data[data['Order_Block']]['datetime'], y=data[data['Order_Block']]['Order_Block_High'], 
                             mode='markers', marker=dict(color='red', symbol='square', size=10), name='Order Block High'))
    fig.add_trace(go.Scatter(x=data[data['Order_Block']]['datetime'], y=data[data['Order_Block']]['Order_Block_Low'], 
                             mode='markers', marker=dict(color='green', symbol='square', size=10), name='Order Block Low'))
    
    # Traditional Buy/Sell Signals
    fig.add_trace(go.Scatter(x=data[data['Final_Signal'] == 1]['datetime'], y=data[data['Final_Signal'] == 1]['close'],
                             mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Traditional Buy Signal'))
    fig.add_trace(go.Scatter(x=data[data['Final_Signal'] == -1]['datetime'], y=data[data['Final_Signal'] == -1]['close'],
                             mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Traditional Sell Signal'))
    
    # ML Buy/Sell Signals
    fig.add_trace(go.Scatter(x=data[data['ML_Signal'] == 1]['datetime'], y=data[data['ML_Signal'] == 1]['close'],
                             mode='markers', marker=dict(color='green', symbol='triangle-up', size=15, line=dict(color='black', width=2)), name='ML Buy Signal'))
    fig.add_trace(go.Scatter(x=data[data['ML_Signal'] == -1]['datetime'], y=data[data['ML_Signal'] == -1]['close'],
                             mode='markers', marker=dict(color='red', symbol='triangle-down', size=15, line=dict(color='black', width=2)), name='ML Sell Signal'))
    
    # Combined Buy/Sell Signals
    fig.add_trace(go.Scatter(x=data[data['Combined_Signal'] == 1]['datetime'], y=data[data['Combined_Signal'] == 1]['close'],
                             mode='markers', marker=dict(color='blue', symbol='triangle-up', size=20, line=dict(color='black', width=2)), name='Combined Buy Signal'))
    fig.add_trace(go.Scatter(x=data[data['Combined_Signal'] == -1]['datetime'], y=data[data['Combined_Signal'] == -1]['close'],
                             mode='markers', marker=dict(color='blue', symbol='triangle-down', size=20, line=dict(color='black', width=2)), name='Combined Sell Signal'))
    
    fig.update_layout(title=f"{symbol} Live Chart with Signals", xaxis_title='Time', yaxis_title='Price')
    return fig