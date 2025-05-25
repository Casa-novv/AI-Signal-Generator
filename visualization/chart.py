# visualization/chart.py
import plotly.graph_objects as go

def create_chart(data, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data['datetime'], open=data['open'], high=data['high'], low=data['low'], close=data['close']))
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['SMA'], line=dict(color='blue'), name='SMA'))
    fig.update_layout(title=f"{symbol} Live Chart", xaxis_title='Time', yaxis_title='Price')
    return fig