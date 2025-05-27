import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_comprehensive_chart(data, symbol="FOREX", timeframe="H1"):
    """Create comprehensive trading chart with indicators"""
    try:
        if data.empty:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Indicators', 'RSI', 'MACD', 'Volume'),
            row_heights=[0.5, 0.2, 0.2, 0.1]
        )
        
        # Main price chart (Candlestick)
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Add SMAs if available
        sma_columns = [col for col in data.columns if col.startswith('SMA_')]
        colors = ['blue', 'orange', 'purple', 'brown']
        for i, sma_col in enumerate(sma_columns[:4]):
            if sma_col in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[sma_col],
                        mode='lines',
                        name=sma_col,
                        line=dict(color=colors[i % len(colors)], width=1)
                    ),
                    row=1, col=1
                )
        
        # Add Bollinger Bands if available
        if all(col in data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Add trading signals
        if 'signal' in data.columns:
            buy_signals = data[data['signal'] == 1]
            sell_signals = data[data['signal'] == -1]
            
            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['close'],
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(
                            symbol='triangle-up',
                            size=10,
                            color='green'
                        )
                    ),
                    row=1, col=1
                )
            
            if not sell_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['close'],
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(
                            symbol='triangle-down',
                            size=10,
                            color='red'
                        )
                    ),
                    row=1, col=1
                )
        
        # RSI subplot
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=2, col=1
            )
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # MACD subplot
        if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue')
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD_Signal'],
                    mode='lines',
                    name='MACD Signal',
                    line=dict(color='red')
                ),
                row=3, col=1
            )
            
            if 'MACD_Histogram' in data.columns:
                colors = ['green' if x >= 0 else 'red' for x in data['MACD_Histogram']]
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['MACD_Histogram'],
                        name='MACD Histogram',
                        marker_color=colors
                    ),
                    row=3, col=1
                )
        
        # Volume subplot
        if 'volume' in data.columns:
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name='Volume',
                    marker_color='lightblue'
                ),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - {timeframe} Chart',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="Volume", row=4, col=1)
        
        return fig
        
    except Exception as e:
        print(f"Error creating comprehensive chart: {e}")
        return None

def create_simple_chart(data, symbol="FOREX"):
    """Create simple price chart"""
    try:
        fig = go.Figure()
        
        if 'close' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue')
                )
            )
        
        fig.update_layout(
            title=f'{symbol} Price Chart',
            xaxis_title='Time',
            yaxis_title='Price',
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating simple chart: {e}")
        return None