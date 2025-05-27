import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_chart(data: pd.DataFrame, symbol: str = ""):
    """Create comprehensive trading chart with signals and indicators"""
    
    if data.empty:
        return go.Figure()
    
    # Create subplots with secondary y-axis
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(
            f'{symbol} - Price Action & Signals',
            'Volume',
            'RSI (14)',
            'MACD'
        ),
        row_heights=[0.5, 0.15, 0.175, 0.175],
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Add SMAs
    sma_columns = [col for col in data.columns if col.startswith('SMA_')]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, sma_col in enumerate(sma_columns):
        if sma_col in data.columns:
            period = sma_col.split('_')[1]
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[sma_col],
                    mode='lines',
                    name=f'SMA {period}',
                    line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.8
                ),
                row=1, col=1
            )
    
    # Add Order Blocks
    if 'Order_Block_High' in data.columns and 'Order_Block_Low' in data.columns:
        order_blocks = data.dropna(subset=['Order_Block_High', 'Order_Block_Low'])
        
        for idx, row in order_blocks.iterrows():
            # Add order block rectangles
            fig.add_shape(
                type="rect",
                x0=idx, x1=data.index[-1],
                y0=row['Order_Block_Low'], y1=row['Order_Block_High'],
                fillcolor="rgba(255, 255, 0, 0.1)",
                line=dict(color="rgba(255, 255, 0, 0.3)", width=1),
                row=1, col=1
            )
    
    # Add Liquidity Zones
    if 'Liquidity_High' in data.columns and 'Liquidity_Low' in data.columns:
        liquidity_zones = data.dropna(subset=['Liquidity_High', 'Liquidity_Low'])
        
        for idx, row in liquidity_zones.iterrows():
            fig.add_shape(
                type="rect",
                x0=idx, x1=data.index[-1],
                y0=row['Liquidity_Low'], y1=row['Liquidity_High'],
                fillcolor="rgba(0, 255, 255, 0.1)",
                line=dict(color="rgba(0, 255, 255, 0.3)", width=1),
                row=1, col=1
            )
    
    # Add Buy Signals
    buy_signals = data[data['signal'] == 1] if 'signal' in data.columns else pd.DataFrame()
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['close'],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='#00ff00',
                    line=dict(color='#ffffff', width=2)
                ),
                name='Buy Signal',
                text=buy_signals.get('signal_reason', ''),
                hovertemplate='<b>BUY SIGNAL</b><br>' +
                             'Price: $%{y:.4f}<br>' +
                             'Strength: %{customdata:.2f}<br>' +
                             'Reason: %{text}<br>' +
                             '<extra></extra>',
                customdata=buy_signals.get('signal_strength', 0)
            ),
            row=1, col=1
        )
    
    # Add Sell Signals
    sell_signals = data[data['signal'] == -1] if 'signal' in data.columns else pd.DataFrame()
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['close'],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='#ff0000',
                    line=dict(color='#ffffff', width=2)
                ),
                name='Sell Signal',
                text=sell_signals.get('signal_reason', ''),
                hovertemplate='<b>SELL SIGNAL</b><br>' +
                             'Price: $%{y:.4f}<br>' +
                             'Strength: %{customdata:.2f}<br>' +
                             'Reason: %{text}<br>' +
                             '<extra></extra>',
                customdata=sell_signals.get('signal_strength', 0)
            ),
            row=1, col=1
        )
    
    # Add Stop Loss and Take Profit levels
    if 'entry_price' in data.columns:
        active_trades = data.dropna(subset=['entry_price'])
        
        for idx, trade in active_trades.iterrows():
            if pd.notna(trade.get('stop_loss')):
                fig.add_hline(
                    y=trade['stop_loss'],
                    line_dash="dot",
                    line_color="red",
                    opacity=0.7,
                    annotation_text=f"SL: {trade['stop_loss']:.4f}",
                    row=1, col=1
                )
            
            if pd.notna(trade.get('take_profit')):
                fig.add_hline(
                    y=trade['take_profit'],
                    line_dash="dot",
                    line_color="green",
                    opacity=0.7,
                    annotation_text=f"TP: {trade['take_profit']:.4f}",
                    row=1, col=1
                )
    
    # Volume chart
    if 'volume' in data.columns:
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(data['close'], data['open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # RSI chart
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=3, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=3, col=1)
        
        # Add RSI signal markers
        if 'rsi_signal' in data.columns:
            rsi_buy = data[data['rsi_signal'] == 1]
            rsi_sell = data[data['rsi_signal'] == -1]
            
            if not rsi_buy.empty:
                fig.add_trace(
                    go.Scatter(
                        x=rsi_buy.index,
                        y=rsi_buy['RSI'],
                        mode='markers',
                        marker=dict(symbol='circle', size=8, color='green'),
                        name='RSI Buy',
                        showlegend=False
                    ),
                    row=3, col=1
                )
            
            if not rsi_sell.empty:
                fig.add_trace(
                    go.Scatter(
                        x=rsi_sell.index,
                        y=rsi_sell['RSI'],
                        mode='markers',
                        marker=dict(symbol='circle', size=8, color='red'),
                        name='RSI Sell',
                        showlegend=False
                    ),
                    row=3, col=1
                )
    
    # MACD chart
    if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD_Signal'],
                mode='lines',
                name='MACD Signal',
                line=dict(color='orange', width=2)
            ),
            row=4, col=1
        )
        
        if 'MACD_Histogram' in data.columns:
            colors = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['MACD_Histogram'],
                    name='MACD Histogram',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=4, col=1
            )
        
        # Add MACD signal markers
        if 'macd_signal' in data.columns:
            macd_buy = data[data['macd_signal'] == 1]
            macd_sell = data[data['macd_signal'] == -1]
            
            if not macd_buy.empty:
                fig.add_trace(
                    go.Scatter(
                        x=macd_buy.index,
                        y=macd_buy['MACD'],
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=10, color='green'),
                        name='MACD Buy',
                        showlegend=False
                    ),
                    row=4, col=1
                )
            
            if not macd_sell.empty:
                fig.add_trace(
                    go.Scatter(
                        x=macd_sell.index,
                        y=macd_sell['MACD'],
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=10, color='red'),
                        name='MACD Sell',
                        showlegend=False
                    ),
                    row=4, col=1
                )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"AI Trading Signals - {symbol}",
            x=0.5,
            font=dict(size=20)
        ),
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    
    # Update x-axes
    fig.update_xaxes(title_text="Time", row=4, col=1)
    
    return fig

def create_signal_summary_chart(data: pd.DataFrame):
    """Create a summary chart showing signal performance"""
    
    if 'signal' not in data.columns:
        return go.Figure()
    
    signals_data = data[data['signal'] != 0].copy()
    
    if signals_data.empty:
        return go.Figure()
    
    # Create signal distribution
    signal_counts = signals_data['signal'].value_counts()
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=['Sell Signals', 'Buy Signals'],
            y=[signal_counts.get(-1, 0), signal_counts.get(1, 0)],
            marker_color=['red', 'green'],
            text=[signal_counts.get(-1, 0), signal_counts.get(1, 0)],
            textposition='auto'
        )
    )
    
    fig.update_layout(
        title="Signal Distribution",
        xaxis_title="Signal Type",
        yaxis_title="Count",
        template="plotly_dark"
    )
    
    return fig
