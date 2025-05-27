import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import custom modules with error handling
try:
    from data import data_loader
    from indicators import sma, rsi, macd, liquidity, order_blocks, adx, sentiment
    from strategy import signal_generator
    from visualization import chart
    from config import settings
    DATA_LOADER_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    DATA_LOADER_AVAILABLE = False

# Check if MT5 is available
MT5_AVAILABLE = getattr(data_loader, 'MT5_AVAILABLE', False) if DATA_LOADER_AVAILABLE else False

# Page configuration
st.set_page_config(
    page_title="AI Forex Signal Generator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data_source(data_source, **kwargs):
    """Load data from specified source"""
    try:
        if data_source == "MT5" and MT5_AVAILABLE:
            symbol = kwargs.get('symbol', 'EURUSD')
            timeframe = kwargs.get('timeframe', 16385)  # H1
            count = kwargs.get('count', 1000)
            
            data = data_loader.fetch_real_time_data(symbol, timeframe, count)
            if data is None:
                st.error("Failed to fetch data from MT5. Please check your MT5 connection.")
                return None
            return data
            
        elif data_source == "CSV":
            csv_file = kwargs.get('csv_file')
            if csv_file:
                data = data_loader.load_csv_data(csv_file)
                if data is None:
                    st.error("Failed to load CSV file. Please check the file format.")
                    return None
                return data
            else:
                st.error("Please select a CSV file.")
                return None
                
        elif data_source == "Sample":
            symbol = kwargs.get('symbol', 'EURUSD')
            days = kwargs.get('days', 30)
            timeframe_minutes = kwargs.get('timeframe_minutes', 60)
            
            data = data_loader.generate_sample_data(symbol, days, timeframe_minutes)
            return data
            
        else:
            st.error(f"Unknown data source: {data_source}")
            return None
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        logger.error(f"Error in load_data_source: {e}")
        return None

def calculate_all_indicators(data):
    """Calculate all indicators for the data"""
    try:
        if data is None or data.empty:
            return data
        
        # Make a copy to avoid modifying original data
        result_data = data.copy()
        
        # Calculate SMA
        result_data = sma.calculate_sma(result_data, windows=[10, 20, 50])
        
        # Calculate RSI
        result_data = rsi.calculate_rsi(result_data, window=14)
        
        # Calculate MACD
        result_data = macd.calculate_macd(result_data, fast=12, slow=26, signal=9)
        
        # Calculate ADX
        result_data = adx.calculate_adx(result_data, window=14)
        
        # Calculate Order Blocks
        result_data = order_blocks.calculate_order_blocks(result_data, lookback=50, threshold=0.01)
        
        # Calculate Liquidity
        result_data = liquidity.calculate_liquidity(result_data, lookback=20, volume_threshold=1.5)
        
        # Calculate Sentiment (if available)
        try:
            result_data = sentiment.calculate_sentiment(result_data)
        except Exception as e:
            logger.warning(f"Sentiment calculation failed: {e}")
        
        return result_data
        
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        logger.error(f"Error in calculate_all_indicators: {e}")
        return data

def display_data_info(data):
    """Display data information in sidebar"""
    if data is None or data.empty:
        return
    
    try:
        data_info = data_loader.get_data_info(data)
        
        st.sidebar.subheader("📊 Data Information")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.metric("Total Bars", data_info.get("total_bars", 0))
            st.metric("Duration (Days)", data_info.get("duration_days", 0))
        
        with col2:
            current_price = data_info.get("price_range", {}).get("current", 0)
            change = data_info.get("price_range", {}).get("change", 0)
            st.metric("Current Price", f"{current_price:.5f}", f"{change:+.5f}")
        
        # Data quality
        quality = data_info.get("data_quality", {})
        quality_score = quality.get("overall_score", 0)
        
        if quality_score >= 90:
            quality_color = "🟢"
        elif quality_score >= 70:
            quality_color = "🟡"
        else:
            quality_color = "🔴"
        
        st.sidebar.metric("Data Quality", f"{quality_color} {quality_score:.1f}%")
        
        # Time range
        st.sidebar.text(f"From: {data_info.get('start_time', 'N/A')}")
        st.sidebar.text(f"To: {data_info.get('end_time', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Error displaying data info: {e}")

def main():
    st.title("🤖 AI Forex Signal Generator")
    st.markdown("---")
    
    if not DATA_LOADER_AVAILABLE:
        st.error("❌ Data loader not available. Please check your installation.")
        st.stop()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data source selection
        data_sources = ["Sample", "CSV"]
        if MT5_AVAILABLE:
            data_sources.insert(0, "MT5")
        
        data_source = st.radio("📊 Select Data Source", data_sources)
        
        # Initialize session state for data
        if 'market_data' not in st.session_state:
            st.session_state.market_data = None
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        
        # Data source specific configuration
        data_kwargs = {}
        
        if data_source == "MT5" and MT5_AVAILABLE:
            st.subheader("🔗 MT5 Settings")
            
            # Time frame selection
            timeframes = {
                "M1": 1,
                "M5": 5,
                "M15": 15,
                "M30": 30,
                "H1": 16385,
                "H4": 16388,
                "D1": 16408
            }
            
            selected_timeframe = st.selectbox("⏰ Time Frame", list(timeframes.keys()))
            mt5_timeframe = timeframes[selected_timeframe]
            
            # Symbol selection
            symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
            symbol = st.selectbox("💱 Symbol", symbols)
            
            # Number of bars
            count = st.slider("📊 Number of Bars", 100, 5000, 1000, 100)
            
            data_kwargs = {
                'symbol': symbol,
                'timeframe': mt5_timeframe,
                'count': count
            }
            
        elif data_source == "CSV":
            st.subheader("📁 CSV Settings")
            
            # Get available CSV files
            csv_files = data_loader.get_available_csv_files()
            
            if not csv_files:
                st.warning("No CSV files found. Creating sample files...")
                if st.button("Create Sample CSV Files"):
                    created_files = data_loader.create_sample_csv_files()
                    if created_files:
                        st.success(f"Created {len(created_files)} sample files")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to create sample files")
            else:
                # Display available files
                file_names = [os.path.basename(f) for f in csv_files]
                selected_file_name = st.selectbox("📄 Select CSV File", file_names)
                
                if selected_file_name:
                    selected_file_path = next(f for f in csv_files if os.path.basename(f) == selected_file_name)
                    data_kwargs = {'csv_file': selected_file_path}
                    
                    # Show file info
                    try:
                        file_info = data_loader.detect_csv_format(selected_file_path)
                        st.text(f"Columns: {len(file_info.get('columns', []))}")
                        st.text(f"Separator: '{file_info.get('separator', ',')}'")
                    except:
                        pass
            
            # File upload option
            st.subheader("📤 Upload CSV File")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                # Save uploaded file
                upload_dir = "data/csv/uploaded"
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                data_kwargs = {'csv_file': file_path}
                st.success(f"File uploaded: {uploaded_file.name}")
                
        elif data_source == "Sample":
            st.subheader("🎲 Sample Data Settings")
            
            symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
            symbol = st.selectbox("💱 Symbol", symbols)
            
            days = st.slider("📅 Days of Data", 7, 90, 30)
            
            timeframe_options = {
                "15 Minutes": 15,
                "30 Minutes": 30,
                "1 Hour": 60,
                "4 Hours": 240,
                "1 Day": 1440
            }
            
            selected_tf = st.selectbox("⏰ Timeframe", list(timeframe_options.keys()))
            timeframe_minutes = timeframe_options[selected_tf]
            
            data_kwargs = {
                'symbol': symbol,
                'days': days,
                'timeframe_minutes': timeframe_minutes
            }
        
        # Load data button
        if st.button("🔄 Load Data", type="primary"):
            with st.spinner("Loading data..."):
                st.session_state.market_data = load_data_source(data_source, **data_kwargs)
                
                if st.session_state.market_data is not None:
                    st.success("✅ Data loaded successfully!")
                    
                    # Calculate indicators
                    with st.spinner("Calculating indicators..."):
                        st.session_state.processed_data = calculate_all_indicators(st.session_state.market_data)
                    
                    st.success("✅ Indicators calculated!")
                else:
                    st.error("❌ Failed to load data")
        
        # Display data info if available
        if st.session_state.market_data is not None:
            display_data_info(st.session_state.market_data)
    
    # Main content area
    if st.session_state.processed_data is not None:
        data = st.session_state.processed_data
        
        # Generate signals
        with st.spinner("Generating trading signals..."):
            try:
                data_with_signals = signal_generator.generate_signals(data)
                
                # Display current signals
                latest_signal = data_with_signals.iloc[-1]
                
                # Signal dashboard
                st.subheader("🎯 Current Trading Signal")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    signal_value = latest_signal.get('signal', 0)
                    if signal_value == 1:
                        st.success("🟢 BUY SIGNAL")
                    elif signal_value == -1:
                        st.error("🔴 SELL SIGNAL")
                    else:
                        st.info("⚪ NO SIGNAL")
                
                with col2:
                    strength = latest_signal.get('signal_strength', 0)
                    st.metric("Signal Strength", f"{strength:.2f}")
                
                with col3:
                    current_price = latest_signal.get('close', 0)
                    st.metric("Current Price", f"{current_price:.5f}")
                
                with col4:
                    entry_price = latest_signal.get('entry_price', 0)
                    if not pd.isna(entry_price):
                        st.metric("Entry Price", f"{entry_price:.5f}")
                    else:
                        st.metric("Entry Price", "N/A")
                
                # Signal reason
                if latest_signal.get('signal_reason'):
                    st.info(f"📋 Signal Reason: {latest_signal['signal_reason']}")
                
                # Charts
                st.subheader("📈 Price Chart with Indicators")
                
                try:
                    fig = chart.create_comprehensive_chart(data_with_signals)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating chart: {e}")
                    logger.error(f"Chart creation error: {e}")
                
                # Signal history table
                st.subheader("📊 Recent Signals")
                
                # Filter signals
                signals_data = data_with_signals[data_with_signals['signal'] != 0].copy()
                
                if not signals_data.empty:
                    # Show last 10 signals
                    recent_signals = signals_data.tail(10)[['time', 'signal', 'signal_strength', 'signal_reason', 'close']].copy()
                    recent_signals['signal_type'] = recent_signals['signal'].map({1: '🟢 BUY', -1: '🔴 SELL'})
                    recent_signals['signal_strength'] = recent_signals['signal_strength'].round(3)
                    recent_signals['close'] = recent_signals['close'].round(5)
                    
                    st.dataframe(
                                                recent_signals[['time', 'signal_type', 'signal_strength', 'close', 'signal_reason']],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No trading signals found in the current dataset.")
                
                # Technical indicators summary
                st.subheader("📊 Technical Indicators Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Moving Averages**")
                    if 'SMA_10' in data_with_signals.columns and 'SMA_20' in data_with_signals.columns:
                        sma10 = latest_signal.get('SMA_10', 0)
                        sma20 = latest_signal.get('SMA_20', 0)
                        st.metric("SMA 10", f"{sma10:.5f}")
                        st.metric("SMA 20", f"{sma20:.5f}")
                        
                        if sma10 > sma20:
                            st.success("🟢 Bullish Trend")
                        else:
                            st.error("🔴 Bearish Trend")
                
                with col2:
                    st.markdown("**Momentum Indicators**")
                    if 'RSI' in data_with_signals.columns:
                        rsi = latest_signal.get('RSI', 50)
                        st.metric("RSI", f"{rsi:.2f}")
                        
                        if rsi > 70:
                            st.warning("⚠️ Overbought")
                        elif rsi < 30:
                            st.warning("⚠️ Oversold")
                        else:
                            st.info("ℹ️ Neutral")
                    
                    if 'MACD' in data_with_signals.columns:
                        macd = latest_signal.get('MACD', 0)
                        macd_signal = latest_signal.get('MACD_Signal', 0)
                        st.metric("MACD", f"{macd:.6f}")
                        
                        if macd > macd_signal:
                            st.success("🟢 Bullish MACD")
                        else:
                            st.error("🔴 Bearish MACD")
                
                with col3:
                    st.markdown("**Trend Strength**")
                    if 'ADX' in data_with_signals.columns:
                        adx = latest_signal.get('ADX', 0)
                        st.metric("ADX", f"{adx:.2f}")
                        
                        if adx > 25:
                            st.success("🟢 Strong Trend")
                        elif adx > 20:
                            st.warning("⚠️ Moderate Trend")
                        else:
                            st.info("ℹ️ Weak Trend")
                
                # Risk Management Section
                st.subheader("⚠️ Risk Management")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    stop_loss = latest_signal.get('stop_loss')
                    if not pd.isna(stop_loss):
                        st.metric("Stop Loss", f"{stop_loss:.5f}")
                    else:
                        st.metric("Stop Loss", "Not Set")
                
                with col2:
                    take_profit = latest_signal.get('take_profit')
                    if not pd.isna(take_profit):
                        st.metric("Take Profit", f"{take_profit:.5f}")
                    else:
                        st.metric("Take Profit", "Not Set")
                
                with col3:
                    # Calculate risk-reward ratio
                    if not pd.isna(stop_loss) and not pd.isna(take_profit) and not pd.isna(entry_price):
                        risk = abs(entry_price - stop_loss)
                        reward = abs(take_profit - entry_price)
                        if risk > 0:
                            rr_ratio = reward / risk
                            st.metric("Risk:Reward", f"1:{rr_ratio:.2f}")
                        else:
                            st.metric("Risk:Reward", "N/A")
                    else:
                        st.metric("Risk:Reward", "N/A")
                
                # Performance Analytics
                st.subheader("📈 Performance Analytics")
                
                # Calculate some basic performance metrics
                signals_only = data_with_signals[data_with_signals['signal'] != 0].copy()
                
                if len(signals_only) > 1:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_signals = len(signals_only)
                        st.metric("Total Signals", total_signals)
                    
                    with col2:
                        buy_signals = len(signals_only[signals_only['signal'] == 1])
                        st.metric("Buy Signals", buy_signals)
                    
                    with col3:
                        sell_signals = len(signals_only[signals_only['signal'] == -1])
                        st.metric("Sell Signals", sell_signals)
                    
                    with col4:
                        avg_strength = signals_only['signal_strength'].mean()
                        st.metric("Avg Strength", f"{avg_strength:.3f}")
                    
                    # Signal distribution chart
                    signal_counts = signals_only['signal'].value_counts()
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Buy Signals', 'Sell Signals'],
                        values=[signal_counts.get(1, 0), signal_counts.get(-1, 0)],
                        hole=.3,
                        marker_colors=['green', 'red']
                    )])
                    
                    fig_pie.update_layout(
                        title="Signal Distribution",
                        height=400
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Data Export Section
                st.subheader("💾 Export Data")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📄 Export to CSV"):
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"trading_data_{timestamp}"
                            
                            export_results = data_loader.export_data_formats(
                                data_with_signals, 
                                filename, 
                                ['csv']
                            )
                            
                            if export_results.get('csv', False):
                                st.success("✅ Data exported to CSV successfully!")
                            else:
                                st.error("❌ Failed to export CSV")
                        except Exception as e:
                            st.error(f"Export error: {e}")
                
                with col2:
                    if st.button("📊 Export to Excel"):
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"trading_data_{timestamp}"
                            
                            export_results = data_loader.export_data_formats(
                                data_with_signals, 
                                filename, 
                                ['excel']
                            )
                            
                            if export_results.get('excel', False):
                                st.success("✅ Data exported to Excel successfully!")
                            else:
                                st.error("❌ Failed to export Excel")
                        except Exception as e:
                            st.error(f"Export error: {e}")
                
                with col3:
                    if st.button("🔄 Export JSON"):
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"trading_data_{timestamp}"
                            
                            export_results = data_loader.export_data_formats(
                                data_with_signals, 
                                filename, 
                                ['json']
                            )
                            
                            if export_results.get('json', False):
                                st.success("✅ Data exported to JSON successfully!")
                            else:
                                st.error("❌ Failed to export JSON")
                        except Exception as e:
                            st.error(f"Export error: {e}")
                
            except Exception as e:
                st.error(f"Error generating signals: {e}")
                logger.error(f"Signal generation error: {e}")
    
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to AI Forex Signal Generator
        
        This application provides advanced forex trading signals using multiple technical indicators and AI-powered analysis.
        
        ### 🚀 Features:
        - **Real-time Data**: Connect to MetaTrader 5 for live market data
        - **Multiple Indicators**: SMA, RSI, MACD, ADX, Order Blocks, Liquidity Zones
        - **Smart Signals**: AI-powered signal generation with confidence scoring
        - **Risk Management**: Automatic stop-loss and take-profit calculations
        - **Interactive Charts**: Professional trading charts with all indicators
        - **Data Export**: Export your analysis to CSV, Excel, or JSON
        
        ### 📊 Supported Data Sources:
        - **MT5**: Live data from MetaTrader 5 (if installed)
        - **CSV Files**: Upload your own historical data
        - **Sample Data**: Generated sample data for testing
        
        ### 🎯 Getting Started:
        1. Select a data source from the sidebar
        2. Configure your settings (symbol, timeframe, etc.)
        3. Click "Load Data" to fetch market data
        4. View generated signals and analysis
        
        **👈 Use the sidebar to configure your data source and start analyzing!**
        """)
        
        # Quick stats about available features
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Indicators", "7+")
        
        with col2:
            st.metric("💱 Symbols", "7+")
        
        with col3:
            st.metric("⏰ Timeframes", "7")
        
        with col4:
            st.metric("📈 Data Sources", "3")
        
        # Sample chart placeholder
        st.subheader("📈 Sample Chart")
        
        # Create a sample chart to show what the interface looks like
        sample_data = data_loader.generate_sample_data("EURUSD", 7, 60)
        if not sample_data.empty:
            try:
                sample_with_indicators = calculate_all_indicators(sample_data)
                sample_with_signals = signal_generator.generate_signals(sample_with_indicators)
                
                fig = chart.create_comprehensive_chart(sample_with_signals)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("👆 This is a sample chart showing EURUSD with technical indicators. Load your own data to start trading!")
                
            except Exception as e:
                st.warning("Sample chart not available. Please load data to see charts.")
                logger.error(f"Sample chart error: {e}")

# Additional utility functions for the app
def display_system_status():
    """Display system status in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 System Status")
    
    # Check MT5 availability
    if MT5_AVAILABLE:
        st.sidebar.success("✅ MT5 Available")
    else:
        st.sidebar.warning("⚠️ MT5 Not Available")
    
    # Check data loader
    if DATA_LOADER_AVAILABLE:
        st.sidebar.success("✅ Data Loader Ready")
    else:
        st.sidebar.error("❌ Data Loader Error")
    
    # Memory usage (if psutil is available)
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        st.sidebar.metric("Memory Usage", f"{memory_percent:.1f}%")
    except ImportError:
        pass

def add_custom_css():
    """Add custom CSS styling"""
    st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    
    .signal-buy {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    
    .signal-sell {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    
    .signal-neutral {
        background-color: #e2e3e5;
        color: #383d41;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #6c757d;
    }
    </style>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    # Add custom styling
    add_custom_css()
    
    # Display system status
    display_system_status()
    
    # Run main application
    main()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🤖 AI Forex Signal Generator | Built with Streamlit & Python</p>
        <p>⚠️ Trading involves risk. This tool is for educational purposes only.</p>
    </div>
    """, unsafe_allow_html=True)