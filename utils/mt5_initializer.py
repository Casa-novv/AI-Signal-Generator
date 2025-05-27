import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import time
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# MT5 Timeframe constants mapping
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1
}

# Reverse mapping for timeframe names
TIMEFRAME_NAMES = {v: k for k, v in TIMEFRAMES.items()}

def initialize_mt5(login: Optional[int] = None, 
                  password: Optional[str] = None, 
                  server: Optional[str] = None,
                  path: Optional[str] = None,
                  timeout: int = 60000) -> bool:
    """
    Initialize MT5 connection with optional credentials
    
    Args:
        login: MT5 account login
        password: MT5 account password
        server: MT5 server name
        path: Path to MT5 terminal
        timeout: Connection timeout in milliseconds
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        # Shutdown any existing connection
        mt5.shutdown()
        
        # Initialize MT5
        if path:
            if not mt5.initialize(path=path, login=login, password=password, server=server, timeout=timeout):
                logger.error(f"MT5 initialization failed with path: {path}")
                return False
        elif login and password and server:
            if not mt5.initialize(login=login, password=password, server=server, timeout=timeout):
                logger.error(f"MT5 initialization failed with credentials")
                return False
        else:
            if not mt5.initialize():
                logger.error("MT5 initialization failed with default settings")
                return False
        
        # Verify connection
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            logger.error("Failed to get terminal info after initialization")
            return False
        
        account_info = mt5.account_info()
        if account_info is None:
            logger.warning("Account info not available, but terminal connected")
        else:
            logger.info(f"MT5 initialized successfully - Account: {account_info.login}, Server: {account_info.server}")
        
        return True
        
    except Exception as e:
        logger.error(f"Exception during MT5 initialization: {e}")
        return False

def shutdown_mt5() -> bool:
    """
    Shutdown MT5 connection
    
    Returns:
        bool: True if shutdown successful
    """
    try:
        mt5.shutdown()
        logger.info("MT5 connection shutdown successfully")
        return True
    except Exception as e:
        logger.error(f"Error shutting down MT5: {e}")
        return False

def get_account_info() -> Optional[Dict]:
    """
    Get MT5 account information
    
    Returns:
        Dict with account info or None if failed
    """
    try:
        account_info = mt5.account_info()
        if account_info is None:
            logger.warning("Account info not available")
            return None
        
        return {
            'login': account_info.login,
            'trade_mode': account_info.trade_mode,
            'name': account_info.name,
            'server': account_info.server,
            'currency': account_info.currency,
            'leverage': account_info.leverage,
            'limit_orders': account_info.limit_orders,
            'margin_so_mode': account_info.margin_so_mode,
            'trade_allowed': account_info.trade_allowed,
            'trade_expert': account_info.trade_expert,
            'margin_mode': account_info.margin_mode,
            'currency_digits': account_info.currency_digits,
            'balance': account_info.balance,
            'credit': account_info.credit,
            'profit': account_info.profit,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'margin_free': account_info.margin_free,
            'margin_level': account_info.margin_level,
            'margin_so_call': account_info.margin_so_call,
            'margin_so_so': account_info.margin_so_so,
            'margin_initial': account_info.margin_initial,
            'margin_maintenance': account_info.margin_maintenance,
            'assets': account_info.assets,
            'liabilities': account_info.liabilities,
            'commission_blocked': account_info.commission_blocked
        }
        
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        return None

def get_terminal_info() -> Optional[Dict]:
    """
    Get MT5 terminal information
    
    Returns:
        Dict with terminal info or None if failed
    """
    try:
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            return None
        
        return {
            'community_account': terminal_info.community_account,
            'community_connection': terminal_info.community_connection,
            'connected': terminal_info.connected,
            'dlls_allowed': terminal_info.dlls_allowed,
            'trade_allowed': terminal_info.trade_allowed,
            'tradeapi_disabled': terminal_info.tradeapi_disabled,
            'email_enabled': terminal_info.email_enabled,
            'ftp_enabled': terminal_info.ftp_enabled,
            'notifications_enabled': terminal_info.notifications_enabled,
            'mqid': terminal_info.mqid,
            'build': terminal_info.build,
            'maxbars': terminal_info.maxbars,
            'codepage': terminal_info.codepage,
            'ping_last': terminal_info.ping_last,
            'community_balance': terminal_info.community_balance,
            'retransmission': terminal_info.retransmission,
            'company': terminal_info.company,
            'name': terminal_info.name,
            'language': terminal_info.language,
            'path': terminal_info.path
        }
        
    except Exception as e:
        logger.error(f"Error getting terminal info: {e}")
        return None

def get_available_symbols(pattern: str = "*") -> List[str]:
    """
    Get list of available symbols
    
    Args:
        pattern: Symbol pattern filter (e.g., "*EUR*", "*USD*")
    
    Returns:
        List of symbol names
    """
    try:
        symbols = mt5.symbols_get(pattern)
        if symbols is None:
            logger.warning(f"No symbols found for pattern: {pattern}")
            return []
        
        symbol_names = [symbol.name for symbol in symbols]
        logger.info(f"Found {len(symbol_names)} symbols for pattern: {pattern}")
        return symbol_names
        
    except Exception as e:
        logger.error(f"Error getting symbols: {e}")
        return []

def get_symbol_info(symbol: str) -> Optional[Dict]:
    """
    Get detailed information about a symbol
    
    Args:
        symbol: Symbol name
    
    Returns:
        Dict with symbol info or None if failed
    """
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"Symbol info not available for: {symbol}")
            return None
        
        return {
            'custom': symbol_info.custom,
            'chart_mode': symbol_info.chart_mode,
            'select': symbol_info.select,
            'visible': symbol_info.visible,
            'session_deals': symbol_info.session_deals,
            'session_buy_orders': symbol_info.session_buy_orders,
            'session_sell_orders': symbol_info.session_sell_orders,
            'volume': symbol_info.volume,
            'volumehigh': symbol_info.volumehigh,
            'volumelow': symbol_info.volumelow,
            'time': symbol_info.time,
            'digits': symbol_info.digits,
            'spread': symbol_info.spread,
            'spread_float': symbol_info.spread_float,
            'ticks_bookdepth': symbol_info.ticks_bookdepth,
            'trade_calc_mode': symbol_info.trade_calc_mode,
            'trade_mode': symbol_info.trade_mode,
            'start_time': symbol_info.start_time,
            'expiration_time': symbol_info.expiration_time,
            'trade_stops_level': symbol_info.trade_stops_level,
            'trade_freeze_level': symbol_info.trade_freeze_level,
            'trade_exemode': symbol_info.trade_exemode,
            'swap_mode': symbol_info.swap_mode,
            'swap_rollover3days': symbol_info.swap_rollover3days,
            'margin_hedged_use_leg': symbol_info.margin_hedged_use_leg,
            'expiration_mode': symbol_info.expiration_mode,
            'filling_mode': symbol_info.filling_mode,
            'order_mode': symbol_info.order_mode,
            'order_gtc_mode': symbol_info.order_gtc_mode,
            'option_mode': symbol_info.option_mode,
            'option_right': symbol_info.option_right,
            'bid': symbol_info.bid,
            'bidhigh': symbol_info.bidhigh,
            'bidlow': symbol_info.bidlow,
            'ask': symbol_info.ask,
            'askhigh': symbol_info.askhigh,
            'asklow': symbol_info.asklow,
            'last': symbol_info.last,
            'lasthigh': symbol_info.lasthigh,
            'lastlow': symbol_info.lastlow,
            'volume_real': symbol_info.volume_real,
            'volumehigh_real': symbol_info.volumehigh_real,
            'volumelow_real': symbol_info.volumelow_real,
            'option_strike': symbol_info.option_strike,
            'point': symbol_info.point,
            'trade_tick_value': symbol_info.trade_tick_value,
            'trade_tick_value_profit': symbol_info.trade_tick_value_profit,
            'trade_tick_value_loss': symbol_info.trade_tick_value_loss,
            'trade_tick_size': symbol_info.trade_tick_size,
            'trade_contract_size': symbol_info.trade_contract_size,
            'trade_accrued_interest': symbol_info.trade_accrued_interest,
            'trade_face_value': symbol_info.trade_face_value,
            'trade_liquidity_rate': symbol_info.trade_liquidity_rate,
            'volume_min': symbol_info.volume_min,
            'volume_max': symbol_info.volume_max,
            'volume_step': symbol_info.volume_step,
            'volume_limit': symbol_info.volume_limit,
            'swap_long': symbol_info.swap_long,
            'swap_short': symbol_info.swap_short,
            'margin_initial': symbol_info.margin_initial,
            'margin_maintenance': symbol_info.margin_maintenance,
            'session_volume': symbol_info.session_volume,
            'session_turnover': symbol_info.session_turnover,
            'session_interest': symbol_info.session_interest,
            'session_buy_orders_volume': symbol_info.session_buy_orders_volume,
            'session_sell_orders_volume': symbol_info.session_sell_orders_volume,
            'session_open': symbol_info.session_open,
            'session_close': symbol_info.session_close,
            'session_aw': symbol_info.session_aw,
            'session_price_settlement': symbol_info.session_price_settlement,
            'session_price_limit_min': symbol_info.session_price_limit_min,
            'session_price_limit_max': symbol_info.session_price_limit_max,
            'margin_hedged': symbol_info.margin_hedged,
            'price_change': symbol_info.price_change,
            'price_volatility': symbol_info.price_volatility,
            'price_theoretical': symbol_info.price_theoretical,
            'price_greeks_delta': symbol_info.price_greeks_delta,
            'price_greeks_theta': symbol_info.price_greeks_theta,
            'price_greeks_gamma': symbol_info.price_greeks_gamma,
            'price_greeks_vega': symbol_info.price_greeks_vega,
            'price_greeks_rho': symbol_info.price_greeks_rho,
            'price_greeks_omega': symbol_info.price_greeks_omega,
            'price_sensitivity': symbol_info.price_sensitivity,
            'basis': symbol_info.basis,
            'category': symbol_info.category,
            'currency_base': symbol_info.currency_base,
            'currency_profit': symbol_info.currency_profit,
            'currency_margin': symbol_info.currency_margin,
            'bank': symbol_info.bank,
            'description': symbol_info.description,
            'exchange': symbol_info.exchange,
            'formula': symbol_info.formula,
            'isin': symbol_info.isin,
            'name': symbol_info.name,
            'page': symbol_info.page,
            'path': symbol_info.path
        }
        
    except Exception as e:
        logger.error(f"Error getting symbol info for {symbol}: {e}")
        return None

def select_symbol(symbol: str) -> bool:
    """
    Select symbol in Market Watch
    
    Args:
        symbol: Symbol name
    
    Returns:
        bool: True if symbol selected successfully
    """
    try:
        if mt5.symbol_select(symbol, True):
            logger.debug(f"Symbol {symbol} selected successfully")
            return True
        else:
            logger.warning(f"Failed to select symbol: {symbol}")
            return False
            
    except Exception as e:
        logger.error(f"Error selecting symbol {symbol}: {e}")
        return False

def get_market_data(symbol: str, timeframe: int, count: int = 1000) -> Optional[List]:
    """
    Get market data for symbol
    
    Args:
        symbol: Symbol name
        timeframe: MT5 timeframe constant
        count: Number of bars to retrieve
    
    Returns:
        List of market data or None if failed
    """
    try:
        if not select_symbol(symbol):
            return None
        
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        
        if rates is None:
            logger.warning(f"No market data available for {symbol}")
            return None
        
        logger.info(f"Retrieved {len(rates)} bars for {symbol}")
        return rates.tolist()
        
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
        return None

def get_market_data_range(symbol: str, timeframe: int, date_from: datetime, date_to: datetime) -> Optional[List]:
    """
    Get market data for symbol within date range
    
    Args:
        symbol: Symbol name
        timeframe: MT5 timeframe constant
        date_from: Start date
        date_to: End date
    
    Returns:
        List of market data or None if failed
    """
    try:
        if not select_symbol(symbol):
            return None
        
        rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
        
        if rates is None:
            logger.warning(f"No market data available for {symbol} in range {date_from} to {date_to}")
            return None
        
        logger.info(f"Retrieved {len(rates)} bars for {symbol} from {date_from} to {date_to}")
        return rates.tolist()
        
    except Exception as e:
        logger.error(f"Error getting market data range for {symbol}: {e}")
        return None

def get_ticks(symbol: str, count: int = 1000) -> Optional[List]:
    """
    Get tick data for symbol
    
    Args:
        symbol: Symbol name
        count: Number of ticks to retrieve
    
    Returns:
        List of tick data or None if failed
    """
    try:
        if not select_symbol(symbol):
            return None
        
        ticks = mt5.copy_ticks_from_pos(symbol, 0, count, mt5.COPY_TICKS_ALL)
        
        if ticks is None:
            logger.warning(f"No tick data available for {symbol}")
            return None
        
        logger.info(f"Retrieved {len(ticks)} ticks for {symbol}")
        return ticks.tolist()
        
    except Exception as e:
        logger.error(f"Error getting tick data for {symbol}: {e}")
        return None

def get_ticks_range(symbol: str, date_from: datetime, date_to: datetime) -> Optional[List]:
    """
    Get tick data for symbol within date range
    
    Args:
        symbol: Symbol name
        date_from: Start date
        date_to: End date
    
    Returns:
        List of tick data or None if failed
    """
    try:
        if not select_symbol(symbol):
            return None
        
        ticks = mt5.copy_ticks_range(symbol, date_from, date_to, mt5.COPY_TICKS_ALL)
        
        if ticks is None:
            logger.warning(f"No tick data available for {symbol} in range {date_from} to {date_to}")
            return None
        
        logger.info(f"Retrieved {len(ticks)} ticks for {symbol} from {date_from} to {date_to}")
        return ticks.tolist()
        
    except Exception as e:
        logger.error(f"Error getting tick data range for {symbol}: {e}")
        return None

def get_current_price(symbol: str) -> Optional[Dict]:
    """
    Get current price information for symbol
    
    Args:
        symbol: Symbol name
    
    Returns:
        Dict with current price info or None if failed
    """
    try:
        if not select_symbol(symbol):
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        
        if tick is None:
            logger.warning(f"No current price available for {symbol}")
            return None
        
        return {
            'time': tick.time,
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'time_msc': tick.time_msc,
            'flags': tick.flags,
            'volume_real': tick.volume_real
        }
        
    except Exception as e:
        logger.error(f"Error getting current price for {symbol}: {e}")
        return None

def check_connection() -> bool:
    """
    Check if MT5 connection is active
    
    Returns:
        bool: True if connected, False otherwise
    """
    try:
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            return False
        
        return terminal_info.connected
        
    except Exception as e:
        logger.error(f"Error checking connection: {e}")
        return False

def reconnect(max_attempts: int = 3, delay: int = 5) -> bool:
    """
    Attempt to reconnect to MT5
    
    Args:
        max_attempts: Maximum number of reconnection attempts
        delay: Delay between attempts in seconds
    
    Returns:
        bool: True if reconnection successful
    """
    for attempt in range(max_attempts):
        try:
            logger.info(f"Reconnection attempt {attempt + 1}/{max_attempts}")
            
            # Shutdown existing connection
            mt5.shutdown()
            time.sleep(delay)
            
            # Try to initialize again
            if initialize_mt5():
                logger.info("Reconnection successful")
                return True
            
            time.sleep(delay)
            
        except Exception as e:
            logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
    
    logger.error("All reconnection attempts failed")
    return False

def get_positions() -> Optional[List[Dict]]:
    """
    Get all open positions
    
    Returns:
        List of position dictionaries or None if failed
    """
    try:
        positions = mt5.positions_get()
        
        if positions is None:
            logger.info("No open positions")
            return []
        
        position_list = []
        for position in positions:
            position_dict = {
                'ticket': position.ticket,
                'time': position.time,
                'time_msc': position.time_msc,
                'time_update': position.time_update,
                'time_update_msc': position.time_update_msc,
                'type': position.type,
                'magic': position.magic,
                'identifier': position.identifier,
                'reason': position.reason,
                'volume': position.volume,
                'price_open': position.price_open,
                'sl': position.sl,
                'tp': position.tp,
                'price_current': position.price_current,
                'swap': position.swap,
                'profit': position.profit,
                'symbol': position.symbol,
                'comment': position.comment,
                'external_id': position.external_id
            }
            position_list.append(position_dict)
        
        logger.info(f"Retrieved {len(position_list)} open positions")
        return position_list
        
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return None

def get_orders() -> Optional[List[Dict]]:
    """
    Get all pending orders
    
    Returns:
        List of order dictionaries or None if failed
    """
    try:
        orders = mt5.orders_get()
        
        if orders is None:
            logger.info("No pending orders")
            return []
        
        order_list = []
        for order in orders:
            order_dict = {
                'ticket': order.ticket,
                'time_setup': order.time_setup,
                'time_setup_msc': order.time_setup_msc,
                'time_expiration': order.time_expiration,
                'type': order.type,
                'type_filling': order.type_filling,
                'type_time': order.type_time,
                'state': order.state,
                'magic': order.magic,
                'position_id': order.position_id,
                'position_by_id': order.position_by_id,
                'reason': order.reason,
                'volume_initial': order.volume_initial,
                'volume_current': order.volume_current,
                'price_open': order.price_open,
                'sl': order.sl,
                'tp': order.tp,
                'price_current': order.price_current,
                'price_stoplimit': order.price_stoplimit,
                'symbol': order.symbol,
                'comment': order.comment,
                'external_id': order.external_id
            }
            order_list.append(order_dict)
        
        logger.info(f"Retrieved {len(order_list)} pending orders")
        return order_list
        
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        return None

def get_history_deals(date_from: datetime, date_to: datetime) -> Optional[List[Dict]]:
    """
    Get historical deals within date range
    
    Args:
        date_from: Start date
        date_to: End date
    
    Returns:
        List of deal dictionaries or None if failed
    """
    try:
        deals = mt5.history_deals_get(date_from, date_to)
        
        if deals is None:
            logger.info(f"No deals found from {date_from} to {date_to}")
            return []
        
        deal_list = []
        for deal in deals:
            deal_dict = {
                'ticket': deal.ticket,
                'order': deal.order,
                'time': deal.time,
                'time_msc': deal.time_msc,
                'type': deal.type,
                'entry': deal.entry,
                'magic': deal.magic,
                'position_id': deal.position_id,
                'reason': deal.reason,
                'volume': deal.volume,
                'price': deal.price,
                'commission': deal.commission,
                'swap': deal.swap,
                'profit': deal.profit,
                'fee': deal.fee,
                'symbol': deal.symbol,
                'comment': deal.comment,
                'external_id': deal.external_id
            }
            deal_list.append(deal_dict)
        
        logger.info(f"Retrieved {len(deal_list)} historical deals")
        return deal_list
        
    except Exception as e:
        logger.error(f"Error getting historical deals: {e}")
        return None

def get_history_orders(date_from: datetime, date_to: datetime) -> Optional[List[Dict]]:
    """
    Get historical orders within date range
    
    Args:
        date_from: Start date
        date_to: End date
    
    Returns:
        List of order dictionaries or None if failed
    """
    try:
        orders = mt5.history_orders_get(date_from, date_to)
        
        if orders is None:
            logger.info(f"No historical orders found from {date_from} to {date_to}")
            return []
        
        order_list = []
        for order in orders:
            order_dict = {
                'ticket': order.ticket,
                'time_setup': order.time_setup,
                'time_setup_msc': order.time_setup_msc,
                'time_done': order.time_done,
                'time_done_msc': order.time_done_msc,
                'time_expiration': order.time_expiration,
                'type': order.type,
                'type_filling': order.type_filling,
                'type_time': order.type_time,
                'state': order.state,
                'magic': order.magic,
                'position_id': order.position_id,
                'position_by_id': order.position_by_id,
                'reason': order.reason,
                'volume_initial': order.volume_initial,
                'volume_current': order.volume_current,
                'price_open': order.price_open,
                'sl': order.sl,
                'tp': order.tp,
                'price_current': order.price_current,
                'price_stoplimit': order.price_stoplimit,
                'symbol': order.symbol,
                'comment': order.comment,
                'external_id': order.external_id
            }
            order_list.append(order_dict)
        
        logger.info(f"Retrieved {len(order_list)} historical orders")
        return order_list
        
    except Exception as e:
        logger.error(f"Error getting historical orders: {e}")
        return None

def send_order(symbol: str, order_type: int, volume: float, price: float = 0.0, 
               sl: float = 0.0, tp: float = 0.0, deviation: int = 20, 
               magic: int = 0, comment: str = "", type_time: int = mt5.ORDER_TIME_GTC,
               type_filling: int = mt5.ORDER_FILLING_IOC) -> Optional[Dict]:
    """
    Send a trading order
    
    Args:
        symbol: Symbol name
        order_type: Order type (mt5.ORDER_TYPE_BUY, etc.)
        volume: Order volume
        price: Order price (0 for market orders)
        sl: Stop loss price
        tp: Take profit price
        deviation: Maximum price deviation for market orders
        magic: Magic number
        comment: Order comment
        type_time: Order time type
        type_filling: Order filling type
    
    Returns:
        Dict with order result or None if failed
    """
    try:
        if not select_symbol(symbol):
            return None
        
        # Get current price if not specified
        if price == 0.0:
            current_price = get_current_price(symbol)
            if current_price is None:
                return None
            
            if order_type == mt5.ORDER_TYPE_BUY:
                price = current_price['ask']
            elif order_type == mt5.ORDER_TYPE_SELL:
                price = current_price['bid']
        
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_time": type_time,
            "type_filling": type_filling,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            logger.error("Order send failed - no result returned")
            return None
        
        result_dict = {
            'retcode': result.retcode,
            'deal': result.deal,
            'order': result.order,
            'volume': result.volume,
            'price': result.price,
            'bid': result.bid,
            'ask': result.ask,
            'comment': result.comment,
            'request_id': result.request_id,
            'retcode_external': result.retcode_external
        }
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Order executed successfully: {result_dict}")
        else:
            logger.error(f"Order failed with retcode {result.retcode}: {result.comment}")
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error sending order: {e}")
        return None
def close_position(ticket: int, deviation: int = 20) -> Optional[Dict]:
    """
    Close an open position
    
    Args:
        ticket: Position ticket
        deviation: Maximum price deviation
    
    Returns:
        Dict with close result or None if failed
    """
    try:
        # Get position info
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.error(f"Position {ticket} not found")
            return None
        
        position = positions[0]
        
        # Determine opposite order type
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
        else:
            order_type = mt5.ORDER_TYPE_BUY
        
        # Get current price
        current_price = get_current_price(position.symbol)
        if current_price is None:
            return None
        
        price = current_price['bid'] if order_type == mt5.ORDER_TYPE_SELL else current_price['ask']
        
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": deviation,
            "magic": position.magic,
            "comment": f"Close position {ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send close order
        result = mt5.order_send(request)
        
        if result is None:
            logger.error("Position close failed - no result returned")
            return None
        
        result_dict = {
            'retcode': result.retcode,
            'deal': result.deal,
            'order': result.order,
            'volume': result.volume,
            'price': result.price,
            'bid': result.bid,
            'ask': result.ask,
            'comment': result.comment,
            'request_id': result.request_id,
            'retcode_external': result.retcode_external
        }
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Position {ticket} closed successfully")
        else:
            logger.error(f"Position close failed with retcode {result.retcode}: {result.comment}")
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error closing position {ticket}: {e}")
        return None

def modify_position(ticket: int, sl: float = 0.0, tp: float = 0.0) -> Optional[Dict]:
    """
    Modify stop loss and take profit of an open position
    
    Args:
        ticket: Position ticket
        sl: New stop loss price (0 to remove)
        tp: New take profit price (0 to remove)
    
    Returns:
        Dict with modification result or None if failed
    """
    try:
        # Get position info
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.error(f"Position {ticket} not found")
            return None
        
        position = positions[0]
        
        # Prepare modification request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }
        
        # Send modification order
        result = mt5.order_send(request)
        
        if result is None:
            logger.error("Position modification failed - no result returned")
            return None
        
        result_dict = {
            'retcode': result.retcode,
            'deal': result.deal,
            'order': result.order,
            'volume': result.volume,
            'price': result.price,
            'bid': result.bid,
            'ask': result.ask,
            'comment': result.comment,
            'request_id': result.request_id,
            'retcode_external': result.retcode_external
        }
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Position {ticket} modified successfully - SL: {sl}, TP: {tp}")
        else:
            logger.error(f"Position modification failed with retcode {result.retcode}: {result.comment}")
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error modifying position {ticket}: {e}")
        return None

def cancel_order(ticket: int) -> Optional[Dict]:
    """
    Cancel a pending order
    
    Args:
        ticket: Order ticket
    
    Returns:
        Dict with cancellation result or None if failed
    """
    try:
        # Prepare cancellation request
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket,
        }
        
        # Send cancellation order
        result = mt5.order_send(request)
        
        if result is None:
            logger.error("Order cancellation failed - no result returned")
            return None
        
        result_dict = {
            'retcode': result.retcode,
            'deal': result.deal,
            'order': result.order,
            'volume': result.volume,
            'price': result.price,
            'bid': result.bid,
            'ask': result.ask,
            'comment': result.comment,
            'request_id': result.request_id,
            'retcode_external': result.retcode_external
        }
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Order {ticket} cancelled successfully")
        else:
            logger.error(f"Order cancellation failed with retcode {result.retcode}: {result.comment}")
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error cancelling order {ticket}: {e}")
        return None

def calculate_lot_size(symbol: str, risk_amount: float, sl_pips: float) -> float:
    """
    Calculate lot size based on risk amount and stop loss in pips
    
    Args:
        symbol: Symbol name
        risk_amount: Risk amount in account currency
        sl_pips: Stop loss in pips
    
    Returns:
        Calculated lot size
    """
    try:
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return 0.0
        
        # Get pip value
        pip_value = symbol_info['trade_tick_value']
        if symbol_info['digits'] == 5 or symbol_info['digits'] == 3:
            pip_value *= 10  # Adjust for 5-digit brokers
        
        # Calculate lot size
        lot_size = risk_amount / (sl_pips * pip_value)
        
        # Round to valid lot size
        volume_step = symbol_info['volume_step']
        lot_size = round(lot_size / volume_step) * volume_step
        
        # Ensure within limits
        lot_size = max(symbol_info['volume_min'], lot_size)
        lot_size = min(symbol_info['volume_max'], lot_size)
        
        return lot_size
        
    except Exception as e:
        logger.error(f"Error calculating lot size for {symbol}: {e}")
        return 0.0

def get_pip_value(symbol: str, lot_size: float = 1.0) -> float:
    """
    Get pip value for a symbol
    
    Args:
        symbol: Symbol name
        lot_size: Lot size
    
    Returns:
        Pip value in account currency
    """
    try:
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return 0.0
        
        pip_value = symbol_info['trade_tick_value'] * lot_size
        
        # Adjust for 5-digit brokers
        if symbol_info['digits'] == 5 or symbol_info['digits'] == 3:
            pip_value *= 10
        
        return pip_value
        
    except Exception as e:
        logger.error(f"Error getting pip value for {symbol}: {e}")
        return 0.0

def get_spread(symbol: str) -> float:
    """
    Get current spread for symbol in pips
    
    Args:
        symbol: Symbol name
    
    Returns:
        Spread in pips
    """
    try:
        current_price = get_current_price(symbol)
        symbol_info = get_symbol_info(symbol)
        
        if current_price is None or symbol_info is None:
            return 0.0
        
        spread_points = current_price['ask'] - current_price['bid']
        point_value = symbol_info['point']
        
        # Convert to pips
        if symbol_info['digits'] == 5 or symbol_info['digits'] == 3:
            spread_pips = spread_points / (point_value * 10)
        else:
            spread_pips = spread_points / point_value
        
        return spread_pips
        
    except Exception as e:
        logger.error(f"Error getting spread for {symbol}: {e}")
        return 0.0

def is_market_open(symbol: str) -> bool:
    """
    Check if market is open for trading
    
    Args:
        symbol: Symbol name
    
    Returns:
        bool: True if market is open
    """
    try:
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return False
        
        # Check if symbol is selected and visible
        if not symbol_info['select'] or not symbol_info['visible']:
            return False
        
        # Check trade mode
        trade_mode = symbol_info['trade_mode']
        if trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            return False
        
        # Get current price to verify market activity
        current_price = get_current_price(symbol)
        if current_price is None:
            return False
        
        # Check if bid and ask are available
        if current_price['bid'] == 0 or current_price['ask'] == 0:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking market status for {symbol}: {e}")
        return False

def get_trading_hours(symbol: str) -> Optional[Dict]:
    """
    Get trading hours information for symbol
    
    Args:
        symbol: Symbol name
    
    Returns:
        Dict with trading hours info or None if failed
    """
    try:
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return None
        
        return {
            'start_time': symbol_info['start_time'],
            'expiration_time': symbol_info['expiration_time'],
            'trade_mode': symbol_info['trade_mode'],
            'trade_stops_level': symbol_info['trade_stops_level'],
            'trade_freeze_level': symbol_info['trade_freeze_level']
        }
        
    except Exception as e:
        logger.error(f"Error getting trading hours for {symbol}: {e}")
        return None

def validate_order_params(symbol: str, volume: float, price: float = 0.0, 
                         sl: float = 0.0, tp: float = 0.0) -> Tuple[bool, str]:
    """
    Validate order parameters
    
    Args:
        symbol: Symbol name
        volume: Order volume
        price: Order price
        sl: Stop loss price
        tp: Take profit price
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return False, f"Symbol {symbol} not found"
        
        # Check volume limits
        if volume < symbol_info['volume_min']:
            return False, f"Volume {volume} below minimum {symbol_info['volume_min']}"
        
        if volume > symbol_info['volume_max']:
            return False, f"Volume {volume} above maximum {symbol_info['volume_max']}"
        
        # Check volume step
        volume_step = symbol_info['volume_step']
        if volume % volume_step != 0:
            return False, f"Volume {volume} not multiple of step {volume_step}"
        
        # Check if market is open
        if not is_market_open(symbol):
            return False, f"Market closed for {symbol}"
        
        # Check stop levels if SL/TP provided
        if sl > 0 or tp > 0:
            stops_level = symbol_info['trade_stops_level']
            current_price = get_current_price(symbol)
            
            if current_price is None:
                return False, "Cannot get current price"
            
            point = symbol_info['point']
            min_distance = stops_level * point
            
            if sl > 0:
                sl_distance = abs(price - sl) if price > 0 else abs(current_price['bid'] - sl)
                if sl_distance < min_distance:
                    return False, f"Stop loss too close to price. Minimum distance: {stops_level} points"
            
            if tp > 0:
                tp_distance = abs(price - tp) if price > 0 else abs(current_price['ask'] - tp)
                if tp_distance < min_distance:
                    return False, f"Take profit too close to price. Minimum distance: {stops_level} points"
        
        return True, "Order parameters valid"
        
    except Exception as e:
        logger.error(f"Error validating order parameters: {e}")
        return False, f"Validation error: {e}"

def get_margin_required(symbol: str, volume: float) -> float:
    """
    Calculate required margin for a position
    
    Args:
        symbol: Symbol name
        volume: Position volume
    
    Returns:
        Required margin in account currency
    """
    try:
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            return 0.0
        
        # Get current price
        current_price = get_current_price(symbol)
        if current_price is None:
            return 0.0
        
        # Calculate margin
        contract_size = symbol_info['trade_contract_size']
        margin_rate = symbol_info['margin_initial']
        
        if margin_rate == 0:
            margin_rate = symbol_info['margin_maintenance']
        
        margin = volume * contract_size * current_price['ask'] * margin_rate
        
        return margin
        
    except Exception as e:
        logger.error(f"Error calculating margin for {symbol}: {e}")
        return 0.0
def get_free_margin() -> float:
    """
    Get free margin from account
    
    Returns:
        Free margin amount
    """
    try:
        account_info = get_account_info()
        if account_info is None:
            return 0.0
        
        return account_info['margin_free']
        
    except Exception as e:
        logger.error(f"Error getting free margin: {e}")
        return 0.0

def check_margin_sufficient(symbol: str, volume: float) -> bool:
    """
    Check if there's sufficient margin for a trade
    
    Args:
        symbol: Symbol name
        volume: Position volume
    
    Returns:
        bool: True if margin is sufficient
    """
    try:
        required_margin = get_margin_required(symbol, volume)
        free_margin = get_free_margin()
        
        return free_margin >= required_margin
        
    except Exception as e:
        logger.error(f"Error checking margin sufficiency: {e}")
        return False

def get_server_time() -> Optional[datetime]:
    """
    Get MT5 server time
    
    Returns:
        Server time as datetime object or None if failed
    """
    try:
        # Get any symbol's current tick to get server time
        symbols = get_available_symbols()
        if not symbols:
            return None
        
        tick = mt5.symbol_info_tick(symbols[0])
        if tick is None:
            return None
        
        return datetime.fromtimestamp(tick.time)
        
    except Exception as e:
        logger.error(f"Error getting server time: {e}")
        return None

def convert_timeframe_to_minutes(timeframe: int) -> int:
    """
    Convert MT5 timeframe constant to minutes
    
    Args:
        timeframe: MT5 timeframe constant
    
    Returns:
        Number of minutes
    """
    timeframe_minutes = {
        mt5.TIMEFRAME_M1: 1,
        mt5.TIMEFRAME_M2: 2,
        mt5.TIMEFRAME_M3: 3,
        mt5.TIMEFRAME_M4: 4,
        mt5.TIMEFRAME_M5: 5,
        mt5.TIMEFRAME_M6: 6,
        mt5.TIMEFRAME_M10: 10,
        mt5.TIMEFRAME_M12: 12,
        mt5.TIMEFRAME_M15: 15,
        mt5.TIMEFRAME_M20: 20,
        mt5.TIMEFRAME_M30: 30,
        mt5.TIMEFRAME_H1: 60,
        mt5.TIMEFRAME_H2: 120,
        mt5.TIMEFRAME_H3: 180,
        mt5.TIMEFRAME_H4: 240,
        mt5.TIMEFRAME_H6: 360,
        mt5.TIMEFRAME_H8: 480,
        mt5.TIMEFRAME_H12: 720,
        mt5.TIMEFRAME_D1: 1440,
        mt5.TIMEFRAME_W1: 10080,
        mt5.TIMEFRAME_MN1: 43200
    }
    
    return timeframe_minutes.get(timeframe, 0)

def get_timeframe_name(timeframe: int) -> str:
    """
    Get timeframe name from MT5 constant
    
    Args:
        timeframe: MT5 timeframe constant
    
    Returns:
        Timeframe name string
    """
    return TIMEFRAME_NAMES.get(timeframe, f"Unknown({timeframe})")

def backup_connection_info() -> Dict:
    """
    Backup current connection information
    
    Returns:
        Dict with connection info
    """
    try:
        return {
            'account_info': get_account_info(),
            'terminal_info': get_terminal_info(),
            'connected': check_connection(),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error backing up connection info: {e}")
        return {}

def get_connection_status() -> Dict:
    """
    Get comprehensive connection status
    
    Returns:
        Dict with detailed connection status
    """
    try:
        terminal_info = get_terminal_info()
        account_info = get_account_info()
        
        status = {
            'connected': check_connection(),
            'terminal_connected': terminal_info['connected'] if terminal_info else False,
            'trade_allowed': terminal_info['trade_allowed'] if terminal_info else False,
            'account_trade_allowed': account_info['trade_allowed'] if account_info else False,
            'account_trade_expert': account_info['trade_expert'] if account_info else False,
            'server_time': get_server_time(),
            'local_time': datetime.now(),
            'ping': terminal_info['ping_last'] if terminal_info else None,
            'build': terminal_info['build'] if terminal_info else None,
            'company': terminal_info['company'] if terminal_info else None,
            'account_server': account_info['server'] if account_info else None,
            'account_login': account_info['login'] if account_info else None
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting connection status: {e}")
        return {'connected': False, 'error': str(e)}

def test_connection() -> Dict:
    """
    Test MT5 connection with comprehensive checks
    
    Returns:
        Dict with test results
    """
    test_results = {
        'initialization': False,
        'terminal_info': False,
        'account_info': False,
        'symbol_access': False,
        'market_data': False,
        'trading_allowed': False,
        'overall_status': False,
        'errors': []
    }
    
    try:
        # Test initialization
        if check_connection():
            test_results['initialization'] = True
        else:
            test_results['errors'].append("MT5 not initialized")
        
        # Test terminal info
        terminal_info = get_terminal_info()
        if terminal_info:
            test_results['terminal_info'] = True
            if terminal_info['connected']:
                test_results['terminal_connected'] = True
            else:
                test_results['errors'].append("Terminal not connected")
        else:
            test_results['errors'].append("Cannot get terminal info")
        
        # Test account info
        account_info = get_account_info()
        if account_info:
            test_results['account_info'] = True
        else:
            test_results['errors'].append("Cannot get account info")
        
        # Test symbol access
        symbols = get_available_symbols()
        if symbols and len(symbols) > 0:
            test_results['symbol_access'] = True
            
            # Test market data with first symbol
            test_symbol = symbols[0]
            market_data = get_market_data(test_symbol, mt5.TIMEFRAME_M1, 10)
            if market_data:
                test_results['market_data'] = True
            else:
                test_results['errors'].append(f"Cannot get market data for {test_symbol}")
        else:
            test_results['errors'].append("No symbols available")
        
        # Test trading permissions
        if terminal_info and account_info:
            if (terminal_info.get('trade_allowed', False) and 
                account_info.get('trade_allowed', False)):
                test_results['trading_allowed'] = True
            else:
                test_results['errors'].append("Trading not allowed")
        
        # Overall status
        critical_tests = ['initialization', 'terminal_info', 'symbol_access', 'market_data']
        test_results['overall_status'] = all(test_results.get(test, False) for test in critical_tests)
        
        if test_results['overall_status']:
            logger.info("MT5 connection test passed")
        else:
            logger.warning(f"MT5 connection test failed: {test_results['errors']}")
        
    except Exception as e:
        test_results['errors'].append(f"Test exception: {e}")
        logger.error(f"Error during connection test: {e}")
    
    return test_results

def get_mt5_version() -> Optional[str]:
    """
    Get MT5 terminal version
    
    Returns:
        Version string or None if failed
    """
    try:
        terminal_info = get_terminal_info()
        if terminal_info:
            return f"Build {terminal_info['build']}"
        return None
        
    except Exception as e:
        logger.error(f"Error getting MT5 version: {e}")
        return None

def cleanup_resources():
    """
    Clean up MT5 resources and connections
    """
    try:
        # Cancel any pending operations
        logger.info("Cleaning up MT5 resources...")
        
        # Shutdown connection
        shutdown_mt5()
        
        logger.info("MT5 resources cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error cleaning up MT5 resources: {e}")

# Context manager for MT5 connection
class MT5Connection:
    """Context manager for MT5 connection"""
    
    def __init__(self, login: Optional[int] = None, password: Optional[str] = None, 
                 server: Optional[str] = None, path: Optional[str] = None):
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.connected = False
    
    def __enter__(self):
        self.connected = initialize_mt5(self.login, self.password, self.server, self.path)
        if not self.connected:
            raise ConnectionError("Failed to initialize MT5 connection")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connected:
            shutdown_mt5()
        
        if exc_type is not None:
            logger.error(f"Exception in MT5 context: {exc_type.__name__}: {exc_val}")
        
        return False  # Don't suppress exceptions

# Utility functions for common operations
def quick_price_check(symbol: str) -> Optional[float]:
    """
    Quick function to get current bid price
    
    Args:
        symbol: Symbol name
    
    Returns:
        Current bid price or None if failed
    """
    try:
        price_info = get_current_price(symbol)
        return price_info['bid'] if price_info else None
    except:
        return None

def quick_spread_check(symbol: str) -> Optional[float]:
    """
    Quick function to get current spread in pips
    
    Args:
        symbol: Symbol name
    
    Returns:
        Current spread in pips or None if failed
    """
    try:
        return get_spread(symbol)
    except:
        return None

def is_symbol_available(symbol: str) -> bool:
    """
    Quick check if symbol is available for trading
    
    Args:
        symbol: Symbol name
    
    Returns:
        bool: True if symbol is available
    """
    try:
        return select_symbol(symbol) and is_market_open(symbol)
    except:
        return False

# Export all public functions
__all__ = [
    'TIMEFRAMES', 'TIMEFRAME_NAMES',
    'initialize_mt5', 'shutdown_mt5', 'check_connection', 'reconnect',
    'get_account_info', 'get_terminal_info', 'get_connection_status',
    'get_available_symbols', 'get_symbol_info', 'select_symbol',
    'get_market_data', 'get_market_data_range', 'get_ticks', 'get_ticks_range',
    'get_current_price', 'get_spread', 'is_market_open', 'get_trading_hours',
    'get_positions', 'get_orders', 'get_history_deals', 'get_history_orders',
    'send_order', 'close_position', 'modify_position', 'cancel_order',
    'calculate_lot_size', 'get_pip_value', 'get_margin_required', 'get_free_margin',
    'check_margin_sufficient', 'validate_order_params',
    'get_server_time', 'convert_timeframe_to_minutes', 'get_timeframe_name',
    'test_connection', 'get_mt5_version', 'cleanup_resources',
    'MT5Connection', 'quick_price_check', 'quick_spread_check', 'is_symbol_available'
]

if __name__ == "__main__":
    # Test the MT5 initializer when run directly
    print("=== MT5 Initializer Test ===")
    
    # Test connection
    print("Testing MT5 connection...")
    test_results = test_connection()
    
    print("\nTest Results:")
    for test, result in test_results.items():
        if test != 'errors':
            status = "PASS" if result else "FAIL"
            print(f"  {test}: {status}")
    
    if test_results['errors']:
        print("\nErrors:")
        for error in test_results['errors']:
            print(f"  - {error}")
    
    # Test basic functionality if connected
    if test_results['overall_status']:
        print("\n=== Basic Functionality Test ===")
        
        # Test symbol access
        symbols = get_available_symbols("*EUR*")
        print(f"EUR symbols found: {len(symbols)}")
        
        if symbols:
            test_symbol = symbols[0]
            print(f"Testing with symbol: {test_symbol}")
            
            # Test current price
            price = quick_price_check(test_symbol)
            print(f"Current price: {price}")
            
            # Test spread
            spread = quick_spread_check(test_symbol)
            print(f"Current spread: {spread} pips")
            
            # Test market data
            data = get_market_data(test_symbol, mt5.TIMEFRAME_H1, 5)
            print(f"Market data bars: {len(data) if data else 0}")
        
        # Test account info
        account = get_account_info()
        if account:
            print(f"Account: {account['login']} on {account['server']}")
            print(f"Balance: {account['balance']} {account['currency']}")
    
    print("\n=== Test Complete ===")



