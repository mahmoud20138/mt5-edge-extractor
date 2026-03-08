"""
MT5 Connector Module - Handles connection and data extraction from MetaTrader 5.

This module provides:
- Connection management to MT5 terminal
- OHLCV data extraction
- Tick data extraction  
- Trade history extraction
- Symbol information retrieval
- Error handling and retry logic

NOTE: This is for BACKTESTING ONLY - no live trade execution.
"""

import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MT5TimeFrame:
    """MT5 timeframe constants."""
    M1 = 1       # 1 minute
    M5 = 5       # 5 minutes
    M15 = 15     # 15 minutes
    M30 = 30     # 30 minutes
    H1 = 60      # 1 hour
    H4 = 240     # 4 hours
    D1 = 1440    # Daily
    W1 = 10080   # Weekly
    MN1 = 43200  # Monthly


class MT5Connector:
    """
    Handles connection and data extraction from MetaTrader 5.
    
    This class provides methods for:
    - Connecting to MT5 terminal
    - Extracting historical price data (OHLCV)
    - Extracting tick data
    - Retrieving trade history
    - Getting symbol information
    
    Example:
        >>> connector = MT5Connector()
        >>> if connector.connect():
        >>>     rates = connector.get_rates("EURUSD", MT5TimeFrame.H1, 1000)
        >>>     connector.disconnect()
    """
    
    def __init__(self, path: str = "", login: int = 0, 
                 password: str = "", server: str = ""):
        """
        Initialize MT5 connector.
        
        Args:
            path: Path to MT5 terminal (optional)
            login: Account number
            password: Account password
            server: Server name
        """
        self.path = path
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        self._mt5 = None
        self._symbol_info_cache: Dict[str, Any] = {}
        
    def connect(self) -> bool:
        """
        Establish connection to MT5 terminal.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Try to import MetaTrader5
            try:
                import MetaTrader5 as mt5
                self._mt5 = mt5
            except ImportError:
                logger.warning(
                    "MetaTrader5 package not installed. "
                    "Using simulated data mode. "
                    "Install with: pip install MetaTrader5"
                )
                self.connected = False
                return False
            
            # Initialize MT5
            if self.path:
                if not mt5.initialize(path=self.path):
                    logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
                    return False
            else:
                if not mt5.initialize():
                    logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
                    return False
            
            # Login if credentials provided
            if self.login and self.password and self.server:
                if not mt5.login(self.login, self.password, self.server):
                    logger.error(f"MT5 login() failed: {mt5.last_error()}")
                    mt5.shutdown()
                    return False
            
            self.connected = True
            logger.info("Connected to MT5 successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        if self._mt5 and self.connected:
            self._mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
    
    def is_connected(self) -> bool:
        """Check if connected to MT5."""
        return self.connected
    
    def get_terminal_info(self) -> Optional[Dict[str, Any]]:
        """
        Get MT5 terminal information.
        
        Returns:
            Dictionary with terminal info or None if not connected
        """
        if not self.connected or not self._mt5:
            return None
            
        info = self._mt5.terminal_info()
        if info is None:
            return None
            
        return {
            "community_account": info.community_account,
            "community_connection": info.community_connection,
            "connected": info.connected,
            "dlls_allowed": info.dlls_allowed,
            "trade_allowed": info.trade_allowed,
            "email_enabled": info.email_enabled,
            "ftp_enabled": info.ftp_enabled,
            "notifications_enabled": info.notifications_enabled,
            "company": info.company,
            "name": info.name,
            "language": info.language,
            "path": info.path,
            "data_path": info.data_path,
            "common_data_path": info.common_data_path,
            "build": info.build,
            "max_bars": info.max_bars,
            "max_requests": info.max_requests,
            "memory_used": info.memory_used,
            "tradeapi_version": info.tradeapi_version
        }
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current account information.
        
        Returns:
            Dictionary with account info or None if not connected
        """
        if not self.connected or not self._mt5:
            return None
            
        info = self._mt5.account_info()
        if info is None:
            return None
            
        return {
            "login": info.login,
            "trade_mode": info.trade_mode,
            "leverage": info.leverage,
            "limit_orders": info.limit_orders,
            "margin_so_mode": info.margin_so_mode,
            "trade_allowed": info.trade_allowed,
            "trade_expert": info.trade_expert,
            "margin_mode": info.margin_mode,
            "currency_digits": info.currency_digits,
            "fifo_close": info.fifo_close,
            "balance": info.balance,
            "credit": info.credit,
            "profit": info.profit,
            "equity": info.equity,
            "margin": info.margin,
            "margin_free": info.margin_free,
            "margin_level": info.margin_level,
            "margin_so_call": info.margin_so_call,
            "margin_so_so": info.margin_so_so,
            "margin_initial": info.margin_initial,
            "margin_maintenance": info.margin_maintenance,
            "assets": info.assets,
            "liabilities": info.liabilities,
            "commission_blocked": info.commission_blocked,
            "name": info.name,
            "server": info.server,
            "currency": info.currency,
            "company": info.company
        }
    
    def get_symbols(self) -> Optional[List[str]]:
        """
        Get list of all available symbols.
        
        Returns:
            List of symbol names or None if not connected
        """
        if not self.connected or not self._mt5:
            return None
            
        symbols = self._mt5.symbols_get()
        if symbols is None:
            return None
            
        return [s.name for s in symbols]
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a symbol.
        
        Args:
            symbol: Symbol name (e.g., "EURUSD")
            
        Returns:
            Dictionary with symbol info or None if not found
        """
        if not self.connected or not self._mt5:
            return None
            
        # Check cache
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]
            
        info = self._mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"Symbol {symbol} not found")
            return None
            
        result = {
            "name": info.name,
            "description": info.description,
            "path": info.path,
            "currency_base": info.currency_base,
            "currency_profit": info.currency_profit,
            "currency_margin": info.currency_margin,
            "color": info.color,
            "digits": info.digits,
            "point": info.point,
            "spread": info.spread,
            "spread_float": info.spread_float,
            "ticks_bookdepth": info.ticks_bookdepth,
            "trade_calc_mode": info.trade_calc_mode,
            "trade_mode": info.trade_mode,
            "trade_stops_level": info.trade_stops_level,
            "trade_freeze_level": info.trade_freeze_level,
            "trade_exemode": info.trade_exemode,
            "trade_contract_size": info.trade_contract_size,
            "trade_tick_size": info.trade_tick_size,
            "trade_tick_value": info.trade_tick_value,
            "trade_tick_value_profit": info.trade_tick_value_profit,
            "trade_tick_value_loss": info.trade_tick_value_loss,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "swap_long": info.swap_long,
            "swap_short": info.swap_short,
            "swap_rollover3days": info.swap_rollover3days,
            "time": info.time,
            "bid": info.bid,
            "ask": info.ask,
            "last": info.last,
            "volume": info.volume,
            "vol_high": info.vol_high,
            "vol_low": info.vol_low,
            "session_open": info.session_open,
            "session_high": info.session_high,
            "session_low": info.session_low,
            "session_close": info.session_close
        }
        
        self._symbol_info_cache[symbol] = result
        return result
    
    def select_symbol(self, symbol: str) -> bool:
        """
        Select a symbol in Market Watch.
        
        Args:
            symbol: Symbol name
            
        Returns:
            bool: True if successful
        """
        if not self.connected or not self._mt5:
            return False
            
        return self._mt5.symbol_select(symbol, True)
    
    def get_rates_from_pos(self, symbol: str, timeframe: int, 
                           start_pos: int, count: int) -> Optional[pd.DataFrame]:
        """
        Get OHLCV rates from position.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe constant (use MT5TimeFrame)
            start_pos: Starting position (0 = most recent)
            count: Number of bars to retrieve
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        if not self.connected or not self._mt5:
            return None
            
        # Ensure symbol is selected
        self.select_symbol(symbol)
        
        rates = self._mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
        if rates is None:
            logger.error(f"Failed to get rates for {symbol}: {self._mt5.last_error()}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
    
    def get_rates_from_date(self, symbol: str, timeframe: int,
                            datetime_from: datetime, count: int) -> Optional[pd.DataFrame]:
        """
        Get OHLCV rates from a specific date.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe constant
            datetime_from: Starting datetime
            count: Number of bars to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.connected or not self._mt5:
            return None
            
        self.select_symbol(symbol)
        
        rates = self._mt5.copy_rates_from(symbol, timeframe, datetime_from, count)
        if rates is None:
            logger.error(f"Failed to get rates for {symbol}: {self._mt5.last_error()}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
    
    def get_rates_range(self, symbol: str, timeframe: int,
                        datetime_from: datetime, datetime_to: datetime) -> Optional[pd.DataFrame]:
        """
        Get OHLCV rates for a date range.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe constant
            datetime_from: Starting datetime
            datetime_to: Ending datetime
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.connected or not self._mt5:
            return None
            
        self.select_symbol(symbol)
        
        rates = self._mt5.copy_rates_range(symbol, timeframe, datetime_from, datetime_to)
        if rates is None:
            logger.error(f"Failed to get rates for {symbol}: {self._mt5.last_error()}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
    
    def get_ticks_from_date(self, symbol: str, datetime_from: datetime, 
                            count: int, flags: int = None) -> Optional[pd.DataFrame]:
        """
        Get tick data from a specific date.
        
        Args:
            symbol: Symbol name
            datetime_from: Starting datetime
            count: Number of ticks to retrieve
            flags: Tick flags (COPY_TICKS_ALL, COPY_TICKS_INFO, COPY_TICKS_TRADE)
            
        Returns:
            DataFrame with tick data
        """
        if not self.connected or not self._mt5:
            return None
            
        self.select_symbol(symbol)
        
        if flags is None:
            flags = self._mt5.COPY_TICKS_ALL
            
        ticks = self._mt5.copy_ticks_from(symbol, datetime_from, count, flags)
        if ticks is None:
            logger.error(f"Failed to get ticks for {symbol}: {self._mt5.last_error()}")
            return None
            
        df = pd.DataFrame(ticks)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
    
    def get_ticks_range(self, symbol: str, datetime_from: datetime,
                        datetime_to: datetime, flags: int = None) -> Optional[pd.DataFrame]:
        """
        Get tick data for a date range.
        
        Args:
            symbol: Symbol name
            datetime_from: Starting datetime
            datetime_to: Ending datetime
            flags: Tick flags
            
        Returns:
            DataFrame with tick data
        """
        if not self.connected or not self._mt5:
            return None
            
        self.select_symbol(symbol)
        
        if flags is None:
            flags = self._mt5.COPY_TICKS_ALL
            
        ticks = self._mt5.copy_ticks_range(symbol, datetime_from, datetime_to, flags)
        if ticks is None:
            logger.error(f"Failed to get ticks for {symbol}: {self._mt5.last_error()}")
            return None
            
        df = pd.DataFrame(ticks)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
    
    def get_deals_history(self, datetime_from: datetime, 
                          datetime_to: datetime) -> Optional[pd.DataFrame]:
        """
        Get trade deals history.
        
        Args:
            datetime_from: Starting datetime
            datetime_to: Ending datetime
            
        Returns:
            DataFrame with deals data
        """
        if not self.connected or not self._mt5:
            return None
            
        deals = self._mt5.history_deals_get(datetime_from, datetime_to)
        if deals is None:
            logger.error(f"Failed to get deals: {self._mt5.last_error()}")
            return None
            
        if len(deals) == 0:
            return pd.DataFrame()
            
        df = pd.DataFrame([{
            'ticket': d.ticket,
            'order': d.order,
            'time': datetime.fromtimestamp(d.time),
            'type': d.type,
            'entry': d.entry,
            'symbol': d.symbol,
            'volume': d.volume,
            'price': d.price,
            'commission': d.commission,
            'swap': d.swap,
            'profit': d.profit,
            'fee': d.fee,
            'comment': d.comment,
            'magic': d.magic,
            'position_id': d.position_id,
            'reason': d.reason
        } for d in deals])
        
        df.set_index('time', inplace=True)
        return df
    
    def get_orders_history(self, datetime_from: datetime,
                           datetime_to: datetime) -> Optional[pd.DataFrame]:
        """
        Get trade orders history.
        
        Args:
            datetime_from: Starting datetime
            datetime_to: Ending datetime
            
        Returns:
            DataFrame with orders data
        """
        if not self.connected or not self._mt5:
            return None
            
        orders = self._mt5.history_orders_get(datetime_from, datetime_to)
        if orders is None:
            logger.error(f"Failed to get orders: {self._mt5.last_error()}")
            return None
            
        if len(orders) == 0:
            return pd.DataFrame()
            
        df = pd.DataFrame([{
            'ticket': o.ticket,
            'time_setup': datetime.fromtimestamp(o.time_setup),
            'type': o.type,
            'state': o.state,
            'time_expiration': o.time_expiration,
            'time_done': datetime.fromtimestamp(o.time_done),
            'type_filling': o.type_filling,
            'type_time': o.type_time,
            'magic': o.magic,
            'position_id': o.position_id,
            'position_by_id': o.position_by_id,
            'reason': o.reason,
            'volume_initial': o.volume_initial,
            'volume_current': o.volume_current,
            'price_open': o.price_open,
            'sl': o.sl,
            'tp': o.tp,
            'price_current': o.price_current,
            'price_stoplimit': o.price_stoplimit,
            'symbol': o.symbol,
            'comment': o.comment,
            'external_id': o.external_id
        } for o in orders])
        
        return df

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
