"""
Data Loader Module - Handles loading and caching of market data.

This module provides:
- Data loading from MT5
- Data loading from CSV files
- Simulated data generation for testing
- Data caching for performance
- Multi-timeframe data handling
"""

import logging
from typing import Optional, Dict, List, Union
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from config import Config, TimeFrame
from data.mt5_connector import MT5Connector, MT5TimeFrame

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def get_rates(self, symbol: str, timeframe: TimeFrame, 
                  start: datetime, end: datetime) -> pd.DataFrame:
        """Get OHLCV rates for the specified parameters."""
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if data source is available."""
        pass


class MT5DataSource(DataSource):
    """Data source using MetaTrader 5."""
    
    def __init__(self, connector: MT5Connector):
        self.connector = connector
        self._tf_map = {
            TimeFrame.M1: MT5TimeFrame.M1,
            TimeFrame.M5: MT5TimeFrame.M5,
            TimeFrame.M15: MT5TimeFrame.M15,
            TimeFrame.M30: MT5TimeFrame.M30,
            TimeFrame.H1: MT5TimeFrame.H1,
            TimeFrame.H4: MT5TimeFrame.H4,
            TimeFrame.D1: MT5TimeFrame.D1,
            TimeFrame.W1: MT5TimeFrame.W1,
            TimeFrame.MN1: MT5TimeFrame.MN1,
        }
    
    def get_rates(self, symbol: str, timeframe: TimeFrame,
                  start: datetime, end: datetime) -> pd.DataFrame:
        """Get rates from MT5."""
        mt5_tf = self._tf_map.get(timeframe, MT5TimeFrame.H1)
        return self.connector.get_rates_range(symbol, mt5_tf, start, end)
    
    def get_available_symbols(self) -> List[str]:
        """Get available symbols from MT5."""
        symbols = self.connector.get_symbols()
        return symbols if symbols else []
    
    def is_available(self) -> bool:
        """Check if MT5 connection is available."""
        return self.connector.is_connected()


class CSVDataSource(DataSource):
    """Data source using CSV files."""
    
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def get_rates(self, symbol: str, timeframe: TimeFrame,
                  start: datetime, end: datetime) -> pd.DataFrame:
        """Get rates from CSV file."""
        # Expected file naming: SYMBOL_TIMEFRAME.csv (e.g., EURUSD_H1.csv)
        filename = f"{symbol}_{timeframe.name}.csv"
        filepath = self.data_directory / filename
        
        if not filepath.exists():
            logger.error(f"CSV file not found: {filepath}")
            return pd.DataFrame()
        
        # Load and cache
        cache_key = f"{symbol}_{timeframe.name}"
        if cache_key not in self._cache:
            df = pd.read_csv(filepath, parse_dates=['time'], index_col='time')
            self._cache[cache_key] = df
        
        df = self._cache[cache_key].copy()
        
        # Filter by date range
        mask = (df.index >= start) & (df.index <= end)
        return df[mask]
    
    def get_available_symbols(self) -> List[str]:
        """Get available symbols from CSV files."""
        symbols = set()
        for f in self.data_directory.glob("*.csv"):
            # Extract symbol from filename (SYMBOL_TIMEFRAME.csv)
            parts = f.stem.split('_')
            if len(parts) >= 2:
                symbols.add(parts[0])
        return list(symbols)
    
    def is_available(self) -> bool:
        """Check if data directory exists."""
        return self.data_directory.exists()


class SimulatedDataSource(DataSource):
    """
    Simulated data source for testing and development.
    
    Generates realistic OHLCV data using Geometric Brownian Motion
    with configurable volatility and drift parameters.
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def get_rates(self, symbol: str, timeframe: TimeFrame,
                  start: datetime, end: datetime) -> pd.DataFrame:
        """Generate simulated rates."""
        cache_key = f"{symbol}_{timeframe.name}_{start.date()}_{end.date()}"
        
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # Calculate number of bars based on timeframe
        tf_minutes = timeframe.value
        total_minutes = (end - start).total_seconds() / 60
        num_bars = int(total_minutes / tf_minutes)
        
        # Parameters for simulation
        # Different parameters for different symbols to simulate currency pairs
        symbol_hash = hash(symbol) % 100
        base_price = 1.0 + (symbol_hash / 100)  # Price between 1.0 and 2.0
        annual_drift = (symbol_hash - 50) / 10000  # Small drift
        annual_vol = 0.1 + (symbol_hash / 500)  # Volatility between 10-30%
        
        # Generate timestamps
        timestamps = pd.date_range(start=start, periods=num_bars, 
                                   freq=f'{tf_minutes}min')
        
        # Skip weekends for forex data
        timestamps = timestamps[timestamps.dayofweek < 5]
        num_bars = len(timestamps)
        
        # Generate prices using GBM
        dt = tf_minutes / (252 * 24 * 60)  # Time step in years
        drift = (annual_drift - 0.5 * annual_vol**2) * dt
        diffusion = annual_vol * np.sqrt(dt)
        
        # Generate returns
        returns = np.random.normal(drift, diffusion, num_bars)
        
        # Generate log prices
        log_prices = np.log(base_price) + np.cumsum(returns)
        close_prices = np.exp(log_prices)
        
        # Generate OHLC with realistic relationships
        bar_vol = annual_vol * np.sqrt(dt)
        
        opens = close_prices * np.exp(np.random.normal(0, bar_vol * 0.3, num_bars) - bar_vol * 0.045)
        highs = np.maximum(opens, close_prices) * (1 + np.abs(np.random.normal(0, bar_vol * 0.5, num_bars)))
        lows = np.minimum(opens, close_prices) * (1 - np.abs(np.random.normal(0, bar_vol * 0.5, num_bars)))
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        for i in range(num_bars):
            highs[i] = max(highs[i], opens[i], close_prices[i])
            lows[i] = min(lows[i], opens[i], close_prices[i])
        
        # Generate tick volumes (correlated with volatility)
        ranges = (highs - lows) / lows
        base_volume = 1000 + symbol_hash * 10
        tick_volumes = (base_volume * (1 + ranges * 50) * 
                       np.random.uniform(0.5, 1.5, num_bars)).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': close_prices,
            'tick_volume': tick_volumes,
            'real_volume': tick_volumes * 10,
            'spread': np.random.uniform(1, 3, num_bars)
        }, index=timestamps)
        
        df.index.name = 'time'
        
        # Cache result
        self._cache[cache_key] = df.copy()
        
        return df
    
    def get_available_symbols(self) -> List[str]:
        """Return list of simulated symbols."""
        return ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 
                'USDCHF', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY']
    
    def is_available(self) -> bool:
        """Simulated data is always available."""
        return True


class DataLoader:
    """
    Main data loader class that manages multiple data sources.
    
    Provides a unified interface for loading market data from:
    - MetaTrader 5 (primary)
    - CSV files (backup)
    - Simulated data (testing)
    
    Includes caching and validation features.
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize data loader.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self._mt5_connector: Optional[MT5Connector] = None
        self._mt5_source: Optional[MT5DataSource] = None
        self._csv_source: Optional[CSVDataSource] = None
        self._simulated_source: Optional[SimulatedDataSource] = None
        
        # Cache for loaded data
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
        # Initialize sources
        self._initialize_sources()
    
    def _initialize_sources(self):
        """Initialize data sources."""
        # Try MT5 first
        try:
            self._mt5_connector = MT5Connector(
                path=self.config.mt5.path,
                login=self.config.mt5.login,
                password=self.config.mt5.password,
                server=self.config.mt5.server
            )
            if self._mt5_connector.connect():
                self._mt5_source = MT5DataSource(self._mt5_connector)
                logger.info("MT5 data source initialized")
        except Exception as e:
            logger.warning(f"MT5 initialization failed: {e}")
        
        # Initialize simulated source as fallback
        self._simulated_source = SimulatedDataSource()
    
    def set_csv_source(self, directory: str):
        """Set CSV data source directory."""
        self._csv_source = CSVDataSource(directory)
    
    def load_data(self, symbol: str, timeframe: TimeFrame = None,
                  start: datetime = None, end: datetime = None,
                  years: int = None, use_cache: bool = True) -> pd.DataFrame:
        """
        Load market data for a symbol.
        
        Args:
            symbol: Symbol name (e.g., 'EURUSD')
            timeframe: Timeframe (default from config)
            start: Start datetime
            end: End datetime
            years: Number of years to load (alternative to start/end)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data
        """
        # Set defaults
        timeframe = timeframe or TimeFrame(self.config.data.default_timeframe.value)
        
        if end is None:
            end = datetime.now()
        if start is None:
            if years is not None:
                start = end - timedelta(days=years * 365)
            else:
                start = end - timedelta(days=self.config.data.lookback_years * 365)
        
        # Check cache
        cache_key = f"{symbol}_{timeframe.name}_{start.date()}_{end.date()}"
        if use_cache and cache_key in self._data_cache:
            logger.info(f"Using cached data for {symbol}")
            return self._data_cache[cache_key].copy()
        
        # Try data sources in order
        df = pd.DataFrame()
        
        # Try MT5 first
        if self._mt5_source and self._mt5_source.is_available():
            logger.info(f"Loading {symbol} from MT5...")
            df = self._mt5_source.get_rates(symbol, timeframe, start, end)
            if df is None:
                df = pd.DataFrame()
        
        # Fall back to CSV
        if (df is None or df.empty) and self._csv_source and self._csv_source.is_available():
            logger.info(f"Loading {symbol} from CSV...")
            df = self._csv_source.get_rates(symbol, timeframe, start, end)
            if df is None:
                df = pd.DataFrame()
        
        # Fall back to simulated
        if df is None or df.empty:
            logger.info(f"Using simulated data for {symbol}...")
            df = self._simulated_source.get_rates(symbol, timeframe, start, end)
        
        # Cache result
        if df is not None and not df.empty and use_cache:
            self._data_cache[cache_key] = df.copy()
        
        return df
    
    def load_multi_timeframe(self, symbol: str, timeframes: List[TimeFrame],
                            start: datetime = None, end: datetime = None,
                            years: int = None) -> Dict[TimeFrame, pd.DataFrame]:
        """
        Load data for multiple timeframes.
        
        Args:
            symbol: Symbol name
            timeframes: List of timeframes
            start: Start datetime
            end: End datetime
            years: Number of years
            
        Returns:
            Dictionary mapping timeframes to DataFrames
        """
        result = {}
        for tf in timeframes:
            result[tf] = self.load_data(symbol, tf, start, end, years)
        return result
    
    def load_multiple_symbols(self, symbols: List[str], 
                             timeframe: TimeFrame = None,
                             start: datetime = None, end: datetime = None,
                             years: int = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.
        
        Args:
            symbols: List of symbol names
            timeframe: Timeframe
            start: Start datetime
            end: End datetime
            years: Number of years
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        result = {}
        for symbol in symbols:
            result[symbol] = self.load_data(symbol, timeframe, start, end, years)
        return result
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from all sources."""
        symbols = []
        
        if self._mt5_source and self._mt5_source.is_available():
            symbols.extend(self._mt5_source.get_available_symbols())
        
        if self._csv_source and self._csv_source.is_available():
            symbols.extend(self._csv_source.get_available_symbols())
        
        if self._simulated_source:
            symbols.extend(self._simulated_source.get_available_symbols())
        
        return list(set(symbols))  # Remove duplicates
    
    def clear_cache(self):
        """Clear all cached data."""
        self._data_cache.clear()
    
    def disconnect(self):
        """Disconnect from MT5."""
        if self._mt5_connector:
            self._mt5_connector.disconnect()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.disconnect()
