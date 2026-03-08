"""
MT5 Trading Edge Extractor - Backtesting System
================================================

A comprehensive system for extracting and validating trading edges from 
MetaTrader 5 historical data. This system is designed for BACKTESTING ONLY
and does NOT execute live trades.

Key Features:
- MT5 data extraction (OHLCV, ticks, trade history)
- Feature engineering (100+ technical indicators)
- Multiple edge detection categories
- Statistical validation with walk-forward testing
- Transaction cost analysis
- Comprehensive reporting and visualization

Author: Edge Extraction System
Version: 1.0.0
"""

from .config import Config
from .data.mt5_connector import MT5Connector
from .data.data_loader import DataLoader
from .data.preprocessing import DataPreprocessor
from .engine.backtest_engine import BacktestEngine

__version__ = "1.0.0"
__all__ = [
    "Config",
    "MT5Connector", 
    "DataLoader",
    "DataPreprocessor",
    "BacktestEngine"
]
