"""
Feature Engineering Package - Technical indicators and derived features.

This package provides modules for:
- Momentum indicators (RSI, MACD, Stochastic, etc.)
- Trend indicators (MA, ADX, etc.)
- Volatility indicators (ATR, Bollinger, etc.)
- Volume indicators (OBV, Volume Profile, etc.)
- Candlestick pattern detection
"""

from .momentum import MomentumFeatures
from .trend import TrendFeatures
from .volatility_features import VolatilityFeatures
from .volume import VolumeFeatures
from .candle_patterns import CandlePatternFeatures

__all__ = [
    "MomentumFeatures",
    "TrendFeatures", 
    "VolatilityFeatures",
    "VolumeFeatures",
    "CandlePatternFeatures"
]
