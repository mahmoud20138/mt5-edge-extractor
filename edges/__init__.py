"""
Edge Detection Package - Identify potential trading edges.

This package provides modules for detecting various types of edges:
- Time-based edges (hour effects, day effects, seasonal patterns)
- Trend/momentum edges (crossovers, breakouts, momentum)
- Mean-reversion edges (z-score, oscillator extremes, gaps)
- Volatility edges (squeeze, regime detection)
- Candlestick pattern edges
- Market structure edges (support/resistance, breakouts)
- Pairs/correlation edges
- Machine learning edges
"""

from .time_based import TimeBasedEdges
from .trend_momentum import TrendMomentumEdges
from .mean_reversion import MeanReversionEdges
from .volatility_edges import VolatilityEdges
from .market_structure import MarketStructureEdges
from .pairs import PairsEdgeDetector, CorrelationEdges, CointegrationEdges
from .machine_learning import MLEdgeDetector, FeatureImportanceAnalyzer

__all__ = [
    "TimeBasedEdges",
    "TrendMomentumEdges",
    "MeanReversionEdges", 
    "VolatilityEdges",
    "MarketStructureEdges",
    "PairsEdgeDetector",
    "CorrelationEdges",
    "CointegrationEdges",
    "MLEdgeDetector",
    "FeatureImportanceAnalyzer"
]
