"""
Configuration module for MT5 Edge Extractor.

Contains all configurable parameters for:
- MT5 connection settings
- Data extraction parameters
- Feature engineering settings
- Edge detection thresholds
- Statistical testing parameters
- Transaction cost assumptions
- Reporting options
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum


class TimeFrame(Enum):
    """MT5 Timeframe enumeration."""
    M1 = 1       # 1 minute
    M5 = 5       # 5 minutes
    M15 = 15     # 15 minutes
    M30 = 30     # 30 minutes
    H1 = 60      # 1 hour
    H4 = 240     # 4 hours
    D1 = 1440    # Daily
    W1 = 10080   # Weekly
    MN1 = 43200  # Monthly


class Session(Enum):
    """Trading sessions (UTC times)."""
    ASIAN = ("Asian", 0, 8)
    LONDON = ("London", 7, 16)
    NEW_YORK = ("New York", 12, 21)
    LONDON_NY_OVERLAP = ("London-NY Overlap", 12, 16)
    ASIAN_LONDON_OVERLAP = ("Asian-London Overlap", 7, 8)


@dataclass
class MT5Settings:
    """MT5 connection settings."""
    path: str = ""
    login: int = 0
    password: str = ""
    server: str = ""
    timeout: int = 60000  # 60 seconds
    portable: bool = False


@dataclass
class DataSettings:
    """Data extraction settings."""
    default_timeframe: TimeFrame = TimeFrame.H1
    default_symbol: str = "EURUSD"
    lookback_years: int = 5
    include_ticks: bool = False
    include_trade_history: bool = True
    

@dataclass
class FeatureSettings:
    """Feature engineering settings."""
    # Returns-based
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 60])
    
    # Volatility-based
    volatility_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    atr_period: int = 14
    
    # Momentum-based
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Trend-based
    sma_periods: List[int] = field(default_factory=lambda: [20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26, 50, 200])
    adx_period: int = 14
    
    # Volume-based
    volume_sma_periods: List[int] = field(default_factory=lambda: [10, 20, 50])


@dataclass
class EdgeSettings:
    """Edge detection thresholds."""
    # Statistical significance
    min_p_value: float = 0.05
    min_sample_size: int = 100
    
    # Z-score thresholds for mean reversion
    zscore_thresholds: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5])
    
    # RSI thresholds
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    
    # Breakout lookback periods
    breakout_periods: List[int] = field(default_factory=lambda: [10, 20, 50])
    
    # Consecutive candles for reversal
    min_consecutive_candles: int = 3
    max_consecutive_candles: int = 7
    
    # Hurst exponent thresholds
    hurst_trending: float = 0.55
    hurst_mean_reverting: float = 0.45


@dataclass
class ValidationSettings:
    """Validation and testing settings."""
    # Data splits
    in_sample_pct: float = 0.60
    validation_pct: float = 0.20
    test_pct: float = 0.20
    
    # Walk-forward optimization
    wfo_training_years: int = 2
    wfo_test_months: int = 6
    wfo_step_months: int = 3
    
    # Bootstrap
    bootstrap_samples: int = 10000
    
    # Monte Carlo
    monte_carlo_runs: int = 1000


@dataclass
class TransactionCostSettings:
    """Transaction cost assumptions."""
    # Spread (in points/pips)
    default_spread: float = 1.5
    spread_multiplier_during_news: float = 3.0
    
    # Commission (per lot, per side)
    commission_per_lot: float = 7.0  # USD
    
    # Slippage (in points/pips)
    default_slippage: float = 0.5
    slippage_during_news: float = 5.0
    
    # Swap (overnight financing)
    swap_long: float = 0.0
    swap_short: float = 0.0
    
    # Point value
    point_value: float = 10.0  # USD per pip per standard lot


@dataclass
class ReportingSettings:
    """Reporting and visualization settings."""
    output_directory: str = "./reports"
    generate_plots: bool = True
    generate_excel: bool = True
    generate_html: bool = True
    plot_style: str = "seaborn-v0_8-darkgrid"
    figure_size: tuple = (12, 6)
    dpi: int = 150


@dataclass
class Config:
    """Main configuration class."""
    mt5: MT5Settings = field(default_factory=MT5Settings)
    data: DataSettings = field(default_factory=DataSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    edges: EdgeSettings = field(default_factory=EdgeSettings)
    validation: ValidationSettings = field(default_factory=ValidationSettings)
    costs: TransactionCostSettings = field(default_factory=TransactionCostSettings)
    reporting: ReportingSettings = field(default_factory=ReportingSettings)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "Config":
        """Create configuration from dictionary."""
        config = cls()
        
        if "mt5" in config_dict:
            config.mt5 = MT5Settings(**config_dict["mt5"])
        if "data" in config_dict:
            config.data = DataSettings(**config_dict["data"])
        if "features" in config_dict:
            config.features = FeatureSettings(**config_dict["features"])
        if "edges" in config_dict:
            config.edges = EdgeSettings(**config_dict["edges"])
        if "validation" in config_dict:
            config.validation = ValidationSettings(**config_dict["validation"])
        if "costs" in config_dict:
            config.costs = TransactionCostSettings(**config_dict["costs"])
        if "reporting" in config_dict:
            config.reporting = ReportingSettings(**config_dict["reporting"])
            
        return config
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "mt5": self.mt5.__dict__,
            "data": self.data.__dict__,
            "features": self.features.__dict__,
            "edges": self.edges.__dict__,
            "validation": self.validation.__dict__,
            "costs": self.costs.__dict__,
            "reporting": self.reporting.__dict__
        }


# Default configuration instance
DEFAULT_CONFIG = Config()
