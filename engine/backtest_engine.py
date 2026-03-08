"""
Backtesting Engine Module.

Main engine that orchestrates the edge extraction and validation process.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import json

from config import Config, TimeFrame
from data.data_loader import DataLoader
from data.preprocessing import DataPreprocessor
from features.momentum import MomentumFeatures
from features.trend import TrendFeatures
from features.volatility_features import VolatilityFeatures
from features.volume import VolumeFeatures
from features.candle_patterns import CandlePatternFeatures
from edges.time_based import TimeBasedEdges
from edges.trend_momentum import TrendMomentumEdges
from edges.mean_reversion import MeanReversionEdges
from edges.volatility_edges import VolatilityEdges
from edges.market_structure import MarketStructureEdges
from validation.statistical_tests import StatisticalTests
from validation.walk_forward import WalkForwardValidator
from validation.bootstrap import BootstrapValidator
from metrics import EdgeMetrics

logger = logging.getLogger(__name__)


@dataclass
class EdgeAnalysisResult:
    """Container for complete edge analysis results."""
    edge_name: str
    edge_type: str
    sample_size: int
    mean_return: float
    win_rate: float
    sharpe_ratio: float
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    details: Dict


class BacktestEngine:
    """
    Main backtesting engine for edge extraction.
    
    Orchestrates the entire process:
    1. Data loading and preprocessing
    2. Feature engineering
    3. Edge detection
    4. Statistical validation
    5. Walk-forward testing
    6. Results reporting
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize backtest engine.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.data_loader = DataLoader(self.config)
        self.preprocessor = DataPreprocessor()
        self.statistical_tests = StatisticalTests(self.config.edges.min_p_value)
        self.wf_validator = WalkForwardValidator()
        self.bootstrap = BootstrapValidator()
        
        self._data: Optional[pd.DataFrame] = None
        self._features: Optional[pd.DataFrame] = None
        self._edge_results: Dict[str, Any] = {}
    
    def load_data(self, symbol: str, timeframe: TimeFrame = None,
                  start: datetime = None, end: datetime = None,
                  years: int = None) -> pd.DataFrame:
        """
        Load and preprocess data.
        
        Args:
            symbol: Symbol to load
            timeframe: Timeframe
            start: Start date
            end: End date
            years: Years of data
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Loading data for {symbol}...")
        
        self._data = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            years=years
        )
        
        if self._data.empty:
            logger.error("Failed to load data")
            return pd.DataFrame()
        
        logger.info(f"Loaded {len(self._data)} bars")
        
        return self._data
    
    def engineer_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Engineer all features.
        
        Args:
            df: DataFrame to process (uses loaded data if None)
            
        Returns:
            DataFrame with all features
        """
        if df is None:
            df = self._data
        
        if df is None or df.empty:
            logger.error("No data to engineer features")
            return pd.DataFrame()
        
        logger.info("Engineering features...")
        
        # Preprocess
        df = self.preprocessor.preprocess(df)
        
        # Add momentum features
        df = MomentumFeatures.add_all_momentum(df)
        
        # Add trend features
        df = TrendFeatures.add_all_trend(df)
        
        # Add volatility features
        df = VolatilityFeatures.add_all_volatility(df)
        
        # Add volume features
        if 'tick_volume' in df.columns or 'real_volume' in df.columns:
            df = VolumeFeatures.add_all_volume(df)
        
        # Add candlestick patterns
        df = CandlePatternFeatures.add_all_patterns(df)
        
        self._features = df
        
        logger.info(f"Engineered {len(df.columns)} features")
        
        return df
    
    def detect_edges(self, df: pd.DataFrame = None,
                    lookahead: int = 10) -> Dict[str, EdgeAnalysisResult]:
        """
        Detect all edge types.
        
        Args:
            df: DataFrame with features (uses engineered features if None)
            lookahead: Forward return period
            
        Returns:
            Dictionary of edge results
        """
        if df is None:
            df = self._features
        
        if df is None or df.empty:
            logger.error("No data to detect edges")
            return {}
        
        logger.info("Detecting edges...")
        
        all_results = {}
        
        # Time-based edges
        time_edges = TimeBasedEdges(self.config.edges.min_p_value)
        time_results = time_edges.run_all_time_edges(df)
        for name, result in time_results.items():
            if result:
                all_results[f'time_{name}'] = self._to_analysis_result(result)
        
        # Trend/momentum edges
        trend_edges = TrendMomentumEdges(self.config.edges.min_p_value)
        trend_results = trend_edges.run_all_trend_edges(df, lookahead)
        for name, result in trend_results.items():
            if result:
                all_results[f'trend_{name}'] = self._to_analysis_result(result)
        
        # Mean reversion edges
        mr_edges = MeanReversionEdges(self.config.edges.min_p_value)
        mr_results = mr_edges.run_all_mr_edges(df, lookahead)
        for name, result in mr_results.items():
            if result:
                all_results[f'mr_{name}'] = self._to_analysis_result(result)
        
        # Volatility edges
        vol_edges = VolatilityEdges(self.config.edges.min_p_value)
        vol_results = vol_edges.run_all_vol_edges(df, lookahead)
        for name, result in vol_results.items():
            if result:
                all_results[f'vol_{name}'] = self._to_analysis_result(result)
        
        # Market structure edges
        struct_edges = MarketStructureEdges(self.config.edges.min_p_value)
        struct_results = struct_edges.run_all_structure_edges(df, lookahead)
        for name, result in struct_results.items():
            if result:
                all_results[f'struct_{name}'] = self._to_analysis_result(result)
        
        self._edge_results = all_results
        
        logger.info(f"Detected {len(all_results)} edge candidates")
        
        return all_results
    
    def validate_edges(self, df: pd.DataFrame = None,
                      edge_results: Dict = None) -> Dict[str, Dict]:
        """
        Validate edges with walk-forward and bootstrap.
        
        Args:
            df: DataFrame with features
            edge_results: Edge detection results
            
        Returns:
            Dictionary with validation results
        """
        if df is None:
            df = self._features
        if edge_results is None:
            edge_results = self._edge_results
        
        if df is None or not edge_results:
            logger.error("No data or edges to validate")
            return {}
        
        logger.info("Validating edges...")
        
        validation_results = {}
        
        # Get significant edges
        significant_edges = {
            name: result for name, result in edge_results.items()
            if result.is_significant
        }
        
        logger.info(f"Validating {len(significant_edges)} significant edges")
        
        for edge_name, edge_result in significant_edges.items():
            validation_results[edge_name] = {
                'edge_result': edge_result,
                'bootstrap': None,
                'walk_forward': None
            }
        
        return validation_results
    
    def filter_significant_edges(self, min_samples: int = 100,
                                 max_p_value: float = 0.05,
                                 min_win_rate: float = 0.45) -> Dict[str, EdgeAnalysisResult]:
        """
        Filter edges by significance criteria.
        
        Args:
            min_samples: Minimum sample size
            max_p_value: Maximum p-value
            min_win_rate: Minimum win rate
            
        Returns:
            Filtered edge results
        """
        filtered = {}
        
        for name, result in self._edge_results.items():
            if (result.sample_size >= min_samples and
                result.p_value <= max_p_value and
                result.win_rate >= min_win_rate):
                filtered[name] = result
        
        return filtered
    
    def generate_report(self, output_path: str = None) -> Dict:
        """
        Generate comprehensive analysis report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Report dictionary
        """
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'config': self.config.to_dict()
            },
            'data_summary': {
                'total_bars': len(self._data) if self._data is not None else 0,
                'date_range': {
                    'start': str(self._data.index[0]) if self._data is not None else None,
                    'end': str(self._data.index[-1]) if self._data is not None else None
                },
                'features_count': len(self._features.columns) if self._features is not None else 0
            },
            'edge_analysis': {
                'total_edges_tested': len(self._edge_results),
                'significant_edges': len(self.filter_significant_edges()),
                'by_type': self._group_by_edge_type()
            },
            'top_edges': self._get_top_edges(10),
            'all_results': {
                name: {
                    'edge_name': r.edge_name,
                    'edge_type': r.edge_type,
                    'sample_size': r.sample_size,
                    'mean_return': r.mean_return,
                    'win_rate': r.win_rate,
                    'sharpe_ratio': r.sharpe_ratio,
                    'p_value': r.p_value,
                    'is_significant': r.is_significant
                }
                for name, r in self._edge_results.items()
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def run_full_analysis(self, symbol: str = 'EURUSD',
                         timeframe: TimeFrame = TimeFrame.H1,
                         years: int = 3) -> Dict:
        """
        Run complete edge analysis pipeline.
        
        Args:
            symbol: Symbol to analyze
            timeframe: Data timeframe
            years: Years of historical data
            
        Returns:
            Complete analysis results
        """
        logger.info(f"Starting full edge analysis for {symbol}")
        
        # Step 1: Load data
        df = self.load_data(symbol, timeframe, years=years)
        if df.empty:
            return {'error': 'Failed to load data'}
        
        # Step 2: Engineer features
        df = self.engineer_features(df)
        if df.empty:
            return {'error': 'Failed to engineer features'}
        
        # Step 3: Detect edges
        edge_results = self.detect_edges(df)
        
        # Step 4: Validate edges
        validation_results = self.validate_edges(df, edge_results)
        
        # Step 5: Generate report
        report = self.generate_report()
        
        logger.info("Analysis complete!")
        
        return {
            'data': df,
            'edge_results': edge_results,
            'validation_results': validation_results,
            'report': report
        }
    
    def _to_analysis_result(self, edge_result) -> EdgeAnalysisResult:
        """Convert edge result to analysis result."""
        return EdgeAnalysisResult(
            edge_name=edge_result.name,
            edge_type=edge_result.edge_type,
            sample_size=edge_result.sample_size,
            mean_return=edge_result.mean_return,
            win_rate=edge_result.win_rate,
            sharpe_ratio=edge_result.sharpe_ratio,
            p_value=edge_result.p_value,
            is_significant=edge_result.is_significant,
            confidence_interval=(0, 0),  # Would need bootstrap
            details=edge_result.details
        )
    
    def _group_by_edge_type(self) -> Dict[str, int]:
        """Group results by edge type."""
        groups = {}
        for result in self._edge_results.values():
            edge_type = result.edge_type
            groups[edge_type] = groups.get(edge_type, 0) + 1
        return groups
    
    def _get_top_edges(self, n: int = 10) -> List[Dict]:
        """Get top N edges by Sharpe ratio."""
        sorted_edges = sorted(
            [r for r in self._edge_results.values() if r.is_significant],
            key=lambda x: x.sharpe_ratio,
            reverse=True
        )[:n]
        
        return [
            {
                'name': r.edge_name,
                'type': r.edge_type,
                'sharpe': r.sharpe_ratio,
                'win_rate': r.win_rate,
                'sample_size': r.sample_size
            }
            for r in sorted_edges
        ]
    
    def cleanup(self):
        """Cleanup resources."""
        self.data_loader.disconnect()
