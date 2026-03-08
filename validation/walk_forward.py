"""
Walk-Forward Validation Module.

Provides walk-forward validation for edge testing:
- Rolling window validation
- Anchored validation
- Time-series cross-validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class WFOResult:
    """Container for walk-forward optimization results."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    parameters: Dict[str, Any]


class WalkForwardValidator:
    """
    Perform walk-forward validation.
    
    Walk-forward validation is the proper way to validate
    trading strategies by testing on data that was not
    used for optimization.
    """
    
    def __init__(self, train_size: int = 252, 
                 test_size: int = 63,
                 step_size: int = 21):
        """
        Initialize walk-forward validator.
        
        Args:
            train_size: Number of bars in training window
            test_size: Number of bars in test window
            step_size: Number of bars to step forward
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
    
    def generate_splits(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test splits.
        
        Args:
            df: Full dataset
            
        Returns:
            List of (train_df, test_df) tuples
        """
        n = len(df)
        splits = []
        
        start = 0
        while start + self.train_size + self.test_size <= n:
            train_end = start + self.train_size
            test_end = train_end + self.test_size
            
            train_df = df.iloc[start:train_end]
            test_df = df.iloc[train_end:test_end]
            
            splits.append((train_df, test_df))
            
            start += self.step_size
        
        return splits
    
    def validate_edge(self, df: pd.DataFrame,
                     edge_func: Callable,
                     metric_func: Callable) -> List[WFOResult]:
        """
        Validate an edge using walk-forward.
        
        Args:
            df: Full dataset
            edge_func: Function that finds edge in training data
            metric_func: Function that calculates metrics
            
        Returns:
            List of WFOResult for each fold
        """
        splits = self.generate_splits(df)
        results = []
        
        for train_df, test_df in splits:
            # Find edge in training data
            edge_result = edge_func(train_df)
            
            if edge_result is None:
                continue
            
            # Calculate metrics
            train_metrics = metric_func(train_df, edge_result)
            test_metrics = metric_func(test_df, edge_result)
            
            results.append(WFOResult(
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0],
                test_end=test_df.index[-1],
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                parameters=edge_result.details if hasattr(edge_result, 'details') else {}
            ))
        
        return results
    
    def calculate_efficiency(self, results: List[WFOResult],
                            metric: str = 'sharpe_ratio') -> float:
        """
        Calculate walk-forward efficiency.
        
        WFE = OOS_metric / IS_metric
        
        WFE > 0.5 indicates a robust edge.
        
        Args:
            results: List of WFOResult
            metric: Metric to compare
            
        Returns:
            Efficiency ratio
        """
        if not results:
            return 0.0
        
        is_values = [r.train_metrics.get(metric, 0) for r in results]
        oos_values = [r.test_metrics.get(metric, 0) for r in results]
        
        avg_is = np.mean(is_values)
        avg_oos = np.mean(oos_values)
        
        if avg_is == 0:
            return 0.0
        
        return avg_oos / avg_is
    
    def summary_statistics(self, results: List[WFOResult]) -> Dict[str, Any]:
        """
        Calculate summary statistics from walk-forward results.
        
        Args:
            results: List of WFOResult
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}
        
        # Aggregate metrics
        is_sharpes = [r.train_metrics.get('sharpe_ratio', 0) for r in results]
        oos_sharpes = [r.test_metrics.get('sharpe_ratio', 0) for r in results]
        is_returns = [r.train_metrics.get('mean_return', 0) for r in results]
        oos_returns = [r.test_metrics.get('mean_return', 0) for r in results]
        is_win_rates = [r.train_metrics.get('win_rate', 0) for r in results]
        oos_win_rates = [r.test_metrics.get('win_rate', 0) for r in results]
        
        return {
            'n_folds': len(results),
            'is_sharpe_mean': np.mean(is_sharpes),
            'is_sharpe_std': np.std(is_sharpes),
            'oos_sharpe_mean': np.mean(oos_sharpes),
            'oos_sharpe_std': np.std(oos_sharpes),
            'is_return_mean': np.mean(is_returns),
            'oos_return_mean': np.mean(oos_returns),
            'is_win_rate_mean': np.mean(is_win_rates),
            'oos_win_rate_mean': np.mean(oos_win_rates),
            'efficiency': self.calculate_efficiency(results, 'sharpe_ratio'),
            'sharpe_degradation': np.mean(is_sharpes) - np.mean(oos_sharpes) if np.mean(is_sharpes) != 0 else 0,
            'positive_folds': sum(1 for s in oos_sharpes if s > 0),
            'fold_consistency': sum(1 for s in oos_sharpes if s > 0) / len(results)
        }
    
    def check_overfitting(self, results: List[WFOResult],
                          threshold: float = 0.5) -> Dict[str, Any]:
        """
        Check for overfitting indicators.
        
        Args:
            results: Walk-forward results
            threshold: Efficiency threshold
            
        Returns:
            Dictionary with overfitting analysis
        """
        efficiency = self.calculate_efficiency(results, 'sharpe_ratio')
        
        # Calculate degradation
        is_sharpes = [r.train_metrics.get('sharpe_ratio', 0) for r in results]
        oos_sharpes = [r.test_metrics.get('sharpe_ratio', 0) for r in results]
        
        degradation = np.mean(is_sharpes) - np.mean(oos_sharpes)
        degradation_pct = degradation / np.mean(is_sharpes) if np.mean(is_sharpes) != 0 else 0
        
        # Check consistency
        positive_oos = sum(1 for s in oos_sharpes if s > 0)
        consistency = positive_oos / len(results) if results else 0
        
        return {
            'efficiency': efficiency,
            'degradation': degradation,
            'degradation_pct': degradation_pct,
            'consistency': consistency,
            'is_overfitted': efficiency < threshold,
            'warnings': self._generate_warnings(efficiency, degradation_pct, consistency)
        }
    
    def _generate_warnings(self, efficiency: float, 
                          degradation_pct: float,
                          consistency: float) -> List[str]:
        """Generate warning messages."""
        warnings = []
        
        if efficiency < 0.3:
            warnings.append("Very low efficiency - severe overfitting likely")
        elif efficiency < 0.5:
            warnings.append("Low efficiency - possible overfitting")
        
        if degradation_pct > 0.7:
            warnings.append("Large performance degradation in OOS")
        
        if consistency < 0.5:
            warnings.append("Edge inconsistent across folds")
        
        return warnings


class TimeSeriesSplit:
    """
    Time Series Cross-Validation.
    
    Scikit-learn compatible time series splitter.
    """
    
    def __init__(self, n_splits: int = 5, 
                 train_size: int = None,
                 test_size: int = None,
                 gap: int = 0):
        """
        Initialize splitter.
        
        Args:
            n_splits: Number of splits
            train_size: Training window size
            test_size: Test window size
            gap: Gap between train and test
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size or 1
        self.gap = gap
    
    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices.
        
        Args:
            X: Dataset
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        n_samples = len(X)
        
        if self.train_size is None:
            train_size = n_samples // (self.n_splits + 1)
        else:
            train_size = self.train_size
        
        indices = np.arange(n_samples)
        splits = []
        
        test_start = train_size + self.gap
        
        for i in range(self.n_splits):
            if test_start + self.test_size > n_samples:
                break
            
            train_idx = indices[:train_size + i * (n_samples - train_size - self.test_size - self.gap) // (self.n_splits)]
            test_idx = indices[test_start + i * self.test_size:test_start + (i + 1) * self.test_size]
            
            if len(test_idx) == 0:
                break
            
            splits.append((train_idx, test_idx))
        
        return splits


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation.
    
    Prevents data leakage by purging samples near the
    train/test boundary and adding embargo periods.
    """
    
    def __init__(self, n_splits: int = 5,
                 purge_size: int = 10,
                 embargo: int = 5):
        """
        Initialize purged K-fold.
        
        Args:
            n_splits: Number of folds
            purge_size: Number of samples to purge
            embargo: Embargo period between folds
        """
        self.n_splits = n_splits
        self.purge_size = purge_size
        self.embargo = embargo
    
    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged train/test indices.
        
        Args:
            X: Dataset
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits
        
        splits = []
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            test_idx = indices[test_start:test_end]
            
            # Purge samples near test set
            train_idx = np.concatenate([
                indices[:max(0, test_start - self.purge_size)],
                indices[min(n_samples, test_end + self.embargo):]
            ])
            
            splits.append((train_idx, test_idx))
        
        return splits
