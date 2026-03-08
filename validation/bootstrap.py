"""
Bootstrap Validation Module.

Provides bootstrap methods for edge validation:
- Bootstrap confidence intervals
- Monte Carlo simulation
- Permutation tests
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from scipy import stats
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    """Container for bootstrap results."""
    statistic_name: str
    observed_value: float
    bootstrap_mean: float
    bootstrap_std: float
    confidence_interval: Tuple[float, float]
    p_value: float
    n_bootstrap: int


class BootstrapValidator:
    """
    Perform bootstrap validation.
    
    Bootstrap methods provide robust confidence intervals
    and hypothesis testing without distributional assumptions.
    """
    
    def __init__(self, n_bootstrap: int = 10000, 
                 confidence_level: float = 0.95,
                 random_state: int = 42):
        """
        Initialize bootstrap validator.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        np.random.seed(random_state)
    
    def bootstrap_mean(self, data: pd.Series) -> BootstrapResult:
        """
        Bootstrap confidence interval for mean.
        
        Args:
            data: Sample data
            
        Returns:
            BootstrapResult with confidence interval
        """
        data = data.dropna().values
        n = len(data)
        
        if n < 10:
            return BootstrapResult(
                statistic_name="mean",
                observed_value=np.mean(data) if n > 0 else 0,
                bootstrap_mean=0,
                bootstrap_std=0,
                confidence_interval=(0, 0),
                p_value=1.0,
                n_bootstrap=0
            )
        
        # Bootstrap samples
        bootstrap_means = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means[i] = np.mean(sample)
        
        # Observed value
        observed_mean = np.mean(data)
        
        # Confidence interval (percentile method)
        alpha = 1 - self.confidence_level
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        # P-value (proportion of bootstrap means more extreme than 0)
        p_value = np.mean(bootstrap_means <= 0) if observed_mean > 0 else np.mean(bootstrap_means >= 0)
        p_value = 2 * min(p_value, 1 - p_value)  # Two-tailed
        
        return BootstrapResult(
            statistic_name="mean",
            observed_value=observed_mean,
            bootstrap_mean=np.mean(bootstrap_means),
            bootstrap_std=np.std(bootstrap_means),
            confidence_interval=(lower, upper),
            p_value=p_value,
            n_bootstrap=self.n_bootstrap
        )
    
    def bootstrap_sharpe(self, returns: pd.Series) -> BootstrapResult:
        """
        Bootstrap confidence interval for Sharpe ratio.
        
        Args:
            returns: Return series
            
        Returns:
            BootstrapResult with Sharpe confidence interval
        """
        returns = returns.dropna().values
        n = len(returns)
        
        if n < 30:
            return BootstrapResult(
                statistic_name="sharpe_ratio",
                observed_value=0,
                bootstrap_mean=0,
                bootstrap_std=0,
                confidence_interval=(0, 0),
                p_value=1.0,
                n_bootstrap=0
            )
        
        def calc_sharpe(r):
            return np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        
        # Observed Sharpe
        observed_sharpe = calc_sharpe(returns)
        
        # Bootstrap
        bootstrap_sharpes = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            sample = np.random.choice(returns, size=n, replace=True)
            bootstrap_sharpes[i] = calc_sharpe(sample)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        lower = np.percentile(bootstrap_sharpes, alpha/2 * 100)
        upper = np.percentile(bootstrap_sharpes, (1 - alpha/2) * 100)
        
        # P-value
        p_value = np.mean(bootstrap_sharpes <= 0) if observed_sharpe > 0 else np.mean(bootstrap_sharpes >= 0)
        p_value = 2 * min(p_value, 1 - p_value)
        
        return BootstrapResult(
            statistic_name="sharpe_ratio",
            observed_value=observed_sharpe,
            bootstrap_mean=np.mean(bootstrap_sharpes),
            bootstrap_std=np.std(bootstrap_sharpes),
            confidence_interval=(lower, upper),
            p_value=p_value,
            n_bootstrap=self.n_bootstrap
        )
    
    def bootstrap_win_rate(self, returns: pd.Series) -> BootstrapResult:
        """
        Bootstrap confidence interval for win rate.
        
        Args:
            returns: Return series
            
        Returns:
            BootstrapResult with win rate confidence interval
        """
        returns = returns.dropna().values
        n = len(returns)
        
        if n < 30:
            return BootstrapResult(
                statistic_name="win_rate",
                observed_value=0,
                bootstrap_mean=0,
                bootstrap_std=0,
                confidence_interval=(0, 0),
                p_value=1.0,
                n_bootstrap=0
            )
        
        # Observed win rate
        observed_wr = np.mean(returns > 0)
        
        # Bootstrap
        bootstrap_wrs = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            sample = np.random.choice(returns, size=n, replace=True)
            bootstrap_wrs[i] = np.mean(sample > 0)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        lower = np.percentile(bootstrap_wrs, alpha/2 * 100)
        upper = np.percentile(bootstrap_wrs, (1 - alpha/2) * 100)
        
        # P-value (test against 50%)
        p_value = np.mean(bootstrap_wrs <= 0.5) if observed_wr > 0.5 else np.mean(bootstrap_wrs >= 0.5)
        p_value = 2 * min(p_value, 1 - p_value)
        
        return BootstrapResult(
            statistic_name="win_rate",
            observed_value=observed_wr,
            bootstrap_mean=np.mean(bootstrap_wrs),
            bootstrap_std=np.std(bootstrap_wrs),
            confidence_interval=(lower, upper),
            p_value=p_value,
            n_bootstrap=self.n_bootstrap
        )
    
    def monte_carlo_returns(self, returns: pd.Series,
                            n_periods: int = 252) -> Dict[str, Any]:
        """
        Monte Carlo simulation of equity curves.
        
        Simulates possible future equity curves by randomly
        sampling from historical returns.
        
        Args:
            returns: Historical returns
            n_periods: Number of periods to simulate
            
        Returns:
            Dictionary with simulation results
        """
        returns = returns.dropna().values
        
        if len(returns) < 30:
            return {'error': 'Insufficient data'}
        
        # Simulate equity curves
        equity_curves = np.zeros((self.n_bootstrap, n_periods + 1))
        equity_curves[:, 0] = 1.0  # Start at 1
        
        for i in range(self.n_bootstrap):
            sample_returns = np.random.choice(returns, size=n_periods, replace=True)
            equity_curves[i, 1:] = (1 + sample_returns).cumprod()
        
        # Calculate statistics
        final_values = equity_curves[:, -1]
        
        # Percentiles
        percentiles = {
            '5th': np.percentile(final_values, 5),
            '25th': np.percentile(final_values, 25),
            '50th': np.percentile(final_values, 50),
            '75th': np.percentile(final_values, 75),
            '95th': np.percentile(final_values, 95)
        }
        
        # Drawdown analysis
        max_drawdowns = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            cummax = np.maximum.accumulate(equity_curves[i])
            drawdowns = (equity_curves[i] - cummax) / cummax
            max_drawdowns[i] = drawdowns.min()
        
        return {
            'equity_curves': equity_curves,
            'final_values': final_values,
            'percentiles': percentiles,
            'mean_final': np.mean(final_values),
            'std_final': np.std(final_values),
            'prob_profit': np.mean(final_values > 1),
            'expected_max_dd': np.mean(max_drawdowns),
            'worst_case_dd': np.percentile(max_drawdowns, 5)
        }
    
    def permutation_test(self, group1: pd.Series, 
                        group2: pd.Series,
                        statistic: str = 'mean') -> BootstrapResult:
        """
        Permutation test for group comparison.
        
        Tests whether two groups have significantly different
        statistics by randomly permuting group labels.
        
        Args:
            group1: First group
            group2: Second group
            statistic: Statistic to compare ('mean' or 'median')
            
        Returns:
            BootstrapResult with permutation test results
        """
        group1 = group1.dropna().values
        group2 = group2.dropna().values
        
        n1, n2 = len(group1), len(group2)
        
        if n1 + n2 < 30:
            return BootstrapResult(
                statistic_name=f"permutation_{statistic}",
                observed_value=0,
                bootstrap_mean=0,
                bootstrap_std=0,
                confidence_interval=(0, 0),
                p_value=1.0,
                n_bootstrap=0
            )
        
        # Calculate statistic function
        if statistic == 'mean':
            calc_stat = lambda x: np.mean(x)
        else:
            calc_stat = lambda x: np.median(x)
        
        # Observed difference
        observed_diff = calc_stat(group1) - calc_stat(group2)
        
        # Permutation test
        combined = np.concatenate([group1, group2])
        perm_diffs = np.zeros(self.n_bootstrap)
        
        for i in range(self.n_bootstrap):
            np.random.shuffle(combined)
            perm_diffs[i] = calc_stat(combined[:n1]) - calc_stat(combined[n1:])
        
        # P-value
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        
        return BootstrapResult(
            statistic_name=f"permutation_{statistic}",
            observed_value=observed_diff,
            bootstrap_mean=np.mean(perm_diffs),
            bootstrap_std=np.std(perm_diffs),
            confidence_interval=(
                np.percentile(perm_diffs, 2.5),
                np.percentile(perm_diffs, 97.5)
            ),
            p_value=p_value,
            n_bootstrap=self.n_bootstrap
        )
    
    def bootstrap_edge_validation(self, returns: pd.Series) -> Dict[str, BootstrapResult]:
        """
        Comprehensive bootstrap validation of an edge.
        
        Args:
            returns: Trade returns or signal returns
            
        Returns:
            Dictionary with all bootstrap results
        """
        results = {
            'mean': self.bootstrap_mean(returns),
            'sharpe': self.bootstrap_sharpe(returns),
            'win_rate': self.bootstrap_win_rate(returns)
        }
        
        return results
    
    def probability_of_profit(self, returns: pd.Series,
                             target_return: float = 0.0) -> float:
        """
        Calculate probability of achieving target return.
        
        Args:
            returns: Historical returns
            target_return: Target return threshold
            
        Returns:
            Probability of achieving target
        """
        returns = returns.dropna().values
        
        if len(returns) < 30:
            return 0.0
        
        # Bootstrap
        bootstrap_means = np.zeros(self.n_bootstrap)
        n = len(returns)
        
        for i in range(self.n_bootstrap):
            sample = np.random.choice(returns, size=n, replace=True)
            bootstrap_means[i] = np.mean(sample)
        
        return np.mean(bootstrap_means > target_return)
