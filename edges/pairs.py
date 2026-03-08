"""
Pairs and Correlation Edge Detection Module.

Identifies edges based on:
- Currency correlations
- Lead-lag relationships
- Cointegration
- Spread trading opportunities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass
import warnings

from .time_based import EdgeResult

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class PairResult:
    """Container for pair analysis results."""
    pair1: str
    pair2: str
    correlation: float
    cointegration_pvalue: Optional[float]
    half_life: Optional[float]
    spread_sharpe: Optional[float]
    is_tradeable: bool


class CorrelationEdges:
    """
    Detect correlation-based trading edges.
    
    Identifies:
    - Rolling correlation breakdowns
    - Lead-lag relationships
    - Correlation clustering
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize correlation edge detector.
        
        Args:
            significance_level: P-value threshold
        """
        self.significance_level = significance_level
    
    def rolling_correlation(self, series1: pd.Series, series2: pd.Series,
                           window: int = 20) -> pd.Series:
        """
        Calculate rolling correlation.
        
        Args:
            series1: First series
            series2: Second series
            window: Rolling window
            
        Returns:
            Series of rolling correlations
        """
        # Align series
        common_idx = series1.dropna().index.intersection(series2.dropna().index)
        s1 = series1.loc[common_idx]
        s2 = series2.loc[common_idx]
        
        # Calculate rolling correlation
        rolling_corr = s1.rolling(window).corr(s2)
        
        return rolling_corr
    
    def correlation_breakdown(self, series1: pd.Series, series2: pd.Series,
                             window: int = 20,
                             threshold: float = 0.5) -> EdgeResult:
        """
        Detect correlation breakdown signals.
        
        When correlation breaks down significantly, it may indicate
        a trading opportunity as correlation is expected to revert.
        
        Args:
            series1: First price series
            series2: Second price series
            window: Correlation window
            threshold: Correlation threshold for breakdown
            
        Returns:
            EdgeResult with breakdown analysis
        """
        # Calculate returns
        ret1 = series1.pct_change()
        ret2 = series2.pct_change()
        
        # Rolling correlation
        rolling_corr = self.rolling_correlation(ret1, ret2, window)
        
        # Long-term average correlation
        avg_corr = rolling_corr.rolling(window * 5).mean()
        
        # Correlation breakdown: short-term correlation drops significantly
        breakdown = (rolling_corr < avg_corr - threshold) | (rolling_corr < -threshold)
        
        # Forward returns (expect correlation to revert)
        fwd_ret1 = series1.pct_change(10).shift(-10)
        fwd_ret2 = series2.pct_change(10).shift(-10)
        
        # Calculate spread returns
        breakdown_returns = []
        normal_returns = []
        
        for idx in breakdown[breakdown].index:
            if idx in fwd_ret1.index and idx in fwd_ret2.index:
                r1 = fwd_ret1.loc[idx]
                r2 = fwd_ret2.loc[idx]
                if not pd.isna(r1) and not pd.isna(r2):
                    # Spread trade: long one, short other based on sign
                    spread_ret = abs(r1 - r2)
                    breakdown_returns.append(spread_ret)
        
        for idx in breakdown[~breakdown].index:
            if idx in fwd_ret1.index and idx in fwd_ret2.index:
                r1 = fwd_ret1.loc[idx]
                r2 = fwd_ret2.loc[idx]
                if not pd.isna(r1) and not pd.isna(r2):
                    spread_ret = abs(r1 - r2)
                    normal_returns.append(spread_ret)
        
        if len(breakdown_returns) < 30:
            return None
        
        breakdown_returns = pd.Series(breakdown_returns)
        normal_returns = pd.Series(normal_returns)
        
        t_stat, p_val = stats.ttest_ind(breakdown_returns, normal_returns)
        
        return EdgeResult(
            name="Correlation Breakdown",
            edge_type="correlation",
            sample_size=len(breakdown_returns),
            mean_return=breakdown_returns.mean() - normal_returns.mean(),
            std_return=breakdown_returns.std(),
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=(breakdown_returns > normal_returns.mean()).mean(),
            sharpe_ratio=0,
            is_significant=p_val < self.significance_level,
            details={
                'avg_breakdown_spread': breakdown_returns.mean(),
                'avg_normal_spread': normal_returns.mean(),
                'correlation_window': window
            }
        )
    
    def lead_lag_relationship(self, series1: pd.Series, series2: pd.Series,
                              max_lag: int = 10) -> Dict[str, EdgeResult]:
        """
        Detect lead-lag relationships between pairs.
        
        Tests whether one series leads the other.
        
        Args:
            series1: First price series (potential leader)
            series2: Second price series (potential lagger)
            max_lag: Maximum lag to test
            
        Returns:
            Dictionary with lead-lag analysis
        """
        results = {}
        
        # Calculate returns
        ret1 = series1.pct_change()
        ret2 = series2.pct_change()
        
        # Align
        common_idx = ret1.dropna().index.intersection(ret2.dropna().index)
        ret1 = ret1.loc[common_idx]
        ret2 = ret2.loc[common_idx]
        
        # Test cross-correlation at different lags
        for lag in range(1, max_lag + 1):
            # ret1 leads ret2
            lagged_corr, p_val = stats.pearsonr(
                ret1.iloc[:-lag].values,
                ret2.iloc[lag:].values
            )
            
            results[f'lead_{lag}'] = EdgeResult(
                name=f"Lead-Lag ({lag} bars)",
                edge_type="lead_lag",
                sample_size=len(ret1) - lag,
                mean_return=lagged_corr,
                std_return=0,
                t_statistic=lagged_corr / (1 / np.sqrt(len(ret1) - lag)),
                p_value=p_val,
                win_rate=0,
                sharpe_ratio=0,
                is_significant=abs(lagged_corr) > 0.1 and p_val < self.significance_level,
                details={
                    'lag': lag,
                    'direction': f'Series1 leads Series2 by {lag} bars'
                }
            )
        
        return results


class CointegrationEdges:
    """
    Detect cointegration-based trading edges.
    
    Identifies mean-reverting spreads between cointegrated pairs.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize cointegration edge detector.
        
        Args:
            significance_level: P-value threshold
        """
        self.significance_level = significance_level
    
    def engle_granger_test(self, series1: pd.Series, 
                          series2: pd.Series) -> Tuple[float, float]:
        """
        Perform Engle-Granger cointegration test.
        
        Args:
            series1: First price series
            series2: Second price series
            
        Returns:
            Tuple of (cointegration statistic, p-value)
        """
        try:
            from statsmodels.tsa.stattools import coint
            stat, pvalue, _ = coint(series1, series2)
            return stat, pvalue
        except ImportError:
            # Fallback: simple OLS residual test
            return self._simple_coint_test(series1, series2)
    
    def _simple_coint_test(self, series1: pd.Series, 
                          series2: pd.Series) -> Tuple[float, float]:
        """Simple cointegration test without statsmodels."""
        from numpy.linalg import lstsq
        
        # Align series
        common_idx = series1.dropna().index.intersection(series2.dropna().index)
        y = series1.loc[common_idx].values
        x = series2.loc[common_idx].values
        
        # OLS regression
        X = np.column_stack([np.ones(len(x)), x])
        beta, _, _, _ = lstsq(X, y, rcond=None)
        
        # Residuals
        residuals = y - X @ beta
        
        # ADF-like test on residuals
        diff_resid = np.diff(residuals)
        lag_resid = residuals[:-1]
        
        # Simple test
        corr, p_val = stats.pearsonr(diff_resid, lag_resid)
        
        # More negative correlation = more mean-reverting
        return corr, p_val
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion.
        
        Args:
            spread: Spread series
            
        Returns:
            Half-life in bars
        """
        spread = spread.dropna()
        
        # Calculate spread changes
        lag_spread = spread.shift(1).dropna()
        delta_spread = spread.diff().dropna()
        
        # Align
        common_idx = lag_spread.index.intersection(delta_spread.index)
        lag_spread = lag_spread.loc[common_idx]
        delta_spread = delta_spread.loc[common_idx]
        
        # Regression: delta = lambda * lag + error
        X = np.column_stack([np.ones(len(lag_spread)), lag_spread.values])
        beta, _, _, _ = np.linalg.lstsq(X, delta_spread.values, rcond=None)
        
        # Half-life
        lambda_val = beta[1]
        
        if lambda_val >= 0:
            return np.inf  # No mean reversion
        
        half_life = -np.log(2) / lambda_val
        
        return half_life
    
    def find_cointegrated_pairs(self, data_dict: Dict[str, pd.Series],
                                p_threshold: float = 0.05) -> List[PairResult]:
        """
        Find cointegrated pairs from multiple series.
        
        Args:
            data_dict: Dictionary of symbol -> price series
            p_threshold: P-value threshold for cointegration
            
        Returns:
            List of PairResult with cointegration analysis
        """
        symbols = list(data_dict.keys())
        results = []
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym1, sym2 = symbols[i], symbols[j]
                
                try:
                    series1 = data_dict[sym1].dropna()
                    series2 = data_dict[sym2].dropna()
                    
                    # Align
                    common_idx = series1.index.intersection(series2.index)
                    if len(common_idx) < 100:
                        continue
                    
                    s1 = series1.loc[common_idx]
                    s2 = series2.loc[common_idx]
                    
                    # Test cointegration
                    stat, pvalue = self.engle_granger_test(s1, s2)
                    
                    if pvalue < p_threshold:
                        # Calculate hedge ratio
                        X = np.column_stack([np.ones(len(s2)), s2.values])
                        beta, _, _, _ = np.linalg.lstsq(X, s1.values, rcond=None)
                        hedge_ratio = beta[1]
                        
                        # Calculate spread
                        spread = s1 - hedge_ratio * s2
                        
                        # Half-life
                        half_life = self.calculate_half_life(spread)
                        
                        # Spread trading potential
                        spread_returns = spread.pct_change().dropna()
                        sharpe = spread_returns.mean() / spread_returns.std() * np.sqrt(252) if spread_returns.std() > 0 else 0
                        
                        results.append(PairResult(
                            pair1=sym1,
                            pair2=sym2,
                            correlation=np.corrcoef(s1, s2)[0, 1],
                            cointegration_pvalue=pvalue,
                            half_life=half_life,
                            spread_sharpe=sharpe,
                            is_tradeable=5 < half_life < 100
                        ))
                
                except Exception as e:
                    continue
        
        # Sort by cointegration strength
        results.sort(key=lambda x: x.cointegration_pvalue)
        
        return results
    
    def spread_zscore_signal(self, spread: pd.Series,
                             lookback: int = 20,
                             threshold: float = 2.0) -> EdgeResult:
        """
        Test spread z-score trading signal.
        
        Args:
            spread: Spread series
            lookback: Lookback for z-score
            threshold: Z-score threshold
            
        Returns:
            EdgeResult with spread signal analysis
        """
        spread = spread.dropna()
        
        if len(spread) < lookback + 30:
            return None
        
        # Calculate z-score
        mean = spread.rolling(lookback).mean()
        std = spread.rolling(lookback).std()
        zscore = (spread - mean) / std
        
        # Forward returns (expect mean reversion)
        fwd_returns = spread.shift(-10) / spread - 1
        
        # Signal conditions
        oversold = zscore < -threshold  # Buy spread
        overbought = zscore > threshold  # Sell spread
        
        # Returns
        oversold_returns = fwd_returns[oversold].dropna()
        overbought_returns = -fwd_returns[overbought].dropna()
        
        all_returns = pd.concat([oversold_returns, overbought_returns])
        
        if len(all_returns) < 30:
            return None
        
        t_stat, p_val = stats.ttest_1samp(all_returns, 0)
        
        mean_ret = all_returns.mean()
        std_ret = all_returns.std()
        win_rate = (all_returns > 0).mean()
        sharpe = mean_ret / std_ret * np.sqrt(252 / 10) if std_ret > 0 else 0
        
        return EdgeResult(
            name=f"Spread Z-Score ({threshold}σ)",
            edge_type="pairs",
            sample_size=len(all_returns),
            mean_return=mean_ret,
            std_return=std_ret,
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            is_significant=p_val < self.significance_level,
            details={
                'oversold_count': len(oversold_returns),
                'overbought_count': len(overbought_returns),
                'lookback': lookback
            }
        )


class CurrencyStrengthEdges:
    """
    Detect currency strength-based edges.
    
    Identifies strong vs weak currencies for pair selection.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize currency strength detector."""
        self.significance_level = significance_level
    
    def calculate_currency_strength(self, 
                                   data_dict: Dict[str, pd.Series],
                                   lookback: int = 20) -> pd.DataFrame:
        """
        Calculate relative currency strength.
        
        Args:
            data_dict: Dictionary of pair -> price series
            lookback: Lookback period
            
        Returns:
            DataFrame with currency strength scores
        """
        # Extract currencies from pairs
        currencies = set()
        for pair in data_dict.keys():
            # Assume 6-character pairs like EURUSD
            if len(pair) >= 6:
                currencies.add(pair[:3])
                currencies.add(pair[3:6])
        
        currencies = sorted(currencies)
        
        # Calculate returns for each pair
        returns_dict = {}
        for pair, prices in data_dict.items():
            returns_dict[pair] = prices.pct_change(lookback)
        
        # Calculate strength for each currency
        strength_data = {}
        
        for currency in currencies:
            strength = pd.Series(0.0, index=list(data_dict.values())[0].index)
            count = 0
            
            for pair, ret in returns_dict.items():
                if len(pair) >= 6:
                    base = pair[:3]
                    quote = pair[3:6]
                    
                    if base == currency:
                        strength += ret
                        count += 1
                    elif quote == currency:
                        strength -= ret
                        count += 1
            
            if count > 0:
                strength_data[currency] = strength / count
        
        return pd.DataFrame(strength_data)
    
    def strongest_weakest_strategy(self, 
                                  data_dict: Dict[str, pd.Series],
                                  lookback: int = 20,
                                  lookahead: int = 10) -> EdgeResult:
        """
        Test strongest vs weakest currency strategy.
        
        Buy the strongest currency, sell the weakest.
        
        Args:
            data_dict: Dictionary of pair -> price series
            lookback: Strength lookback
            lookahead: Forward return period
            
        Returns:
            EdgeResult with strategy analysis
        """
        strength_df = self.calculate_currency_strength(data_dict, lookback)
        
        if strength_df.empty:
            return None
        
        # Find strongest and weakest at each point
        strongest = strength_df.idxmax(axis=1)
        weakest = strength_df.idxmin(axis=1)
        
        # Calculate forward returns for each pair
        pair_fwd_returns = {}
        for pair, prices in data_dict.items():
            pair_fwd_returns[pair] = prices.pct_change(lookahead).shift(-lookahead)
        
        # Build strategy returns
        strategy_returns = []
        
        for date in strength_df.index:
            strong = strongest.loc[date]
            weak = weakest.loc[date]
            
            # Find pair that goes long strong, short weak
            long_pair = None
            short_pair = None
            
            for pair in data_dict.keys():
                if len(pair) >= 6:
                    base = pair[:3]
                    quote = pair[3:6]
                    
                    if base == strong and quote == weak:
                        long_pair = pair
                        break
                    elif base == weak and quote == strong:
                        short_pair = pair
                        break
            
            if long_pair and date in pair_fwd_returns[long_pair].index:
                ret = pair_fwd_returns[long_pair].loc[date]
                if not pd.isna(ret):
                    strategy_returns.append(ret)
            elif short_pair and date in pair_fwd_returns[short_pair].index:
                ret = -pair_fwd_returns[short_pair].loc[date]
                if not pd.isna(ret):
                    strategy_returns.append(ret)
        
        if len(strategy_returns) < 30:
            return None
        
        strategy_returns = pd.Series(strategy_returns)
        
        t_stat, p_val = stats.ttest_1samp(strategy_returns, 0)
        
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        win_rate = (strategy_returns > 0).mean()
        sharpe = mean_ret / std_ret * np.sqrt(252 / lookahead) if std_ret > 0 else 0
        
        return EdgeResult(
            name="Strongest vs Weakest",
            edge_type="currency_strength",
            sample_size=len(strategy_returns),
            mean_return=mean_ret,
            std_return=std_ret,
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            is_significant=p_val < self.significance_level,
            details={
                'lookback': lookback,
                'lookahead': lookahead
            }
        )


class PairsEdgeDetector:
    """
    Main class for detecting pairs-based edges.
    
    Combines correlation, cointegration, and currency strength analysis.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize pairs edge detector."""
        self.correlation = CorrelationEdges(significance_level)
        self.cointegration = CointegrationEdges(significance_level)
        self.currency_strength = CurrencyStrengthEdges(significance_level)
    
    def run_all_pairs_edges(self, data_dict: Dict[str, pd.Series],
                           single_pair: Tuple[str, str] = None) -> Dict[str, EdgeResult]:
        """
        Run all pairs-based edge tests.
        
        Args:
            data_dict: Dictionary of symbol -> price series
            single_pair: Optional specific pair to test
            
        Returns:
            Dictionary with all edge results
        """
        all_results = {}
        
        if single_pair:
            sym1, sym2 = single_pair
            series1 = data_dict.get(sym1)
            series2 = data_dict.get(sym2)
            
            if series1 is not None and series2 is not None:
                # Correlation breakdown
                result = self.correlation.correlation_breakdown(series1, series2)
                if result:
                    all_results['correlation_breakdown'] = result
                
                # Lead-lag
                lead_lag = self.correlation.lead_lag_relationship(series1, series2)
                all_results.update({f'lead_lag_{k}': v for k, v in lead_lag.items() if v.is_significant})
        
        # Currency strength
        result = self.currency_strength.strongest_weakest_strategy(data_dict)
        if result:
            all_results['currency_strength'] = result
        
        return all_results
