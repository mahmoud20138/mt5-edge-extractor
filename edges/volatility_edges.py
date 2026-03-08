"""
Volatility Edge Detection Module.

Identifies edges based on volatility patterns:
- Volatility clustering
- Volatility contraction/expansion
- Regime detection
- Squeeze patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass

from .time_based import EdgeResult


class VolatilityEdges:
    """
    Detect volatility-based trading edges.
    
    Volatility edges arise from the tendency of volatility
    to cluster and alternate between high and low regimes.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize volatility edge detector.
        
        Args:
            significance_level: P-value threshold for significance
        """
        self.significance_level = significance_level
    
    def volatility_clustering(self, df: pd.DataFrame,
                              period: int = 20) -> EdgeResult:
        """
        Test for volatility clustering.
        
        Tests whether high/low volatility periods tend to persist.
        
        Args:
            df: DataFrame with price data
            period: Rolling period for volatility
            
        Returns:
            EdgeResult with clustering analysis
        """
        # Calculate returns
        returns = df['close'].pct_change()
        
        # Absolute returns (proxy for volatility)
        abs_returns = returns.abs()
        
        # Autocorrelation of absolute returns
        autocorr = abs_returns.rolling(period).apply(
            lambda x: pd.Series(x).autocorr() if len(x) > 1 else np.nan
        )
        
        avg_autocorr = autocorr.dropna().mean()
        
        # Ljung-Box test for volatility clustering
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        try:
            lb_result = acorr_ljungbox(abs_returns.dropna(), lags=[10], return_df=True)
            lb_pvalue = lb_result['lb_pvalue'].iloc[0]
        except:
            lb_pvalue = np.nan
        
        # Determine if clustering exists
        is_clustered = avg_autocorr > 0.2 and (lb_pvalue < 0.05 if not np.isnan(lb_pvalue) else False)
        
        return EdgeResult(
            name="Volatility Clustering",
            edge_type="volatility",
            sample_size=len(returns.dropna()),
            mean_return=avg_autocorr,
            std_return=autocorr.std(),
            t_statistic=avg_autocorr / autocorr.std() if autocorr.std() > 0 else 0,
            p_value=lb_pvalue,
            win_rate=0,
            sharpe_ratio=0,
            is_significant=is_clustered,
            details={
                'avg_autocorrelation': avg_autocorr,
                'ljung_box_pvalue': lb_pvalue,
                'interpretation': 'clustering_detected' if is_clustered else 'no_significant_clustering'
            }
        )
    
    def volatility_contraction_expansion(self, df: pd.DataFrame,
                                         lookback: int = 20,
                                         percentile_threshold: float = 20,
                                         lookahead: int = 20) -> EdgeResult:
        """
        Test volatility contraction -> expansion edge.
        
        Tests whether low volatility periods are followed
        by larger than average moves.
        
        Args:
            df: DataFrame with OHLC data
            lookback: Period for volatility calculation
            percentile_threshold: Percentile for low volatility
            lookahead: Forward period for range measurement
            
        Returns:
            EdgeResult with contraction analysis
        """
        # Calculate volatility (ATR-based)
        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)
        
        atr = tr.rolling(lookback).mean()
        atr_pct = atr / df['close']
        
        # Volatility percentile rank
        vol_rank = atr_pct.rolling(lookback * 5).rank(pct=True) * 100
        
        # Low volatility condition
        low_vol = vol_rank < percentile_threshold
        
        # Forward range
        fwd_high = df['high'].shift(-1).rolling(lookahead).max()
        fwd_low = df['low'].shift(-1).rolling(lookahead).min()
        fwd_range = (fwd_high - fwd_low) / df['close']
        
        # Average forward range
        avg_fwd_range = fwd_range.rolling(lookback * 5).mean()
        
        # Compare forward range after low volatility
        low_vol_fwd_range = fwd_range[low_vol].dropna()
        normal_fwd_range = fwd_range[~low_vol].dropna()
        
        if len(low_vol_fwd_range) < 30:
            return None
        
        # Test if forward range after low vol is larger
        t_stat, p_val = stats.ttest_ind(low_vol_fwd_range, normal_fwd_range)
        
        return EdgeResult(
            name=f"Vol Contraction ({percentile_threshold}th pct)",
            edge_type="volatility",
            sample_size=len(low_vol_fwd_range),
            mean_return=low_vol_fwd_range.mean() / avg_fwd_range.mean() - 1,
            std_return=low_vol_fwd_range.std(),
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=(low_vol_fwd_range > avg_fwd_range.mean()).mean(),
            sharpe_ratio=0,
            is_significant=p_val < self.significance_level and low_vol_fwd_range.mean() > normal_fwd_range.mean(),
            details={
                'avg_range_after_contraction': low_vol_fwd_range.mean(),
                'avg_normal_range': normal_fwd_range.mean(),
                'expansion_factor': low_vol_fwd_range.mean() / normal_fwd_range.mean()
            }
        )
    
    def bollinger_squeeze(self, df: pd.DataFrame,
                          bb_period: int = 20,
                          kc_period: int = 20,
                          lookahead: int = 20) -> EdgeResult:
        """
        Test Bollinger Band squeeze edge.
        
        Tests whether BB inside KC (squeeze) predicts
        subsequent large moves.
        
        Args:
            df: DataFrame with OHLC data
            bb_period: BB period
            kc_period: KC period
            lookahead: Forward period for measuring moves
            
        Returns:
            EdgeResult with squeeze analysis
        """
        from features.volatility_features import VolatilityFeatures
        
        # Calculate BB and KC
        bb = VolatilityFeatures.bollinger_bands(df, bb_period)
        kc = VolatilityFeatures.keltner_channels(df, kc_period)
        
        # Squeeze condition
        squeeze = (bb['bb_upper'] < kc['kc_upper']) & (bb['bb_lower'] > kc['kc_lower'])
        
        # Forward range
        fwd_high = df['high'].shift(-1).rolling(lookahead).max()
        fwd_low = df['low'].shift(-1).rolling(lookahead).min()
        fwd_range = (fwd_high - fwd_low) / df['close']
        
        # Compare
        squeeze_range = fwd_range[squeeze].dropna()
        no_squeeze_range = fwd_range[~squeeze].dropna()
        
        if len(squeeze_range) < 30:
            return None
        
        t_stat, p_val = stats.ttest_ind(squeeze_range, no_squeeze_range)
        
        # Direction bias after squeeze
        fwd_return = df['close'].shift(-lookahead) / df['close'] - 1
        squeeze_direction = fwd_return[squeeze].dropna()
        
        return EdgeResult(
            name="BB Squeeze",
            edge_type="volatility",
            sample_size=len(squeeze_range),
            mean_return=squeeze_range.mean(),
            std_return=squeeze_range.std(),
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=(squeeze_range > no_squeeze_range.mean()).mean(),
            sharpe_ratio=0,
            is_significant=p_val < self.significance_level and squeeze_range.mean() > no_squeeze_range.mean(),
            details={
                'squeeze_count': len(squeeze_range),
                'avg_range_after_squeeze': squeeze_range.mean(),
                'avg_normal_range': no_squeeze_range.mean(),
                'direction_bias': squeeze_direction.mean() if len(squeeze_direction) > 0 else 0
            }
        )
    
    def nr4_nr7_pattern(self, df: pd.DataFrame,
                        lookahead: int = 10) -> Tuple[EdgeResult, EdgeResult]:
        """
        Test NR4/NR7 (Narrow Range) patterns.
        
        NR4/NR7 are days with the narrowest range of the
        past 4/7 days, often preceding breakouts.
        
        Args:
            df: DataFrame with OHLC data
            lookahead: Forward period for range
            
        Returns:
            Tuple of (NR4 result, NR7 result)
        """
        # Calculate range
        range_ = df['high'] - df['low']
        
        # NR4: today's range is smallest of past 4 days
        nr4 = (range_ == range_.rolling(4).min())
        
        # NR7: today's range is smallest of past 7 days
        nr7 = (range_ == range_.rolling(7).min())
        
        # Forward range
        fwd_high = df['high'].shift(-1).rolling(lookahead).max()
        fwd_low = df['low'].shift(-1).rolling(lookahead).min()
        fwd_range = (fwd_high - fwd_low) / df['close']
        
        avg_range = fwd_range.rolling(50).mean()
        
        results = []
        
        for name, condition in [('NR4', nr4), ('NR7', nr7)]:
            cond_range = fwd_range[condition].dropna()
            normal_range = fwd_range[~condition].dropna()
            
            if len(cond_range) < 30:
                results.append(None)
                continue
            
            t_stat, p_val = stats.ttest_ind(cond_range, normal_range)
            
            results.append(EdgeResult(
                name=name,
                edge_type="volatility",
                sample_size=len(cond_range),
                mean_return=cond_range.mean() / avg_range.mean() - 1,
                std_return=cond_range.std(),
                t_statistic=t_stat,
                p_value=p_val,
                win_rate=(cond_range > avg_range.mean()).mean(),
                sharpe_ratio=0,
                is_significant=p_val < self.significance_level,
                details={
                    'pattern_count': len(cond_range),
                    'avg_forward_range': cond_range.mean()
                }
            ))
        
        return tuple(results)
    
    def atr_breakout(self, df: pd.DataFrame,
                     atr_period: int = 14,
                     atr_multiplier: float = 1.5,
                     lookahead: int = 10) -> EdgeResult:
        """
        Test ATR-based breakout edge.
        
        Tests whether price breaking out by N x ATR
        predicts continuation.
        
        Args:
            df: DataFrame with OHLC data
            atr_period: ATR period
            atr_multiplier: ATR multiplier for breakout
            lookahead: Forward period for returns
            
        Returns:
            EdgeResult with ATR breakout analysis
        """
        from features.volatility_features import VolatilityFeatures
        
        # Calculate ATR
        atr = VolatilityFeatures.atr(df, atr_period)
        
        # Previous close
        prev_close = df['close'].shift(1)
        
        # Breakout levels
        upper_level = prev_close + atr_multiplier * atr.shift(1)
        lower_level = prev_close - atr_multiplier * atr.shift(1)
        
        # Breakout conditions
        breakout_up = df['close'] > upper_level
        breakout_down = df['close'] < lower_level
        
        # Forward returns
        fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Analyze
        up_returns = fwd_returns[breakout_up].dropna()
        down_returns = -fwd_returns[breakout_down].dropna()
        
        all_returns = pd.concat([up_returns, down_returns])
        
        if len(all_returns) < 30:
            return None
        
        t_stat, p_val = stats.ttest_1samp(all_returns, 0)
        
        mean_ret = all_returns.mean()
        std_ret = all_returns.std()
        win_rate = (all_returns > 0).mean()
        sharpe = mean_ret / std_ret * np.sqrt(252 / lookahead) if std_ret > 0 else 0
        
        return EdgeResult(
            name=f"ATR Breakout ({atr_multiplier}x)",
            edge_type="volatility",
            sample_size=len(all_returns),
            mean_return=mean_ret,
            std_return=std_ret,
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            is_significant=p_val < self.significance_level,
            details={
                'up_breakouts': len(up_returns),
                'down_breakouts': len(down_returns)
            }
        )
    
    def regime_detection(self, df: pd.DataFrame,
                         period: int = 100) -> Dict[str, EdgeResult]:
        """
        Detect volatility regimes.
        
        Classifies market into low/normal/high volatility regimes.
        
        Args:
            df: DataFrame with OHLC data
            period: Lookback period for regime classification
            
        Returns:
            Dictionary with regime results
        """
        from features.volatility_features import VolatilityFeatures
        
        # Calculate volatility
        vol = VolatilityFeatures.historical_volatility(df, 20)
        
        # Regime classification
        vol_low = vol.rolling(period).quantile(0.25)
        vol_high = vol.rolling(period).quantile(0.75)
        
        low_vol_regime = vol < vol_low
        high_vol_regime = vol > vol_high
        normal_vol_regime = ~low_vol_regime & ~high_vol_regime
        
        # Forward returns
        fwd_returns = df['close'].shift(-10) / df['close'] - 1
        
        results = {}
        
        for name, regime in [('low', low_vol_regime), ('normal', normal_vol_regime), ('high', high_vol_regime)]:
            regime_returns = fwd_returns[regime].dropna()
            
            if len(regime_returns) < 30:
                results[name] = None
                continue
            
            t_stat, p_val = stats.ttest_1samp(regime_returns, 0)
            
            results[name] = EdgeResult(
                name=f"{name.title()} Vol Regime",
                edge_type="volatility",
                sample_size=len(regime_returns),
                mean_return=regime_returns.mean(),
                std_return=regime_returns.std(),
                t_statistic=t_stat,
                p_value=p_val,
                win_rate=(regime_returns > 0).mean(),
                sharpe_ratio=regime_returns.mean() / regime_returns.std() * np.sqrt(252 / 10) if regime_returns.std() > 0 else 0,
                is_significant=p_val < self.significance_level,
                details={'regime': name}
            )
        
        return results
    
    def run_all_vol_edges(self, df: pd.DataFrame,
                          lookahead: int = 10) -> Dict[str, EdgeResult]:
        """
        Run all volatility edge tests.
        
        Args:
            df: DataFrame with OHLCV data
            lookahead: Forward period for returns
            
        Returns:
            Dictionary with all edge results
        """
        all_results = {}
        
        # Clustering
        all_results['vol_clustering'] = self.volatility_clustering(df)
        
        # Contraction/Expansion
        for pct in [10, 20, 30]:
            result = self.volatility_contraction_expansion(df, percentile_threshold=pct, lookahead=lookahead)
            if result:
                all_results[f'vol_contraction_{pct}'] = result
        
        # BB Squeeze
        result = self.bollinger_squeeze(df, lookahead=lookahead)
        if result:
            all_results['bb_squeeze'] = result
        
        # NR4/NR7
        nr4, nr7 = self.nr4_nr7_pattern(df, lookahead)
        if nr4:
            all_results['nr4'] = nr4
        if nr7:
            all_results['nr7'] = nr7
        
        # ATR Breakout
        for mult in [1.0, 1.5, 2.0]:
            result = self.atr_breakout(df, atr_multiplier=mult, lookahead=lookahead)
            if result:
                all_results[f'atr_breakout_{mult}'] = result
        
        # Regime
        regime_results = self.regime_detection(df)
        for name, result in regime_results.items():
            if result:
                all_results[f'regime_{name}'] = result
        
        return all_results
