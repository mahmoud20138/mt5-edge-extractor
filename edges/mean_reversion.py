"""
Mean Reversion Edge Detection Module.

Identifies edges based on mean reversion:
- Z-score mean reversion
- Oscillator extremes (RSI, Stochastic)
- Gap fill patterns
- Overextension reversal
- Pairs/spread trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass

from .time_based import EdgeResult


class MeanReversionEdges:
    """
    Detect mean reversion-based trading edges.
    
    Mean reversion edges arise from the tendency of prices
    to revert to their mean after extreme moves. This is
    particularly effective in range-bound markets.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize mean reversion edge detector.
        
        Args:
            significance_level: P-value threshold for significance
        """
        self.significance_level = significance_level
    
    def zscore_mean_reversion(self, df: pd.DataFrame,
                              lookback: int = 20,
                              z_threshold: float = 2.0,
                              lookahead: int = 10) -> EdgeResult:
        """
        Test z-score mean reversion edge.
        
        Tests whether extreme z-scores predict price reversion.
        
        Args:
            df: DataFrame with price data
            lookback: Lookback period for mean/std
            z_threshold: Z-score threshold for extremes
            lookahead: Forward period for returns
            
        Returns:
            EdgeResult with z-score analysis
        """
        # Calculate z-score
        mean = df['close'].rolling(lookback).mean()
        std = df['close'].rolling(lookback).std()
        zscore = (df['close'] - mean) / std
        
        # Forward returns
        fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Extreme conditions
        oversold = zscore < -z_threshold  # Expect reversion up
        overbought = zscore > z_threshold  # Expect reversion down
        
        # Returns: buy oversold, sell overbought
        oversold_returns = fwd_returns[oversold].dropna()  # Expect positive
        overbought_returns = -fwd_returns[overbought].dropna()  # Expect positive (inverted)
        
        all_returns = pd.concat([oversold_returns, overbought_returns])
        
        if len(all_returns) < 30:
            return None
        
        t_stat, p_val = stats.ttest_1samp(all_returns, 0)
        
        mean_ret = all_returns.mean()
        std_ret = all_returns.std()
        win_rate = (all_returns > 0).mean()
        sharpe = mean_ret / std_ret * np.sqrt(252 / lookahead) if std_ret > 0 else 0
        
        return EdgeResult(
            name=f"Z-Score MR ({lookback}, ±{z_threshold}σ)",
            edge_type="mean_reversion",
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
                'oversold_win_rate': (oversold_returns > 0).mean() if len(oversold_returns) > 0 else 0,
                'overbought_win_rate': (overbought_returns > 0).mean() if len(overbought_returns) > 0 else 0
            }
        )
    
    def zscore_quintile_analysis(self, df: pd.DataFrame,
                                  lookback: int = 20,
                                  lookahead: int = 10) -> EdgeResult:
        """
        Perform quintile analysis on z-scores.
        
        Analyzes forward returns by z-score quintile to
        identify optimal thresholds.
        
        Args:
            df: DataFrame with price data
            lookback: Lookback period
            lookahead: Forward period
            
        Returns:
            EdgeResult with quintile analysis
        """
        # Calculate z-score
        mean = df['close'].rolling(lookback).mean()
        std = df['close'].rolling(lookback).std()
        zscore = (df['close'] - mean) / std
        
        # Forward returns
        fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Create analysis DataFrame
        analysis_df = pd.DataFrame({
            'zscore': zscore,
            'fwd_return': fwd_returns
        }).dropna()
        
        if len(analysis_df) < 100:
            return None
        
        # Quintile analysis
        analysis_df['quintile'] = pd.qcut(analysis_df['zscore'], 5, labels=False)
        
        quintile_stats = {}
        for q in range(5):
            q_data = analysis_df[analysis_df['quintile'] == q]['fwd_return']
            quintile_stats[f'Q{q+1}'] = {
                'mean': q_data.mean(),
                'std': q_data.std(),
                'win_rate': (q_data > 0).mean(),
                'count': len(q_data)
            }
        
        # Test bottom quintile (most oversold) for positive returns
        bottom_q = analysis_df[analysis_df['quintile'] == 0]['fwd_return']
        top_q = analysis_df[analysis_df['quintile'] == 4]['fwd_return']
        
        # Mean reversion strategy: long bottom, short top
        mr_returns = bottom_q - top_q.values.mean()  # Simplified
        
        t_stat, p_val = stats.ttest_1samp(bottom_q, 0)
        
        return EdgeResult(
            name=f"Z-Score Quintile ({lookback})",
            edge_type="mean_reversion",
            sample_size=len(bottom_q),
            mean_return=bottom_q.mean(),
            std_return=bottom_q.std(),
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=(bottom_q > 0).mean(),
            sharpe_ratio=bottom_q.mean() / bottom_q.std() * np.sqrt(252 / lookahead) if bottom_q.std() > 0 else 0,
            is_significant=p_val < self.significance_level and bottom_q.mean() > 0,
            details={
                'quintiles': quintile_stats,
                'optimal_quintile': 'Q1' if quintile_stats['Q1']['mean'] > quintile_stats['Q5']['mean'] else 'Q5'
            }
        )
    
    def rsi_extremes(self, df: pd.DataFrame,
                     oversold: float = 30,
                     overbought: float = 70,
                     rsi_period: int = 14,
                     lookahead: int = 10) -> EdgeResult:
        """
        Test RSI extremes mean reversion.
        
        Tests whether RSI oversold/overbought levels
        predict price reversals.
        
        Args:
            df: DataFrame with price data
            oversold: RSI oversold threshold
            overbought: RSI overbought threshold
            rsi_period: RSI period
            lookahead: Forward period for returns
            
        Returns:
            EdgeResult with RSI analysis
        """
        # Calculate RSI if not present
        if 'rsi_14' not in df.columns:
            from features.momentum import MomentumFeatures
            df = df.copy()
            df['rsi_14'] = MomentumFeatures.rsi(df, rsi_period)
        
        rsi_col = f'rsi_{rsi_period}' if f'rsi_{rsi_period}' in df.columns else 'rsi_14'
        
        # Forward returns
        fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Extreme conditions
        is_oversold = df[rsi_col] < oversold
        is_overbought = df[rsi_col] > overbought
        
        # Returns
        oversold_returns = fwd_returns[is_oversold].dropna()
        overbought_returns = -fwd_returns[is_overbought].dropna()
        
        all_returns = pd.concat([oversold_returns, overbought_returns])
        
        if len(all_returns) < 30:
            return None
        
        t_stat, p_val = stats.ttest_1samp(all_returns, 0)
        
        mean_ret = all_returns.mean()
        std_ret = all_returns.std()
        win_rate = (all_returns > 0).mean()
        sharpe = mean_ret / std_ret * np.sqrt(252 / lookahead) if std_ret > 0 else 0
        
        return EdgeResult(
            name=f"RSI MR ({rsi_period}, {oversold}/{overbought})",
            edge_type="mean_reversion",
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
                'oversold_win_rate': (oversold_returns > 0).mean() if len(oversold_returns) > 0 else 0,
                'overbought_win_rate': (overbought_returns > 0).mean() if len(overbought_returns) > 0 else 0,
                'avg_rsi_oversold': df.loc[is_oversold, rsi_col].mean(),
                'avg_rsi_overbought': df.loc[is_overbought, rsi_col].mean()
            }
        )
    
    def stochastic_extremes(self, df: pd.DataFrame,
                            oversold: float = 20,
                            overbought: float = 80,
                            lookahead: int = 10) -> EdgeResult:
        """
        Test Stochastic extremes mean reversion.
        
        Args:
            df: DataFrame with OHLC data
            oversold: Stochastic oversold threshold
            overbought: Stochastic overbought threshold
            lookahead: Forward period for returns
            
        Returns:
            EdgeResult with Stochastic analysis
        """
        # Calculate Stochastic if not present
        if 'stoch_k' not in df.columns:
            from features.momentum import MomentumFeatures
            stoch = MomentumFeatures.stochastic(df)
            df = df.copy()
            df['stoch_k'] = stoch['stoch_k']
        
        # Forward returns
        fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Extreme conditions
        is_oversold = df['stoch_k'] < oversold
        is_overbought = df['stoch_k'] > overbought
        
        # Returns
        oversold_returns = fwd_returns[is_oversold].dropna()
        overbought_returns = -fwd_returns[is_overbought].dropna()
        
        all_returns = pd.concat([oversold_returns, overbought_returns])
        
        if len(all_returns) < 30:
            return None
        
        t_stat, p_val = stats.ttest_1samp(all_returns, 0)
        
        mean_ret = all_returns.mean()
        std_ret = all_returns.std()
        win_rate = (all_returns > 0).mean()
        sharpe = mean_ret / std_ret * np.sqrt(252 / lookahead) if std_ret > 0 else 0
        
        return EdgeResult(
            name=f"Stoch MR ({oversold}/{overbought})",
            edge_type="mean_reversion",
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
                'overbought_count': len(overbought_returns)
            }
        )
    
    def bollinger_band_reversion(self, df: pd.DataFrame,
                                  period: int = 20,
                                  std_dev: float = 2.0,
                                  lookahead: int = 10) -> EdgeResult:
        """
        Test Bollinger Band mean reversion edge.
        
        Tests whether touching outer bands predicts reversion.
        
        Args:
            df: DataFrame with OHLC data
            period: BB period
            std_dev: Standard deviation multiplier
            lookahead: Forward period for returns
            
        Returns:
            EdgeResult with BB reversion analysis
        """
        # Bollinger Bands
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        
        # %B indicator
        percent_b = (df['close'] - lower) / (upper - lower)
        
        # Forward returns
        fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Extreme conditions
        lower_touch = df['close'] <= lower  # Expect reversion up
        upper_touch = df['close'] >= upper  # Expect reversion down
        
        # Returns
        lower_returns = fwd_returns[lower_touch].dropna()
        upper_returns = -fwd_returns[upper_touch].dropna()
        
        all_returns = pd.concat([lower_returns, upper_returns])
        
        if len(all_returns) < 30:
            return None
        
        t_stat, p_val = stats.ttest_1samp(all_returns, 0)
        
        mean_ret = all_returns.mean()
        std_ret = all_returns.std()
        win_rate = (all_returns > 0).mean()
        sharpe = mean_ret / std_ret * np.sqrt(252 / lookahead) if std_ret > 0 else 0
        
        return EdgeResult(
            name=f"BB Reversion ({period}, {std_dev}σ)",
            edge_type="mean_reversion",
            sample_size=len(all_returns),
            mean_return=mean_ret,
            std_return=std_ret,
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            is_significant=p_val < self.significance_level,
            details={
                'lower_touch_count': len(lower_returns),
                'upper_touch_count': len(upper_returns)
            }
        )
    
    def consecutive_candles_reversal(self, df: pd.DataFrame,
                                     min_consecutive: int = 3,
                                     max_consecutive: int = 7,
                                     lookahead: int = 5) -> EdgeResult:
        """
        Test consecutive candles reversal edge.
        
        Tests whether consecutive bullish/bearish candles
        predict reversals.
        
        Args:
            df: DataFrame with OHLC data
            min_consecutive: Minimum consecutive candles
            max_consecutive: Maximum consecutive candles
            lookahead: Forward period for returns
            
        Returns:
            EdgeResult with reversal analysis
        """
        # Direction
        bullish = df['close'] > df['open']
        
        # Count consecutive
        def count_consecutive(series):
            groups = (series != series.shift(1)).cumsum()
            return series.groupby(groups).cumsum()
        
        consecutive_bullish = count_consecutive(bullish)
        consecutive_bearish = count_consecutive(~bullish)
        
        # Forward returns
        fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Reversal conditions
        overextended_bullish = (consecutive_bullish >= min_consecutive) & \
                               (consecutive_bullish <= max_consecutive)
        overextended_bearish = (consecutive_bearish >= min_consecutive) & \
                               (consecutive_bearish <= max_consecutive)
        
        # Returns (expect reversal)
        bullish_reversal_returns = -fwd_returns[overextended_bullish].dropna()
        bearish_reversal_returns = fwd_returns[overextended_bearish].dropna()
        
        all_returns = pd.concat([bullish_reversal_returns, bearish_reversal_returns])
        
        if len(all_returns) < 30:
            return None
        
        t_stat, p_val = stats.ttest_1samp(all_returns, 0)
        
        mean_ret = all_returns.mean()
        std_ret = all_returns.std()
        win_rate = (all_returns > 0).mean()
        sharpe = mean_ret / std_ret * np.sqrt(252 / lookahead) if std_ret > 0 else 0
        
        return EdgeResult(
            name=f"Consecutive Reversal ({min_consecutive}-{max_consecutive})",
            edge_type="mean_reversion",
            sample_size=len(all_returns),
            mean_return=mean_ret,
            std_return=std_ret,
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            is_significant=p_val < self.significance_level,
            details={
                'bullish_reversal_count': len(bullish_reversal_returns),
                'bearish_reversal_count': len(bearish_reversal_returns)
            }
        )
    
    def vwap_deviation(self, df: pd.DataFrame,
                       deviation_threshold: float = 0.002,
                       lookahead: int = 10) -> EdgeResult:
        """
        Test VWAP deviation mean reversion.
        
        Tests whether deviation from VWAP predicts reversion.
        
        Args:
            df: DataFrame with OHLCV data
            deviation_threshold: Minimum deviation from VWAP
            lookahead: Forward period for returns
            
        Returns:
            EdgeResult with VWAP deviation analysis
        """
        # Calculate VWAP if not present
        if 'vwap' not in df.columns:
            from features.volume import VolumeFeatures
            df = df.copy()
            df['vwap'] = VolumeFeatures.vwap(df)
        
        # Deviation from VWAP
        deviation = (df['close'] - df['vwap']) / df['vwap']
        
        # Forward returns
        fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Extreme conditions
        above_vwap = deviation > deviation_threshold  # Expect reversion down
        below_vwap = deviation < -deviation_threshold  # Expect reversion up
        
        # Returns
        above_returns = -fwd_returns[above_vwap].dropna()
        below_returns = fwd_returns[below_vwap].dropna()
        
        all_returns = pd.concat([above_returns, below_returns])
        
        if len(all_returns) < 30:
            return None
        
        t_stat, p_val = stats.ttest_1samp(all_returns, 0)
        
        mean_ret = all_returns.mean()
        std_ret = all_returns.std()
        win_rate = (all_returns > 0).mean()
        sharpe = mean_ret / std_ret * np.sqrt(252 / lookahead) if std_ret > 0 else 0
        
        return EdgeResult(
            name=f"VWAP Deviation ({deviation_threshold*100:.1f}%)",
            edge_type="mean_reversion",
            sample_size=len(all_returns),
            mean_return=mean_ret,
            std_return=std_ret,
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            is_significant=p_val < self.significance_level,
            details={
                'above_count': len(above_returns),
                'below_count': len(below_returns)
            }
        )
    
    def run_all_mr_edges(self, df: pd.DataFrame,
                         lookahead: int = 10) -> Dict[str, EdgeResult]:
        """
        Run all mean reversion edge tests.
        
        Args:
            df: DataFrame with OHLCV data
            lookahead: Forward period for returns
            
        Returns:
            Dictionary with all edge results
        """
        all_results = {}
        
        # Z-score mean reversion
        for lookback in [10, 20, 50]:
            for threshold in [1.5, 2.0, 2.5]:
                result = self.zscore_mean_reversion(df, lookback, threshold, lookahead)
                if result:
                    all_results[f'zscore_{lookback}_{threshold}'] = result
        
        # Z-score quintile
        for lookback in [20, 50]:
            result = self.zscore_quintile_analysis(df, lookback, lookahead)
            if result:
                all_results[f'zscore_quintile_{lookback}'] = result
        
        # RSI extremes
        for oversold, overbought in [(30, 70), (25, 75), (20, 80)]:
            result = self.rsi_extremes(df, oversold, overbought, lookahead=lookahead)
            if result:
                all_results[f'rsi_{oversold}_{overbought}'] = result
        
        # Stochastic extremes
        result = self.stochastic_extremes(df, lookahead=lookahead)
        if result:
            all_results['stoch_extremes'] = result
        
        # Bollinger reversion
        result = self.bollinger_band_reversion(df, lookahead=lookahead)
        if result:
            all_results['bb_reversion'] = result
        
        # Consecutive candles
        for min_c in [3, 4, 5]:
            result = self.consecutive_candles_reversal(df, min_c, min_c + 4, lookahead)
            if result:
                all_results[f'consecutive_{min_c}'] = result
        
        # VWAP deviation
        if 'tick_volume' in df.columns or 'real_volume' in df.columns:
            result = self.vwap_deviation(df, lookahead=lookahead)
            if result:
                all_results['vwap_deviation'] = result
        
        return all_results
