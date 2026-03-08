"""
Trend and Momentum Edge Detection Module.

Identifies edges based on trend and momentum:
- Moving average crossovers
- Price momentum
- Breakout patterns
- Trend strength filters
- Multi-timeframe momentum
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass

from .time_based import EdgeResult


class TrendMomentumEdges:
    """
    Detect trend and momentum-based trading edges.
    
    Trend and momentum edges arise from the tendency of prices
    to continue moving in the same direction (momentum) or
    the tendency of trends to persist (trend following).
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize trend/momentum edge detector.
        
        Args:
            significance_level: P-value threshold for significance
        """
        self.significance_level = significance_level
    
    def moving_average_crossover(self, df: pd.DataFrame,
                                 fast_period: int = 50,
                                 slow_period: int = 200,
                                 lookahead: int = 20) -> EdgeResult:
        """
        Test moving average crossover edge.
        
        Tests whether golden/death crosses predict
        subsequent price movements.
        
        Args:
            df: DataFrame with OHLC data
            fast_period: Fast MA period
            slow_period: Slow MA period
            lookahead: Bars to measure forward returns
            
        Returns:
            EdgeResult with crossover analysis
        """
        # Calculate MAs
        fast_ma = df['close'].rolling(fast_period).mean()
        slow_ma = df['close'].rolling(slow_period).mean()
        
        # Find crossovers
        trend = fast_ma > slow_ma
        golden_cross = trend & ~trend.shift(1).fillna(False)
        death_cross = ~trend & trend.shift(1).fillna(False)
        
        # Calculate forward returns
        fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Golden cross returns (expect positive)
        golden_returns = fwd_returns[golden_cross].dropna()
        
        # Death cross returns (expect negative, so invert)
        death_returns = -fwd_returns[death_cross].dropna()
        
        # Combine
        all_returns = pd.concat([golden_returns, death_returns])
        
        if len(all_returns) < 30:
            return None
        
        t_stat, p_val = stats.ttest_1samp(all_returns, 0)
        
        mean_ret = all_returns.mean()
        std_ret = all_returns.std()
        win_rate = (all_returns > 0).mean()
        sharpe = mean_ret / std_ret * np.sqrt(252 / lookahead) if std_ret > 0 else 0
        
        return EdgeResult(
            name=f"MA Crossover ({fast_period}/{slow_period})",
            edge_type="trend_crossover",
            sample_size=len(all_returns),
            mean_return=mean_ret,
            std_return=std_ret,
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            is_significant=p_val < self.significance_level,
            details={
                'golden_crosses': len(golden_returns),
                'death_crosses': len(death_returns),
                'golden_win_rate': (golden_returns > 0).mean() if len(golden_returns) > 0 else 0,
                'death_win_rate': (death_returns > 0).mean() if len(death_returns) > 0 else 0
            }
        )
    
    def n_bar_momentum(self, df: pd.DataFrame,
                       lookback: int = 20,
                       lookahead: int = 10,
                       quintile_analysis: bool = True) -> EdgeResult:
        """
        Test N-bar momentum edge.
        
        Tests whether past returns predict future returns
        (momentum continuation vs mean reversion).
        
        Args:
            df: DataFrame with price data
            lookback: Lookback period for momentum
            lookahead: Forward period for returns
            quintile_analysis: Whether to do quintile analysis
            
        Returns:
            EdgeResult with momentum analysis
        """
        # Calculate momentum
        momentum = df['close'] / df['close'].shift(lookback) - 1
        
        # Forward returns
        fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Create analysis DataFrame
        analysis_df = pd.DataFrame({
            'momentum': momentum,
            'fwd_return': fwd_returns
        }).dropna()
        
        if len(analysis_df) < 100:
            return None
        
        # Correlation between momentum and forward returns
        corr, p_val = stats.pearsonr(analysis_df['momentum'], analysis_df['fwd_return'])
        
        # Quintile analysis
        quintile_results = {}
        if quintile_analysis:
            analysis_df['momentum_quintile'] = pd.qcut(analysis_df['momentum'], 5, labels=False)
            
            for q in range(5):
                q_returns = analysis_df[analysis_df['momentum_quintile'] == q]['fwd_return']
                quintile_results[f'Q{q+1}'] = {
                    'mean': q_returns.mean(),
                    'win_rate': (q_returns > 0).mean(),
                    'count': len(q_returns)
                }
        
        # Test top vs bottom quintile
        top_quintile = analysis_df[analysis_df['momentum'] >= analysis_df['momentum'].quantile(0.8)]['fwd_return']
        bottom_quintile = analysis_df[analysis_df['momentum'] <= analysis_df['momentum'].quantile(0.2)]['fwd_return']
        
        # Momentum strategy: go long top quintile (expect continuation)
        momentum_returns = top_quintile
        
        t_stat, t_pval = stats.ttest_1samp(momentum_returns, 0)
        
        mean_ret = momentum_returns.mean()
        std_ret = momentum_returns.std()
        win_rate = (momentum_returns > 0).mean()
        sharpe = mean_ret / std_ret * np.sqrt(252 / lookahead) if std_ret > 0 else 0
        
        return EdgeResult(
            name=f"{lookback}-Bar Momentum",
            edge_type="momentum",
            sample_size=len(momentum_returns),
            mean_return=mean_ret,
            std_return=std_ret,
            t_statistic=t_stat,
            p_value=t_pval,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            is_significant=t_pval < self.significance_level and corr > 0,
            details={
                'autocorrelation': corr,
                'autocorr_p_value': p_val,
                'quintiles': quintile_results,
                'top_win_rate': (top_quintile > 0).mean(),
                'bottom_win_rate': (bottom_quintile > 0).mean()
            }
        )
    
    def hurst_exponent(self, df: pd.DataFrame,
                       max_lag: int = 20) -> Tuple[float, EdgeResult]:
        """
        Calculate Hurst exponent for trend/mean-reversion detection.
        
        H > 0.5: Trending (momentum works)
        H = 0.5: Random walk
        H < 0.5: Mean-reverting
        
        Args:
            df: DataFrame with price data
            max_lag: Maximum lag for calculation
            
        Returns:
            Tuple of (Hurst value, EdgeResult)
        """
        prices = df['close'].values
        log_returns = np.log(prices[1:] / prices[:-1])
        
        # Calculate variance for different lags
        lags = range(2, max_lag + 1)
        tau = [np.std(np.subtract(log_returns[lag:], log_returns[:-lag])) 
               for lag in lags]
        
        # Linear regression of log(tau) vs log(lag)
        log_lags = np.log(lags)
        log_tau = np.log(tau)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_tau)
        
        hurst = slope
        
        # Determine regime
        if hurst > 0.55:
            regime = "trending"
            edge_direction = "momentum"
        elif hurst < 0.45:
            regime = "mean_reverting"
            edge_direction = "mean_reversion"
        else:
            regime = "random"
            edge_direction = "none"
        
        return hurst, EdgeResult(
            name="Hurst Exponent",
            edge_type="regime",
            sample_size=len(log_returns),
            mean_return=hurst - 0.5,  # Deviation from random
            std_return=std_err,
            t_statistic=(hurst - 0.5) / std_err if std_err > 0 else 0,
            p_value=p_value,
            win_rate=0,  # Not applicable
            sharpe_ratio=0,
            is_significant=abs(hurst - 0.5) > 0.05,
            details={
                'hurst': hurst,
                'regime': regime,
                'edge_direction': edge_direction,
                'r_squared': r_value ** 2
            }
        )
    
    def donchian_breakout(self, df: pd.DataFrame,
                          lookback: int = 20,
                          lookahead: int = 10) -> EdgeResult:
        """
        Test Donchian channel breakout edge.
        
        Tests whether breakouts above/below N-bar highs/lows
        predict subsequent price movements.
        
        Args:
            df: DataFrame with OHLC data
            lookback: Lookback period for channel
            lookahead: Forward period for returns
            
        Returns:
            EdgeResult with breakout analysis
        """
        # Donchian channels
        upper = df['high'].rolling(lookback).max()
        lower = df['low'].rolling(lookback).min()
        
        # Breakouts
        breakout_up = df['close'] > upper.shift(1)
        breakout_down = df['close'] < lower.shift(1)
        
        # Forward returns
        fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Up breakout returns (expect continuation)
        up_returns = fwd_returns[breakout_up].dropna()
        
        # Down breakout returns (expect continuation, so invert)
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
            name=f"Donchian Breakout ({lookback})",
            edge_type="breakout",
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
                'down_breakouts': len(down_returns),
                'up_win_rate': (up_returns > 0).mean() if len(up_returns) > 0 else 0,
                'down_win_rate': (down_returns > 0).mean() if len(down_returns) > 0 else 0
            }
        )
    
    def bollinger_breakout(self, df: pd.DataFrame,
                           period: int = 20,
                           std_dev: float = 2.0,
                           lookahead: int = 10) -> EdgeResult:
        """
        Test Bollinger Band breakout edge.
        
        Tests whether breakouts from Bollinger Bands
        predict subsequent price movements.
        
        Args:
            df: DataFrame with OHLC data
            period: BB period
            std_dev: Standard deviation multiplier
            lookahead: Forward period for returns
            
        Returns:
            EdgeResult with BB breakout analysis
        """
        # Bollinger Bands
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        
        # Breakouts
        breakout_up = df['close'] > upper
        breakout_down = df['close'] < lower
        
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
            name=f"BB Breakout ({period}, {std_dev}σ)",
            edge_type="breakout",
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
    
    def adx_filter(self, df: pd.DataFrame,
                   adx_threshold: int = 25,
                   lookahead: int = 10) -> Tuple[EdgeResult, EdgeResult]:
        """
        Test ADX trend filter edge.
        
        Tests whether ADX can identify trending vs ranging
        conditions for strategy selection.
        
        Args:
            df: DataFrame with ADX calculated
            adx_threshold: ADX threshold for trend
            lookahead: Forward period for returns
            
        Returns:
            Tuple of (trending EdgeResult, ranging EdgeResult)
        """
        if 'adx' not in df.columns:
            # Calculate ADX if not present
            from features.trend import TrendFeatures
            adx_df = TrendFeatures.adx(df)
            df = df.copy()
            df['adx'] = adx_df['adx']
        
        # Forward returns
        fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Trending vs Ranging
        trending = df['adx'] > adx_threshold
        ranging = df['adx'] < 20
        
        # Absolute returns in each regime
        trending_abs_returns = fwd_returns[trending].abs().dropna()
        ranging_abs_returns = fwd_returns[ranging].abs().dropna()
        
        # For trending: test if returns are more directional
        trending_returns = fwd_returns[trending].dropna()
        trending_directional = trending_returns.abs() * np.sign(trending_returns.mean())
        
        # Trending result
        if len(trending_returns) >= 30:
            t_stat_t, p_val_t = stats.ttest_1samp(trending_returns, 0)
            mean_ret_t = trending_returns.mean()
            std_ret_t = trending_returns.std()
            sharpe_t = mean_ret_t / std_ret_t * np.sqrt(252 / lookahead) if std_ret_t > 0 else 0
        else:
            t_stat_t, p_val_t, mean_ret_t, std_ret_t, sharpe_t = 0, 1, 0, 0, 0
        
        trending_result = EdgeResult(
            name=f"Trending (ADX>{adx_threshold})",
            edge_type="trend_filter",
            sample_size=len(trending_returns),
            mean_return=mean_ret_t,
            std_return=std_ret_t,
            t_statistic=t_stat_t,
            p_value=p_val_t,
            win_rate=(trending_returns > 0).mean() if len(trending_returns) > 0 else 0.5,
            sharpe_ratio=sharpe_t,
            is_significant=p_val_t < self.significance_level,
            details={'avg_adx': df.loc[trending, 'adx'].mean()}
        )
        
        # Ranging result
        ranging_returns = fwd_returns[ranging].dropna()
        
        if len(ranging_returns) >= 30:
            t_stat_r, p_val_r = stats.ttest_1samp(ranging_returns, 0)
            mean_ret_r = ranging_returns.mean()
            std_ret_r = ranging_returns.std()
            sharpe_r = mean_ret_r / std_ret_r * np.sqrt(252 / lookahead) if std_ret_r > 0 else 0
        else:
            t_stat_r, p_val_r, mean_ret_r, std_ret_r, sharpe_r = 0, 1, 0, 0, 0
        
        ranging_result = EdgeResult(
            name="Ranging (ADX<20)",
            edge_type="trend_filter",
            sample_size=len(ranging_returns),
            mean_return=mean_ret_r,
            std_return=std_ret_r,
            t_statistic=t_stat_r,
            p_value=p_val_r,
            win_rate=(ranging_returns > 0).mean() if len(ranging_returns) > 0 else 0.5,
            sharpe_ratio=sharpe_r,
            is_significant=p_val_r < self.significance_level,
            details={'avg_adx': df.loc[ranging, 'adx'].mean()}
        )
        
        return trending_result, ranging_result
    
    def runs_test(self, df: pd.DataFrame) -> EdgeResult:
        """
        Perform Wald-Wolfowitz runs test for randomness.
        
        Tests whether price changes are random or show
        non-random patterns (indicating momentum or mean-reversion).
        
        Args:
            df: DataFrame with price data
            
        Returns:
            EdgeResult with runs test results
        """
        returns = df['close'].pct_change().dropna()
        
        # Convert to signs
        signs = np.sign(returns)
        signs = signs[signs != 0]
        
        # Count runs
        runs = 1
        for i in range(1, len(signs)):
            if signs.iloc[i] != signs.iloc[i-1]:
                runs += 1
        
        # Expected runs under randomness
        n_pos = (signs > 0).sum()
        n_neg = (signs < 0).sum()
        n = len(signs)
        
        expected_runs = (2 * n_pos * n_neg / n) + 1
        std_runs = np.sqrt(
            (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / 
            (n * n * (n - 1))
        )
        
        # Z-statistic
        z = (runs - expected_runs) / std_runs if std_runs > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Interpretation
        if z < -1.96:  # Too few runs = momentum
            pattern = "momentum"
        elif z > 1.96:  # Too many runs = mean-reversion
            pattern = "mean_reversion"
        else:
            pattern = "random"
        
        return EdgeResult(
            name="Runs Test",
            edge_type="randomness",
            sample_size=n,
            mean_return=z,
            std_return=std_runs,
            t_statistic=z,
            p_value=p_val,
            win_rate=0,
            sharpe_ratio=0,
            is_significant=abs(z) > 1.96,
            details={
                'runs': runs,
                'expected_runs': expected_runs,
                'pattern': pattern
            }
        )
    
    def run_all_trend_edges(self, df: pd.DataFrame,
                           lookahead: int = 10) -> Dict[str, EdgeResult]:
        """
        Run all trend/momentum edge tests.
        
        Args:
            df: DataFrame with OHLC data
            lookahead: Forward period for returns
            
        Returns:
            Dictionary with all edge results
        """
        all_results = {}
        
        # MA crossovers
        for fast, slow in [(10, 20), (20, 50), (50, 200)]:
            result = self.moving_average_crossover(df, fast, slow, lookahead)
            if result:
                all_results[f'ma_cross_{fast}_{slow}'] = result
        
        # Momentum
        for lookback in [5, 10, 20]:
            result = self.n_bar_momentum(df, lookback, lookahead)
            if result:
                all_results[f'momentum_{lookback}'] = result
        
        # Hurst
        hurst, result = self.hurst_exponent(df)
        all_results['hurst'] = result
        
        # Breakouts
        for lookback in [10, 20, 50]:
            result = self.donchian_breakout(df, lookback, lookahead)
            if result:
                all_results[f'donchian_{lookback}'] = result
        
        # Bollinger breakout
        result = self.bollinger_breakout(df, lookahead=lookahead)
        if result:
            all_results['bb_breakout'] = result
        
        # ADX filter
        trending, ranging = self.adx_filter(df, lookahead=lookahead)
        all_results['adx_trending'] = trending
        all_results['adx_ranging'] = ranging
        
        # Runs test
        all_results['runs_test'] = self.runs_test(df)
        
        return all_results
