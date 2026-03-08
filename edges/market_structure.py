"""
Market Structure Edge Detection Module.

Identifies edges based on market structure:
- Support/Resistance levels
- Breakout patterns
- Higher timeframe context
- Swing highs/lows
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass

from .time_based import EdgeResult


class MarketStructureEdges:
    """
    Detect market structure-based trading edges.
    
    Market structure edges arise from the interaction of
    price with key levels and structural patterns.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize market structure edge detector.
        
        Args:
            significance_level: P-value threshold for significance
        """
        self.significance_level = significance_level
    
    def round_number_effect(self, df: pd.DataFrame,
                            pip_distance: int = 50,
                            lookahead: int = 10) -> EdgeResult:
        """
        Test round number support/resistance effect.
        
        Tests whether price tends to bounce at round numbers.
        
        Args:
            df: DataFrame with price data
            pip_distance: Distance to round number (in pips)
            lookahead: Forward period for returns
            
        Returns:
            EdgeResult with round number analysis
        """
        # Assume 5-digit pricing (0.00001 = 1 pip)
        pip = 0.0001  # Standard pip for EURUSD
        
        # Find round numbers
        def nearest_round(price, distance):
            return round(price / (distance * pip)) * (distance * pip)
        
        # Calculate distance to nearest round number
        df = df.copy()
        df['nearest_round'] = df['close'].apply(lambda x: nearest_round(x, pip_distance))
        df['dist_to_round'] = abs(df['close'] - df['nearest_round']) / df['close']
        
        # Near round number condition
        near_round = df['dist_to_round'] < pip_distance * pip * 0.5 / df['close']
        
        # Forward returns
        fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Compare returns near vs away from round numbers
        near_returns = fwd_returns[near_round].dropna().abs()
        away_returns = fwd_returns[~near_round].dropna().abs()
        
        if len(near_returns) < 30:
            return None
        
        # Test for reduced volatility near round numbers
        t_stat, p_val = stats.ttest_ind(near_returns, away_returns)
        
        # Test for reversal tendency
        # Check if price moves away from round number
        above_round = df['close'] > df['nearest_round']
        below_round = df['close'] < df['nearest_round']
        
        near_above = near_round & above_round
        near_below = near_round & below_round
        
        above_returns = fwd_returns[near_above].dropna()
        below_returns = fwd_returns[near_below].dropna()
        
        # Expect reversal: above round -> negative, below round -> positive
        reversal_returns = pd.concat([
            -above_returns,  # Expect downward
            below_returns    # Expect upward
        ])
        
        reversal_t, reversal_p = stats.ttest_1samp(reversal_returns, 0)
        
        return EdgeResult(
            name=f"Round Number ({pip_distance} pips)",
            edge_type="structure",
            sample_size=len(near_returns),
            mean_return=reversal_returns.mean(),
            std_return=reversal_returns.std(),
            t_statistic=reversal_t,
            p_value=reversal_p,
            win_rate=(reversal_returns > 0).mean(),
            sharpe_ratio=reversal_returns.mean() / reversal_returns.std() * np.sqrt(252 / lookahead) if reversal_returns.std() > 0 else 0,
            is_significant=reversal_p < self.significance_level,
            details={
                'near_round_count': len(near_returns),
                'reversal_win_rate': (reversal_returns > 0).mean(),
                'avg_volatility_near': near_returns.mean(),
                'avg_volatility_away': away_returns.mean()
            }
        )
    
    def previous_day_levels(self, df: pd.DataFrame,
                           lookahead: int = 10) -> Dict[str, EdgeResult]:
        """
        Test previous day high/low/close as support/resistance.
        
        Args:
            df: DataFrame with OHLC data
            lookahead: Forward period for returns
            
        Returns:
            Dictionary with level results
        """
        df = df.copy()
        
        # Calculate daily levels
        df['date'] = df.index.date
        daily = df.groupby('date').agg({
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).shift(1)
        
        daily.columns = ['prev_high', 'prev_low', 'prev_close']
        
        # Merge back to intraday
        df = df.merge(daily, left_on='date', right_index=True, how='left')
        
        results = {}
        
        for level_name, level_col in [('high', 'prev_high'), ('low', 'prev_low'), ('close', 'prev_close')]:
            # Distance to level
            df[f'dist_{level_col}'] = abs(df['close'] - df[level_col]) / df['close']
            
            # Near level condition (within 0.1%)
            near_level = df[f'dist_{level_col}'] < 0.001
            
            # Forward returns
            fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
            
            near_returns = fwd_returns[near_level].dropna()
            
            if len(near_returns) < 30:
                results[f'prev_{level_name}'] = None
                continue
            
            # Test for reversal
            t_stat, p_val = stats.ttest_1samp(near_returns.abs(), fwd_returns.dropna().abs().mean())
            
            results[f'prev_{level_name}'] = EdgeResult(
                name=f"Previous Day {level_name.title()}",
                edge_type="structure",
                sample_size=len(near_returns),
                mean_return=near_returns.mean(),
                std_return=near_returns.std(),
                t_statistic=t_stat,
                p_value=p_val,
                win_rate=(near_returns > 0).mean(),
                sharpe_ratio=near_returns.mean() / near_returns.std() * np.sqrt(252 / lookahead) if near_returns.std() > 0 else 0,
                is_significant=p_val < self.significance_level,
                details={'level': level_name}
            )
        
        return results
    
    def swing_levels(self, df: pd.DataFrame,
                     swing_period: int = 5,
                     lookahead: int = 10) -> EdgeResult:
        """
        Test swing high/low levels as support/resistance.
        
        Args:
            df: DataFrame with OHLC data
            swing_period: Bars on each side to define swing
            lookahead: Forward period for returns
            
        Returns:
            EdgeResult with swing level analysis
        """
        # Find swing highs and lows
        df = df.copy()
        
        # Swing high: high is highest of surrounding bars
        df['swing_high'] = (
            (df['high'] == df['high'].rolling(2*swing_period+1, center=True).max())
        ).astype(int)
        
        # Swing low: low is lowest of surrounding bars
        df['swing_low'] = (
            (df['low'] == df['low'].rolling(2*swing_period+1, center=True).min())
        ).astype(int)
        
        # Create level columns
        df['swing_high_level'] = df['high'].where(df['swing_high'] == 1)
        df['swing_low_level'] = df['low'].where(df['swing_low'] == 1)
        
        # Forward fill levels
        df['nearest_swing_high'] = df['swing_high_level'].ffill()
        df['nearest_swing_low'] = df['swing_low_level'].ffill()
        
        # Distance to swing levels
        df['dist_swing_high'] = abs(df['close'] - df['nearest_swing_high']) / df['close']
        df['dist_swing_low'] = abs(df['close'] - df['nearest_swing_low']) / df['close']
        
        # Near swing level condition
        near_swing_high = df['dist_swing_high'] < 0.001
        near_swing_low = df['dist_swing_low'] < 0.001
        
        # Forward returns
        fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Analyze reversals
        high_reversal = -fwd_returns[near_swing_high].dropna()  # Expect downward
        low_reversal = fwd_returns[near_swing_low].dropna()     # Expect upward
        
        all_returns = pd.concat([high_reversal, low_reversal])
        
        if len(all_returns) < 30:
            return None
        
        t_stat, p_val = stats.ttest_1samp(all_returns, 0)
        
        mean_ret = all_returns.mean()
        std_ret = all_returns.std()
        win_rate = (all_returns > 0).mean()
        sharpe = mean_ret / std_ret * np.sqrt(252 / lookahead) if std_ret > 0 else 0
        
        return EdgeResult(
            name=f"Swing Levels ({swing_period})",
            edge_type="structure",
            sample_size=len(all_returns),
            mean_return=mean_ret,
            std_return=std_ret,
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            is_significant=p_val < self.significance_level,
            details={
                'swing_high_tests': len(high_reversal),
                'swing_low_tests': len(low_reversal),
                'swing_high_win_rate': (high_reversal > 0).mean() if len(high_reversal) > 0 else 0,
                'swing_low_win_rate': (low_reversal > 0).mean() if len(low_reversal) > 0 else 0
            }
        )
    
    def breakout_retest(self, df: pd.DataFrame,
                        lookback: int = 20,
                        lookahead: int = 10) -> EdgeResult:
        """
        Test breakout retest pattern.
        
        Tests whether price tends to retest broken levels
        and continue in breakout direction.
        
        Args:
            df: DataFrame with OHLC data
            lookback: Lookback for breakout
            lookahead: Forward period for returns
            
        Returns:
            EdgeResult with retest analysis
        """
        # Find breakouts
        high_level = df['high'].rolling(lookback).max().shift(1)
        low_level = df['low'].rolling(lookback).min().shift(1)
        
        breakout_up = df['close'] > high_level.shift(1)
        breakout_down = df['close'] < low_level.shift(1)
        
        # Track if we're near a recently broken level
        df = df.copy()
        df['broken_high'] = np.where(breakout_up, high_level, np.nan)
        df['broken_low'] = np.where(breakout_down, low_level, np.nan)
        
        df['broken_high'] = df['broken_high'].ffill()
        df['broken_low'] = df['broken_low'].ffill()
        
        # Retest condition
        retest_high = (
            abs(df['close'] - df['broken_high']) / df['close'] < 0.002
        ) & (df['close'] > df['broken_high'])
        
        retest_low = (
            abs(df['close'] - df['broken_low']) / df['close'] < 0.002
        ) & (df['close'] < df['broken_low'])
        
        # Forward returns
        fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Expect continuation after retest
        high_retest_returns = fwd_returns[retest_high].dropna()  # Expect continuation up
        low_retest_returns = -fwd_returns[retest_low].dropna()   # Expect continuation down
        
        all_returns = pd.concat([high_retest_returns, low_retest_returns])
        
        if len(all_returns) < 30:
            return None
        
        t_stat, p_val = stats.ttest_1samp(all_returns, 0)
        
        mean_ret = all_returns.mean()
        std_ret = all_returns.std()
        win_rate = (all_returns > 0).mean()
        sharpe = mean_ret / std_ret * np.sqrt(252 / lookahead) if std_ret > 0 else 0
        
        return EdgeResult(
            name=f"Breakout Retest ({lookback})",
            edge_type="structure",
            sample_size=len(all_returns),
            mean_return=mean_ret,
            std_return=std_ret,
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            is_significant=p_val < self.significance_level,
            details={
                'high_retests': len(high_retest_returns),
                'low_retests': len(low_retest_returns)
            }
        )
    
    def failed_breakout(self, df: pd.DataFrame,
                        lookback: int = 20,
                        failure_threshold: float = 0.002,
                        lookahead: int = 10) -> EdgeResult:
        """
        Test failed breakout reversal pattern.
        
        Tests whether failed breakouts (price breaks level then
        reverses back) predict strong counter-moves.
        
        Args:
            df: DataFrame with OHLC data
            lookback: Lookback for breakout level
            failure_threshold: How far back price must come
            lookahead: Forward period for returns
            
        Returns:
            EdgeResult with failed breakout analysis
        """
        # Find breakouts
        high_level = df['high'].rolling(lookback).max().shift(1)
        low_level = df['low'].rolling(lookback).min().shift(1)
        
        breakout_up = df['high'] > high_level
        breakout_down = df['low'] < low_level
        
        # Failed breakout: price breaks but closes back inside
        failed_up = breakout_up & (df['close'] < high_level)
        failed_down = breakout_down & (df['close'] > low_level)
        
        # Forward returns
        fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Expect reversal after failed breakout
        failed_up_returns = -fwd_returns[failed_up].dropna()  # Expect down
        failed_down_returns = fwd_returns[failed_down].dropna()  # Expect up
        
        all_returns = pd.concat([failed_up_returns, failed_down_returns])
        
        if len(all_returns) < 30:
            return None
        
        t_stat, p_val = stats.ttest_1samp(all_returns, 0)
        
        mean_ret = all_returns.mean()
        std_ret = all_returns.std()
        win_rate = (all_returns > 0).mean()
        sharpe = mean_ret / std_ret * np.sqrt(252 / lookahead) if std_ret > 0 else 0
        
        return EdgeResult(
            name=f"Failed Breakout ({lookback})",
            edge_type="structure",
            sample_size=len(all_returns),
            mean_return=mean_ret,
            std_return=std_ret,
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            is_significant=p_val < self.significance_level,
            details={
                'failed_up_breakouts': len(failed_up_returns),
                'failed_down_breakouts': len(failed_down_returns)
            }
        )
    
    def higher_highs_lows_trend(self, df: pd.DataFrame,
                                swing_period: int = 5,
                                lookahead: int = 10) -> EdgeResult:
        """
        Test higher highs/lows trend continuation.
        
        Tests whether higher highs and higher lows pattern
        predicts trend continuation.
        
        Args:
            df: DataFrame with OHLC data
            swing_period: Period for swing detection
            lookahead: Forward period for returns
            
        Returns:
            EdgeResult with trend analysis
        """
        # Find swing points
        df = df.copy()
        
        df['swing_high'] = df['high'].rolling(2*swing_period+1, center=True).max() == df['high']
        df['swing_low'] = df['low'].rolling(2*swing_period+1, center=True).min() == df['low']
        
        # Get swing values
        swing_highs = df['high'].where(df['swing_high'])
        swing_lows = df['low'].where(df['swing_low'])
        
        # Higher highs and higher lows
        df['prev_swing_high'] = swing_highs.ffill().shift(1)
        df['prev_swing_low'] = swing_lows.ffill().shift(1)
        
        higher_high = (df['high'] > df['prev_swing_high']) & df['swing_high']
        higher_low = (df['low'] > df['prev_swing_low']) & df['swing_low']
        
        lower_high = (df['high'] < df['prev_swing_high']) & df['swing_high']
        lower_low = (df['low'] < df['prev_swing_low']) & df['swing_low']
        
        # Forward returns
        fwd_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Uptrend: higher highs + higher lows -> expect continuation up
        uptrend = higher_high | higher_low
        # Downtrend: lower highs + lower lows -> expect continuation down
        downtrend = lower_high | lower_low
        
        uptrend_returns = fwd_returns[uptrend].dropna()
        downtrend_returns = -fwd_returns[downtrend].dropna()
        
        all_returns = pd.concat([uptrend_returns, downtrend_returns])
        
        if len(all_returns) < 30:
            return None
        
        t_stat, p_val = stats.ttest_1samp(all_returns, 0)
        
        mean_ret = all_returns.mean()
        std_ret = all_returns.std()
        win_rate = (all_returns > 0).mean()
        sharpe = mean_ret / std_ret * np.sqrt(252 / lookahead) if std_ret > 0 else 0
        
        return EdgeResult(
            name=f"HH/HL Trend ({swing_period})",
            edge_type="structure",
            sample_size=len(all_returns),
            mean_return=mean_ret,
            std_return=std_ret,
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            is_significant=p_val < self.significance_level,
            details={
                'uptrend_signals': len(uptrend_returns),
                'downtrend_signals': len(downtrend_returns)
            }
        )
    
    def run_all_structure_edges(self, df: pd.DataFrame,
                                lookahead: int = 10) -> Dict[str, EdgeResult]:
        """
        Run all market structure edge tests.
        
        Args:
            df: DataFrame with OHLCV data
            lookahead: Forward period for returns
            
        Returns:
            Dictionary with all edge results
        """
        all_results = {}
        
        # Round numbers
        for pips in [25, 50, 100]:
            result = self.round_number_effect(df, pips, lookahead)
            if result:
                all_results[f'round_{pips}'] = result
        
        # Previous day levels
        prev_results = self.previous_day_levels(df, lookahead)
        all_results.update({f'prev_{k}': v for k, v in prev_results.items() if v})
        
        # Swing levels
        for period in [3, 5, 10]:
            result = self.swing_levels(df, period, lookahead)
            if result:
                all_results[f'swing_{period}'] = result
        
        # Breakout retest
        result = self.breakout_retest(df, lookahead=lookahead)
        if result:
            all_results['breakout_retest'] = result
        
        # Failed breakout
        result = self.failed_breakout(df, lookahead=lookahead)
        if result:
            all_results['failed_breakout'] = result
        
        # HH/HL trend
        result = self.higher_highs_lows_trend(df, lookahead=lookahead)
        if result:
            all_results['hh_hl_trend'] = result
        
        return all_results
