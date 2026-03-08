"""
Time-Based Edge Detection Module.

Identifies edges based on temporal patterns:
- Hour-of-day effects
- Day-of-week effects
- Monthly/seasonal patterns
- Session-based patterns
- Intraday patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass


@dataclass
class EdgeResult:
    """Container for edge analysis results."""
    name: str
    edge_type: str
    sample_size: int
    mean_return: float
    std_return: float
    t_statistic: float
    p_value: float
    win_rate: float
    sharpe_ratio: float
    is_significant: bool
    details: Dict


class TimeBasedEdges:
    """
    Detect time-based trading edges.
    
    Time-based edges arise from recurring patterns in market
    behavior at specific times, which can be due to:
    - Market session overlaps
    - Economic data releases
    - Institutional trading schedules
    - Trader psychology patterns
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize time-based edge detector.
        
        Args:
            significance_level: P-value threshold for significance
        """
        self.significance_level = significance_level
    
    def hour_of_day_effect(self, df: pd.DataFrame,
                           return_column: str = 'return_1') -> Dict[int, EdgeResult]:
        """
        Analyze returns by hour of day.
        
        Tests whether returns at specific hours are significantly
        different from zero.
        
        Args:
            df: DataFrame with time index and returns
            return_column: Column name for returns
            
        Returns:
            Dictionary mapping hours to EdgeResult
        """
        results = {}
        
        if 'hour' not in df.columns:
            df = df.copy()
            df['hour'] = df.index.hour
        
        for hour in range(24):
            hour_data = df[df['hour'] == hour][return_column].dropna()
            
            if len(hour_data) < 30:
                continue
            
            # T-test against zero
            t_stat, p_val = stats.ttest_1samp(hour_data, 0)
            
            mean_ret = hour_data.mean()
            std_ret = hour_data.std()
            win_rate = (hour_data > 0).mean()
            
            # Annualized Sharpe (assuming hourly data)
            sharpe = (mean_ret * 24 * 252) / (std_ret * np.sqrt(24 * 252)) if std_ret > 0 else 0
            
            results[hour] = EdgeResult(
                name=f"Hour {hour:02d}:00",
                edge_type="time_hour",
                sample_size=len(hour_data),
                mean_return=mean_ret,
                std_return=std_ret,
                t_statistic=t_stat,
                p_value=p_val,
                win_rate=win_rate,
                sharpe_ratio=sharpe,
                is_significant=p_val < self.significance_level,
                details={'hour': hour}
            )
        
        return results
    
    def day_of_week_effect(self, df: pd.DataFrame,
                           return_column: str = 'return_1') -> Dict[int, EdgeResult]:
        """
        Analyze returns by day of week.
        
        Tests for day-specific patterns like Monday effect,
        Friday position squaring, etc.
        
        Args:
            df: DataFrame with time index and returns
            return_column: Column name for returns
            
        Returns:
            Dictionary mapping days to EdgeResult
        """
        results = {}
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        if 'day_of_week' not in df.columns:
            df = df.copy()
            df['day_of_week'] = df.index.dayofweek
        
        for day in range(5):  # Monday=0 to Friday=4
            day_data = df[df['day_of_week'] == day][return_column].dropna()
            
            if len(day_data) < 30:
                continue
            
            t_stat, p_val = stats.ttest_1samp(day_data, 0)
            
            mean_ret = day_data.mean()
            std_ret = day_data.std()
            win_rate = (day_data > 0).mean()
            
            sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
            
            results[day] = EdgeResult(
                name=day_names[day],
                edge_type="time_day",
                sample_size=len(day_data),
                mean_return=mean_ret,
                std_return=std_ret,
                t_statistic=t_stat,
                p_value=p_val,
                win_rate=win_rate,
                sharpe_ratio=sharpe,
                is_significant=p_val < self.significance_level,
                details={'day': day, 'day_name': day_names[day]}
            )
        
        return results
    
    def session_effect(self, df: pd.DataFrame,
                       return_column: str = 'return_1') -> Dict[str, EdgeResult]:
        """
        Analyze returns by trading session.
        
        Sessions defined in UTC:
        - Asian: 00:00-08:00
        - London: 07:00-16:00
        - New York: 12:00-21:00
        - London-NY Overlap: 12:00-16:00
        
        Args:
            df: DataFrame with time index and returns
            return_column: Column name for returns
            
        Returns:
            Dictionary mapping sessions to EdgeResult
        """
        results = {}
        
        sessions = {
            'asian': (0, 8),
            'london': (7, 16),
            'new_york': (12, 21),
            'london_ny_overlap': (12, 16),
            'asian_london_overlap': (7, 8)
        }
        
        if 'hour' not in df.columns:
            df = df.copy()
            df['hour'] = df.index.hour
        
        for session_name, (start_hour, end_hour) in sessions.items():
            mask = (df['hour'] >= start_hour) & (df['hour'] < end_hour)
            session_data = df[mask][return_column].dropna()
            
            if len(session_data) < 30:
                continue
            
            t_stat, p_val = stats.ttest_1samp(session_data, 0)
            
            mean_ret = session_data.mean()
            std_ret = session_data.std()
            win_rate = (session_data > 0).mean()
            
            sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
            
            results[session_name] = EdgeResult(
                name=session_name.replace('_', ' ').title(),
                edge_type="time_session",
                sample_size=len(session_data),
                mean_return=mean_ret,
                std_return=std_ret,
                t_statistic=t_stat,
                p_value=p_val,
                win_rate=win_rate,
                sharpe_ratio=sharpe,
                is_significant=p_val < self.significance_level,
                details={'session': session_name, 'hours': (start_hour, end_hour)}
            )
        
        return results
    
    def month_effect(self, df: pd.DataFrame,
                     return_column: str = 'return_1') -> Dict[int, EdgeResult]:
        """
        Analyze returns by month.
        
        Tests for seasonal patterns like January effect,
        month-end rebalancing, etc.
        
        Args:
            df: DataFrame with time index and returns
            return_column: Column name for returns
            
        Returns:
            Dictionary mapping months to EdgeResult
        """
        results = {}
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        if 'month' not in df.columns:
            df = df.copy()
            df['month'] = df.index.month
        
        for month in range(1, 13):
            month_data = df[df['month'] == month][return_column].dropna()
            
            if len(month_data) < 30:
                continue
            
            t_stat, p_val = stats.ttest_1samp(month_data, 0)
            
            mean_ret = month_data.mean()
            std_ret = month_data.std()
            win_rate = (month_data > 0).mean()
            
            sharpe = mean_ret / std_ret * np.sqrt(12) if std_ret > 0 else 0
            
            results[month] = EdgeResult(
                name=month_names[month-1],
                edge_type="time_month",
                sample_size=len(month_data),
                mean_return=mean_ret,
                std_return=std_ret,
                t_statistic=t_stat,
                p_value=p_val,
                win_rate=win_rate,
                sharpe_ratio=sharpe,
                is_significant=p_val < self.significance_level,
                details={'month': month, 'month_name': month_names[month-1]}
            )
        
        return results
    
    def opening_range_breakout(self, df: pd.DataFrame,
                               range_hours: int = 1,
                               lookahead_bars: int = 4) -> EdgeResult:
        """
        Test opening range breakout edge.
        
        Tests whether breakout from opening range predicts
        subsequent price movement.
        
        Args:
            df: DataFrame with OHLC data
            range_hours: Hours to define opening range
            lookahead_bars: Bars to measure forward returns
            
        Returns:
            EdgeResult with breakout analysis
        """
        if 'hour' not in df.columns:
            df = df.copy()
            df['hour'] = df.index.hour
        
        # Identify first bar of each day
        df['date'] = df.index.date
        first_bar = df.groupby('date')['hour'].transform('min')
        
        # Calculate opening range
        range_data = []
        
        for date in df['date'].unique():
            day_df = df[df['date'] == date]
            
            if len(day_df) < range_hours + lookahead_bars:
                continue
            
            # Opening range (first N hours)
            range_df = day_df.head(range_hours)
            range_high = range_df['high'].max()
            range_low = range_df['low'].min()
            
            # Subsequent bars
            subsequent = day_df.iloc[range_hours:]
            
            for i, (idx, row) in enumerate(subsequent.iterrows()):
                if i >= lookahead_bars:
                    break
                
                # Check for breakout
                breakout_up = row['close'] > range_high
                breakout_down = row['close'] < range_low
                
                # Forward return
                if i + lookahead_bars < len(subsequent):
                    fwd_return = subsequent.iloc[i + lookahead_bars]['close'] / row['close'] - 1
                    
                    range_data.append({
                        'date': date,
                        'breakout_up': breakout_up,
                        'breakout_down': breakout_down,
                        'fwd_return': fwd_return
                    })
        
        if not range_data:
            return None
        
        results_df = pd.DataFrame(range_data)
        
        # Analyze upward breakouts
        up_breakouts = results_df[results_df['breakout_up']]['fwd_return']
        down_breakouts = results_df[results_df['breakout_down']]['fwd_return']
        
        # Combine for analysis
        breakout_returns = pd.concat([
            up_breakouts,  # Expect continuation
            -down_breakouts  # Expect continuation (so invert)
        ])
        
        if len(breakout_returns) < 30:
            return None
        
        t_stat, p_val = stats.ttest_1samp(breakout_returns, 0)
        
        mean_ret = breakout_returns.mean()
        std_ret = breakout_returns.std()
        win_rate = (breakout_returns > 0).mean()
        sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
        
        return EdgeResult(
            name=f"Opening Range Breakout ({range_hours}h)",
            edge_type="time_pattern",
            sample_size=len(breakout_returns),
            mean_return=mean_ret,
            std_return=std_ret,
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            is_significant=p_val < self.significance_level,
            details={
                'up_breakouts': len(up_breakouts),
                'down_breakouts': len(down_breakouts),
                'up_win_rate': (up_breakouts > 0).mean(),
                'down_win_rate': (down_breakouts > 0).mean()
            }
        )
    
    def weekend_gap_analysis(self, df: pd.DataFrame,
                             timeframe: str = 'H1') -> EdgeResult:
        """
        Analyze weekend gap fill tendency.
        
        Tests whether price tends to fill weekend gaps.
        
        Args:
            df: DataFrame with OHLC data
            timeframe: Timeframe string
            
        Returns:
            EdgeResult with gap analysis
        """
        # Find Monday opens vs Friday closes
        if 'day_of_week' not in df.columns:
            df = df.copy()
            df['day_of_week'] = df.index.dayofweek
        
        gaps = []
        
        # Group by week
        df['week'] = df.index.isocalendar().week
        df['year'] = df.index.year
        
        for (year, week), week_df in df.groupby(['year', 'week']):
            friday = week_df[week_df['day_of_week'] == 4]
            monday = week_df[week_df['day_of_week'] == 0]
            
            if len(friday) == 0 or len(monday) == 0:
                continue
            
            friday_close = friday.iloc[-1]['close']
            monday_open = monday.iloc[0]['open']
            
            # Gap calculation
            gap = (monday_open - friday_close) / friday_close
            
            # Check if gap filled within Monday
            monday_low = monday['low'].min()
            monday_high = monday['high'].max()
            
            gap_filled_up = gap > 0 and monday_low <= friday_close
            gap_filled_down = gap < 0 and monday_high >= friday_close
            gap_filled = gap_filled_up or gap_filled_down
            
            gaps.append({
                'year': year,
                'week': week,
                'gap': gap,
                'gap_filled': gap_filled,
                'gap_direction': 'up' if gap > 0 else 'down' if gap < 0 else 'none'
            })
        
        if len(gaps) < 30:
            return None
        
        gaps_df = pd.DataFrame(gaps)
        
        # Filter non-zero gaps
        significant_gaps = gaps_df[gaps_df['gap_direction'] != 'none']
        
        if len(significant_gaps) < 30:
            return None
        
        fill_rate = significant_gaps['gap_filled'].mean()
        
        # Chi-squared test for gap fill rate > 50%
        observed = significant_gaps['gap_filled'].sum()
        expected = len(significant_gaps) * 0.5
        chi2, p_val = stats.chisquare([observed, len(significant_gaps) - observed],
                                       [expected, expected])
        
        return EdgeResult(
            name="Weekend Gap Fill",
            edge_type="time_pattern",
            sample_size=len(significant_gaps),
            mean_return=fill_rate - 0.5,  # Edge over random
            std_return=np.sqrt(fill_rate * (1 - fill_rate)),
            t_statistic=chi2,
            p_value=p_val,
            win_rate=fill_rate,
            sharpe_ratio=0,  # Not applicable
            is_significant=p_val < self.significance_level and fill_rate > 0.55,
            details={
                'fill_rate': fill_rate,
                'up_gap_fill_rate': significant_gaps[significant_gaps['gap_direction'] == 'up']['gap_filled'].mean(),
                'down_gap_fill_rate': significant_gaps[significant_gaps['gap_direction'] == 'down']['gap_filled'].mean(),
                'avg_gap_size': significant_gaps['gap'].abs().mean()
            }
        )
    
    def first_last_hour_effect(self, df: pd.DataFrame,
                               return_column: str = 'return_1') -> Dict[str, EdgeResult]:
        """
        Analyze first and last hour trading effects.
        
        Args:
            df: DataFrame with time index and returns
            return_column: Column name for returns
            
        Returns:
            Dictionary with first/last hour results
        """
        results = {}
        
        if 'hour' not in df.columns:
            df = df.copy()
            df['hour'] = df.index.hour
        
        # Group by date to find first/last hours
        df['date'] = df.index.date
        
        first_hour_returns = []
        last_hour_returns = []
        
        for date, day_df in df.groupby('date'):
            if len(day_df) < 2:
                continue
            
            first_hour = day_df.iloc[0]['hour']
            last_hour = day_df.iloc[-1]['hour']
            
            first_hour_returns.append(day_df.iloc[0][return_column])
            last_hour_returns.append(day_df.iloc[-1][return_column])
        
        # First hour analysis
        if len(first_hour_returns) >= 30:
            first_hour_data = pd.Series(first_hour_returns).dropna()
            t_stat, p_val = stats.ttest_1samp(first_hour_data, 0)
            
            results['first_hour'] = EdgeResult(
                name="First Hour Effect",
                edge_type="time_pattern",
                sample_size=len(first_hour_data),
                mean_return=first_hour_data.mean(),
                std_return=first_hour_data.std(),
                t_statistic=t_stat,
                p_value=p_val,
                win_rate=(first_hour_data > 0).mean(),
                sharpe_ratio=first_hour_data.mean() / first_hour_data.std() * np.sqrt(252) if first_hour_data.std() > 0 else 0,
                is_significant=p_val < self.significance_level,
                details={}
            )
        
        # Last hour analysis
        if len(last_hour_returns) >= 30:
            last_hour_data = pd.Series(last_hour_returns).dropna()
            t_stat, p_val = stats.ttest_1samp(last_hour_data, 0)
            
            results['last_hour'] = EdgeResult(
                name="Last Hour Effect",
                edge_type="time_pattern",
                sample_size=len(last_hour_data),
                mean_return=last_hour_data.mean(),
                std_return=last_hour_data.std(),
                t_statistic=t_stat,
                p_value=p_val,
                win_rate=(last_hour_data > 0).mean(),
                sharpe_ratio=last_hour_data.mean() / last_hour_data.std() * np.sqrt(252) if last_hour_data.std() > 0 else 0,
                is_significant=p_val < self.significance_level,
                details={}
            )
        
        return results
    
    def run_all_time_edges(self, df: pd.DataFrame,
                           return_column: str = 'return_1') -> Dict[str, EdgeResult]:
        """
        Run all time-based edge tests.
        
        Args:
            df: DataFrame with OHLC data and features
            return_column: Column name for returns
            
        Returns:
            Dictionary with all edge results
        """
        all_results = {}
        
        # Hour effects
        hour_results = self.hour_of_day_effect(df, return_column)
        for hour, result in hour_results.items():
            all_results[f'hour_{hour}'] = result
        
        # Day effects
        day_results = self.day_of_week_effect(df, return_column)
        for day, result in day_results.items():
            all_results[f'day_{day}'] = result
        
        # Session effects
        session_results = self.session_effect(df, return_column)
        for session, result in session_results.items():
            all_results[f'session_{session}'] = result
        
        # Month effects
        month_results = self.month_effect(df, return_column)
        for month, result in month_results.items():
            all_results[f'month_{month}'] = result
        
        # First/last hour
        fl_results = self.first_last_hour_effect(df, return_column)
        all_results.update(fl_results)
        
        # Weekend gap
        gap_result = self.weekend_gap_analysis(df)
        if gap_result:
            all_results['weekend_gap'] = gap_result
        
        # Opening range
        orb_result = self.opening_range_breakout(df)
        if orb_result:
            all_results['opening_range_breakout'] = orb_result
        
        return all_results
