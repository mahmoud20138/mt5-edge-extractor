"""
Data Preprocessing Module - Data cleaning and feature engineering.

This module provides:
- Data cleaning and validation
- Time feature extraction
- Derived price calculations
- Missing data handling
- Multi-timeframe alignment
"""

import logging
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data cleaning and basic feature engineering.
    
    Provides methods for:
    - Data validation and cleaning
    - Time feature extraction
    - Derived price calculations
    - Missing data handling
    - Multi-timeframe alignment
    """
    
    def __init__(self):
        """Initialize preprocessor."""
        pass
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate OHLCV data integrity.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check required columns
        required = ['open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]
        if missing:
            issues.append(f"Missing required columns: {missing}")
        
        if not df.empty:
            # Check OHLC relationships
            invalid_high = df[df['high'] < df[['open', 'close']].max(axis=1)]
            if not invalid_high.empty:
                issues.append(f"Invalid high prices in {len(invalid_high)} rows")
            
            invalid_low = df[df['low'] > df[['open', 'close']].min(axis=1)]
            if not invalid_low.empty:
                issues.append(f"Invalid low prices in {len(invalid_low)} rows")
            
            # Check for negative prices
            for col in required:
                if col in df.columns and (df[col] <= 0).any():
                    issues.append(f"Negative or zero prices in {col}")
            
            # Check for NaN values
            nan_counts = df[required].isna().sum()
            for col, count in nan_counts.items():
                if count > 0:
                    issues.append(f"{count} NaN values in {col}")
        
        return len(issues) == 0, issues
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare OHLCV data.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            else:
                logger.error("No datetime index or 'time' column found")
                return df
        
        # Sort by time
        df.sort_index(inplace=True)
        
        # Remove duplicates
        duplicates = df.index.duplicated()
        if duplicates.any():
            logger.warning(f"Removing {duplicates.sum()} duplicate timestamps")
            df = df[~duplicates]
        
        # Fix OHLC relationships
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        # Forward fill missing values (common approach for OHLCV)
        df = df.ffill()
        
        # Drop any remaining NaN rows
        df.dropna(inplace=True)
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features to the DataFrame.
        
        Args:
            df: OHLCV DataFrame with datetime index
            
        Returns:
            DataFrame with added time features
        """
        df = df.copy()
        
        # Basic time features
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['week_of_year'] = df.index.isocalendar().week
        df['year'] = df.index.year
        
        # Trading session features (UTC times)
        # Asian session: 00:00-08:00 UTC
        df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        
        # London session: 07:00-16:00 UTC
        df['is_london_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
        
        # New York session: 12:00-21:00 UTC
        df['is_ny_session'] = ((df['hour'] >= 12) & (df['hour'] < 21)).astype(int)
        
        # London-NY overlap: 12:00-16:00 UTC
        df['is_london_ny_overlap'] = ((df['hour'] >= 12) & (df['hour'] < 16)).astype(int)
        
        # Asian-London overlap: 07:00-08:00 UTC
        df['is_asian_london_overlap'] = (df['hour'] == 7).astype(int)
        
        # Session name
        def get_session(hour):
            if 0 <= hour < 8:
                return 'asian'
            elif 7 <= hour < 16:
                return 'london'
            elif 12 <= hour < 21:
                return 'ny'
            else:
                return 'other'
        
        df['session'] = df['hour'].apply(get_session)
        
        # Day of week names
        day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        df['day_name'] = df['day_of_week'].apply(lambda x: day_names[x])
        
        # Is weekend (for forex, should be removed)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Is month end (last 3 trading days)
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
        
        # Is quarter end
        df['is_quarter_end'] = ((df['month'].isin([3, 6, 9, 12])) & 
                                (df['day_of_month'] >= 28)).astype(int)
        
        # First/last hour of trading day
        df['is_first_hour'] = (df['hour'] == df.groupby(df.index.date)['hour'].transform('min')).astype(int)
        df['is_last_hour'] = (df['hour'] == df.groupby(df.index.date)['hour'].transform('max')).astype(int)
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived price features.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with added price features
        """
        df = df.copy()
        
        # Basic price features
        df['mid_price'] = (df['high'] + df['low']) / 2
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['weighted_price'] = (df['high'] + df['low'] + df['close'] * 2) / 4
        
        # Candle features
        df['body'] = abs(df['close'] - df['open'])
        df['body_pct'] = df['body'] / df['close'] * 100
        df['range'] = df['high'] - df['low']
        df['range_pct'] = df['range'] / df['close'] * 100
        
        # Body to range ratio
        df['body_range_ratio'] = df['body'] / df['range'].replace(0, np.nan)
        
        # Wick features
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upper_wick_pct'] = df['upper_wick'] / df['range'].replace(0, np.nan)
        df['lower_wick_pct'] = df['lower_wick'] / df['range'].replace(0, np.nan)
        
        # Direction
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['is_bearish'] = (df['close'] < df['open']).astype(int)
        df['is_doji'] = (df['body'] / df['range'].replace(0, np.nan) < 0.1).astype(int)
        
        # Gap from previous close
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1) * 100
        
        # Candle relative size (vs N-period average)
        for period in [5, 10, 20]:
            avg_range = df['range'].rolling(period).mean()
            df[f'rel_range_{period}'] = df['range'] / avg_range
        
        return df
    
    def add_returns(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Add return calculations.
        
        Args:
            df: DataFrame with close prices
            periods: List of periods for returns (default: [1, 5, 10, 20])
            
        Returns:
            DataFrame with return features
        """
        df = df.copy()
        periods = periods or [1, 5, 10, 20]
        
        for period in periods:
            # Simple returns
            df[f'return_{period}'] = df['close'].pct_change(period)
            
            # Log returns
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
            
            # Forward returns (for prediction targets)
            df[f'fwd_return_{period}'] = df['close'].shift(-period) / df['close'] - 1
            df[f'fwd_log_return_{period}'] = np.log(df['close'].shift(-period) / df['close'])
        
        # Cumulative returns
        df['cum_return'] = (1 + df['return_1']).cumprod() - 1
        
        # Rolling cumulative returns
        for period in [5, 10, 20]:
            df[f'rolling_return_{period}'] = df['return_1'].rolling(period).sum()
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame, 
                                periods: List[int] = None) -> pd.DataFrame:
        """
        Add volatility features.
        
        Args:
            df: OHLCV DataFrame
            periods: List of periods for rolling calculations
            
        Returns:
            DataFrame with volatility features
        """
        df = df.copy()
        periods = periods or [5, 10, 20, 50]
        
        for period in periods:
            # Rolling std of returns
            df[f'vol_std_{period}'] = df['return_1'].rolling(period).std()
            
            # Rolling std of log returns
            df[f'vol_log_{period}'] = df['log_return_1'].rolling(period).std()
            
            # Parkinson volatility (high-low based)
            df[f'parkinson_vol_{period}'] = np.sqrt(
                (1 / (4 * period * np.log(2))) * 
                (np.log(df['high'] / df['low']) ** 2).rolling(period).sum()
            )
            
            # Average True Range
            tr = pd.DataFrame({
                'hl': df['high'] - df['low'],
                'hc': abs(df['high'] - df['close'].shift(1)),
                'lc': abs(df['low'] - df['close'].shift(1))
            }).max(axis=1)
            df[f'atr_{period}'] = tr.rolling(period).mean()
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close']
        
        # Garman-Klass volatility
        df['garman_klass_vol'] = np.sqrt(
            0.5 * (np.log(df['high'] / df['low']) ** 2) -
            (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2)
        ).rolling(20).mean()
        
        # Volatility ratio (short/long term)
        df['vol_ratio'] = df['vol_std_5'] / df['vol_std_20'].replace(0, np.nan)
        
        # Volatility percentile rank
        for period in [20, 50]:
            df[f'vol_rank_{period}'] = df['vol_std_20'].rolling(period).rank(pct=True)
        
        return df
    
    def detect_gaps(self, df: pd.DataFrame, timeframe_minutes: int = 60) -> pd.DataFrame:
        """
        Detect gaps in data (missing bars).
        
        Args:
            df: DataFrame with datetime index
            timeframe_minutes: Expected timeframe in minutes
            
        Returns:
            DataFrame with gap information
        """
        df = df.copy()
        
        # Calculate expected vs actual time differences
        time_diffs = df.index.to_series().diff().dropna()
        expected_diff = pd.Timedelta(minutes=timeframe_minutes)
        
        # Find gaps (where difference > expected * 1.5, excluding weekends)
        gaps = time_diffs[time_diffs > expected_diff * 1.5]
        
        gap_info = pd.DataFrame({
            'gap_start': gaps.index - gaps,
            'gap_end': gaps.index,
            'gap_duration': gaps
        })
        
        if not gap_info.empty:
            # Remove weekend gaps
            gap_info['is_weekend'] = (
                (gap_info['gap_start'].dt.dayofweek >= 5) |
                (gap_info['gap_end'].dt.dayofweek >= 5)
            )
            gap_info = gap_info[~gap_info['is_weekend']]
        
        return gap_info
    
    def align_timeframes(self, data_dict: Dict[str, pd.DataFrame],
                         target_timeframe: str = None) -> pd.DataFrame:
        """
        Align data from multiple timeframes.
        
        Args:
            data_dict: Dictionary mapping timeframe names to DataFrames
            target_timeframe: Target timeframe for alignment
            
        Returns:
            Aligned DataFrame with features from all timeframes
        """
        if not data_dict:
            return pd.DataFrame()
        
        # Use first dataframe as base
        base_tf = list(data_dict.keys())[0]
        base_df = data_dict[base_tf].copy()
        
        for tf, df in data_dict.items():
            if tf == base_tf:
                continue
            
            # Merge using asof (nearest timestamp)
            suffix = f'_{tf}'
            base_df = pd.merge_asof(
                base_df, df,
                left_index=True, right_index=True,
                suffixes=('', suffix),
                direction='nearest'
            )
        
        return base_df
    
    def preprocess(self, df: pd.DataFrame, 
                   add_time: bool = True,
                   add_price: bool = True,
                   add_returns: bool = True,
                   add_volatility: bool = True) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline.
        
        Args:
            df: Raw OHLCV DataFrame
            add_time: Add time features
            add_price: Add price features
            add_returns: Add return features
            add_volatility: Add volatility features
            
        Returns:
            Fully preprocessed DataFrame
        """
        # Validate and clean
        is_valid, issues = self.validate_data(df)
        if issues:
            for issue in issues:
                logger.warning(f"Data validation: {issue}")
        
        df = self.clean_data(df)
        
        # Add features
        if add_time:
            df = self.add_time_features(df)
        if add_price:
            df = self.add_price_features(df)
        if add_returns:
            df = self.add_returns(df)
        if add_volatility:
            df = self.add_volatility_features(df)
        
        logger.info(f"Preprocessing complete: {len(df)} rows, {len(df.columns)} columns")
        
        return df
