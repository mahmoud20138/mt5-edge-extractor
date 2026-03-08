"""
Volatility Feature Engineering Module.

Provides volatility-based technical indicators:
- ATR (Average True Range)
- Bollinger Bands
- Keltner Channels
- Historical volatility
- Volatility ratios and percentile ranks
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


class VolatilityFeatures:
    """
    Calculate volatility-based technical indicators.
    
    Volatility indicators measure the rate of price movements
    and help identify potential breakouts and mean-reversion opportunities.
    """
    
    @staticmethod
    def true_range(df: pd.DataFrame) -> pd.Series:
        """
        Calculate True Range.
        
        True Range is the greatest of:
        - Current high - current low
        - Abs(current high - previous close)
        - Abs(current low - previous close)
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Series with True Range values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        ATR measures market volatility by decomposing the entire
        range of an asset price for that period.
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        tr = VolatilityFeatures.true_range(df)
        atr = tr.rolling(window=period, min_periods=period).mean()
        
        return atr
    
    @staticmethod
    def atr_multiple_periods(df: pd.DataFrame, 
                            periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate ATR for multiple periods.
        
        Args:
            df: DataFrame with OHLC data
            periods: List of periods
            
        Returns:
            DataFrame with ATR columns
        """
        periods = periods or [7, 14, 21]
        result = pd.DataFrame(index=df.index)
        
        for period in periods:
            result[f'atr_{period}'] = VolatilityFeatures.atr(df, period)
            result[f'atr_pct_{period}'] = result[f'atr_{period}'] / df['close']
        
        return result
    
    @staticmethod
    def bollinger_bands(df: pd.DataFrame, period: int = 20,
                        std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Bollinger Bands consist of a middle band (SMA) and two
        outer bands that are standard deviations away from the middle.
        
        Args:
            df: DataFrame with price data
            period: SMA period
            std_dev: Standard deviation multiplier
            
        Returns:
            DataFrame with Bollinger Band components
        """
        close = df['close']
        
        # Middle band (SMA)
        middle_band = close.rolling(window=period, min_periods=period).mean()
        
        # Standard deviation
        std = close.rolling(window=period, min_periods=period).std()
        
        # Upper and lower bands
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        # Bandwidth
        bandwidth = (upper_band - lower_band) / middle_band
        
        # %B (position within bands)
        percent_b = (close - lower_band) / (upper_band - lower_band)
        
        return pd.DataFrame({
            'bb_middle': middle_band,
            'bb_upper': upper_band,
            'bb_lower': lower_band,
            'bb_bandwidth': bandwidth,
            'bb_percent_b': percent_b
        }, index=df.index)
    
    @staticmethod
    def keltner_channels(df: pd.DataFrame, period: int = 20,
                         atr_mult: float = 2.0) -> pd.DataFrame:
        """
        Calculate Keltner Channels.
        
        Keltner Channels are volatility-based envelopes set above
        and below an exponential moving average.
        
        Args:
            df: DataFrame with OHLC data
            period: EMA period
            atr_mult: ATR multiplier
            
        Returns:
            DataFrame with Keltner Channel components
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Middle line (EMA of typical price)
        middle = typical_price.ewm(span=period, min_periods=period).mean()
        
        # ATR
        atr = VolatilityFeatures.atr(df, period)
        
        # Upper and lower channels
        upper = middle + (atr_mult * atr)
        lower = middle - (atr_mult * atr)
        
        return pd.DataFrame({
            'kc_middle': middle,
            'kc_upper': upper,
            'kc_lower': lower,
            'kc_width': (upper - lower) / middle
        }, index=df.index)
    
    @staticmethod
    def donchian_channels(df: pd.DataFrame, 
                          period: int = 20) -> pd.DataFrame:
        """
        Calculate Donchian Channels.
        
        Donchian Channels show the highest high and lowest low
        over a specified period.
        
        Args:
            df: DataFrame with OHLC data
            period: Lookback period
            
        Returns:
            DataFrame with Donchian Channel components
        """
        upper = df['high'].rolling(window=period, min_periods=period).max()
        lower = df['low'].rolling(window=period, min_periods=period).min()
        middle = (upper + lower) / 2
        
        return pd.DataFrame({
            'donchian_upper': upper,
            'donchian_lower': lower,
            'donchian_middle': middle,
            'donchian_width': (upper - lower) / middle
        }, index=df.index)
    
    @staticmethod
    def historical_volatility(df: pd.DataFrame, period: int = 20,
                              trading_periods: int = 252) -> pd.Series:
        """
        Calculate Historical Volatility (annualized).
        
        Args:
            df: DataFrame with price data
            period: Lookback period
            trading_periods: Number of trading periods per year
            
        Returns:
            Series with annualized volatility
        """
        log_returns = np.log(df['close'] / df['close'].shift(1))
        
        # Rolling standard deviation
        vol = log_returns.rolling(window=period, min_periods=period).std()
        
        # Annualize
        annualized_vol = vol * np.sqrt(trading_periods)
        
        return annualized_vol
    
    @staticmethod
    def parkinson_volatility(df: pd.DataFrame, 
                             period: int = 20) -> pd.Series:
        """
        Calculate Parkinson Volatility.
        
        Uses high-low prices to estimate volatility, which is more
        efficient for continuous markets.
        
        Args:
            df: DataFrame with OHLC data
            period: Lookback period
            
        Returns:
            Series with Parkinson volatility
        """
        hl_ratio = np.log(df['high'] / df['low'])
        squared = hl_ratio ** 2
        
        # Parkinson volatility
        pv = np.sqrt(
            squared.rolling(window=period, min_periods=period).sum() / 
            (4 * period * np.log(2))
        )
        
        return pv
    
    @staticmethod
    def garman_klass_volatility(df: pd.DataFrame, 
                                period: int = 20) -> pd.Series:
        """
        Calculate Garman-Klass Volatility.
        
        A more efficient estimator that uses OHLC data.
        
        Args:
            df: DataFrame with OHLC data
            period: Lookback period
            
        Returns:
            Series with Garman-Klass volatility
        """
        hl_ratio = np.log(df['high'] / df['low'])
        co_ratio = np.log(df['close'] / df['open'])
        
        gk = 0.5 * (hl_ratio ** 2) - (2 * np.log(2) - 1) * (co_ratio ** 2)
        
        vol = np.sqrt(gk.rolling(window=period, min_periods=period).mean())
        
        return vol
    
    @staticmethod
    def yang_zhang_volatility(df: pd.DataFrame, 
                              period: int = 20) -> pd.Series:
        """
        Calculate Yang-Zhang Volatility.
        
        Handles both overnight and intraday volatility.
        
        Args:
            df: DataFrame with OHLC data
            period: Lookback period
            
        Returns:
            Series with Yang-Zhang volatility
        """
        # Overnight returns
        overnight_returns = np.log(df['open'] / df['close'].shift(1))
        
        # Intraday returns  
        intraday_returns = np.log(df['close'] / df['open'])
        
        # Rogers-Satchell term
        hl_log = np.log(df['high'] / df['low'])
        co_log = np.log(df['close'] / df['open'])
        rs_term = hl_log ** 2 - 2 * co_log ** 2
        
        # Yang-Zhang estimator
        k = 0.34 / (1.34 + (period + 1) / (period - 1))
        
        vol_overnight = overnight_returns.rolling(window=period, min_periods=period).var()
        vol_intraday = intraday_returns.rolling(window=period, min_periods=period).var()
        vol_rs = rs_term.rolling(window=period, min_periods=period).mean()
        
        yz_vol = np.sqrt(vol_overnight + k * vol_intraday + (1 - k) * vol_rs)
        
        return yz_vol
    
    @staticmethod
    def volatility_ratio(df: pd.DataFrame, short_period: int = 5,
                         long_period: int = 20) -> pd.Series:
        """
        Calculate volatility ratio (short/long term).
        
        Ratio > 1 indicates increasing volatility
        Ratio < 1 indicates decreasing volatility
        
        Args:
            df: DataFrame with price data
            short_period: Short-term period
            long_period: Long-term period
            
        Returns:
            Series with volatility ratio
        """
        log_returns = np.log(df['close'] / df['close'].shift(1))
        
        short_vol = log_returns.rolling(window=short_period, min_periods=short_period).std()
        long_vol = log_returns.rolling(window=long_period, min_periods=long_period).std()
        
        ratio = short_vol / long_vol.replace(0, np.nan)
        
        return ratio
    
    @staticmethod
    def volatility_percentile(df: pd.DataFrame, period: int = 20,
                              lookback: int = 252) -> pd.Series:
        """
        Calculate volatility percentile rank.
        
        Shows where current volatility ranks historically.
        
        Args:
            df: DataFrame with price data
            period: Volatility calculation period
            lookback: Percentile lookback period
            
        Returns:
            Series with percentile rank (0-100)
        """
        log_returns = np.log(df['close'] / df['close'].shift(1))
        vol = log_returns.rolling(window=period, min_periods=period).std()
        
        # Percentile rank
        percentile = vol.rolling(window=lookback, min_periods=lookback).rank(pct=True) * 100
        
        return percentile
    
    @staticmethod
    def detect_squeeze(df: pd.DataFrame, bb_period: int = 20,
                       kc_period: int = 20) -> pd.DataFrame:
        """
        Detect volatility squeeze (BB inside KC).
        
        Squeezes often precede significant moves.
        
        Args:
            df: DataFrame with OHLC data
            bb_period: Bollinger Band period
            kc_period: Keltner Channel period
            
        Returns:
            DataFrame with squeeze indicators
        """
        bb = VolatilityFeatures.bollinger_bands(df, bb_period)
        kc = VolatilityFeatures.keltner_channels(df, kc_period)
        
        # Squeeze condition: BB width < KC width
        squeeze = (bb['bb_upper'] < kc['kc_upper']) & (bb['bb_lower'] > kc['kc_lower'])
        
        return pd.DataFrame({
            'bb_width': bb['bb_bandwidth'],
            'kc_width': kc['kc_width'],
            'squeeze': squeeze.astype(int)
        }, index=df.index)
    
    @staticmethod
    def add_all_volatility(df: pd.DataFrame, 
                          atr_periods: List[int] = None) -> pd.DataFrame:
        """
        Add all volatility indicators to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            atr_periods: List of ATR periods
            
        Returns:
            DataFrame with added volatility features
        """
        df = df.copy()
        atr_periods = atr_periods or [7, 14, 21]
        
        # ATR
        for period in atr_periods:
            df[f'atr_{period}'] = VolatilityFeatures.atr(df, period)
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close']
        
        # Bollinger Bands
        bb = VolatilityFeatures.bollinger_bands(df)
        df['bb_upper'] = bb['bb_upper']
        df['bb_lower'] = bb['bb_lower']
        df['bb_middle'] = bb['bb_middle']
        df['bb_bandwidth'] = bb['bb_bandwidth']
        df['bb_percent_b'] = bb['bb_percent_b']
        
        # BB signals
        df['bb_squeeze'] = (df['bb_bandwidth'] < df['bb_bandwidth'].rolling(20).quantile(0.2)).astype(int)
        df['bb_upper_touch'] = (df['close'] >= df['bb_upper'] * 0.99).astype(int)
        df['bb_lower_touch'] = (df['close'] <= df['bb_lower'] * 1.01).astype(int)
        
        # Keltner Channels
        kc = VolatilityFeatures.keltner_channels(df)
        df['kc_upper'] = kc['kc_upper']
        df['kc_lower'] = kc['kc_lower']
        df['kc_middle'] = kc['kc_middle']
        
        # Donchian Channels
        dc = VolatilityFeatures.donchian_channels(df)
        df['donchian_upper'] = dc['donchian_upper']
        df['donchian_lower'] = dc['donchian_lower']
        df['donchian_middle'] = dc['donchian_middle']
        
        # Volatility ratios
        df['vol_ratio'] = VolatilityFeatures.volatility_ratio(df, 5, 20)
        df['vol_percentile'] = VolatilityFeatures.volatility_percentile(df, 20, 252)
        
        # Historical volatility
        df['hist_vol_20'] = VolatilityFeatures.historical_volatility(df, 20)
        
        # Squeeze detection
        sq = VolatilityFeatures.detect_squeeze(df)
        df['vol_squeeze'] = sq['squeeze']
        
        # Volatility regime
        df['vol_regime'] = pd.cut(
            df['vol_percentile'], 
            bins=[0, 25, 75, 100],
            labels=['low', 'normal', 'high']
        ).astype(str)
        
        return df
