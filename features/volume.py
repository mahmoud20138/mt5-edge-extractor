"""
Volume Feature Engineering Module.

Provides volume-based technical indicators:
- Volume SMA and ratios
- OBV (On-Balance Volume)
- Volume Profile
- Volume-weighted price
- Volume anomalies
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict


class VolumeFeatures:
    """
    Calculate volume-based technical indicators.
    
    Volume indicators help confirm price movements and identify
    potential reversals or breakouts.
    """
    
    @staticmethod
    def get_volume(df: pd.DataFrame) -> pd.Series:
        """
        Get volume series from DataFrame.
        
        Prefers tick_volume over real_volume if both exist.
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            Series with volume values
        """
        if 'tick_volume' in df.columns:
            return df['tick_volume']
        elif 'real_volume' in df.columns:
            return df['real_volume']
        else:
            return pd.Series(1, index=df.index)  # Default to 1 if no volume
    
    @staticmethod
    def volume_sma(df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate Volume Simple Moving Averages.
        
        Args:
            df: DataFrame with volume data
            periods: List of periods (default [10, 20, 50])
            
        Returns:
            DataFrame with volume SMA columns
        """
        periods = periods or [10, 20, 50]
        volume = VolumeFeatures.get_volume(df)
        result = pd.DataFrame(index=df.index)
        
        for period in periods:
            result[f'volume_sma_{period}'] = volume.rolling(window=period, min_periods=period).mean()
            result[f'volume_ratio_{period}'] = volume / result[f'volume_sma_{period}'].replace(0, np.nan)
        
        return result
    
    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        OBV is a cumulative indicator that uses volume to predict
        changes in stock price.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with OBV values
        """
        close = df['close']
        volume = VolumeFeatures.get_volume(df)
        
        # Direction
        direction = np.sign(close.diff())
        
        # OBV
        obv = (direction * volume).cumsum()
        
        return obv
    
    @staticmethod
    def obv_signal(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Calculate OBV with signal line.
        
        Args:
            df: DataFrame with OHLCV data
            period: SMA period for signal
            
        Returns:
            DataFrame with OBV and signal
        """
        obv = VolumeFeatures.obv(df)
        signal = obv.rolling(window=period, min_periods=period).mean()
        
        return pd.DataFrame({
            'obv': obv,
            'obv_signal': signal,
            'obv_trend': (obv > signal).astype(int)
        }, index=df.index)
    
    @staticmethod
    def vwap_rolling(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate rolling VWAP.
        
        Args:
            df: DataFrame with OHLCV data
            period: Rolling period
            
        Returns:
            Series with rolling VWAP values
        """
        tp = (df['high'] + df['low'] + df['close']) / 3
        volume = VolumeFeatures.get_volume(df)
        
        vwap = (tp * volume).rolling(window=period, min_periods=period).sum() / \
               volume.rolling(window=period, min_periods=period).sum()
        
        return vwap
    
    @staticmethod
    def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index.
        
        MFI uses both price and volume to measure buying and selling pressure.
        
        Args:
            df: DataFrame with OHLCV data
            period: MFI period
            
        Returns:
            Series with MFI values (0-100)
        """
        tp = (df['high'] + df['low'] + df['close']) / 3
        volume = VolumeFeatures.get_volume(df)
        
        raw_money_flow = tp * volume
        
        positive_flow = raw_money_flow.where(tp > tp.shift(1), 0)
        negative_flow = raw_money_flow.where(tp < tp.shift(1), 0)
        
        positive_sum = positive_flow.rolling(window=period, min_periods=period).sum()
        negative_sum = negative_flow.rolling(window=period, min_periods=period).sum()
        
        money_ratio = positive_sum / negative_sum.replace(0, np.nan)
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    @staticmethod
    def volume_profile(df: pd.DataFrame, bins: int = 20,
                       lookback: int = 100) -> pd.DataFrame:
        """
        Calculate Volume Profile.
        
        Shows volume traded at each price level over a lookback period.
        
        Args:
            df: DataFrame with OHLCV data
            bins: Number of price bins
            lookback: Lookback period
            
        Returns:
            DataFrame with price levels and volumes
        """
        volume = VolumeFeatures.get_volume(df)
        
        # Use recent data
        recent_df = df.tail(lookback).copy()
        recent_vol = volume.tail(lookback)
        
        # Create price bins
        price_range = recent_df['high'].max() - recent_df['low'].min()
        bin_size = price_range / bins
        
        price_levels = []
        volumes = []
        
        for i in range(bins):
            lower = recent_df['low'].min() + i * bin_size
            upper = lower + bin_size
            
            # Approximate volume at each price level using candle range
            mask = (recent_df['low'] <= upper) & (recent_df['high'] >= lower)
            vol_at_level = recent_vol[mask].sum()
            
            price_levels.append((lower + upper) / 2)
            volumes.append(vol_at_level)
        
        result = pd.DataFrame({
            'price_level': price_levels,
            'volume': volumes
        })
        
        # Find POC (Point of Control) - highest volume level
        poc_idx = result['volume'].idxmax()
        result['is_poc'] = (result.index == poc_idx).astype(int)
        
        return result
    
    @staticmethod
    def volume_anomalies(df: pd.DataFrame, 
                         threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect volume anomalies.
        
        Identifies bars with unusually high or low volume.
        
        Args:
            df: DataFrame with volume data
            threshold: Standard deviation threshold
            
        Returns:
            DataFrame with anomaly indicators
        """
        volume = VolumeFeatures.get_volume(df)
        
        vol_mean = volume.rolling(window=20, min_periods=20).mean()
        vol_std = volume.rolling(window=20, min_periods=20).std()
        
        z_score = (volume - vol_mean) / vol_std.replace(0, np.nan)
        
        return pd.DataFrame({
            'volume_zscore': z_score,
            'high_volume': (z_score > threshold).astype(int),
            'low_volume': (z_score < -threshold).astype(int)
        }, index=df.index)
    
    @staticmethod
    def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line.
        
        A volume-based indicator designed to measure the cumulative
        flow of money into and out of a security.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with A/D Line values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        volume = VolumeFeatures.get_volume(df)
        
        # CLV (Close Location Value)
        clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        
        # A/D Line
        ad = (clv * volume).cumsum()
        
        return ad
    
    @staticmethod
    def chaikin_oscillator(df: pd.DataFrame, fast: int = 3,
                           slow: int = 10) -> pd.Series:
        """
        Calculate Chaikin Oscillator.
        
        Applies MACD to the Accumulation/Distribution Line.
        
        Args:
            df: DataFrame with OHLCV data
            fast: Fast EMA period
            slow: Slow EMA period
            
        Returns:
            Series with Chaikin Oscillator values
        """
        ad = VolumeFeatures.accumulation_distribution(df)
        
        fast_ema = ad.ewm(span=fast, min_periods=fast).mean()
        slow_ema = ad.ewm(span=slow, min_periods=slow).mean()
        
        return fast_ema - slow_ema
    
    @staticmethod
    def ease_of_movement(df: pd.DataFrame, 
                         period: int = 14) -> pd.Series:
        """
        Calculate Ease of Movement (EMV).
        
        Relates price change to volume to quantify the ease
        of price movement.
        
        Args:
            df: DataFrame with OHLCV data
            period: Smoothing period
            
        Returns:
            Series with EMV values
        """
        high = df['high']
        low = df['low']
        volume = VolumeFeatures.get_volume(df)
        
        # Midpoint move
        midpt_move = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        
        # Box ratio (scaled for readability)
        box_ratio = (volume / 100000000) / (high - low)
        
        # EMV
        emv = midpt_move / box_ratio.replace(0, np.nan)
        
        # Smoothed
        emv_smooth = emv.rolling(window=period, min_periods=period).mean()
        
        return emv_smooth
    
    @staticmethod
    def volume_price_trend(df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Price Trend (VPT).
        
        Similar to OBV but accounts for the magnitude of price changes.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with VPT values
        """
        close = df['close']
        volume = VolumeFeatures.get_volume(df)
        
        pct_change = close.pct_change()
        
        vpt = (volume * pct_change).cumsum()
        
        return vpt
    
    @staticmethod
    def negative_volume_index(df: pd.DataFrame) -> pd.Series:
        """
        Calculate Negative Volume Index (NVI).
        
        NVI focuses on days when volume decreases from the previous day.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with NVI values
        """
        close = df['close']
        volume = VolumeFeatures.get_volume(df)
        
        nvi = pd.Series(1000, index=df.index)  # Starting value
        
        for i in range(1, len(df)):
            if volume.iloc[i] < volume.iloc[i-1]:
                nvi.iloc[i] = nvi.iloc[i-1] * (close.iloc[i] / close.iloc[i-1])
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        
        return nvi
    
    @staticmethod
    def add_all_volume(df: pd.DataFrame, 
                       sma_periods: List[int] = None) -> pd.DataFrame:
        """
        Add all volume indicators to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            sma_periods: List of SMA periods
            
        Returns:
            DataFrame with added volume features
        """
        df = df.copy()
        sma_periods = sma_periods or [10, 20, 50]
        
        volume = VolumeFeatures.get_volume(df)
        
        # Volume SMAs and ratios
        for period in sma_periods:
            df[f'volume_sma_{period}'] = volume.rolling(window=period, min_periods=period).mean()
            df[f'volume_ratio_{period}'] = volume / df[f'volume_sma_{period}'].replace(0, np.nan)
        
        # OBV
        df['obv'] = VolumeFeatures.obv(df)
        df['obv_sma_20'] = df['obv'].rolling(20).mean()
        df['obv_trend'] = (df['obv'] > df['obv_sma_20']).astype(int)
        
        # VWAP
        df['vwap_20'] = VolumeFeatures.vwap_rolling(df, 20)
        df['price_above_vwap'] = (df['close'] > df['vwap_20']).astype(int)
        
        # MFI
        df['mfi'] = VolumeFeatures.mfi(df)
        df['mfi_overbought'] = (df['mfi'] > 80).astype(int)
        df['mfi_oversold'] = (df['mfi'] < 20).astype(int)
        
        # Volume anomalies
        vol_anom = VolumeFeatures.volume_anomalies(df)
        df['volume_zscore'] = vol_anom['volume_zscore']
        df['high_volume'] = vol_anom['high_volume']
        df['low_volume'] = vol_anom['low_volume']
        
        # A/D Line
        df['ad_line'] = VolumeFeatures.accumulation_distribution(df)
        
        # Chaikin Oscillator
        df['chaikin_osc'] = VolumeFeatures.chaikin_oscillator(df)
        
        return df
