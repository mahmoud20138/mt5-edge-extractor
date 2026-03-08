"""
Trend Feature Engineering Module.

Provides trend-based technical indicators:
- Simple Moving Averages (SMA)
- Exponential Moving Averages (EMA)
- ADX (Average Directional Index)
- Ichimoku Cloud components
- Supertrend
- VWAP (Volume Weighted Average Price)
- Linear regression slope
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


class TrendFeatures:
    """
    Calculate trend-based technical indicators.
    
    Trend indicators help identify the direction of market momentum
    and the strength of trends.
    """
    
    @staticmethod
    def sma(df: pd.DataFrame, periods: List[int] = None,
            column: str = 'close') -> pd.DataFrame:
        """
        Calculate Simple Moving Averages.
        
        Args:
            df: DataFrame with price data
            periods: List of periods (default [20, 50, 100, 200])
            column: Price column
            
        Returns:
            DataFrame with SMA columns
        """
        periods = periods or [20, 50, 100, 200]
        result = pd.DataFrame(index=df.index)
        
        for period in periods:
            result[f'sma_{period}'] = df[column].rolling(window=period, min_periods=period).mean()
        
        return result
    
    @staticmethod
    def ema(df: pd.DataFrame, periods: List[int] = None,
            column: str = 'close') -> pd.DataFrame:
        """
        Calculate Exponential Moving Averages.
        
        Args:
            df: DataFrame with price data
            periods: List of periods (default [12, 26, 50, 200])
            column: Price column
            
        Returns:
            DataFrame with EMA columns
        """
        periods = periods or [12, 26, 50, 200]
        result = pd.DataFrame(index=df.index)
        
        for period in periods:
            result[f'ema_{period}'] = df[column].ewm(span=period, min_periods=period).mean()
        
        return result
    
    @staticmethod
    def price_vs_ma(df: pd.DataFrame, ma_column: str,
                    column: str = 'close') -> pd.Series:
        """
        Calculate price distance from moving average (z-score).
        
        Args:
            df: DataFrame with price and MA data
            ma_column: Name of MA column
            column: Price column
            
        Returns:
            Series with z-score values
        """
        diff = df[column] - df[ma_column]
        std = diff.rolling(window=20).std()
        zscore = diff / std.replace(0, np.nan)
        
        return zscore
    
    @staticmethod
    def sma_crossover(df: pd.DataFrame, fast: int = 50, 
                      slow: int = 200) -> pd.DataFrame:
        """
        Calculate SMA crossover signals.
        
        Args:
            df: DataFrame with price data
            fast: Fast SMA period
            slow: Slow SMA period
            
        Returns:
            DataFrame with crossover features
        """
        fast_sma = df['close'].rolling(window=fast, min_periods=fast).mean()
        slow_sma = df['close'].rolling(window=slow, min_periods=slow).mean()
        
        result = pd.DataFrame(index=df.index)
        result[f'sma_{fast}'] = fast_sma
        result[f'sma_{slow}'] = slow_sma
        result['sma_trend'] = (fast_sma > slow_sma).astype(int)
        
        # Golden cross (fast crosses above slow)
        result['golden_cross'] = ((fast_sma > slow_sma) & 
                                  (fast_sma.shift(1) <= slow_sma.shift(1))).astype(int)
        
        # Death cross (fast crosses below slow)
        result['death_cross'] = ((fast_sma < slow_sma) & 
                                 (fast_sma.shift(1) >= slow_sma.shift(1))).astype(int)
        
        return result
    
    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX).
        
        ADX measures the strength of a trend, regardless of direction.
        
        Args:
            df: DataFrame with OHLC data
            period: ADX period
            
        Returns:
            DataFrame with ADX, +DI, -DI
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = tr.rolling(window=period, min_periods=period).mean()
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=period, min_periods=period).mean()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=period, min_periods=period).mean()
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm_smooth / atr.replace(0, np.nan)
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(window=period, min_periods=period).mean()
        
        return pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }, index=df.index)
    
    @staticmethod
    def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26,
                 senkou: int = 52) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud components.
        
        Args:
            df: DataFrame with OHLC data
            tenkan: Tenkan-sen period
            kijun: Kijun-sen period
            senkou: Senkou span B period
            
        Returns:
            DataFrame with Ichimoku components
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan, min_periods=tenkan).max()
        tenkan_low = low.rolling(window=tenkan, min_periods=tenkan).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun, min_periods=kijun).max()
        kijun_low = low.rolling(window=kijun, min_periods=kijun).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        
        # Senkou Span B (Leading Span B)
        senkou_high = high.rolling(window=senkou, min_periods=senkou).max()
        senkou_low = low.rolling(window=senkou, min_periods=senkou).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(kijun)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-kijun)
        
        result = pd.DataFrame({
            'ichimoku_tenkan': tenkan_sen,
            'ichimoku_kijun': kijun_sen,
            'ichimoku_senkou_a': senkou_span_a,
            'ichimoku_senkou_b': senkou_span_b,
            'ichimoku_chikou': chikou_span
        }, index=df.index)
        
        # Cloud color (1 = bullish, -1 = bearish)
        result['ichimoku_cloud'] = np.where(
            senkou_span_a > senkou_span_b, 1, -1
        )
        
        # Price position relative to cloud
        result['price_above_cloud'] = (
            (close > senkou_span_a) & (close > senkou_span_b)
        ).astype(int)
        result['price_below_cloud'] = (
            (close < senkou_span_a) & (close < senkou_span_b)
        ).astype(int)
        
        # TK cross
        result['tk_cross_up'] = ((tenkan_sen > kijun_sen) & 
                                 (tenkan_sen.shift(1) <= kijun_sen.shift(1))).astype(int)
        result['tk_cross_down'] = ((tenkan_sen < kijun_sen) & 
                                   (tenkan_sen.shift(1) >= kijun_sen.shift(1))).astype(int)
        
        return result
    
    @staticmethod
    def supertrend(df: pd.DataFrame, period: int = 10,
                   multiplier: float = 3.0) -> pd.DataFrame:
        """
        Calculate Supertrend indicator.
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period
            multiplier: ATR multiplier
            
        Returns:
            DataFrame with Supertrend values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        
        # Basic bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Supertrend logic
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=float)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(df)):
            if close.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif close.iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
        
        return pd.DataFrame({
            'supertrend': supertrend,
            'supertrend_direction': direction
        }, index=df.index)
    
    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """
        Calculate VWAP (Volume Weighted Average Price).
        
        VWAP is calculated by adding up the dollars traded for every
        transaction (price multiplied by the number of shares traded)
        and then dividing by the total shares traded.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with VWAP values
        """
        # Typical price
        tp = (df['high'] + df['low'] + df['close']) / 3
        
        # Use tick_volume or real_volume
        volume = df.get('tick_volume', df.get('real_volume', 1))
        
        # VWAP calculation (daily reset)
        # Group by date for daily VWAP
        df_temp = pd.DataFrame({
            'tp': tp,
            'volume': volume,
            'date': df.index.date
        })
        
        vwap = df_temp.groupby('date').apply(
            lambda x: (x['tp'] * x['volume']).cumsum() / x['volume'].cumsum()
        ).reset_index(level=0, drop=True)
        
        vwap.index = df.index
        
        return vwap
    
    @staticmethod
    def linear_regression_slope(df: pd.DataFrame, period: int = 20,
                                column: str = 'close') -> pd.Series:
        """
        Calculate linear regression slope.
        
        Args:
            df: DataFrame with price data
            period: Regression period
            column: Price column
            
        Returns:
            Series with slope values
        """
        def calc_slope(y):
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
            return slope
        
        slope = df[column].rolling(window=period, min_periods=period).apply(calc_slope)
        
        return slope
    
    @staticmethod
    def add_all_trend(df: pd.DataFrame, sma_periods: List[int] = None,
                      ema_periods: List[int] = None) -> pd.DataFrame:
        """
        Add all trend indicators to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            sma_periods: List of SMA periods
            ema_periods: List of EMA periods
            
        Returns:
            DataFrame with added trend features
        """
        df = df.copy()
        sma_periods = sma_periods or [20, 50, 100, 200]
        ema_periods = ema_periods or [12, 26, 50, 200]
        
        # SMAs
        for period in sma_periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period, min_periods=period).mean()
            if period in [20, 50]:
                df[f'price_above_sma{period}'] = (df['close'] > df[f'sma_{period}']).astype(int)
                df[f'price_z_sma{period}'] = TrendFeatures.price_vs_ma(df, f'sma_{period}')
        
        # EMAs
        for period in ema_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, min_periods=period).mean()
        
        # SMA Crossovers
        if 50 in sma_periods and 200 in sma_periods:
            cross_df = TrendFeatures.sma_crossover(df, 50, 200)
            df['sma_trend_50_200'] = cross_df['sma_trend']
            df['golden_cross'] = cross_df['golden_cross']
            df['death_cross'] = cross_df['death_cross']
        
        # ADX
        adx_df = TrendFeatures.adx(df)
        df['adx'] = adx_df['adx']
        df['plus_di'] = adx_df['plus_di']
        df['minus_di'] = adx_df['minus_di']
        df['is_trending'] = (df['adx'] > 25).astype(int)
        df['is_ranging'] = (df['adx'] < 20).astype(int)
        
        # Ichimoku
        ichi_df = TrendFeatures.ichimoku(df)
        for col in ichi_df.columns:
            df[col] = ichi_df[col]
        
        # Supertrend
        st_df = TrendFeatures.supertrend(df)
        df['supertrend'] = st_df['supertrend']
        df['supertrend_direction'] = st_df['supertrend_direction']
        
        # VWAP (if volume available)
        if 'tick_volume' in df.columns or 'real_volume' in df.columns:
            df['vwap'] = TrendFeatures.vwap(df)
            df['price_above_vwap'] = (df['close'] > df['vwap']).astype(int)
        
        # Linear regression slope
        df['lr_slope_20'] = TrendFeatures.linear_regression_slope(df, 20)
        df['lr_slope_50'] = TrendFeatures.linear_regression_slope(df, 50)
        
        return df
