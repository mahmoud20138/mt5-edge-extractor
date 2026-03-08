"""
Momentum Feature Engineering Module.

Provides momentum-based technical indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)
- ROC (Rate of Change)
- MFI (Money Flow Index)
- TSI (True Strength Index)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


class MomentumFeatures:
    """
    Calculate momentum-based technical indicators.
    
    Momentum indicators help identify the speed of price movements
    and potential overbought/oversold conditions.
    """
    
    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI measures the speed and magnitude of recent price changes
        to evaluate overbought or oversold conditions.
        
        Args:
            df: DataFrame with price data
            period: RSI period (default 14)
            column: Price column to use
            
        Returns:
            Series with RSI values (0-100)
        """
        delta = df[column].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        # Use exponential smoothing for subsequent values
        for i in range(period, len(df)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def rsi_multiple_periods(df: pd.DataFrame, periods: List[int] = None,
                            column: str = 'close') -> pd.DataFrame:
        """
        Calculate RSI for multiple periods.
        
        Args:
            df: DataFrame with price data
            periods: List of periods (default [7, 14, 21])
            column: Price column
            
        Returns:
            DataFrame with RSI values for each period
        """
        periods = periods or [7, 14, 21]
        result = pd.DataFrame(index=df.index)
        
        for period in periods:
            result[f'rsi_{period}'] = MomentumFeatures.rsi(df, period, column)
        
        return result
    
    @staticmethod
    def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, 
             signal: int = 9, column: str = 'close') -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD shows the relationship between two exponential moving averages
        of a security's price.
        
        Args:
            df: DataFrame with price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            column: Price column
            
        Returns:
            DataFrame with MACD, Signal, and Histogram
        """
        ema_fast = df[column].ewm(span=fast, min_periods=fast).mean()
        ema_slow = df[column].ewm(span=slow, min_periods=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_hist': histogram
        }, index=df.index)
    
    @staticmethod
    def stochastic(df: pd.DataFrame, k_period: int = 14, 
                   d_period: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        The stochastic oscillator compares a particular closing price
        of a security to a range of its prices over a certain period.
        
        Args:
            df: DataFrame with OHLC data
            k_period: %K period
            d_period: %D period (smoothing)
            
        Returns:
            DataFrame with %K and %D values
        """
        low_min = df['low'].rolling(window=k_period, min_periods=k_period).min()
        high_max = df['high'].rolling(window=k_period, min_periods=k_period).max()
        
        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period, min_periods=d_period).mean()
        
        return pd.DataFrame({
            'stoch_k': k,
            'stoch_d': d
        }, index=df.index)
    
    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Williams %R.
        
        Williams %R is a momentum indicator that measures overbought
        and oversold levels.
        
        Args:
            df: DataFrame with OHLC data
            period: Lookback period
            
        Returns:
            Series with Williams %R values (0 to -100)
        """
        high_max = df['high'].rolling(window=period, min_periods=period).max()
        low_min = df['low'].rolling(window=period, min_periods=period).min()
        
        wr = -100 * (high_max - df['close']) / (high_max - low_min)
        
        return wr
    
    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).
        
        CCI measures the current price level relative to an average
        price level over a given period.
        
        Args:
            df: DataFrame with OHLC data
            period: CCI period
            
        Returns:
            Series with CCI values
        """
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period, min_periods=period).mean()
        mad = tp.rolling(window=period, min_periods=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        cci = (tp - sma) / (0.015 * mad)
        
        return cci
    
    @staticmethod
    def roc(df: pd.DataFrame, period: int = 10, 
            column: str = 'close') -> pd.Series:
        """
        Calculate Rate of Change (ROC).
        
        ROC measures the percentage change in price from one period
        to the next.
        
        Args:
            df: DataFrame with price data
            period: Lookback period
            column: Price column
            
        Returns:
            Series with ROC values
        """
        roc = 100 * (df[column] - df[column].shift(period)) / df[column].shift(period)
        return roc
    
    @staticmethod
    def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index (MFI).
        
        MFI uses both price and volume to measure buying and selling pressure.
        
        Args:
            df: DataFrame with OHLCV data
            period: MFI period
            
        Returns:
            Series with MFI values (0-100)
        """
        tp = (df['high'] + df['low'] + df['close']) / 3
        
        # Use tick_volume or real_volume
        volume = df.get('tick_volume', df.get('real_volume', 1))
        
        raw_money_flow = tp * volume
        
        positive_flow = raw_money_flow.where(tp > tp.shift(1), 0)
        negative_flow = raw_money_flow.where(tp < tp.shift(1), 0)
        
        positive_sum = positive_flow.rolling(window=period, min_periods=period).sum()
        negative_sum = negative_flow.rolling(window=period, min_periods=period).sum()
        
        money_ratio = positive_sum / negative_sum.replace(0, np.nan)
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    @staticmethod
    def tsi(df: pd.DataFrame, fast: int = 13, slow: int = 25,
            column: str = 'close') -> pd.Series:
        """
        Calculate True Strength Index (TSI).
        
        TSI is a momentum oscillator that uses double smoothed price
        changes to minimize noise.
        
        Args:
            df: DataFrame with price data
            fast: Fast EMA period
            slow: Slow EMA period
            column: Price column
            
        Returns:
            Series with TSI values
        """
        momentum = df[column].diff(1)
        
        abs_momentum = abs(momentum)
        
        # Double smoothing
        smoothed_momentum = momentum.ewm(span=slow, min_periods=slow).mean()
        double_smoothed_momentum = smoothed_momentum.ewm(span=fast, min_periods=fast).mean()
        
        smoothed_abs_momentum = abs_momentum.ewm(span=slow, min_periods=slow).mean()
        double_smoothed_abs_momentum = smoothed_abs_momentum.ewm(span=fast, min_periods=fast).mean()
        
        tsi = 100 * (double_smoothed_momentum / double_smoothed_abs_momentum)
        
        return tsi
    
    @staticmethod
    def add_all_momentum(df: pd.DataFrame, 
                        rsi_periods: List[int] = None,
                        macd_params: Tuple[int, int, int] = (12, 26, 9)) -> pd.DataFrame:
        """
        Add all momentum indicators to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            rsi_periods: List of RSI periods
            macd_params: (fast, slow, signal) for MACD
            
        Returns:
            DataFrame with added momentum features
        """
        df = df.copy()
        rsi_periods = rsi_periods or [7, 14, 21]
        
        # RSI
        for period in rsi_periods:
            df[f'rsi_{period}'] = MomentumFeatures.rsi(df, period)
        
        # RSI extremes
        df['rsi_14_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_14_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_14_extreme'] = ((df['rsi_14'] > 70) | (df['rsi_14'] < 30)).astype(int)
        
        # MACD
        macd_df = MomentumFeatures.macd(df, *macd_params)
        df['macd'] = macd_df['macd']
        df['macd_signal'] = macd_df['macd_signal']
        df['macd_hist'] = macd_df['macd_hist']
        df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                               (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & 
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        
        # Stochastic
        stoch_df = MomentumFeatures.stochastic(df)
        df['stoch_k'] = stoch_df['stoch_k']
        df['stoch_d'] = stoch_df['stoch_d']
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        
        # Williams %R
        df['williams_r'] = MomentumFeatures.williams_r(df)
        
        # CCI
        df['cci'] = MomentumFeatures.cci(df)
        df['cci_extreme'] = ((df['cci'] > 200) | (df['cci'] < -200)).astype(int)
        
        # ROC
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = MomentumFeatures.roc(df, period)
        
        # MFI (if volume available)
        if 'tick_volume' in df.columns or 'real_volume' in df.columns:
            df['mfi'] = MomentumFeatures.mfi(df)
        
        # TSI
        df['tsi'] = MomentumFeatures.tsi(df)
        
        return df
