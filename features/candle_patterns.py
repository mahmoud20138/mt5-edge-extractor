"""
Candlestick Pattern Detection Module.

Provides detection for:
- Single candle patterns (Doji, Hammer, etc.)
- Multi-candle patterns (Engulfing, Morning Star, etc.)
- Pattern context and confirmation
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


class CandlePatternFeatures:
    """
    Detect candlestick patterns.
    
    Candlestick patterns can provide insights into market psychology
    and potential reversals or continuations.
    """
    
    @staticmethod
    def get_candle_properties(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic candle properties.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with candle properties
        """
        o = df['open']
        h = df['high']
        l = df['low']
        c = df['close']
        
        body = c - o
        body_abs = body.abs()
        range_ = h - l
        
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - l
        
        return pd.DataFrame({
            'body': body,
            'body_abs': body_abs,
            'range': range_,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'is_bullish': (body > 0).astype(int),
            'body_ratio': body_abs / range_.replace(0, np.nan),
            'upper_wick_ratio': upper_wick / range_.replace(0, np.nan),
            'lower_wick_ratio': lower_wick / range_.replace(0, np.nan)
        }, index=df.index)
    
    # ============ SINGLE CANDLE PATTERNS ============
    
    @staticmethod
    def doji(df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """
        Detect Doji pattern.
        
        A Doji has a very small body relative to its range,
        indicating indecision in the market.
        
        Args:
            df: DataFrame with OHLC data
            threshold: Maximum body/range ratio to be considered doji
            
        Returns:
            Series with Doji signals (1 = Doji, 0 = not)
        """
        props = CandlePatternFeatures.get_candle_properties(df)
        
        return (props['body_ratio'] < threshold).astype(int)
    
    @staticmethod
    def hammer(df: pd.DataFrame, body_threshold: float = 0.3,
               wick_threshold: float = 2.0) -> pd.Series:
        """
        Detect Hammer pattern.
        
        Hammer has a small body at the top and a long lower wick,
        appearing in downtrends as potential reversal signals.
        
        Args:
            df: DataFrame with OHLC data
            body_threshold: Maximum body/range ratio
            wick_threshold: Minimum lower_wick/body ratio
            
        Returns:
            Series with Hammer signals
        """
        props = CandlePatternFeatures.get_candle_properties(df)
        
        conditions = (
            (props['body_ratio'] < body_threshold) &  # Small body
            (props['lower_wick'] > props['body_abs'] * wick_threshold) &  # Long lower wick
            (props['upper_wick_ratio'] < 0.1)  # Small or no upper wick
        )
        
        return conditions.astype(int)
    
    @staticmethod
    def hanging_man(df: pd.DataFrame, body_threshold: float = 0.3,
                    wick_threshold: float = 2.0) -> pd.Series:
        """
        Detect Hanging Man pattern.
        
        Same shape as Hammer but appears in uptrends.
        
        Args:
            df: DataFrame with OHLC data
            body_threshold: Maximum body/range ratio
            wick_threshold: Minimum lower_wick/body ratio
            
        Returns:
            Series with Hanging Man signals
        """
        return CandlePatternFeatures.hammer(df, body_threshold, wick_threshold)
    
    @staticmethod
    def inverted_hammer(df: pd.DataFrame, body_threshold: float = 0.3,
                        wick_threshold: float = 2.0) -> pd.Series:
        """
        Detect Inverted Hammer pattern.
        
        Small body at bottom with long upper wick.
        
        Args:
            df: DataFrame with OHLC data
            body_threshold: Maximum body/range ratio
            wick_threshold: Minimum upper_wick/body ratio
            
        Returns:
            Series with Inverted Hammer signals
        """
        props = CandlePatternFeatures.get_candle_properties(df)
        
        conditions = (
            (props['body_ratio'] < body_threshold) &
            (props['upper_wick'] > props['body_abs'] * wick_threshold) &
            (props['lower_wick_ratio'] < 0.1)
        )
        
        return conditions.astype(int)
    
    @staticmethod
    def shooting_star(df: pd.DataFrame, body_threshold: float = 0.3,
                      wick_threshold: float = 2.0) -> pd.Series:
        """
        Detect Shooting Star pattern.
        
        Same shape as Inverted Hammer but appears in uptrends.
        
        Args:
            df: DataFrame with OHLC data
            body_threshold: Maximum body/range ratio
            wick_threshold: Minimum upper_wick/body ratio
            
        Returns:
            Series with Shooting Star signals
        """
        return CandlePatternFeatures.inverted_hammer(df, body_threshold, wick_threshold)
    
    @staticmethod
    def marubozu(df: pd.DataFrame, wick_threshold: float = 0.05) -> pd.Series:
        """
        Detect Marubozu pattern.
        
        A candle with very little or no wicks, indicating strong
        momentum in one direction.
        
        Args:
            df: DataFrame with OHLC data
            wick_threshold: Maximum wick/range ratio
            
        Returns:
            Series with Marubozu signals
        """
        props = CandlePatternFeatures.get_candle_properties(df)
        
        conditions = (
            (props['upper_wick_ratio'] < wick_threshold) &
            (props['lower_wick_ratio'] < wick_threshold) &
            (props['body_ratio'] > 0.9)
        )
        
        return conditions.astype(int)
    
    @staticmethod
    def spinning_top(df: pd.DataFrame, body_threshold: float = 0.3,
                     wick_threshold: float = 0.3) -> pd.Series:
        """
        Detect Spinning Top pattern.
        
        Small body with wicks on both sides, indicating indecision.
        
        Args:
            df: DataFrame with OHLC data
            body_threshold: Maximum body/range ratio
            wick_threshold: Minimum wick/range ratio for both wicks
            
        Returns:
            Series with Spinning Top signals
        """
        props = CandlePatternFeatures.get_candle_properties(df)
        
        conditions = (
            (props['body_ratio'] < body_threshold) &
            (props['upper_wick_ratio'] > wick_threshold) &
            (props['lower_wick_ratio'] > wick_threshold)
        )
        
        return conditions.astype(int)
    
    # ============ MULTI-CANDLE PATTERNS ============
    
    @staticmethod
    def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
        """
        Detect Bullish Engulfing pattern.
        
        A bullish candle that completely engulfs the previous
        bearish candle's body.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Series with Bullish Engulfing signals
        """
        o, h, l, c = df['open'], df['high'], df['low'], df['close']
        
        # Current candle is bullish, previous is bearish
        current_bullish = c > o
        prev_bearish = o.shift(1) > c.shift(1)
        
        # Current body engulfs previous body
        engulfs = (o < o.shift(1)) & (o < c.shift(1)) & (c > o.shift(1)) & (c > c.shift(1))
        
        return (current_bullish & prev_bearish & engulfs).astype(int)
    
    @staticmethod
    def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
        """
        Detect Bearish Engulfing pattern.
        
        A bearish candle that completely engulfs the previous
        bullish candle's body.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Series with Bearish Engulfing signals
        """
        o, h, l, c = df['open'], df['high'], df['low'], df['close']
        
        # Current candle is bearish, previous is bullish
        current_bearish = o > c
        prev_bullish = c.shift(1) > o.shift(1)
        
        # Current body engulfs previous body
        engulfs = (c < o.shift(1)) & (c < c.shift(1)) & (o > o.shift(1)) & (o > c.shift(1))
        
        return (current_bearish & prev_bullish & engulfs).astype(int)
    
    @staticmethod
    def morning_star(df: pd.DataFrame) -> pd.Series:
        """
        Detect Morning Star pattern.
        
        Three-candle bullish reversal pattern:
        1. Large bearish candle
        2. Small candle (gap down)
        3. Large bullish candle (gap up)
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Series with Morning Star signals
        """
        o, c = df['open'], df['close']
        
        # First candle: bearish
        first_bearish = o.shift(2) > c.shift(2)
        
        # Second candle: small body, gaps down
        second_small = (c.shift(1) - o.shift(1)).abs() < (o.shift(2) - c.shift(2)) * 0.5
        second_gap_down = c.shift(1) < c.shift(2)
        
        # Third candle: bullish, gaps up
        third_bullish = c > o
        third_gap_up = o > c.shift(1)
        third_large = (c - o) > (o.shift(2) - c.shift(2)) * 0.5
        
        return (first_bearish & second_small & second_gap_down & 
                third_bullish & third_gap_up & third_large).astype(int)
    
    @staticmethod
    def evening_star(df: pd.DataFrame) -> pd.Series:
        """
        Detect Evening Star pattern.
        
        Three-candle bearish reversal pattern:
        1. Large bullish candle
        2. Small candle (gap up)
        3. Large bearish candle (gap down)
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Series with Evening Star signals
        """
        o, c = df['open'], df['close']
        
        # First candle: bullish
        first_bullish = c.shift(2) > o.shift(2)
        
        # Second candle: small body, gaps up
        second_small = (c.shift(1) - o.shift(1)).abs() < (c.shift(2) - o.shift(2)) * 0.5
        second_gap_up = c.shift(1) > c.shift(2)
        
        # Third candle: bearish, gaps down
        third_bearish = o > c
        third_gap_down = o < c.shift(1)
        third_large = (o - c) > (c.shift(2) - o.shift(2)) * 0.5
        
        return (first_bullish & second_small & second_gap_up & 
                third_bearish & third_gap_down & third_large).astype(int)
    
    @staticmethod
    def three_white_soldiers(df: pd.DataFrame) -> pd.Series:
        """
        Detect Three White Soldiers pattern.
        
        Three consecutive bullish candles, each opening within
        the previous body and closing higher.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Series with Three White Soldiers signals
        """
        o, c = df['open'], df['close']
        
        # All three candles are bullish
        all_bullish = (c > o) & (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2))
        
        # Each opens within previous body
        open_within = (o > o.shift(1)) & (o < c.shift(1))
        open_within_prev = (o.shift(1) > o.shift(2)) & (o.shift(1) < c.shift(2))
        
        # Each closes higher
        close_higher = c > c.shift(1)
        close_higher_prev = c.shift(1) > c.shift(2)
        
        return (all_bullish & open_within & open_within_prev & 
                close_higher & close_higher_prev).astype(int)
    
    @staticmethod
    def three_black_crows(df: pd.DataFrame) -> pd.Series:
        """
        Detect Three Black Crows pattern.
        
        Three consecutive bearish candles, each opening within
        the previous body and closing lower.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Series with Three Black Crows signals
        """
        o, c = df['open'], df['close']
        
        # All three candles are bearish
        all_bearish = (o > c) & (o.shift(1) > c.shift(1)) & (o.shift(2) > c.shift(2))
        
        # Each opens within previous body
        open_within = (o < o.shift(1)) & (o > c.shift(1))
        open_within_prev = (o.shift(1) < o.shift(2)) & (o.shift(1) > c.shift(2))
        
        # Each closes lower
        close_lower = c < c.shift(1)
        close_lower_prev = c.shift(1) < c.shift(2)
        
        return (all_bearish & open_within & open_within_prev & 
                close_lower & close_lower_prev).astype(int)
    
    @staticmethod
    def harami(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Harami (inside bar) pattern.
        
        A small candle whose body is contained within the
        previous candle's body.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with bullish and bearish Harami signals
        """
        o, c = df['open'], df['close']
        
        # Previous candle has larger body
        prev_body = (o.shift(1) - c.shift(1)).abs()
        curr_body = (o - c).abs()
        smaller_body = curr_body < prev_body * 0.5
        
        # Current body is inside previous body
        inside = (o > c.shift(1).min()) & (o < o.shift(1).max()) & \
                 (c > c.shift(1).min()) & (c < o.shift(1).max())
        
        # Bullish harami: previous bearish, current bullish
        bullish = (o.shift(1) > c.shift(1)) & (c > o)
        
        # Bearish harami: previous bullish, current bearish
        bearish = (c.shift(1) > o.shift(1)) & (o > c)
        
        return pd.DataFrame({
            'bullish_harami': (smaller_body & inside & bullish).astype(int),
            'bearish_harami': (smaller_body & inside & bearish).astype(int)
        }, index=df.index)
    
    @staticmethod
    def tweezer_top(df: pd.DataFrame, tolerance: float = 0.001) -> pd.Series:
        """
        Detect Tweezer Top pattern.
        
        Two candles with matching highs, indicating resistance.
        
        Args:
            df: DataFrame with OHLC data
            tolerance: Maximum difference ratio for matching highs
            
        Returns:
            Series with Tweezer Top signals
        """
        h = df['high']
        
        # Highs are approximately equal
        high_match = abs(h - h.shift(1)) / h < tolerance
        
        # First candle bullish, second bearish (ideal)
        first_bullish = df['close'].shift(1) > df['open'].shift(1)
        second_bearish = df['open'] > df['close']
        
        return (high_match & first_bullish & second_bearish).astype(int)
    
    @staticmethod
    def tweezer_bottom(df: pd.DataFrame, tolerance: float = 0.001) -> pd.Series:
        """
        Detect Tweezer Bottom pattern.
        
        Two candles with matching lows, indicating support.
        
        Args:
            df: DataFrame with OHLC data
            tolerance: Maximum difference ratio for matching lows
            
        Returns:
            Series with Tweezer Bottom signals
        """
        l = df['low']
        
        # Lows are approximately equal
        low_match = abs(l - l.shift(1)) / l < tolerance
        
        # First candle bearish, second bullish (ideal)
        first_bearish = df['open'].shift(1) > df['close'].shift(1)
        second_bullish = df['close'] > df['open']
        
        return (low_match & first_bearish & second_bullish).astype(int)
    
    @staticmethod
    def piercing_line(df: pd.DataFrame) -> pd.Series:
        """
        Detect Piercing Line pattern.
        
        Two-candle bullish reversal:
        1. Bearish candle
        2. Bullish candle opening below prior low,
           closing above prior midpoint
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Series with Piercing Line signals
        """
        o, c = df['open'], df['close']
        
        # First candle bearish
        first_bearish = o.shift(1) > c.shift(1)
        
        # Second opens below first's low
        opens_below_low = o < df['low'].shift(1)
        
        # Second closes above first's midpoint
        midpoint = (o.shift(1) + c.shift(1)) / 2
        closes_above_mid = c > midpoint
        
        # Second candle bullish
        second_bullish = c > o
        
        return (first_bearish & opens_below_low & 
                closes_above_mid & second_bullish).astype(int)
    
    @staticmethod
    def dark_cloud_cover(df: pd.DataFrame) -> pd.Series:
        """
        Detect Dark Cloud Cover pattern.
        
        Two-candle bearish reversal:
        1. Bullish candle
        2. Bearish candle opening above prior high,
           closing below prior midpoint
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Series with Dark Cloud Cover signals
        """
        o, c = df['open'], df['close']
        
        # First candle bullish
        first_bullish = c.shift(1) > o.shift(1)
        
        # Second opens above first's high
        opens_above_high = o > df['high'].shift(1)
        
        # Second closes below first's midpoint
        midpoint = (o.shift(1) + c.shift(1)) / 2
        closes_below_mid = c < midpoint
        
        # Second candle bearish
        second_bearish = o > c
        
        return (first_bullish & opens_above_high & 
                closes_below_mid & second_bearish).astype(int)
    
    # ============ CONSECUTIVE PATTERNS ============
    
    @staticmethod
    def consecutive_candles(df: pd.DataFrame, 
                           min_count: int = 3,
                           max_count: int = 7) -> pd.DataFrame:
        """
        Count consecutive bullish/bearish candles.
        
        Args:
            df: DataFrame with OHLC data
            min_count: Minimum consecutive candles
            max_count: Maximum to count
            
        Returns:
            DataFrame with consecutive counts
        """
        c, o = df['close'], df['open']
        
        bullish = (c > o).astype(int)
        bearish = (o > c).astype(int)
        
        # Count consecutive
        def count_consecutive(series):
            groups = (series != series.shift(1)).cumsum()
            return series.groupby(groups).cumsum()
        
        bullish_count = count_consecutive(bullish)
        bearish_count = count_consecutive(bearish)
        
        return pd.DataFrame({
            'consecutive_bullish': bullish_count.where(bullish == 1, 0),
            'consecutive_bearish': bearish_count.where(bearish == 1, 0)
        }, index=df.index)
    
    @staticmethod
    def add_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all candlestick patterns to DataFrame.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with pattern columns added
        """
        df = df.copy()
        
        # Single candle patterns
        df['doji'] = CandlePatternFeatures.doji(df)
        df['hammer'] = CandlePatternFeatures.hammer(df)
        df['inverted_hammer'] = CandlePatternFeatures.inverted_hammer(df)
        df['shooting_star'] = CandlePatternFeatures.shooting_star(df)
        df['marubozu'] = CandlePatternFeatures.marubozu(df)
        df['spinning_top'] = CandlePatternFeatures.spinning_top(df)
        
        # Multi-candle patterns
        df['bullish_engulfing'] = CandlePatternFeatures.bullish_engulfing(df)
        df['bearish_engulfing'] = CandlePatternFeatures.bearish_engulfing(df)
        df['morning_star'] = CandlePatternFeatures.morning_star(df)
        df['evening_star'] = CandlePatternFeatures.evening_star(df)
        df['three_white_soldiers'] = CandlePatternFeatures.three_white_soldiers(df)
        df['three_black_crows'] = CandlePatternFeatures.three_black_crows(df)
        
        harami = CandlePatternFeatures.harami(df)
        df['bullish_harami'] = harami['bullish_harami']
        df['bearish_harami'] = harami['bearish_harami']
        
        df['tweezer_top'] = CandlePatternFeatures.tweezer_top(df)
        df['tweezer_bottom'] = CandlePatternFeatures.tweezer_bottom(df)
        df['piercing_line'] = CandlePatternFeatures.piercing_line(df)
        df['dark_cloud_cover'] = CandlePatternFeatures.dark_cloud_cover(df)
        
        # Consecutive candles
        consec = CandlePatternFeatures.consecutive_candles(df)
        df['consecutive_bullish'] = consec['consecutive_bullish']
        df['consecutive_bearish'] = consec['consecutive_bearish']
        
        # Reversal signals (any pattern suggesting reversal)
        df['bullish_reversal_pattern'] = (
            df['hammer'] | df['inverted_hammer'] | df['bullish_engulfing'] |
            df['morning_star'] | df['three_white_soldiers'] | df['bullish_harami'] |
            df['tweezer_bottom'] | df['piercing_line']
        ).astype(int)
        
        df['bearish_reversal_pattern'] = (
            df['shooting_star'] | df['bearish_engulfing'] |
            df['evening_star'] | df['three_black_crows'] | df['bearish_harami'] |
            df['tweezer_top'] | df['dark_cloud_cover']
        ).astype(int)
        
        return df
