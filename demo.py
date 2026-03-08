#!/usr/bin/env python3
"""
Quick Demo - MT5 Edge Extraction System

This script demonstrates the edge extraction system with simulated data.
Run this to see the system in action without MT5 connection.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()


def generate_realistic_fx_data(days: int = 500, 
                                symbol: str = 'EURUSD',
                                seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic forex data for testing.
    
    Creates OHLCV data with realistic properties:
    - Different volatility by session
    - Weekend gaps
    - Trends and mean-reversion regimes
    """
    np.random.seed(seed)
    
    # Generate hourly timestamps (skip weekends)
    start = datetime(2022, 1, 1)
    timestamps = []
    current = start
    
    while len(timestamps) < days * 24:
        if current.weekday() < 5:  # Skip weekends
            timestamps.append(current)
        current += timedelta(hours=1)
    
    n = len(timestamps)
    
    # Base parameters
    base_price = 1.1000 if 'EURUSD' in symbol else 1.0
    annual_vol = 0.10
    hourly_vol = annual_vol / np.sqrt(252 * 24)
    
    # Generate returns with session effects
    returns = np.zeros(n)
    
    for i in range(n):
        hour = timestamps[i].hour
        
        # Session volatility multiplier
        if 7 <= hour < 16:  # London session
            vol_mult = 1.5
        elif 12 <= hour < 21:  # NY session
            vol_mult = 1.4
        elif 12 <= hour < 16:  # Overlap
            vol_mult = 1.8
        else:  # Asian/other
            vol_mult = 0.7
        
        # Add slight mean-reversion
        if i > 0:
            recent_return = returns[max(0, i-5):i].sum()
            mr_factor = -0.1 * recent_return
        else:
            mr_factor = 0
        
        # Generate return
        returns[i] = np.random.normal(0, hourly_vol * vol_mult) + mr_factor * 0.0001
    
    # Generate prices
    log_prices = np.log(base_price) + np.cumsum(returns)
    close_prices = np.exp(log_prices)
    
    # Generate OHLC
    open_prices = close_prices * (1 + np.random.normal(0, hourly_vol * 0.3, n))
    
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, hourly_vol * 0.5, n)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, hourly_vol * 0.5, n)))
    
    # Generate volume
    base_volume = 1000
    volumes = base_volume * (1 + np.random.uniform(0, 2, n))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'tick_volume': volumes.astype(int)
    }, index=pd.DatetimeIndex(timestamps))
    
    df.index.name = 'time'
    
    return df


def run_demo():
    """Run the edge extraction demo."""
    print("\n" + "="*70)
    print("  MT5 TRADING EDGE EXTRACTOR - DEMO MODE")
    print("  Using Simulated Forex Data")
    print("="*70)
    
    print("\n[1/5] Generating realistic forex data...")
    df = generate_realistic_fx_data(days=400, symbol='EURUSD')
    print(f"      Generated {len(df)} hourly bars from {df.index[0].date()} to {df.index[-1].date()}")
    
    print("\n[2/5] Preprocessing data and adding time features...")
    from mt5_edge_extractor.data.preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess(df, add_time=True, add_price=True, add_returns=True, add_volatility=True)
    print(f"      Added {len(df.columns)} base features")
    
    print("\n[3/5] Engineering technical indicators...")
    from mt5_edge_extractor.features.momentum import MomentumFeatures
    from mt5_edge_extractor.features.trend import TrendFeatures
    from mt5_edge_extractor.features.volatility_features import VolatilityFeatures
    from mt5_edge_extractor.features.candle_patterns import CandlePatternFeatures
    
    df = MomentumFeatures.add_all_momentum(df)
    df = TrendFeatures.add_all_trend(df)
    df = VolatilityFeatures.add_all_volatility(df)
    df = CandlePatternFeatures.add_all_patterns(df)
    print(f"      Total features: {len(df.columns)}")
    
    print("\n[4/5] Detecting trading edges...")
    from mt5_edge_extractor.edges.time_based import TimeBasedEdges
    from mt5_edge_extractor.edges.trend_momentum import TrendMomentumEdges
    from mt5_edge_extractor.edges.mean_reversion import MeanReversionEdges
    from mt5_edge_extractor.edges.volatility_edges import VolatilityEdges
    
    all_edges = {}
    
    # Time-based edges
    time_edges = TimeBasedEdges()
    time_results = time_edges.run_all_time_edges(df)
    for name, result in time_results.items():
        if result:
            all_edges[f'time_{name}'] = result
    
    # Trend edges
    trend_edges = TrendMomentumEdges()
    trend_results = trend_edges.run_all_trend_edges(df, lookahead=10)
    for name, result in trend_results.items():
        if result:
            all_edges[f'trend_{name}'] = result
    
    # Mean reversion edges
    mr_edges = MeanReversionEdges()
    mr_results = mr_edges.run_all_mr_edges(df, lookahead=10)
    for name, result in mr_results.items():
        if result:
            all_edges[f'mr_{name}'] = result
    
    # Volatility edges
    vol_edges = VolatilityEdges()
    vol_results = vol_edges.run_all_vol_edges(df, lookahead=10)
    for name, result in vol_results.items():
        if result:
            all_edges[f'vol_{name}'] = result
    
    print(f"      Tested {len(all_edges)} edge hypotheses")
    
    print("\n[5/5] Analyzing results...")
    
    # Filter significant edges
    significant = {k: v for k, v in all_edges.items() if v.is_significant}
    
    # Print results
    print("\n" + "-"*70)
    print("  RESULTS SUMMARY")
    print("-"*70)
    
    print(f"\nTotal edges tested: {len(all_edges)}")
    print(f"Significant edges (p < 0.05): {len(significant)}")
    
    # By type
    by_type = {}
    for result in all_edges.values():
        by_type[result.edge_type] = by_type.get(result.edge_type, 0) + 1
    print(f"\nEdges by type:")
    for t, count in by_type.items():
        print(f"  - {t}: {count}")
    
    # Top edges
    print("\n" + "-"*70)
    print("  TOP SIGNIFICANT EDGES (by Sharpe Ratio)")
    print("-"*70)
    
    sorted_edges = sorted(significant.values(), key=lambda x: x.sharpe_ratio, reverse=True)[:10]
    
    if sorted_edges:
        print(f"\n{'Edge Name':<45} {'Type':<15} {'Sharpe':>8} {'Win%':>6}")
        print("-"*70)
        for edge in sorted_edges:
            name = edge.name[:43] if len(edge.name) > 43 else edge.name
            # Replace unicode sigma with 's' for Windows console compatibility
            name = name.replace('\u03c3', 's').replace('\u00b1', '+/-')
            edge_type = edge.edge_type.replace('\u03c3', 's').replace('\u00b1', '+/-')
            print(f"{name:<45} {edge_type:<15} {edge.sharpe_ratio:>8.2f} {edge.win_rate*100:>5.1f}%")
    else:
        print("\nNo significant edges found in this dataset.")
    
    # Decision framework
    print("\n" + "-"*70)
    print("  EDGE VALIDATION CHECKLIST")
    print("-"*70)
    
    print("""
    Before considering any edge for live trading:
    
    [ ] Is p-value < 0.05?            (Statistical significance)
    [ ] Is sample size > 100?         (Sufficient data)
    [ ] Survives walk-forward?        (Out-of-sample validation)
    [ ] Net of costs still +EV?       (After spreads, slippage)
    [ ] Logical explanation exists?   (Economic rationale)
    [ ] Works on other symbols?       (Robustness)
    [ ] Works on other timeframes?    (Not overfit to one TF)
    [ ] Stable across years?          (Regime independence)
    
    All YES? -> Paper trade -> Small live -> Scale up
    """)
    
    # Save report
    report = {
        'generated_at': datetime.now().isoformat(),
        'data_summary': {
            'bars': len(df),
            'start': str(df.index[0]),
            'end': str(df.index[-1]),
            'features': len(df.columns)
        },
        'edge_analysis': {
            'total_tested': len(all_edges),
            'significant': len(significant),
            'by_type': by_type
        },
        'top_edges': [
            {
                'name': e.name,
                'type': e.edge_type,
                'sharpe': e.sharpe_ratio,
                'win_rate': e.win_rate,
                'p_value': e.p_value,
                'sample_size': e.sample_size
            }
            for e in sorted_edges[:10]
        ]
    }
    
    report_path = 'edge_analysis_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    print("\n" + "="*70)
    print("  DEMO COMPLETE")
    print("="*70 + "\n")
    
    return all_edges


if __name__ == '__main__':
    run_demo()
