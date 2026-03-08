#!/usr/bin/env python3
"""
Comprehensive Edge Analysis Demo

This script demonstrates all the edge detection capabilities:
- Time-based edges
- Trend/momentum edges
- Mean-reversion edges
- Volatility edges
- Market structure edges
- Pairs/correlation edges
- Machine learning edges
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


def generate_multi_pair_data(days: int = 500, seed: int = 42) -> dict:
    """Generate realistic forex data for multiple pairs."""
    np.random.seed(seed)
    
    start = datetime(2022, 1, 1)
    timestamps = []
    current = start
    
    while len(timestamps) < days * 24:
        if current.weekday() < 5:
            timestamps.append(current)
        current += timedelta(hours=1)
    
    n = len(timestamps)
    
    # Generate correlated pairs
    pairs = {}
    
    # Base EURUSD
    base_price = 1.1000
    hourly_vol = 0.10 / np.sqrt(252 * 24)
    
    returns = np.random.normal(0, hourly_vol, n)
    
    # Add session effects
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        if 7 <= hour < 16:  # London
            returns[i] *= 1.3
        elif 12 <= hour < 21:  # NY
            returns[i] *= 1.2
    
    log_prices = np.log(base_price) + np.cumsum(returns)
    close = np.exp(log_prices)
    
    open_ = close * (1 + np.random.normal(0, hourly_vol * 0.3, n))
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, hourly_vol * 0.5, n)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, hourly_vol * 0.5, n)))
    volume = 1000 * (1 + np.random.uniform(0, 2, n))
    
    pairs['EURUSD'] = pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'tick_volume': volume.astype(int)
    }, index=pd.DatetimeIndex(timestamps))
    
    # GBPUSD correlated with EURUSD
    gbp_base = 1.2500
    correlation = 0.7
    correlated_returns = correlation * returns + np.sqrt(1 - correlation**2) * np.random.normal(0, hourly_vol, n)
    
    log_prices = np.log(gbp_base) + np.cumsum(correlated_returns)
    close = np.exp(log_prices)
    open_ = close * (1 + np.random.normal(0, hourly_vol * 0.3, n))
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, hourly_vol * 0.5, n)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, hourly_vol * 0.5, n)))
    
    pairs['GBPUSD'] = pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'tick_volume': volume.astype(int)
    }, index=pd.DatetimeIndex(timestamps))
    
    # USDJPY inversely correlated
    jpy_base = 130.0
    correlation = -0.5
    correlated_returns = correlation * returns + np.sqrt(1 - correlation**2) * np.random.normal(0, hourly_vol, n)
    
    log_prices = np.log(jpy_base) + np.cumsum(correlated_returns)
    close = np.exp(log_prices)
    open_ = close * (1 + np.random.normal(0, hourly_vol * 0.3, n))
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, hourly_vol * 0.5, n)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, hourly_vol * 0.5, n)))
    
    pairs['USDJPY'] = pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'tick_volume': volume.astype(int)
    }, index=pd.DatetimeIndex(timestamps))
    
    return pairs


def run_comprehensive_demo():
    """Run comprehensive edge detection demo."""
    print("\n" + "="*80)
    print("  MT5 TRADING EDGE EXTRACTOR - COMPREHENSIVE DEMO")
    print("  All Edge Detection Modules")
    print("="*80)
    
    # Generate data
    print("\n[1/8] Generating multi-pair forex data...")
    pairs_data = generate_multi_pair_data(days=400)
    df = pairs_data['EURUSD'].copy()
    print(f"      Generated {len(df)} bars for {len(pairs_data)} pairs")
    
    # Preprocessing
    print("\n[2/8] Preprocessing and feature engineering...")
    from mt5_edge_extractor.data.preprocessing import DataPreprocessor
    from mt5_edge_extractor.features.momentum import MomentumFeatures
    from mt5_edge_extractor.features.trend import TrendFeatures
    from mt5_edge_extractor.features.volatility_features import VolatilityFeatures
    from mt5_edge_extractor.features.candle_patterns import CandlePatternFeatures
    
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess(df)
    df = MomentumFeatures.add_all_momentum(df)
    df = TrendFeatures.add_all_trend(df)
    df = VolatilityFeatures.add_all_volatility(df)
    df = CandlePatternFeatures.add_all_patterns(df)
    print(f"      Total features: {len(df.columns)}")
    
    all_edges = {}
    
    # Time-based edges
    print("\n[3/8] Detecting TIME-BASED edges...")
    from mt5_edge_extractor.edges.time_based import TimeBasedEdges
    time_edges = TimeBasedEdges()
    results = time_edges.run_all_time_edges(df)
    significant_time = sum(1 for r in results.values() if r and r.is_significant)
    all_edges.update({f'time_{k}': v for k, v in results.items() if v})
    print(f"      Found {significant_time} significant time-based edges")
    
    # Trend/momentum edges
    print("\n[4/8] Detecting TREND/MOMENTUM edges...")
    from mt5_edge_extractor.edges.trend_momentum import TrendMomentumEdges
    trend_edges = TrendMomentumEdges()
    results = trend_edges.run_all_trend_edges(df, lookahead=10)
    significant_trend = sum(1 for r in results.values() if r and r.is_significant)
    all_edges.update({f'trend_{k}': v for k, v in results.items() if v})
    print(f"      Found {significant_trend} significant trend edges")
    
    # Mean-reversion edges
    print("\n[5/8] Detecting MEAN-REVERSION edges...")
    from mt5_edge_extractor.edges.mean_reversion import MeanReversionEdges
    mr_edges = MeanReversionEdges()
    results = mr_edges.run_all_mr_edges(df, lookahead=10)
    significant_mr = sum(1 for r in results.values() if r and r.is_significant)
    all_edges.update({f'mr_{k}': v for k, v in results.items() if v})
    print(f"      Found {significant_mr} significant mean-reversion edges")
    
    # Volatility edges
    print("\n[6/8] Detecting VOLATILITY edges...")
    from mt5_edge_extractor.edges.volatility_edges import VolatilityEdges
    vol_edges = VolatilityEdges()
    results = vol_edges.run_all_vol_edges(df, lookahead=10)
    significant_vol = sum(1 for r in results.values() if r and r.is_significant)
    all_edges.update({f'vol_{k}': v for k, v in results.items() if v})
    print(f"      Found {significant_vol} significant volatility edges")
    
    # Market structure edges
    print("\n[7/8] Detecting MARKET STRUCTURE edges...")
    from mt5_edge_extractor.edges.market_structure import MarketStructureEdges
    struct_edges = MarketStructureEdges()
    results = struct_edges.run_all_structure_edges(df, lookahead=10)
    significant_struct = sum(1 for r in results.values() if r and r.is_significant)
    all_edges.update({f'struct_{k}': v for k, v in results.items() if v})
    print(f"      Found {significant_struct} significant structure edges")
    
    # Pairs/correlation edges
    print("\n[8/8] Detecting PAIRS/CORRELATION edges...")
    from mt5_edge_extractor.edges.pairs import PairsEdgeDetector
    pairs_detector = PairsEdgeDetector()
    
    price_dict = {sym: data['close'] for sym, data in pairs_data.items()}
    results = pairs_detector.run_all_pairs_edges(price_dict, single_pair=('EURUSD', 'GBPUSD'))
    significant_pairs = sum(1 for r in results.values() if r and r.is_significant)
    all_edges.update({f'pairs_{k}': v for k, v in results.items() if v})
    print(f"      Found {significant_pairs} significant pairs edges")
    
    # Summary
    total_significant = sum(1 for r in all_edges.values() if r.is_significant)
    
    print("\n" + "-"*80)
    print("  COMPREHENSIVE RESULTS SUMMARY")
    print("-"*80)
    
    print(f"\nTotal edges tested: {len(all_edges)}")
    print(f"Significant edges (p < 0.05): {total_significant}")
    
    # By type
    by_type = {}
    for result in all_edges.values():
        t = result.edge_type
        by_type[t] = by_type.get(t, 0) + 1
    
    print(f"\nBy edge type:")
    for t, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {t}: {count}")
    
    # Top edges
    significant = [r for r in all_edges.values() if r.is_significant]
    sorted_edges = sorted(significant, key=lambda x: x.sharpe_ratio, reverse=True)[:15]
    
    print("\n" + "-"*80)
    print("  TOP 15 SIGNIFICANT EDGES")
    print("-"*80)
    
    print(f"\n{'Edge Name':<45} {'Type':<18} {'Sharpe':>8} {'Win%':>6}")
    print("-"*80)
    
    for edge in sorted_edges:
        name = edge.name[:43] if len(edge.name) > 43 else edge.name
        # Replace unicode sigma with 's' for Windows console compatibility
        name = name.replace('\u03c3', 's').replace('\u00b1', '+/-')
        edge_type = edge.edge_type.replace('\u03c3', 's').replace('\u00b1', '+/-')
        print(f"{name:<45} {edge_type:<18} {edge.sharpe_ratio:>8.2f} {edge.win_rate*100:>5.1f}%")
    
    # Validation checklist
    print("\n" + "-"*80)
    print("  EDGE VALIDATION CHECKLIST")
    print("-"*80)
    
    print("""
    [ ] Is p-value < 0.05?            (Statistical significance)
    [ ] Is sample size > 100?         (Sufficient data)
    [ ] Survives walk-forward?        (Out-of-sample validation)
    [ ] Net of costs still +EV?       (After spreads, slippage)
    [ ] Logical explanation exists?   (Economic rationale)
    [ ] Works on other symbols?       (Robustness)
    [ ] Works on other timeframes?    (Not overfit to one TF)
    [ ] Stable across years?          (Regime independence)
    
    ALL YES? -> Paper trade -> Small live -> Scale up
    """)
    
    # Save comprehensive report
    report = {
        'generated_at': datetime.now().isoformat(),
        'data_summary': {
            'bars': len(df),
            'start': str(df.index[0]),
            'end': str(df.index[-1]),
            'features': len(df.columns),
            'pairs': list(pairs_data.keys())
        },
        'edge_analysis': {
            'total_tested': len(all_edges),
            'significant': total_significant,
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
            for e in sorted_edges
        ]
    }
    
    report_path = 'comprehensive_edge_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    print("\n" + "="*80)
    print("  DEMO COMPLETE - Open the web dashboard for visualization!")
    print("="*80 + "\n")
    
    return all_edges


if __name__ == '__main__':
    run_comprehensive_demo()
