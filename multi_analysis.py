#!/usr/bin/env python3
"""
Multi-Symbol Multi-Timeframe Edge Analysis

Tests edges across multiple currency pairs and timeframes to find
robust edges that work across different markets.
"""

import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.WARNING, format='%(message)s')

from config import Config, TimeFrame
from engine.backtest_engine import BacktestEngine


def run_multi_analysis():
    """Run edge analysis on multiple symbols and timeframes."""
    
    print("\n" + "="*80)
    print("  MULTI-SYMBOL MULTI-TIMEFRAME EDGE ANALYSIS")
    print("="*80)
    
    # Define test matrix
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    timeframes = [
        (TimeFrame.M15, 'M15'),
        (TimeFrame.H1, 'H1'),
        (TimeFrame.H4, 'H4'),
        (TimeFrame.D1, 'D1'),
    ]
    
    results = []
    
    print(f"\nTesting {len(symbols)} symbols x {len(timeframes)} timeframes = {len(symbols) * len(timeframes)} combinations")
    print("-"*80)
    
    for symbol in symbols:
        for tf_enum, tf_name in timeframes:
            print(f"\nAnalyzing {symbol} @ {tf_name}...", end=" ")
            
            try:
                config = Config()
                engine = BacktestEngine(config)
                
                analysis = engine.run_full_analysis(
                    symbol=symbol,
                    timeframe=tf_enum,
                    years=1
                )
                
                if 'error' in analysis:
                    print(f"ERROR: {analysis['error']}")
                    continue
                
                # Get significant edges
                edge_results = analysis.get('edge_results', {})
                significant = [r for r in edge_results.values() if r.is_significant]
                
                # Get top edge
                if significant:
                    top_edge = max(significant, key=lambda x: x.sharpe_ratio)
                    results.append({
                        'symbol': symbol,
                        'timeframe': tf_name,
                        'total_edges': len(edge_results),
                        'significant_edges': len(significant),
                        'top_edge': top_edge.edge_name,
                        'top_edge_type': top_edge.edge_type,
                        'sharpe': top_edge.sharpe_ratio,
                        'win_rate': top_edge.win_rate,
                        'p_value': top_edge.p_value,
                        'sample_size': top_edge.sample_size
                    })
                    print(f"Found {len(significant)} significant edges (Top: {top_edge.edge_name[:30]} Sharpe={top_edge.sharpe_ratio:.2f})")
                else:
                    results.append({
                        'symbol': symbol,
                        'timeframe': tf_name,
                        'total_edges': len(edge_results),
                        'significant_edges': 0,
                        'top_edge': 'None',
                        'top_edge_type': '-',
                        'sharpe': 0,
                        'win_rate': 0,
                        'p_value': 1,
                        'sample_size': 0
                    })
                    print("No significant edges found")
                
                engine.cleanup()
                
            except Exception as e:
                print(f"ERROR: {str(e)[:50]}")
                results.append({
                    'symbol': symbol,
                    'timeframe': tf_name,
                    'total_edges': 0,
                    'significant_edges': 0,
                    'top_edge': f'Error: {str(e)[:20]}',
                    'top_edge_type': '-',
                    'sharpe': 0,
                    'win_rate': 0,
                    'p_value': 1,
                    'sample_size': 0
                })
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*80)
    print("  RESULTS SUMMARY")
    print("="*80)
    
    # Pivot table for significant edges count
    print("\n1. SIGNIFICANT EDGES COUNT (by Symbol x Timeframe)")
    print("-"*60)
    pivot = df.pivot(index='symbol', columns='timeframe', values='significant_edges')
    print(pivot.to_string())
    
    # Best edges per symbol
    print("\n\n2. TOP EDGE PER COMBINATION")
    print("-"*80)
    print(f"{'Symbol':<10} {'TF':<6} {'Top Edge':<35} {'Sharpe':>8} {'Win%':>6}")
    print("-"*80)
    for _, row in df.sort_values('sharpe', ascending=False).head(20).iterrows():
        edge_name = row['top_edge'][:33] if len(str(row['top_edge'])) > 33 else row['top_edge']
        edge_name = edge_name.replace('\u03c3', 's').replace('\u00b1', '+/-')
        print(f"{row['symbol']:<10} {row['timeframe']:<6} {edge_name:<35} {row['sharpe']:>8.2f} {row['win_rate']*100:>5.1f}%")
    
    # Find edges that work across multiple symbols/timeframes
    print("\n\n3. ROBUST EDGES (work across multiple combinations)")
    print("-"*80)
    
    edge_counts = df[df['significant_edges'] > 0].groupby('top_edge_type').agg({
        'symbol': 'count',
        'sharpe': 'mean',
        'win_rate': 'mean'
    }).rename(columns={'symbol': 'count'}).sort_values('count', ascending=False)
    
    print(f"{'Edge Type':<20} {'Appearances':>12} {'Avg Sharpe':>12} {'Avg Win%':>10}")
    print("-"*60)
    for edge_type, row in edge_counts.head(10).iterrows():
        edge_type = str(edge_type).replace('\u03c3', 's')[:18]
        print(f"{edge_type:<20} {int(row['count']):>12} {row['sharpe']:>12.2f} {row['win_rate']*100:>9.1f}%")
    
    # Best timeframe overall
    print("\n\n4. BEST TIMEFRAME (by avg significant edges)")
    print("-"*60)
    tf_summary = df.groupby('timeframe').agg({
        'significant_edges': 'mean',
        'sharpe': 'mean'
    }).sort_values('significant_edges', ascending=False)
    print(f"{'Timeframe':<10} {'Avg Significant':>15} {'Avg Sharpe':>12}")
    print("-"*40)
    for tf, row in tf_summary.iterrows():
        print(f"{tf:<10} {row['significant_edges']:>15.1f} {row['sharpe']:>12.2f}")
    
    # Best symbol overall
    print("\n\n5. BEST SYMBOL (by avg significant edges)")
    print("-"*60)
    symbol_summary = df.groupby('symbol').agg({
        'significant_edges': 'mean',
        'sharpe': 'mean'
    }).sort_values('significant_edges', ascending=False)
    print(f"{'Symbol':<10} {'Avg Significant':>15} {'Avg Sharpe':>12}")
    print("-"*40)
    for sym, row in symbol_summary.iterrows():
        print(f"{sym:<10} {row['significant_edges']:>15.1f} {row['sharpe']:>12.2f}")
    
    # Save results
    df.to_csv('multi_analysis_results.csv', index=False)
    print(f"\n\nFull results saved to: multi_analysis_results.csv")
    
    print("\n" + "="*80)
    print("  ANALYSIS COMPLETE")
    print("="*80 + "\n")
    
    return df


if __name__ == '__main__':
    run_multi_analysis()
