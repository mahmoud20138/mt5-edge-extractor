#!/usr/bin/env python3
"""
MT5 Edge Extraction System - Main Entry Point

This script demonstrates how to use the MT5 Edge Extractor
to find potential trading edges in historical data.

Usage:
    python -m mt5_edge_extractor.main

This is for BACKTESTING ONLY - no live trades are executed.
"""

import sys
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from config import Config, TimeFrame
from engine.backtest_engine import BacktestEngine


def print_header():
    """Print application header."""
    print("\n" + "="*70)
    print("  MT5 TRADING EDGE EXTRACTOR - BACKTESTING SYSTEM")
    print("  For Research Purposes Only - No Live Trading")
    print("="*70 + "\n")


def print_edge_summary(edge_results: dict):
    """Print summary of detected edges."""
    print("\n" + "-"*70)
    print("  DETECTED EDGES SUMMARY")
    print("-"*70)
    
    # Count by type
    by_type = {}
    significant_count = 0
    
    for name, result in edge_results.items():
        edge_type = result.edge_type
        by_type[edge_type] = by_type.get(edge_type, 0) + 1
        if result.is_significant:
            significant_count += 1
    
    print(f"\nTotal edges tested: {len(edge_results)}")
    print(f"Significant edges: {significant_count}")
    print(f"\nBy type:")
    for edge_type, count in by_type.items():
        print(f"  - {edge_type}: {count}")
    
    # Top edges
    print("\n" + "-"*70)
    print("  TOP 10 SIGNIFICANT EDGES (by Sharpe Ratio)")
    print("-"*70)
    
    significant = [r for r in edge_results.values() if r.is_significant]
    sorted_edges = sorted(significant, key=lambda x: x.sharpe_ratio, reverse=True)[:10]
    
    if sorted_edges:
        print(f"\n{'Edge Name':<40} {'Type':<15} {'Sharpe':>8} {'Win%':>6}")
        print("-"*70)
        for edge in sorted_edges:
            # Replace unicode sigma for Windows console compatibility
            name = edge.edge_name[:38].replace('\u03c3', 's').replace('\u00b1', '+/-')
            edge_type = edge.edge_type.replace('\u03c3', 's').replace('\u00b1', '+/-')
            print(f"{name:<40} {edge_type:<15} {edge.sharpe_ratio:>8.2f} {edge.win_rate*100:>5.1f}%")
    else:
        print("\nNo significant edges found.")
    
    print("\n" + "-"*70)
    print("  DECISION FRAMEWORK")
    print("-"*70)
    
    print("""
    Found something interesting?
    
    1. Is p-value < 0.05?          [YES] if any significant edges found
    2. Is sample size > 100?       [CHECK] Verify sample sizes above
    3. Net of costs still +EV?     [CHECK] Factor in spreads/slippage
    4. Works on other symbols?      -> TEST ON MULTIPLE PAIRS
    5. Works on other timeframes?   -> TEST ON MULTIPLE TFs
    6. Stable across years?         -> DO WALK-FORWARD ANALYSIS
    
    ALL YES? -> You may have a real edge.
    -> Paper trade it -> Small live -> Scale up
    """)


def run_analysis():
    """Run the main edge analysis."""
    print_header()
    
    print("Initializing Edge Extractor...")
    
    # Create configuration
    config = Config()
    config.edges.min_p_value = 0.05
    config.validation.bootstrap_samples = 1000
    
    # Create engine
    engine = BacktestEngine(config)
    
    try:
        print("\nLoading data (simulated mode - install MetaTrader5 for live data)...")
        print("This may take a moment...\n")
        
        # Run full analysis
        results = engine.run_full_analysis(
            symbol='EURUSD',
            timeframe=TimeFrame.H1,
            years=2  # 2 years of data
        )
        
        if 'error' in results:
            print(f"\nError: {results['error']}")
            return
        
        # Print summary
        print_edge_summary(results['edge_results'])
        
        # Save report
        report_path = 'edge_analysis_report.json'
        engine.generate_report(report_path)
        print(f"\nFull report saved to: {report_path}")
        
        # Print some statistics
        df = results['data']
        print(f"\nData Statistics:")
        print(f"  - Total bars: {len(df)}")
        print(f"  - Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  - Features: {len(df.columns)}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine.cleanup()
        print("\nAnalysis complete. Thank you for using MT5 Edge Extractor!")


def run_quick_test():
    """Run a quick test with minimal output."""
    print("Running quick edge detection test...")
    
    config = Config()
    engine = BacktestEngine(config)
    
    try:
        results = engine.run_full_analysis(
            symbol='EURUSD',
            timeframe=TimeFrame.H1,
            years=1
        )
        
        significant = [r for r in results['edge_results'].values() if r.is_significant]
        print(f"\nFound {len(significant)} significant edges out of {len(results['edge_results'])} tested")
        
        return len(significant) > 0
        
    finally:
        engine.cleanup()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MT5 Edge Extraction System')
    parser.add_argument('--quick', action='store_true', help='Run quick test')
    parser.add_argument('--symbol', default='EURUSD', help='Symbol to analyze')
    parser.add_argument('--years', type=int, default=2, help='Years of data')
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        run_analysis()
