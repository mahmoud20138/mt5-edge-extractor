# MT5 Edge Extractor

**ML-Powered Trading Edge Discovery and Validation Framework for MetaTrader 5**

A comprehensive backtesting and research system that extracts, validates, and reports potential trading edges from historical MT5 data. Tests 100+ edge hypotheses across 7 categories using statistical significance testing, walk-forward validation, and bootstrap confidence intervals. For research purposes only -- no live trades are executed.

---

## Architecture

```
+----------------+     +------------------+     +------------------+
|   MT5 / CSV    | --> |  Data Loader &   | --> | Feature Engine   |
|  (OHLCV Data)  |     |  Preprocessing   |     | (50+ features)   |
+----------------+     +------------------+     +--------+---------+
                                                         |
                                            +------------+------------+
                                            |                         |
                                   +--------+--------+      +--------+--------+
                                   |  Edge Detectors  |      |   Validation    |
                                   |  (7 categories)  |      | Walk-Forward    |
                                   |  100+ hypotheses  |      | Bootstrap       |
                                   +--------+---------+      | Monte Carlo     |
                                            |                +--------+--------+
                                            +-------+--------+
                                                    |
                                           +--------+--------+
                                           |    Reporting     |
                                           | HTML / JSON / CSV|
                                           +-----------------+
```

---

## Edge Categories

### 1. Time-Based Edges (`time_based.py`)

Detects statistical anomalies tied to time:
- **Hour-of-day effects**: Which hours have directional bias
- **Day-of-week effects**: Monday vs Friday return patterns
- **Month-of-year / seasonal patterns**: Recurring annual effects
- **Session-based edges**: Asian, London, NY session biases

### 2. Trend & Momentum Edges (`trend_momentum.py`)

Classical trend-following and momentum signals:
- **Moving average crossovers**: EMA/SMA cross signals with various periods
- **Breakout strategies**: N-bar high/low breakouts
- **Momentum indicators**: RSI, MACD, Stochastic extremes and crossovers
- **ADX trend strength**: Directional movement filters

### 3. Mean Reversion Edges (`mean_reversion.py`)

Counter-trend strategies exploiting overextension:
- **Z-score reversion**: Price deviation from rolling mean (1.0/1.5/2.0/2.5 sigma)
- **Oscillator extremes**: RSI oversold/overbought reversals
- **Gap fills**: Opening gap reversal statistics
- **Bollinger Band bounces**: Band touch reversals

### 4. Volatility Edges (`volatility_edges.py`)

Strategies based on volatility regime changes:
- **Volatility squeeze**: Low-vol compression followed by expansion
- **ATR regime detection**: High vs low volatility performance differences
- **Range contraction/expansion**: NR4/NR7 patterns

### 5. Market Structure Edges (`market_structure.py`)

Price action and structural patterns:
- **Support/resistance breakouts**: Key level breaks with volume confirmation
- **Consecutive candle patterns**: N consecutive up/down bars reversal probability
- **Inside bars / outside bars**: Consolidation breakout patterns

### 6. Pairs & Correlation Edges (`pairs.py`)

Cross-instrument relationship analysis:
- **Correlation edges**: Decorrelation signals between typically correlated pairs
- **Cointegration edges**: Mean-reversion of cointegrated pair spreads
- **Spread Z-score**: Statistical arbitrage signals

### 7. Machine Learning Edges (`machine_learning.py`)

Data-driven pattern discovery:
- **Feature importance analysis**: Which features actually predict returns
- **ML-based edge detection**: Gradient boosting / random forest classifiers
- **Walk-forward ML validation**: Out-of-sample ML performance

---

## Feature Engineering (50+ Features)

| Module | Features |
|--------|---------|
| `features/momentum.py` | RSI (7/14/21), MACD, Stochastic, Williams %R, CCI, ROC |
| `features/trend.py` | SMA (20/50/100/200), EMA (12/26/50/200), ADX, DI+/DI- |
| `features/volatility_features.py` | ATR(14), Bollinger Bands, historical volatility, Garman-Klass |
| `features/volume.py` | Volume SMA (10/20/50), OBV, Volume ROC, relative volume |
| `features/candle_patterns.py` | Doji, hammer, engulfing, inside bar, pin bar, marubozu |

Returns-based features computed at multiple lookback periods (1, 5, 10, 20, 60 bars).

---

## Validation Pipeline

### Walk-Forward Analysis

Prevents overfitting by testing on unseen data:
- Training window: 2 years (configurable)
- Test window: 6 months
- Step: 3 months
- Each edge must show significance across multiple walk-forward windows

### Bootstrap Confidence Intervals

- 10,000 bootstrap samples (configurable)
- 95% confidence intervals for all metrics
- Prevents cherry-picking results

### Statistical Testing

- **p-value threshold**: 0.05 (configurable)
- **Minimum sample size**: 100 trades
- **Multiple hypothesis correction**: Controls false discovery rate

### Monte Carlo Simulation

- 1,000 random permutation runs
- Validates that edge is not due to random chance
- Compares actual results vs shuffled baseline

---

## Transaction Cost Model

Realistic cost assumptions built into all backtests:

| Cost | Default |
|------|---------|
| Spread | 1.5 pips |
| Commission | $7.00 per lot per side |
| Slippage | 0.5 pips (5.0 during news) |
| Spread during news | 3x normal |

---

## Installation

### Prerequisites

- Python 3.10+
- MetaTrader 5 terminal (optional -- falls back to simulated data)

### Setup

```bash
cd mt5_edge_extractor
pip install -r requirements.txt
```

---

## Usage

### Full Analysis

```bash
# Run complete edge analysis on EURUSD H1 (2 years)
python -m mt5_edge_extractor.main

# Specify symbol and timeframe
python -m mt5_edge_extractor.main --symbol GBPUSD --years 3

# Quick test (1 year, minimal output)
python -m mt5_edge_extractor.main --quick
```

### Multi-Symbol Analysis

```bash
# Scan multiple symbols and generate comparison report
python -m mt5_edge_extractor.multi_analysis
```

### Demo Mode

```bash
# Run with simulated data (no MT5 required)
python -m mt5_edge_extractor.demo

# Comprehensive demo with all edge types
python -m mt5_edge_extractor.comprehensive_demo
```

---

## Results & Output

### Console Output

```
==================================================================
  MT5 TRADING EDGE EXTRACTOR - BACKTESTING SYSTEM
==================================================================

Total edges tested: 127
Significant edges: 14

By type:
  - time_based: 3
  - trend_momentum: 4
  - mean_reversion: 3
  - volatility: 2
  - market_structure: 2

----------------------------------------------------------------------
  TOP 10 SIGNIFICANT EDGES (by Sharpe Ratio)
----------------------------------------------------------------------

Edge Name                                Type              Sharpe  Win%
----------------------------------------------------------------------
RSI_14_oversold_bounce                   mean_reversion      1.85  62.3%
London_session_breakout                  time_based          1.52  58.7%
EMA_12_26_cross                          trend_momentum      1.41  55.2%
BB_lower_bounce                          mean_reversion      1.38  61.1%
...
```

### Decision Framework

The system prints a checklist for each significant edge:
1. Is p-value < 0.05?
2. Is sample size > 100?
3. Net of costs still +EV?
4. Works on other symbols?
5. Works on other timeframes?
6. Stable across years (walk-forward)?

All YES = potential real edge. Paper trade, then small live, then scale.

### Report Files

- `edge_analysis_report.json` -- Full JSON report with all metrics
- `comprehensive_edge_report.json` -- Multi-edge comparison
- `multi_analysis_results.csv` -- Cross-symbol results table
- HTML reports with interactive charts (when `generate_html=True`)

### Key Metrics Per Edge

| Metric | Description |
|--------|------------|
| Sharpe Ratio | Risk-adjusted return |
| Win Rate | Percentage of profitable trades |
| Profit Factor | Gross profit / gross loss |
| p-value | Statistical significance |
| Sample Size | Number of trades tested |
| Max Drawdown | Worst peak-to-trough decline |
| is_significant | Boolean: passes all validation checks |

---

## Project Structure

```
mt5_edge_extractor/
  __init__.py
  main.py                           # Entry point
  config.py                         # All configuration dataclasses
  demo.py                           # Quick demo with simulated data
  comprehensive_demo.py             # Full demo with all edge types
  multi_analysis.py                 # Multi-symbol analysis runner
  data/
    data_loader.py                  # Data loading and caching
    mt5_connector.py                # MT5 terminal wrapper
    preprocessing.py                # Data cleaning and normalization
  features/
    __init__.py
    momentum.py                     # RSI, MACD, Stochastic, etc.
    trend.py                        # SMA, EMA, ADX
    volatility_features.py          # ATR, Bollinger, historical vol
    volume.py                       # OBV, Volume SMA, relative volume
    candle_patterns.py              # Doji, hammer, engulfing, etc.
  edges/
    __init__.py
    time_based.py                   # Hour/day/session effects
    trend_momentum.py               # Crossovers, breakouts, momentum
    mean_reversion.py               # Z-score, oscillator, gap fills
    volatility_edges.py             # Squeeze, regime, range patterns
    market_structure.py             # S/R, consecutive candles
    pairs.py                        # Correlation, cointegration
    machine_learning.py             # ML classifiers, feature importance
  engine/
    backtest_engine.py              # Core backtesting engine
  validation/
    bootstrap.py                    # Bootstrap confidence intervals
    statistical_tests.py            # p-value, significance testing
    walk_forward.py                 # Walk-forward out-of-sample testing
  metrics/
    __init__.py                     # Sharpe, PF, win rate, drawdown
  reporting/
    html_report.py                  # Interactive HTML reports
    visualizer.py                   # Matplotlib/Plotly charts
```
