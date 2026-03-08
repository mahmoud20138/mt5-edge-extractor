# MT5 Trading Edge Extractor

## 🔬 Comprehensive Backtesting System for Extracting Trading Edges

**⚠️ IMPORTANT: This system is for BACKTESTING ONLY. No live trades are executed.**

---

## 📊 Overview

This comprehensive Python-based system extracts and validates potential trading edges from MetaTrader 5 (MT5) historical data. It implements a complete workflow based on professional quantitative trading methodology.

### Key Features

- **🔌 MT5 Data Integration**: Connect to MT5 terminal and extract OHLCV, tick data, and trade history
- **📈 100+ Technical Indicators**: Momentum, trend, volatility, volume, and candlestick pattern features
- **🎯 Multiple Edge Categories**:
  - Time-based edges (hour effects, day effects, seasonal patterns)
  - Trend/momentum edges (crossovers, breakouts, momentum)
  - Mean-reversion edges (z-score, oscillator extremes, gaps)
  - Volatility edges (squeeze, regime detection)
  - Market structure edges (support/resistance, breakouts)
  - Pairs/correlation edges (cointegration, lead-lag)
  - Machine learning edges (classification, regime detection)
- **📊 Statistical Validation**: T-tests, non-parametric tests, multiple testing corrections
- **🔄 Walk-Forward Validation**: Proper out-of-sample testing
- **🎲 Bootstrap Confidence Intervals**: Robust statistical inference
- **🌐 Web Dashboard**: Interactive visualization of results

---

## 📁 Project Structure

```
mt5_edge_extractor/
├── __init__.py              # Package initialization
├── config.py                # Configuration settings
├── requirements.txt         # Python dependencies
├── main.py                  # Main entry point
├── demo.py                  # Quick demo script
├── comprehensive_demo.py    # Full system demo
│
├── data/                    # Data handling
│   ├── mt5_connector.py     # MT5 connection and data extraction
│   ├── data_loader.py       # Unified data loading interface
│   └── preprocessing.py     # Data cleaning and feature engineering
│
├── features/                # Technical indicators
│   ├── momentum.py          # RSI, MACD, Stochastic, etc.
│   ├── trend.py             # MA, ADX, Ichimoku, etc.
│   ├── volatility_features.py # ATR, Bollinger, Keltner, etc.
│   ├── volume.py            # OBV, VWAP, MFI, etc.
│   └── candle_patterns.py   # 20+ candlestick patterns
│
├── edges/                   # Edge detection modules
│   ├── time_based.py        # Hour, day, session, seasonal effects
│   ├── trend_momentum.py    # Crossovers, breakouts, momentum
│   ├── mean_reversion.py    # Z-score, RSI extremes, gaps
│   ├── volatility_edges.py  # Squeeze, regime, ATR breakout
│   ├── market_structure.py  # S/R levels, swing points
│   ├── pairs.py             # Cointegration, correlation, spread trading
│   └── machine_learning.py  # ML classification, regime detection
│
├── validation/              # Statistical validation
│   ├── statistical_tests.py # Hypothesis testing
│   ├── walk_forward.py      # Walk-forward validation
│   └── bootstrap.py         # Bootstrap confidence intervals
│
├── metrics/                 # Performance metrics
│   └── edge_metrics.py      # Trade and portfolio metrics
│
├── reporting/               # Output generation
│   ├── visualizer.py        # Chart generation
│   └── html_report.py       # HTML report generation
│
└── engine/                  # Main orchestration
    └── backtest_engine.py   # Complete analysis pipeline
```

---

## 🚀 Quick Start

### Run Demo with Simulated Data

```bash
cd /home/z/my-project
python mt5_edge_extractor/demo.py
```

### Run Comprehensive Demo

```bash
python mt5_edge_extractor/comprehensive_demo.py
```

### Access Web Dashboard

The web dashboard is available at the root URL when the server is running.

---

## 🎯 Edge Categories

### 1. Time-Based Edges

| Edge Type | Description | Tests |
|-----------|-------------|-------|
| Hour-of-Day | Returns by trading hour | 24 |
| Day-of-Week | Monday effect, Friday squaring | 5 |
| Session Effects | Asian, London, NY, Overlaps | 5 |
| Monthly Effects | January effect, month-end rebalancing | 12 |
| Weekend Gap | Gap fill tendency | 1 |
| Opening Range | First hour range patterns | 1 |

### 2. Trend/Momentum Edges

| Edge Type | Description |
|-----------|-------------|
| MA Crossovers | Golden/Death cross signals |
| N-Bar Momentum | Autocorrelation, Hurst exponent |
| Donchian Breakouts | Channel breakouts |
| Bollinger Breakouts | Band penetration |
| ADX Trend Filter | Trending vs ranging identification |
| Runs Test | Randomness detection |

### 3. Mean-Reversion Edges

| Edge Type | Description |
|-----------|-------------|
| Z-Score MR | Price deviation from mean |
| RSI Extremes | Overbought/oversold reversals |
| Stochastic Extremes | K/D crossovers at extremes |
| Bollinger Band Reversion | Price returning to bands |
| Consecutive Candle Reversal | Overextension patterns |
| VWAP Deviation | Distance from VWAP |

### 4. Volatility Edges

| Edge Type | Description |
|-----------|-------------|
| Volatility Clustering | ATR autocorrelation |
| Contraction-Expansion | Low vol → high vol moves |
| BB Squeeze | Band inside Keltner |
| NR4/NR7 Patterns | Narrowest range patterns |
| ATR Breakouts | Price moves exceeding N×ATR |
| Regime Detection | Low/Normal/High vol regimes |

### 5. Market Structure Edges

| Edge Type | Description |
|-----------|-------------|
| Round Number Effect | S/R at psychological levels |
| Previous Day Levels | High/Low/Close as S/R |
| Swing Levels | Swing high/low as S/R |
| Breakout Retest | Return to broken levels |
| Failed Breakout Reversal | Price rejecting after breakout |
| HH/HL Trend | Higher highs/lows continuation |

### 6. Pairs/Correlation Edges

| Edge Type | Description |
|-----------|-------------|
| Correlation Breakdown | Divergence from correlated pairs |
| Lead-Lag Relationships | One pair leading another |
| Cointegration | Mean-reverting spreads |
| Currency Strength | Strongest vs weakest currency |

### 7. Machine Learning Edges

| Edge Type | Description |
|-----------|-------------|
| Random Forest | Direction classification |
| Gradient Boosting | Ensemble predictions |
| K-Means Regimes | Market state clustering |
| Isolation Forest | Anomaly detection |
| Feature Importance | Predictive feature ranking |

---

## 📊 Sample Results

```
Total edges tested: 109
Significant edges (p < 0.05): 19

Top Significant Edges:
┌─────────────────────────────────────────────┬──────────────────┬────────┬───────┐
│ Edge Name                                   │ Type             │ Sharpe │ Win%  │
├─────────────────────────────────────────────┼──────────────────┼────────┼───────┤
│ Hour 12:00                                  │ time_hour        │   8.24 │ 55.0% │
│ Swing Levels (10)                           │ structure        │   1.20 │ 62.0% │
│ Swing Levels (5)                            │ structure        │   0.97 │ 58.6% │
│ Z-Score MR (10, ±2.0σ)                      │ mean_reversion   │   0.73 │ 57.4% │
│ Z-Score MR (20, ±2.0σ)                      │ mean_reversion   │   0.45 │ 55.6% │
└─────────────────────────────────────────────┴──────────────────┴────────┴───────┘
```

---

## 🔧 API Usage

### Basic Usage

```python
from mt5_edge_extractor.engine.backtest_engine import BacktestEngine
from mt5_edge_extractor.config import Config, TimeFrame

# Create engine
engine = BacktestEngine(Config())

# Run full analysis
results = engine.run_full_analysis(
    symbol='EURUSD',
    timeframe=TimeFrame.H1,
    years=3
)

# Get significant edges
significant = engine.filter_significant_edges(
    min_samples=100,
    max_p_value=0.05,
    min_win_rate=0.50
)

for name, edge in significant.items():
    print(f"{edge.edge_name}: Sharpe={edge.sharpe_ratio:.2f}, Win%={edge.win_rate*100:.1f}")
```

### Feature Engineering

```python
from mt5_edge_extractor.data.preprocessing import DataPreprocessor
from mt5_edge_extractor.features.momentum import MomentumFeatures
from mt5_edge_extractor.features.trend import TrendFeatures

# Preprocess
df = DataPreprocessor().preprocess(raw_df)

# Add features
df = MomentumFeatures.add_all_momentum(df)
df = TrendFeatures.add_all_trend(df)
```

### Statistical Testing

```python
from mt5_edge_extractor.validation import StatisticalTests, BootstrapValidator

tester = StatisticalTests(significance_level=0.05)

# T-test
result = tester.t_test_one_sample(returns)

# Multiple testing correction
adjusted_pvals = tester.benjamini_hochberg(p_values)

# Bootstrap
bootstrap = BootstrapValidator(n_bootstrap=10000)
result = bootstrap.bootstrap_sharpe(returns)
```

### Visualization

```python
from mt5_edge_extractor.reporting import EdgeVisualizer, HTMLReportGenerator

viz = EdgeVisualizer()

# Generate charts
equity_fig = viz.plot_equity_curve(returns)
dd_fig = viz.plot_drawdown(returns)
heatmap_fig = viz.plot_monthly_returns(returns)

# Generate HTML report
reporter = HTMLReportGenerator()
html = reporter.generate_report(edge_results, df, returns)
```

---

## ✅ Decision Framework

Before considering any edge for live trading:

| Criterion | Check |
|-----------|-------|
| Is p-value < 0.05? | Statistical significance |
| Is sample size > 100? | Sufficient data |
| Survives walk-forward? | Out-of-sample validation |
| Net of costs still +EV? | After spreads/slippage |
| Logical explanation exists? | Economic rationale |
| Works on other symbols? | Robustness |
| Works on other timeframes? | Not overfit to one TF |
| Stable across years? | Regime independence |

**ALL YES? → Paper trade → Small live → Scale up**

---

## ⚠️ Common Pitfalls Avoided

This system is designed to avoid common backtesting mistakes:

| Pitfall | Solution |
|---------|----------|
| Data Mining Bias | Multiple testing corrections (Bonferroni, FDR) |
| Look-Ahead Bias | Strict point-in-time data handling |
| Survivorship Bias | Can include delisted symbols |
| Overfitting | Walk-forward validation, out-of-sample testing |
| Selection Bias | Test across all available data |

---

## 📦 Installation

### Requirements

```bash
pip install numpy pandas scipy statsmodels scikit-learn matplotlib seaborn
```

### Optional (for MT5 connection - Windows only)

```bash
pip install MetaTrader5
```

---

## 📋 Output

The system generates:

1. **JSON Report**: Complete analysis with all edge statistics
2. **Console Summary**: Top edges, win rates, Sharpe ratios
3. **HTML Report**: Styled report with visualizations
4. **Web Dashboard**: Interactive visualization

---

## 📜 License

MIT License - Use at your own risk.

---

## ⚠️ Disclaimer

**THIS IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY.**

- No live trading is performed
- Past performance does not guarantee future results
- Always paper trade before risking real capital
- Consider transaction costs, slippage, and market conditions
- Use proper risk management

---

## 📚 References

- Moskowitz et al. (2012): Time-Series Momentum
- Lustig & Verdelhan (2007): Carry Trade
- Mandelbrot (1963): Volatility Clustering
- Gao et al. (2018): Intraday Momentum

---

**MT5 Edge Extraction System v1.0.0**
