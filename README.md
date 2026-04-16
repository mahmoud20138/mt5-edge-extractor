# 🔬 MT5 Edge Extractor ML

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/MetaTrader-5-orange.svg" alt="MT5">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Edges-7+-yellow.svg" alt="7+ Edge Types">
  <img src="https://img.shields.io/badge/Features-50+-red.svg" alt="50+ Features">
</p>

A comprehensive **algorithmic trading edge detection system** that analyzes historical market data to discover statistically significant trading edges. Uses multiple detection methods including statistical analysis, machine learning, and pattern recognition with rigorous validation through bootstrap, walk-forward, and Monte Carlo methods.

> ⚠️ **Disclaimer**: This system is for **BACKTESTING & RESEARCH ONLY** - no live trades are executed.

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 📊 **7 Edge Types** | Time-based, Trend/Momentum, Mean Reversion, Volatility, Market Structure, Pairs, ML |
| 📈 **50+ Features** | Momentum, Trend, Volatility, Volume, Candle Patterns |
| 🧪 **Rigorous Validation** | Bootstrap, Walk-Forward, Monte Carlo, Statistical Tests |
| 📉 **Statistical Testing** | T-tests, Chi-square, ANOVA, Effect Sizes, Multiple Testing Correction |
| 🤖 **Machine Learning** | Random Forest, Gradient Boosting, Feature Importance Analysis |
| 📑 **Comprehensive Reports** | JSON, HTML, Excel, Interactive Visualizations |
| 🔌 **MT5 Integration** | Direct connection to MetaTrader 5 for live data |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EDGE EXTRACTOR ML SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐   │
│  │   MT5 Terminal  │ ───► │   Data Loader   │ ───► │  Preprocessor   │   │
│  │  (Live Data)    │      │  (OHLCV Fetch)  │      │ (Clean/Norm)    │   │
│  └─────────────────┘      └──────────────────┘      └────────┬─────────┘   │
│                                                               │              │
│                                                               ▼              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         FEATURE ENGINEERING                         │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  📈 Momentum    │  📊 Trend    │  🌊 Volatility   │  📉 Volume     │   │
│  │  RSI, MACD     │ EMA/SMA/ADX  │ ATR, BB, Keltner  │ OBV, VWAP      │   │
│  │  Stochastic    │ Ichimoku     │ Asian Range      │ Volume SMA     │   │
│  │  Williams %R   │              │                  │                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        EDGE DETECTION LAYER                          │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                         │
│  │  ⏰ Time-Based    │  📈 Trend/Momentum   │  🔄 Mean Reversion        │
│  │  Hour/Day/Season │ EMA Crossover/Breakout│ Z-Score/Oscillator        │
│  │                  │                      │                           │
│  │  🌊 Volatility   │  🏗️ Market Structure  │  🔗 Pairs/Correlation    │
│  │  BB Squeeze      │ S/R, BOS, Order Blocks│ Correlation/Cointegration │
│  │                  │                      │                           │
│  │  🤖 Machine Learning                                            │   │
│  │  Random Forest, Gradient Boosting, Feature Importance             │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      STATISTICAL VALIDATION                          │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                         │
│  │  📊 Bootstrap     │  🚶 Walk-Forward   │  🎲 Monte Carlo           │
│  │  10,000 samples   │  2yr train/6mo test│  1,000 simulations        │
│  │                  │  Rolling window    │  Confidence intervals     │
│  │                                                                         │
│  │  🧪 Statistical Tests                                                │   │
│  │  T-test, Chi-square, ANOVA, Kruskal-Wallis                          │   │
│  │  Effect Size (Cohen's d), Multiple Testing Correction (Bonferroni)  │   │
│  │                                                                         │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │   📑 JSON   │ │   🌐 HTML   │ │   📊 Excel  │ │   📈 Plots  │          │
│  │   Report    │ │   Report    │ │   Export    │ │   Charts    │          │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MT5 Terminal ─────────────────────────────────────────────► OHLCV Data   │
│       │                                                                   │
│       ▼                                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                      DATA LOADER                                    │   │
│  │  • Fetch OHLCV from MT5                                            │   │
│  │  • Handle multiple timeframes (M1-MN1)                            │   │
│  │  • Cache data for fast re-runs                                    │   │
│  │  • Simulated mode for testing without MT5                         │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│       │                                                                   │
│       ▼                                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                    PREPROCESSOR                                     │   │
│  │  • Handle missing data (ffill, bfill, interpolation)              │   │
│  │  • Remove outliers (Z-score > 3)                                  │   │
│  │  • Normalize/scale features                                        │   │
│  │  • Calculate returns and log returns                              │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│       │                                                                   │
│       ▼                                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                    FEATURE ENGINEERING                              │   │
│  │                                                                         │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │   │
│  │  │  Momentum   │ │    Trend     │ │ Volatility  │ │   Volume    │   │   │
│  │  │             │ │             │ │             │ │             │   │   │
│  │  │ RSI (7,14,21)│ │ EMA (12,26) │ │ ATR (14)    │ │ Volume SMA  │   │   │
│  │  │ MACD        │ │ SMA (20,50) │ │ BB (20,2)  │ │ OBV         │   │   │
│  │  │ Stochastic  │ │ ADX (14)    │ │ Keltner    │ │ VWAP        │   │   │
│  │  │ Williams %R │ │ Ichimoku    │ │ Asian Range│ │ Volume ROC  │   │   │
│  │  │ CCI         │ │             │ │ Gap        │ │             │   │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │   │
│  │                                                                         │
│  │  ┌─────────────┐ ┌─────────────────────────────────────────────┐   │   │
│  │  │  Candle     │ │              Return Features                  │   │   │
│  │  │  Patterns   │ │                                               │   │   │
│  │  │ Doji        │ │ Returns (1,5,10,20,60 periods)               │   │   │
│  │  │ Hammer      │ │ Log Returns                                  │   │   │
│  │  │ Engulfing   │ │ Rolling Returns                              │   │   │
│  │  │ Harami      │ │                                              │   │   │
│  │  └─────────────┘ └─────────────────────────────────────────────┘   │   │
│  │                                                                         │
│  └────────────────────────────────────────────────────────────────────┘   │
│       │                                                                   │
│       ▼                                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                       EDGE DETECTION                                 │   │
│  │  For each edge type:                                               │   │
│  │  1. Generate signals based on edge conditions                      │   │
│  │  2. Calculate returns for each signal                              │   │
│  │  3. Aggregate returns                                              │   │
│  │  4. Calculate metrics (Sharpe, Win Rate, Drawdown)                │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│       │                                                                   │
│       ▼                                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                    STATISTICAL VALIDATION                          │   │
│  │                                                                         │
│  │  1. T-test: Is mean return significantly different from zero?     │   │
│  │  2. Bootstrap: Confidence intervals on metrics                     │   │
│  │  3. Walk-Forward: Does edge work on unseen data?                  │   │
│  │  4. Monte Carlo: Robustness under random variations                │   │
│  │                                                                         │
│  └────────────────────────────────────────────────────────────────────┘   │
│       │                                                                   │
│       ▼                                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                    RESULTS REPORTING                                │   │
│  │  • JSON: Machine-readable results                                  │   │
│  │  • HTML: Interactive dashboard                                      │   │
│  │  • Excel: Detailed metrics                                         │   │
│  │  • Plots: Visual analysis                                          │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Edge Types

### 1. ⏰ Time-Based Edges (`edges/time_based.py`)

| Edge Name | Description | Signal Condition |
|-----------|-------------|-------------------|
| **Hour of Day** | Trading edge by hour | Buy/sell at specific hours |
| **Day of Week** | Day-of-week effect | Certain days outperform |
| **Month of Year** | Seasonal patterns | Monthly returns vary |
| **Quarter End** | Quarter boundary effects | Position around quarters |
| **Session Effects** | Asian/London/NY sessions | Session-specific biases |

### 2. 📈 Trend/Momentum Edges (`edges/trend_momentum.py`)

| Edge Name | Description | Signal Condition |
|-----------|-------------|-------------------|
| **EMA Crossover** | Fast/Slow EMA crossing | Fast EMA crosses above/below slow |
| **SMA Breakout** | Price breaks above/below SMA | Close > MA + threshold |
| **Momentum Burst** | Strong recent momentum | Return > X% over N periods |
| **ADX Trend Strength** | Strong trend indicator | ADX > threshold |
| **MACD Crossover** | MACD signal line cross | MACD crosses signal |

### 3. 🔄 Mean Reversion Edges (`edges/mean_reversion.py`)

| Edge Name | Description | Signal Condition |
|-----------|-------------|-------------------|
| **Z-Score Reversion** | Price reverts to mean | Z-score > threshold, expect reversal |
| **RSI Extremes** | RSI overbought/oversold | RSI < 30 or > 70 |
| **Bollinger Bounce** | Price bounces off BB | Touch lower/upper band |
| **Gap Fill** | Gaps get filled | Open gap from previous close |
| **ATR Reversal** | Extreme ATR expansion | ATR > N * average |

### 4. 🌊 Volatility Edges (`edges/volatility_edges.py`)

| Edge Name | Description | Signal Condition |
|-----------|-------------|-------------------|
| **BB Squeeze** | Low volatility expansion | BB width < threshold, expect move |
| **Volatility Breakout** | High volatility expansion | ATR > N * average |
| **Volatility Regime** | Regime-based trading | Different strategies per regime |
| **NR4/NR7** | Narrow range patterns | 4-bar/7-bar low volatility |

### 5. 🏗️ Market Structure Edges (`edges/market_structure.py`)

| Edge Name | Description | Signal Condition |
|-----------|-------------|-------------------|
| **Support/Resistance** | Bounce off S/R | Price approaches S/R level |
| **Break of Structure (BOS)** | Trend continuation | Price breaks swing point |
| **Order Block** | Institutional order zones | Price enters order block area |
| **Fair Value Gap (FVG)** | Gap in fair value | Price fills FVG |
| **Consecutive Candles** | N consecutive up/down | Reversal probability |

### 6. 🔗 Pairs/Correlation Edges (`edges/pairs.py`)

| Edge Name | Description | Signal Condition |
|-----------|-------------|-------------------|
| **Correlation Edge** | Correlated pairs | Spread deviates from correlation |
| **Cointegration** | Long-term equilibrium | Spread deviates from mean |
| **Spread Trading** | Pairs trading | Long one, short other |

### 7. 🤖 Machine Learning Edges (`edges/machine_learning.py`)

| Edge Name | Description | Signal Condition |
|-----------|-------------|-------------------|
| **Random Forest** | Ensemble tree model | ML prediction > threshold |
| **Gradient Boosting** | Sequential boosting | GB prediction signal |
| **Feature Importance** | Key feature analysis | Trade on top features |

---

## 📈 Feature Engineering (50+ Features)

### Momentum Features (`features/momentum.py`)

| Feature | Description | Periods |
|---------|-------------|---------|
| RSI | Relative Strength Index | 7, 14, 21 |
| MACD | Moving Average Convergence Divergence | 12, 26, 9 |
| Stochastic | Stochastic Oscillator | %K, %D |
| Williams %R | Williams Percent Range | 14 |
| CCI | Commodity Channel Index | 14 |
| ROC | Rate of Change | 10 |

### Trend Features (`features/trend.py`)

| Feature | Description | Periods |
|---------|-------------|---------|
| EMA | Exponential Moving Average | 12, 26, 50, 200 |
| SMA | Simple Moving Average | 20, 50, 100, 200 |
| ADX | Average Directional Index | 14 |
| DI+/DI- | Directional Indicators | 14 |
| Ichimoku | Ichimoku Cloud | Default |

### Volatility Features (`features/volatility_features.py`)

| Feature | Description | Periods |
|---------|-------------|---------|
| ATR | Average True Range | 14 |
| Bollinger Bands | BB Upper/Lower/Mid | 20, 2 |
| Keltner Channel | KC Upper/Lower | 20, 2 |
| Asian Range | High-Low of Asian session | - |
| Gap | Overnight gap calculation | - |
| Historical Vol | Rolling volatility | 20 |

### Volume Features (`features/volume.py`)

| Feature | Description | Periods |
|---------|-------------|---------|
| Volume SMA | Simple Moving Average | 10, 20, 50 |
| OBV | On-Balance Volume | - |
| VWAP | Volume Weighted Average Price | - |
| Volume ROC | Rate of Change | 10 |
| Relative Volume | vs average volume | - |

### Candle Patterns (`features/candle_patterns.py`)

| Pattern | Description |
|---------|-------------|
| Doji | Indecision candle |
| Hammer/Hanging Man | Reversal signal |
| Engulfing | Bullish/Bearish reversal |
| Harami | Inside candle reversal |
| Morning/Evening Star | Three-candle reversal |
| Pin Bar | Wick-heavy reversal |
| Marubozu | Full-bodied candle |

---

## 🧪 Statistical Validation

### Test Methods (`validation/`)

| Method | Module | Description | Purpose |
|--------|--------|-------------|---------|
| **T-Test** | `statistical_tests.py` | One-sample t-test | Is mean return > 0? |
| **Mann-Whitney U** | `statistical_tests.py` | Non-parametric test | Non-normal distributions |
| **Chi-Square** | `statistical_tests.py` | Categorical test | Pattern frequency |
| **ANOVA** | `statistical_tests.py` | Multiple group comparison | Different conditions |
| **Bootstrap** | `bootstrap.py` | Resampling (10,000 samples) | Confidence intervals |
| **Walk-Forward** | `walk_forward.py` | Rolling train/test | Out-of-sample validation |
| **Monte Carlo** | N/A | 1,000 simulations | Robustness testing |

### Multiple Testing Correction

| Method | Description |
|--------|-------------|
| **Bonferroni** | Conservative correction |
| **Benjamini-Hochberg** | FDR control |
| **Holm-Bonferroni** | Step-down method |

### Effect Size Metrics

| Metric | Description | Threshold |
|--------|-------------|------------|
| Cohen's d | Standardized mean diff | 0.2 small, 0.5 medium, 0.8 large |
| Pearson's r | Correlation coefficient | 0.1 small, 0.3 medium, 0.5 large |

---

## 📊 Edge Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Sharpe Ratio** | Risk-adjusted return | (Return - RiskFree) / StdDev |
| **Win Rate** | % profitable trades | Wins / Total Trades |
| **Profit Factor** | Gross profit / loss | Gross Profit / Gross Loss |
| **Max Drawdown** | Largest peak-to-trough | (Peak - Trough) / Peak |
| **CAGR** | Compound Annual Growth | (End/Start)^(1/n) - 1 |
| **Sortino** | Downside Sharpe | (Return - Target) / DownsideStd |
| **Calmar** | Return / Max DD | CAGR / Max Drawdown |
| **p-value** | Statistical significance | From t-test |

---

## 💰 Transaction Cost Model

| Cost | Default | Description |
|------|---------|-------------|
| **Spread** | 1.5 pips | Default spread |
| **Commission** | $7.00/lot/side | Per lot commission |
| **Slippage** | 0.5 pips | Normal slippage |
| **News Slippage** | 5.0 pips | During news events |
| **Spread Multiplier** | 3x | During high-impact news |
| **Swap Long** | 0.0 | Overnight financing long |
| **Swap Short** | 0.0 | Overnight financing short |

---

## 📔 Reporting

### Output Formats (`reporting/`)

| Format | Module | Description |
|--------|--------|-------------|
| **JSON** | `engine` | Machine-readable results |
| **HTML** | `html_report.py` | Interactive dashboard |
| **Excel** | N/A | Detailed metrics export |
| **Plots** | `visualizer.py` | Matplotlib/Plotly charts |

### Report Contents

- Executive summary with top edges
- Detailed edge-by-edge analysis
- Statistical test results
- Walk-forward validation results
- Bootstrap confidence intervals
- Monte Carlo simulation results

---

## 🚀 Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| 🐍 Python | 3.10+ |
| 🖥️ OS | Windows/Linux/MacOS |
| 📊 MT5 | Optional (for live data) |

### Setup

```bash
# Clone repository
git clone https://github.com/mahmoud20138/Edge-Extractor-ML.git
cd Edge-Extractor-ML

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
openpyxl>=3.1.0
MetaTrader5>=5.0.45  # Optional
jinja2>=3.1.0
```

---

## 📖 Usage

### Basic Usage

```bash
# Full analysis (2 years of data)
python main.py

# Quick test (1 year)
python main.py --quick

# Custom symbol and timeframe
python main.py --symbol GBPUSD --years 3

# Demo with simulated data
python demo.py

# Comprehensive demo
python comprehensive_demo.py

# Multi-symbol analysis
python multi_analysis.py
```

### Python API

```python
from config import Config, TimeFrame
from engine.backtest_engine import BacktestEngine

# Create configuration
config = Config()
config.edges.min_p_value = 0.05
config.validation.bootstrap_samples = 1000

# Create engine
engine = BacktestEngine(config)

# Run analysis
results = engine.run_full_analysis(
    symbol='EURUSD',
    timeframe=TimeFrame.H1,
    years=2
)

# Get significant edges
significant = [r for r in results['edge_results'].values() 
                if r.is_significant]
print(f"Found {len(significant)} significant edges")

# Generate report
engine.generate_report('my_report.json')
```

### Configuration Options

```python
# MT5 Connection
config.mt5.path = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
config.mt5.login = 12345678
config.mt5.password = "your_password"
config.mt5.server = "YourBroker-Server"

# Data Settings
config.data.default_timeframe = TimeFrame.H1
config.data.lookback_years = 5
config.data.default_symbol = "EURUSD"

# Edge Detection
config.edges.min_p_value = 0.05
config.edges.min_sample_size = 100
config.edges.zscore_thresholds = [1.0, 1.5, 2.0, 2.5]
config.edges.rsi_oversold = 30.0
config.edges.rsi_overbought = 70.0

# Validation
config.validation.bootstrap_samples = 10000
config.validation.wfo_training_years = 2
config.validation.wfo_test_months = 6
config.validation.monte_carlo_runs = 1000

# Transaction Costs
config.costs.default_spread = 1.5
config.costs.commission_per_lot = 7.0
config.costs.default_slippage = 0.5
```

---

## 📁 Project Structure

```
Edge-Extractor-ML/
├── main.py                          # 🎯 Entry point
├── config.py                       # ⚙️ Configuration (all settings)
├── requirements.txt                # 📦 Dependencies
├── LICENSE                         # 📜 MIT License
├── CONTRIBUTING.md                 # 🤝 Contributing guidelines
├── demo.py                         # 🚀 Quick demo script
├── comprehensive_demo.py           # 📊 Full demo
├── multi_analysis.py              # 📈 Multi-symbol analysis
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
│
├── engine/                         # 🧠 Core engine
│   └── backtest_engine.py         # Main orchestration
│
├── edges/                         # 🎯 Edge detection
│   ├── __init__.py
│   ├── time_based.py              # ⏰ Hour/Day/Seasonal effects
│   ├── trend_momentum.py          # 📈 Trend following edges
│   ├── mean_reversion.py          # 🔄 Mean reversion edges
│   ├── volatility_edges.py       # 🌊 Volatility-based edges
│   ├── market_structure.py       # 🏗️ S/R, BOS, OB, FVG
│   ├── pairs.py                  # 🔗 Correlation/Cointegration
│   └── machine_learning.py       # 🤖 ML-based edges
│
├── features/                      # 📈 Feature engineering
│   ├── __init__.py
│   ├── momentum.py               # RSI, MACD, Stochastic
│   ├── trend.py                  # EMA, SMA, ADX, Ichimoku
│   ├── volatility_features.py   # ATR, Bollinger, Keltner
│   ├── volume.py                 # Volume SMA, OBV, VWAP
│   └── candle_patterns.py       # Doji, Engulfing, etc.
│
├── data/                         # 📊 Data handling
│   ├── data_loader.py            # MT5 data fetching
│   ├── preprocessing.py          # Clean, normalize, transform
│   └── mt5_connector.py          # MT5 connection wrapper
│
├── validation/                   # 🧪 Statistical validation
│   ├── statistical_tests.py     # T-test, Chi-square, ANOVA
│   ├── walk_forward.py          # Walk-forward validation
│   └── bootstrap.py             # Bootstrap confidence intervals
│
├── metrics/                      # 📉 Performance metrics
│   └── __init__.py              # Sharpe, Drawdown, etc.
│
├── reporting/                    # 📑 Report generation
│   ├── visualizer.py            # Plotly/Matplotlib charts
│   └── html_report.py           # HTML dashboard
│
└── validation/                   # Additional validation
    └── ...
```

---

## 🎯 Decision Framework

The system provides a decision checklist for evaluating edges:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EDGE DECISION FRAMEWORK                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1. STATISTICAL SIGNIFICANCE                                         │   │
│  │     p-value < 0.05?                                                  │   │
│  └────────────────────────────┬────────────────────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  2. SAMPLE SIZE                                                      │   │
│  │     Is sample size > 100?                                           │   │
│  └────────────────────────────┬────────────────────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  3. TRANSACTION COSTS                                                │   │
│  │     After spreads/slippage, still +EV?                             │   │
│  └────────────────────────────┬────────────────────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  4. MULTIPLE INSTRUMENTS                                             │   │
│  │     Does it work on other symbols?                                 │   │
│  └────────────────────────────┬────────────────────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  5. MULTIPLE TIMEFRAMES                                              │   │
│  │     Does it work on other timeframes?                              │   │
│  └────────────────────────────┬────────────────────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  6. WALK-FORWARD VALIDATION                                          │   │
│  │     Is it stable across different time periods?                    │   │
│  └────────────────────────────┬────────────────────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     ALL YES? → YOU HAVE A REAL EDGE                  │   │
│  │                     → Paper Trade → Small Live → Scale Up            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Sample Output

```
======================================================================
  MT5 TRADING EDGE EXTRACTOR - BACKTESTING SYSTEM
  For Research Purposes Only - No Live Trading
======================================================================

Initializing Edge Extractor...
Loading data (simulated mode - install MetaTrader5 for live data)...

----------------------------------------------------------------------
  DETECTED EDGES SUMMARY
----------------------------------------------------------------------

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

Edge Name                                Type            Sharpe   Win%
----------------------------------------------------------------------
RSI_14_oversold_bounce                   mean_reversion    1.85  62.3%
London_session_breakout                  time_based        1.52  58.7%
EMA_12_26_cross                          trend_momentum    1.41  55.2%
BB_lower_bounce                          mean_reversion    1.38  61.1%
...
```

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute
- 🐛 Report bugs via GitHub Issues
- ✨ Suggest features or improvements
- 📝 Improve documentation
- 🔧 Submit pull requests

---

## 📜 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

---

## ⚠️ Disclaimer

**⚠️ WARNING: This system is for educational and research purposes only.**

- **BACKTESTING ONLY** - no live trading
- Past performance does not guarantee future results
- Always paper trade before live trading
- The authors assume no liability for trading losses

---

## 🙏 Acknowledgments

- MetaTrader 5 Python API
- SciPy for statistical testing
- Scikit-learn for ML
- Plotly for visualizations
- Seaborn for charting

---

<p align="center">
  Made with 🔬 for algorithmic traders
</p>