# USDBRL Mean Reversion Strategy

A quantitative trading strategy for the USD/BRL exchange rate, combining RSI and Z-score signals with volatility-adjusted position sizing (ATR-based risk management). Built end-to-end in Python with `yfinance`, following a strict in-sample / out-of-sample validation methodology to avoid overfitting.

> ⚠️ **Disclaimer:** This project is for educational and portfolio purposes only. It does not constitute investment advice. Past backtest performance does not guarantee future results.

---

## Table of Contents

- [Motivation](#motivation)
- [Strategy Overview](#strategy-overview)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Results](#results)
- [How to Run](#how-to-run)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Author](#author)

---

## Motivation

This project was built to demonstrate practical skills at the intersection of software engineering and quantitative finance: data pipelines, statistical reasoning, risk management, and honest backtest evaluation. Every design decision below — from indicator choice to position sizing — is documented with the reasoning behind it, not just the implementation.

## Strategy Overview

**Asset:** USD/BRL (`USDBRL=X`)
**Type:** Mean reversion, long-short
**Signal frequency:** Daily

### Indicators

| Indicator | Role |
|---|---|
| RSI (14) | Confirmation filter for overextended price moves |
| Z-score (rolling mean/std) | Primary entry trigger |
| ATR (14) | Volatility measure used for stop-loss distance and position sizing |

### Entry Rules

Initial thresholds (Z-score ±2, RSI 30/70) were the starting specification. After in-sample calibration (see [Methodology](#methodology)), the frozen, final thresholds are:

| Direction | Condition |
|---|---|
| Long (buy USD) | Z-score < -2.5 **and** RSI < 35 |
| Short (sell USD) | Z-score > +2.5 **and** RSI > 65 |

### Exit Rules

- **Mean reversion target:** Z-score returns to the [-0.5, +0.5] range
- **Dynamic stop-loss:** 2× ATR(14) from entry price

### Position Sizing

Volatility-based sizing: each trade risks a fixed **1% of capital**. Position size is derived from the ATR-based stop distance, so exposure automatically shrinks in volatile regimes and grows in calmer ones.

### Transaction Costs

A simulated cost of **0.05% per leg** (entry and exit) is applied to approximate spread and slippage.

## Methodology

To avoid overfitting — one of the most common pitfalls in retail quant projects — parameters are calibrated exclusively on an in-sample period and validated once, without further adjustment, on an out-of-sample period:

| Period | Range | Purpose |
|---|---|---|
| In-sample | 2010–2021 | Parameter tuning and iteration |
| Out-of-sample | 2022–2026 | Single blind validation run |

Performance is benchmarked against a simple **buy-and-hold** position over the same periods.

### Parameter Calibration

Thresholds were calibrated exclusively on the in-sample period using a small, hypothesis-driven parameter search (not an exhaustive grid) — each range tested had a specific rationale (e.g. wider stops to reduce premature stop-outs, more selective entries, symmetric RSI bands) rather than being chosen by brute force. This keeps the search modest enough to limit the "multiple comparisons" risk of a parameter combination looking good purely by chance. See `calibration.py` for the full search process. Once frozen, parameters were **not** adjusted based on out-of-sample results.

### Evaluation Metrics

- CAGR (annualized return)
- Annualized volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor

## Project Structure

```
usdbrl-mean-reversion/
├── data/                  # Cached raw/processed price data
├── src/
│   ├── data_loader.py     # yfinance data fetching and cleaning
│   ├── indicators.py      # RSI, Z-score, ATR calculations
│   ├── strategy.py        # Signal generation (entry/exit logic)
│   ├── risk.py            # Position sizing and stop-loss logic
│   ├── backtester.py      # Trade simulation engine
│   ├── metrics.py         # Performance metrics and reporting
│   ├── pipeline.py        # End-to-end orchestration (run this)
│   └── calibration.py     # In-sample parameter search
├── notebooks/
│   └── analysis.ipynb     # Exploratory analysis and results visualization
├── tests/                 # Unit tests (108+ tests across all modules)
├── requirements.txt
└── README.md
```

## Tech Stack

- **Python 3.11+**
- `yfinance` — market data
- `pandas` / `numpy` — data manipulation
- `matplotlib` / `plotly` — visualization
- `pytest` — testing

## Results

### In-Sample (2010–2021, calibration period)

| Metric | Strategy | Buy-and-Hold |
|---|---|---|
| CAGR | 0.20% | 9.46% |
| Sharpe Ratio | 0.14 | 0.60 |
| Sortino Ratio | 0.20 | 0.88 |
| Max Drawdown | -6.39% | -26.80% |
| Win Rate | 64.4% | — |
| Profit Factor | 1.18 | — |
| Total Trades | 45 | — |

The strategy underperformed buy-and-hold on raw return in-sample — 2010–2021 was a sustained, multi-year USD appreciation trend (multiple crises reinforcing the direction), a regime that structurally disadvantages a symmetric mean-reversion approach. What the strategy did deliver was a much shallower drawdown (-6.39% vs. -26.80%) and a positive Profit Factor, at the cost of most of the upside.

### Out-of-Sample (2022–2026, single blind validation run)

| Metric | Strategy | Buy-and-Hold |
|---|---|---|
| CAGR | **1.16%** | -1.14% |
| Sharpe Ratio | **0.79** | -0.01 |
| Sortino Ratio | **1.29** | -0.02 |
| Max Drawdown | **-1.81%** | -22.12% |
| Win Rate | 78.6% | — |
| Profit Factor | 3.54 | — |
| Total Trades | 14 | — |

The strategy beat the benchmark on every metric in the out-of-sample period, with a notably shallow drawdown. This is consistent with a plausible regime change: 2022–2026 was more range-bound/volatile (high interest rates, elections, a change in government) than a sustained directional trend — exactly the kind of environment mean reversion is expected to do well in.

### Honest caveats

- **Small sample.** 14 out-of-sample trades is not enough to claim statistical confidence in the edge — treat the Sharpe of 0.79 as a plausible, not proven, signal.
- **Out-of-sample outperforming in-sample is unusual.** Normally you'd expect some degradation, since parameters were tuned on the in-sample data. The reversal here is consistent with a regime shift, but also warrants healthy skepticism rather than overclaiming.
- **This is not a "get rich" strategy.** Absolute returns are modest (CAGR ~1%). The result that stands out is capital preservation (much shallower drawdowns than buy-and-hold), not high absolute returns.
- Full parameter search process — including the round that ruled out several hypotheses — is in `calibration.py`.

## How to Run

```bash
git clone https://github.com/<your-username>/usdbrl-mean-reversion.git
cd usdbrl-mean-reversion
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m src.pipeline
```

This runs the full pipeline (data → indicators → signals → risk → backtest → metrics) for both the in-sample and out-of-sample periods, with the frozen calibrated parameters, printing strategy vs. buy-and-hold reports for each.

To reproduce the in-sample parameter search:
```bash
python -m src.calibration
```

## Limitations

- Transaction costs are simplified (fixed %, not order-book based)
- No consideration of interest rate differentials (carry), which materially affect FX positions held over time
- Signals assume next-bar execution at closing price
- Backtest does not account for market impact at scale

## Roadmap

- [x] Core backtest engine (RSI + Z-score + ATR sizing)
- [x] In-sample parameter tuning
- [x] Out-of-sample validation
- [ ] Extend to a basket of assets for robustness testing
- [ ] Compare RSI-only vs. Bollinger-only vs. combined signal approaches
- [ ] Interactive dashboard (Streamlit) for exploring results

## Author

**Matheus** — Software Engineering student (FIAP) with a background in ETL, financial certifications (CPA-10, fixed income, derivatives), and hands-on experience in data/finance tooling.

[LinkedIn](#) · [GitHub](#)
