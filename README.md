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

| Direction | Condition |
|---|---|
| Long (buy USD) | Z-score < -2 **and** RSI < 30 |
| Short (sell USD) | Z-score > +2 **and** RSI > 70 |

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
│   ├── backtester.py      # Trade simulation engine
│   ├── risk.py            # Position sizing and stop-loss logic
│   └── metrics.py         # Performance metrics and reporting
├── notebooks/
│   └── analysis.ipynb     # Exploratory analysis and results visualization
├── tests/                 # Unit tests
├── requirements.txt
└── README.md
```

*(structure will evolve as the project is built)*

## Tech Stack

- **Python 3.11+**
- `yfinance` — market data
- `pandas` / `numpy` — data manipulation
- `matplotlib` / `plotly` — visualization
- `pytest` — testing

## Results

🚧 *In progress — results and performance charts will be added as the backtest engine is completed.*

## How to Run

```bash
git clone https://github.com/<your-username>/usdbrl-mean-reversion.git
cd usdbrl-mean-reversion
pip install -r requirements.txt
python src/backtester.py
```

*(instructions will be finalized once the codebase is complete)*

## Limitations

- Transaction costs are simplified (fixed %, not order-book based)
- No consideration of interest rate differentials (carry), which materially affect FX positions held over time
- Signals assume next-bar execution at closing price
- Backtest does not account for market impact at scale

## Roadmap

- [ ] Core backtest engine (RSI + Z-score + ATR sizing)
- [ ] In-sample parameter tuning
- [ ] Out-of-sample validation
- [ ] Extend to a basket of assets for robustness testing
- [ ] Compare RSI-only vs. Bollinger-only vs. combined signal approaches
- [ ] Interactive dashboard (Streamlit) for exploring results

## Author

**Matheus** — Software Engineering student (FIAP) with a background in ETL, financial certifications (CPA, fixed income, derivatives), and hands-on experience in data/finance tooling.
