"""
calibration.py

Parameter exploration on the IN-SAMPLE period only. This module must
never be pointed at the out-of-sample date range — doing so would defeat
the entire point of having a held-out validation period (you'd be
fitting parameters to the same data you later "test" on).

The grid search here is deliberately modest in size and grounded in
specific hypotheses about what might be underperforming, rather than a
brute-force search over every imaginable combination — a smaller,
reasoned search is less prone to overfitting to noise in the in-sample
period than an exhaustive one.
"""

import logging
from itertools import product

import pandas as pd

from src.backtester import run_backtest
from src.data_loader import load_price_data
from src.indicators import add_indicators
from src.metrics import generate_report
from src.risk import apply_risk_management
from src.strategy import generate_signals

logger = logging.getLogger(__name__)


def run_parameter_grid(
    ticker: str,
    start: str,
    end: str,
    param_grid: dict,
    initial_capital: float = 100_000,
    risk_per_trade: float = 0.01,
    cost_pct: float = 0.0005,
) -> pd.DataFrame:
    """
    Run the pipeline once per combination in param_grid (the Cartesian
    product of all provided value lists), on the same price data fetched
    once and reused across combinations for speed.

    Any parameter not included in param_grid keeps its default from the
    underlying functions (indicators.add_indicators,
    strategy.generate_signals, risk.apply_risk_management).

    Args:
        ticker: Yahoo Finance ticker.
        start, end: Date range — MUST be the in-sample period only.
        param_grid: Dict mapping parameter name -> list of values to try.
            Recognized keys: rsi_period, zscore_period, atr_period,
            zscore_entry, rsi_lower, rsi_upper, zscore_exit_band,
            atr_multiplier.
        initial_capital: Starting capital, held constant across runs.
        risk_per_trade: Risk per trade, held constant across runs.
        cost_pct: Transaction cost per leg, held constant across runs.

    Returns:
        A DataFrame with one row per combination, the parameter values
        used, and every metric from metrics.generate_report — sorted by
        Sharpe Ratio, descending (best risk-adjusted result first).
    """
    prices_raw = load_price_data(ticker, start, end)

    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))
    logger.info("Running grid search: %d combinations", len(combinations))

    records = []
    for i, combo in enumerate(combinations, start=1):
        params = dict(zip(keys, combo))
        logger.info("[%d/%d] %s", i, len(combinations), params)

        prices = add_indicators(
            prices_raw,
            rsi_period=params.get("rsi_period", 14),
            zscore_period=params.get("zscore_period", 20),
            atr_period=params.get("atr_period", 14),
        )
        prices = generate_signals(
            prices,
            zscore_entry=params.get("zscore_entry", 2.0),
            rsi_lower=params.get("rsi_lower", 30),
            rsi_upper=params.get("rsi_upper", 70),
            zscore_exit_band=params.get("zscore_exit_band", 0.5),
        )
        prices = apply_risk_management(
            prices,
            capital=initial_capital,
            risk_per_trade=risk_per_trade,
            atr_multiplier=params.get("atr_multiplier", 2.0),
        )
        result = run_backtest(prices, initial_capital=initial_capital, cost_pct=cost_pct)
        report = generate_report(result.equity_curve, result.trades)

        records.append({**params, **report})

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values("Sharpe Ratio", ascending=False).reset_index(drop=True)
    return results_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Hypothesis-driven grid: wider stops (less premature stop-outs),
    # more selective entries, and a tighter exit band — see the
    # conversation for the reasoning behind each range.
    grid = {
        "atr_multiplier": [1.5, 2.0, 3.0],
        "zscore_entry": [1.5, 2.0, 2.5],
        "zscore_exit_band": [0.25, 0.5],
    }

    results = run_parameter_grid(
        ticker="USDBRL=X",
        start="2010-01-01",
        end="2021-12-31",
        param_grid=grid,
    )

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\nTop 10 combinations by Sharpe Ratio:")
    print(results.head(10)[
        ["atr_multiplier", "zscore_entry", "zscore_exit_band",
         "CAGR", "Sharpe Ratio", "Max Drawdown", "Win Rate", "Profit Factor", "Total Trades"]
    ].to_string(index=False))
