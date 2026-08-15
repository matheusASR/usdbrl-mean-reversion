"""
pipeline.py

End-to-end orchestration script: ties together every module in the
project (data_loader -> indicators -> strategy -> risk -> backtester ->
metrics) into a single reusable function, and runs it for the in-sample
period by default when executed directly.

This is where the strategy specification from the README becomes an
actual, runnable result on real USDBRL data.
"""

import logging

from src.backtester import run_backtest
from src.data_loader import DataLoadError, load_price_data
from src.indicators import add_indicators
from src.metrics import generate_report
from src.risk import apply_risk_management
from src.strategy import generate_signals

logger = logging.getLogger(__name__)


def run_pipeline(
    ticker: str,
    start: str,
    end: str,
    initial_capital: float = 100_000,
    risk_per_trade: float = 0.01,
    rsi_period: int = 14,
    zscore_period: int = 20,
    atr_period: int = 14,
    zscore_entry: float = 2.0,
    rsi_lower: float = 30,
    rsi_upper: float = 70,
    zscore_exit_band: float = 0.5,
    atr_multiplier: float = 2.0,
    cost_pct: float = 0.0005,
) -> dict:
    """
    Run the full strategy pipeline end-to-end for a given ticker and
    date range: fetch data, compute indicators, generate signals, apply
    risk management, run the backtest, and evaluate performance.

    Every parameter here maps directly to a decision documented in the
    README — this function is the single place where they all come
    together.

    Args:
        ticker: Yahoo Finance ticker (e.g. "USDBRL=X").
        start: Start date ("YYYY-MM-DD").
        end: End date ("YYYY-MM-DD").
        initial_capital: Starting capital for the backtest.
        risk_per_trade: Fraction of capital risked per trade.
        rsi_period, zscore_period, atr_period: Indicator lookback windows.
        zscore_entry, rsi_lower, rsi_upper: Entry signal thresholds.
        zscore_exit_band: Mean-reversion exit threshold.
        atr_multiplier: Stop-loss distance, in multiples of ATR.
        cost_pct: Transaction cost per leg.

    Returns:
        A dict with:
            - "prices": the full DataFrame with indicators/signals/risk
              parameters (useful for plotting/debugging).
            - "backtest_result": the BacktestResult (trades + equity
              curve) from backtester.run_backtest.
            - "report": the metrics summary dict from
              metrics.generate_report.
    """
    logger.info("Running pipeline for %s (%s to %s)", ticker, start, end)

    prices = load_price_data(ticker, start, end)
    prices = add_indicators(
        prices, rsi_period=rsi_period, zscore_period=zscore_period, atr_period=atr_period
    )
    prices = generate_signals(
        prices,
        zscore_entry=zscore_entry,
        rsi_lower=rsi_lower,
        rsi_upper=rsi_upper,
        zscore_exit_band=zscore_exit_band,
    )
    prices = apply_risk_management(
        prices,
        capital=initial_capital,
        risk_per_trade=risk_per_trade,
        atr_multiplier=atr_multiplier,
    )

    result = run_backtest(prices, initial_capital=initial_capital, cost_pct=cost_pct)
    report = generate_report(result.equity_curve, result.trades)

    return {"prices": prices, "backtest_result": result, "report": report}


def print_report(report: dict, title: str = "Performance Report") -> None:
    """Pretty-print a metrics report dict to the console."""
    print(f"\n{'=' * 50}")
    print(title)
    print("=" * 50)

    percent_keys = {"CAGR", "Annualized Volatility", "Max Drawdown", "Win Rate"}
    for key, value in report.items():
        if key in percent_keys:
            print(f"{key:.<30} {value:.2%}" if value == value else f"{key:.<30} N/A")  # NaN check
        elif isinstance(value, float):
            print(f"{key:.<30} {value:.2f}" if value != float("inf") else f"{key:.<30} inf")
        else:
            print(f"{key:.<30} {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    IN_SAMPLE_START = "2010-01-01"
    IN_SAMPLE_END = "2021-12-31"

    try:
        results = run_pipeline(
            ticker="USDBRL=X",
            start=IN_SAMPLE_START,
            end=IN_SAMPLE_END,
        )
    except DataLoadError as exc:
        print(f"Could not run the pipeline: {exc}")
        raise SystemExit(1)

    print_report(results["report"], title=f"In-Sample ({IN_SAMPLE_START} to {IN_SAMPLE_END})")

    n_trades = len(results["backtest_result"].trades)
    print(f"\nTotal signals generated: {(results['prices']['entry_signal'] != 0).sum()}")
    print(f"Total trades executed: {n_trades}")
