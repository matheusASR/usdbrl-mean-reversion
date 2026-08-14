"""
metrics.py

Performance evaluation: turns the equity curve and trade log from
backtester.py into the metrics that actually answer "is this strategy
good?" — CAGR, volatility, Sharpe, Sortino, Max Drawdown, Win Rate, and
Profit Factor.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def calculate_daily_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate simple day-over-day percentage returns from an equity curve.

    Args:
        equity_curve: Daily equity series (see backtester.build_equity_curve).

    Returns:
        A Series of daily returns (e.g. 0.01 = +1%). One element shorter
        than the input, since the first day has no prior value to
        compare against.
    """
    returns = equity_curve.pct_change().dropna()
    returns.name = "daily_return"
    return returns


def calculate_cagr(equity_curve: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Calculate the Compound Annual Growth Rate (CAGR) — the constant
    annual growth rate that would take the starting equity to the
    ending equity over the observed period.

        CAGR = (final / initial) ^ (1 / years) - 1

    Args:
        equity_curve: Daily equity series.
        periods_per_year: Trading periods per year (default 252).

    Returns:
        CAGR as a decimal (e.g. 0.15 = 15% per year). Returns -1.0 (total
        loss) if final equity is zero or negative, since the formula is
        undefined in that case (no real root of a negative number).
    """
    n_periods = len(equity_curve) - 1
    if n_periods <= 0:
        return 0.0

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    if total_return <= 0:
        return -1.0

    years = n_periods / periods_per_year
    return total_return ** (1 / years) - 1


def calculate_annualized_volatility(
    daily_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Annualize the standard deviation of daily returns using the
    square-root-of-time rule: since variance scales linearly with time
    (under the standard i.i.d. assumption), standard deviation scales
    with the square root of time.

        annual_vol = daily_std * sqrt(periods_per_year)

    Args:
        daily_returns: Daily returns (see calculate_daily_returns).
        periods_per_year: Trading periods per year (default 252).

    Returns:
        Annualized volatility as a decimal (e.g. 0.12 = 12% per year).
    """
    return daily_returns.std() * np.sqrt(periods_per_year)


if __name__ == "__main__":
    # Synthetic equity curve: ~2 years, modest upward drift with noise
    rng = np.random.default_rng(seed=99)
    n = 504  # ~2 trading years
    dates = pd.bdate_range("2024-01-01", periods=n)
    daily_ret = rng.normal(0.0004, 0.006, size=n)  # ~10%/yr drift, realistic daily vol
    equity = pd.Series(100_000 * np.cumprod(1 + daily_ret), index=dates, name="equity")

    returns = calculate_daily_returns(equity)
    cagr = calculate_cagr(equity)
    vol = calculate_annualized_volatility(returns)

    print(f"Final equity: {equity.iloc[-1]:.2f}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Annualized volatility: {vol:.2%}")
