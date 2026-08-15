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

# Floating-point comparisons to zero should use a tolerance, not exact
# equality — even mathematically identical inputs can leave a tiny
# non-zero residue after operations like std() (sums/subtractions in
# floating point rarely cancel to a perfect, bit-exact 0.0).
_ZERO_TOLERANCE = 1e-10


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


def calculate_sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate the annualized Sharpe Ratio: return per unit of total risk
    (volatility, both upside and downside).

    Uses a risk-free rate of 0% by default — a common simplification in
    FX backtests, since "the risk-free rate" is ambiguous when the asset
    itself is an exchange rate between two currencies. Override
    risk_free_rate if you want to net out a specific reference rate
    (e.g. CDI).

    Args:
        daily_returns: Daily returns (see calculate_daily_returns).
        risk_free_rate: Annual risk-free rate as a decimal (default 0.0).
        periods_per_year: Trading periods per year (default 252).

    Returns:
        Annualized Sharpe Ratio. Returns 0.0 if returns have zero
        volatility (a degenerate case — no risk-adjusted signal to
        compute).
    """
    daily_rf = risk_free_rate / periods_per_year
    excess_returns = daily_returns - daily_rf

    std = excess_returns.std()
    if std < _ZERO_TOLERANCE:
        return 0.0

    return (excess_returns.mean() / std) * np.sqrt(periods_per_year)


def calculate_downside_deviation(
    daily_returns: pd.Series,
    target_return: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate the annualized downside deviation: the risk measure Sortino
    uses instead of total volatility.

    Important implementation detail: this is computed over ALL returns,
    not just the negative ones. Returns above the target contribute a
    deviation of exactly 0 (rather than being dropped from the set)
    before averaging. Filtering to only the negative days and taking
    their standard deviation — a common shortcut — divides by a smaller
    N and inflates the result; that is NOT the standard definition.

    Args:
        daily_returns: Daily returns.
        target_return: The threshold below which a return counts as
            "downside" (default 0.0).
        periods_per_year: Trading periods per year (default 252).

    Returns:
        Annualized downside deviation (always >= 0).
    """
    downside_diff = np.minimum(daily_returns - target_return, 0)
    downside_variance = (downside_diff ** 2).mean()
    daily_downside_dev = np.sqrt(downside_variance)
    return daily_downside_dev * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate the annualized Sortino Ratio: return per unit of downside
    risk only (see calculate_downside_deviation).

    Args:
        daily_returns: Daily returns.
        risk_free_rate: Annual risk-free rate as a decimal (default 0.0).
        target_return: Threshold below which a return counts as downside
            (default 0.0).
        periods_per_year: Trading periods per year (default 252).

    Returns:
        Annualized Sortino Ratio. Returns np.inf if there were no
        downside days at all in the sample (a real, if rare, possibility
        over a short backtest — not a computation error, so it is NOT
        silently mapped to 0.0. Callers displaying this value should
        handle infinity explicitly, e.g. showing "∞").
    """
    daily_rf = risk_free_rate / periods_per_year
    excess_return_annualized = (daily_returns.mean() - daily_rf) * periods_per_year

    downside_dev = calculate_downside_deviation(daily_returns, target_return, periods_per_year)
    if downside_dev < _ZERO_TOLERANCE:
        return np.inf

    return excess_return_annualized / downside_dev


def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate the drawdown at every point in the equity curve: how far
    below its highest-ever value (up to that day) the equity currently
    sits.

    Args:
        equity_curve: Daily equity series.

    Returns:
        A Series of drawdown values, always <= 0 (e.g. -0.15 = 15% below
        the running peak; 0 = at a new all-time high).
    """
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    drawdown.name = "drawdown"
    return drawdown


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate the Maximum Drawdown: the single worst peak-to-trough
    decline observed over the whole equity curve.

    Args:
        equity_curve: Daily equity series.

    Returns:
        Max drawdown as a decimal, always <= 0 (e.g. -0.22 = -22%).
    """
    return calculate_drawdown_series(equity_curve).min()


def calculate_win_rate(trades: pd.DataFrame) -> float:
    """
    Calculate the fraction of trades that closed with a positive net P&L.

    Args:
        trades: Trade log DataFrame (see backtester.trades_to_dataframe),
            must contain a net_pnl column.

    Returns:
        Win rate as a decimal (e.g. 0.55 = 55%). Returns np.nan if there
        were no trades at all — a win rate is undefined without any
        trades, not "0%".
    """
    if len(trades) == 0:
        return np.nan
    return (trades["net_pnl"] > 0).mean()


def calculate_profit_factor(trades: pd.DataFrame) -> float:
    """
    Calculate the Profit Factor: total profit from winning trades divided
    by the absolute total loss from losing trades.

    A value above 1.0 means the strategy is net profitable; above 2.0 is
    generally considered strong.

    Args:
        trades: Trade log DataFrame, must contain a net_pnl column.

    Returns:
        Profit Factor. Returns np.nan if there are no trades at all.
        Returns np.inf if there were winning trades but zero losing
        trades — a real (if rare) outcome, not a computation error, so
        it is not silently mapped to 0.
    """
    if len(trades) == 0:
        return np.nan

    gross_profit = trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum()
    gross_loss = trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum()  # <= 0

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan

    return gross_profit / abs(gross_loss)


def generate_report(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """
    Single entry point tying together every metric in this module — the
    same set defined in the project's README.

    Args:
        equity_curve: Daily equity series (see
            backtester.build_equity_curve).
        trades: Trade log DataFrame (see backtester.trades_to_dataframe).
        risk_free_rate: Annual risk-free rate as a decimal (default 0.0).
        periods_per_year: Trading periods per year (default 252).

    Returns:
        A dict with CAGR, Annualized Volatility, Sharpe Ratio, Sortino
        Ratio, Max Drawdown, Win Rate, Profit Factor, and Total Trades.
    """
    daily_returns = calculate_daily_returns(equity_curve)

    return {
        "CAGR": calculate_cagr(equity_curve, periods_per_year),
        "Annualized Volatility": calculate_annualized_volatility(daily_returns, periods_per_year),
        "Sharpe Ratio": calculate_sharpe_ratio(daily_returns, risk_free_rate, periods_per_year),
        "Sortino Ratio": calculate_sortino_ratio(daily_returns, risk_free_rate, 0.0, periods_per_year),
        "Max Drawdown": calculate_max_drawdown(equity_curve),
        "Win Rate": calculate_win_rate(trades),
        "Profit Factor": calculate_profit_factor(trades),
        "Total Trades": len(trades),
    }


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
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd = calculate_max_drawdown(equity)

    print(f"Final equity: {equity.iloc[-1]:.2f}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Annualized volatility: {vol:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Sortino Ratio: {sortino:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")

    # Synthetic trade log, just for this demo (a real one comes from
    # backtester.run_backtest).
    trades = pd.DataFrame({
        "net_pnl": [500, -200, 800, -150, 300, -100, 950, 200, -400, 600],
    })

    print("\n--- Full report (generate_report) ---")
    report = generate_report(equity, trades)
    for key, value in report.items():
        if isinstance(value, float) and abs(value) < 10:
            print(f"{key}: {value:.2%}" if "Rate" in key or key in ("CAGR", "Annualized Volatility", "Max Drawdown") else f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")