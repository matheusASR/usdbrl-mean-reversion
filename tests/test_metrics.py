"""
tests/test_metrics.py

Unit tests for metrics.py. Golden values here were computed once with
the same formulas on small controlled series (see the conversation) and
hardcoded as fixed references, the same pattern used for indicators.py.
"""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    calculate_annualized_volatility,
    calculate_cagr,
    calculate_daily_returns,
    calculate_downside_deviation,
    calculate_drawdown_series,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_win_rate,
    generate_report,
)


# ---------------------------------------------------------------------------
# calculate_daily_returns
# ---------------------------------------------------------------------------

class TestCalculateDailyReturns:
    def test_simple_returns(self):
        equity = pd.Series([100, 110, 99])
        returns = calculate_daily_returns(equity)
        assert list(returns) == pytest.approx([0.10, -0.10])

    def test_one_shorter_than_input(self):
        equity = pd.Series([100, 105, 110, 108])
        returns = calculate_daily_returns(equity)
        assert len(returns) == len(equity) - 1


# ---------------------------------------------------------------------------
# calculate_cagr
# ---------------------------------------------------------------------------

class TestCalculateCAGR:
    def test_doubling_over_exactly_one_year(self):
        # 253 points = 252 periods = exactly one trading year
        dates = pd.bdate_range("2024-01-01", periods=253)
        equity = pd.Series(np.linspace(100, 200, 253), index=dates)
        cagr = calculate_cagr(equity)
        assert cagr == pytest.approx(1.0, rel=1e-6)

    def test_flat_equity_gives_zero_cagr(self):
        dates = pd.bdate_range("2024-01-01", periods=253)
        equity = pd.Series([100.0] * 253, index=dates)
        assert calculate_cagr(equity) == pytest.approx(0.0)

    def test_zero_final_equity_returns_total_loss(self):
        equity = pd.Series([100.0, 50.0, 0.0])
        assert calculate_cagr(equity) == -1.0

    def test_negative_final_equity_returns_total_loss(self):
        equity = pd.Series([100.0, -20.0])
        assert calculate_cagr(equity) == -1.0

    def test_single_point_returns_zero(self):
        equity = pd.Series([100.0])
        assert calculate_cagr(equity) == 0.0


# ---------------------------------------------------------------------------
# calculate_annualized_volatility
# ---------------------------------------------------------------------------

class TestCalculateAnnualizedVolatility:
    def test_scales_by_sqrt_of_periods(self):
        returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02])
        vol = calculate_annualized_volatility(returns, periods_per_year=252)
        assert vol == pytest.approx(returns.std() * np.sqrt(252))

    def test_zero_volatility_for_constant_returns(self):
        returns = pd.Series([0.001] * 10)
        assert calculate_annualized_volatility(returns) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# calculate_sharpe_ratio
# ---------------------------------------------------------------------------

class TestCalculateSharpeRatio:
    def test_positive_returns_give_positive_sharpe(self):
        rng = np.random.default_rng(seed=1)
        returns = pd.Series(rng.normal(0.001, 0.01, size=252))
        assert calculate_sharpe_ratio(returns) > 0

    def test_zero_std_returns_zero_not_error(self):
        returns = pd.Series([0.001] * 10)
        assert calculate_sharpe_ratio(returns) == 0.0

    def test_higher_risk_free_rate_lowers_sharpe(self):
        returns = pd.Series([0.001, 0.002, 0.0015, 0.0018, 0.0012])
        low_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        high_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.10)
        assert high_rf < low_rf


# ---------------------------------------------------------------------------
# calculate_downside_deviation / calculate_sortino_ratio
# ---------------------------------------------------------------------------

class TestDownsideDeviationAndSortino:
    def test_downside_deviation_golden_value(self):
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
        dd = calculate_downside_deviation(returns)
        assert dd == pytest.approx(0.15874507866387544, rel=1e-6)

    def test_downside_deviation_uses_all_periods_not_just_negatives(self):
        # Regression-style check for the "correct" formula: dividing by
        # the full N (5), not just the count of negative days (2), so
        # the result must NOT match a naive std() of only the losses.
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
        naive_wrong = returns[returns < 0].std()  # what NOT to compute
        dd = calculate_downside_deviation(returns, periods_per_year=1)
        assert dd != pytest.approx(naive_wrong)

    def test_sortino_is_inf_when_no_downside_days(self):
        returns = pd.Series([0.01, 0.02, 0.015, 0.03])  # all positive
        assert calculate_sortino_ratio(returns) == np.inf

    def test_sortino_exceeds_sharpe_when_upside_skewed(self):
        # A return series with big up days and small, consistent down
        # days should score higher on Sortino than Sharpe, since Sortino
        # doesn't penalize the upside volatility.
        returns = pd.Series([0.05, -0.01, 0.04, -0.01, 0.06, -0.01, 0.03])
        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)
        assert sortino > sharpe


# ---------------------------------------------------------------------------
# calculate_drawdown_series / calculate_max_drawdown
# ---------------------------------------------------------------------------

class TestDrawdown:
    def test_max_drawdown_golden_value(self):
        equity = pd.Series([100, 110, 105, 90, 95, 120])
        assert calculate_max_drawdown(equity) == pytest.approx(-0.18181818, rel=1e-6)

    def test_drawdown_is_zero_at_new_highs(self):
        equity = pd.Series([100, 110, 120, 130])  # strictly increasing
        drawdown = calculate_drawdown_series(equity)
        assert (drawdown == 0).all()

    def test_drawdown_never_positive(self):
        rng = np.random.default_rng(seed=3)
        equity = pd.Series(100 + np.cumsum(rng.normal(0, 1, size=100)))
        drawdown = calculate_drawdown_series(equity)
        assert (drawdown <= 0).all()


# ---------------------------------------------------------------------------
# calculate_win_rate / calculate_profit_factor
# ---------------------------------------------------------------------------

class TestWinRateAndProfitFactor:
    def _make_trades(self, pnls):
        return pd.DataFrame({"net_pnl": pnls})

    def test_win_rate_known_value(self):
        trades = self._make_trades([10, -5, 20, -3])
        assert calculate_win_rate(trades) == pytest.approx(0.5)

    def test_win_rate_nan_when_no_trades(self):
        trades = self._make_trades([])
        assert np.isnan(calculate_win_rate(trades))

    def test_profit_factor_known_value(self):
        trades = self._make_trades([10, -5, 20, -3])
        # gross_profit=30, gross_loss=8 -> 30/8 = 3.75
        assert calculate_profit_factor(trades) == pytest.approx(3.75)

    def test_profit_factor_inf_when_no_losses(self):
        trades = self._make_trades([10, 20, 5])
        assert calculate_profit_factor(trades) == np.inf

    def test_profit_factor_nan_when_no_trades(self):
        trades = self._make_trades([])
        assert np.isnan(calculate_profit_factor(trades))


# ---------------------------------------------------------------------------
# generate_report (orchestration)
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_contains_all_expected_keys(self):
        dates = pd.bdate_range("2024-01-01", periods=100)
        equity = pd.Series(np.linspace(100_000, 110_000, 100), index=dates)
        trades = pd.DataFrame({"net_pnl": [500, -200, 800, -150]})

        report = generate_report(equity, trades)
        expected_keys = {
            "CAGR", "Annualized Volatility", "Sharpe Ratio", "Sortino Ratio",
            "Max Drawdown", "Win Rate", "Profit Factor", "Total Trades",
        }
        assert expected_keys.issubset(report.keys())

    def test_total_trades_matches_input(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        equity = pd.Series(np.linspace(100_000, 101_000, 10), index=dates)
        trades = pd.DataFrame({"net_pnl": [100, -50, 200]})

        report = generate_report(equity, trades)
        assert report["Total Trades"] == 3
