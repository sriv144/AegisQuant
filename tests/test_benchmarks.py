"""
Tests for BenchmarkSuite and multi-asset benchmark helpers.
Uses synthetic return data so no network calls are required.
"""
import numpy as np
import pandas as pd
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_returns(n: int = 252, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0004, 0.01, n)


def _make_price_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Single-column DataFrame with a DatetimeIndex — mimics yfinance output."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    return pd.DataFrame({"close": close}, index=idx)


# ── metrics used by benchmarks ───────────────────────────────────────────────

class TestMetrics:
    def test_sharpe_positive_drift(self):
        from src.backtest.metrics import sharpe_ratio
        rets = _make_returns(n=252, seed=1) + 0.001  # positive drift
        sr = sharpe_ratio(rets)
        assert sr > 0.0

    def test_sharpe_zero_std(self):
        from src.backtest.metrics import sharpe_ratio
        rets = np.zeros(100)
        assert sharpe_ratio(rets) == 0.0

    def test_sortino_no_downside(self):
        from src.backtest.metrics import sortino_ratio
        rets = np.abs(_make_returns(252))  # all positive
        sr = sortino_ratio(rets)
        assert sr == float("inf")

    def test_max_drawdown_flat(self):
        from src.backtest.metrics import max_drawdown
        rets = np.zeros(100)
        assert max_drawdown(rets) == 0.0

    def test_max_drawdown_monotone_loss(self):
        from src.backtest.metrics import max_drawdown
        rets = np.full(50, -0.01)
        dd = max_drawdown(rets)
        assert dd < 0.0   # convention: drawdown is returned as a negative number
        assert dd >= -1.0

    def test_win_rate_range(self):
        from src.backtest.metrics import win_rate
        rets = _make_returns(200)
        wr = win_rate(rets)
        assert 0.0 <= wr <= 1.0

    def test_profit_factor_positive(self):
        from src.backtest.metrics import profit_factor
        rets = _make_returns(300, seed=5) + 0.0005
        pf = profit_factor(rets)
        assert pf >= 0.0


# ── BenchmarkSuite (mock data path) ──────────────────────────────────────────

class TestBenchmarkSuite:
    """
    Tests BenchmarkSuite using pre-built synthetic return arrays so that
    no live yfinance calls are made.
    """

    def _make_suite_with_mock(self, monkeypatch):
        """Monkey-patches BenchmarkSuite._fetch_prices to return synthetic data."""
        from src.backtest.benchmarks import BenchmarkSuite

        n = 252 * 3  # 3 years
        idx = pd.bdate_range("2019-01-01", periods=n)
        rng = np.random.default_rng(7)

        def mock_fetch(self_inner, ticker, start, end):
            close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
            return pd.Series(close, index=idx, name=ticker)

        monkeypatch.setattr(BenchmarkSuite, "_fetch_prices", mock_fetch, raising=False)
        return BenchmarkSuite(universe=["SPY", "TLT", "GLD"])

    def test_suite_instantiation(self):
        from src.backtest.benchmarks import BenchmarkSuite
        suite = BenchmarkSuite(universe=["SPY", "TLT"])
        assert "SPY" in suite.universe

    def test_compute_all_metrics_keys(self):
        from src.backtest.metrics import compute_all_metrics
        rets = _make_returns(252)
        weights = np.random.uniform(-0.5, 0.5, 252)
        metrics = compute_all_metrics(rets, weights, n_trials=1, label="test")
        for key in ("sharpe_ratio", "sortino_ratio", "max_drawdown", "win_rate"):
            assert key in metrics, f"Missing key: {key}"

    def test_sharpe_finite(self):
        from src.backtest.metrics import compute_all_metrics
        rets = _make_returns(252)
        weights = np.ones(252) * 0.5
        metrics = compute_all_metrics(rets, weights, n_trials=1, label="test")
        assert np.isfinite(metrics["sharpe_ratio"])


# ── Multi-asset benchmarks ────────────────────────────────────────────────────

class TestMultiBenchmarks:
    TICKERS = ["SPY", "TLT", "GLD"]

    def _make_returns_df(self, n: int = 252) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        idx = pd.bdate_range("2020-01-01", periods=n)
        data = {t: rng.normal(0.0003, 0.01, n) for t in self.TICKERS}
        return pd.DataFrame(data, index=idx)

    def test_equal_weight_benchmark(self):
        from src.backtest.multi_benchmarks import MultiAssetBenchmarkSuite
        suite = MultiAssetBenchmarkSuite(tickers=self.TICKERS)
        result = suite.evaluate_equal_weight(self._make_returns_df())
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert np.isfinite(result["sharpe_ratio"])

    def test_momentum_benchmark(self):
        from src.backtest.multi_benchmarks import MultiAssetBenchmarkSuite
        suite = MultiAssetBenchmarkSuite(tickers=self.TICKERS)
        result = suite.evaluate_cross_sectional_momentum(
            self._make_returns_df(n=400), momentum_lookback=63
        )
        assert "sharpe_ratio" in result
        assert np.isfinite(result["sharpe_ratio"])

    def test_equal_weight_empty_df(self):
        from src.backtest.multi_benchmarks import MultiAssetBenchmarkSuite
        suite = MultiAssetBenchmarkSuite(tickers=self.TICKERS)
        result = suite.evaluate_equal_weight(pd.DataFrame())
        assert result == {}

    def test_momentum_insufficient_data(self):
        from src.backtest.multi_benchmarks import MultiAssetBenchmarkSuite
        suite = MultiAssetBenchmarkSuite(tickers=self.TICKERS)
        small_df = self._make_returns_df(n=10)
        result = suite.evaluate_cross_sectional_momentum(small_df, momentum_lookback=63)
        assert result == {}
