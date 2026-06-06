"""
Sleeve Backtester
=================

Walks a Sleeve through historical data with point-in-time correctness for
PRICE-BASED factors. Outputs per-sleeve metrics: Sharpe, max drawdown,
Calmar, turnover, deflated Sharpe.

Algorithm
---------
1. For each rebalance date r_i in [start, end]:
   - Call sleeve.weights(as_of=r_i)  -> {ticker -> weight}
   - These weights are held until r_{i+1}
2. Compute the portfolio's daily return between rebalances
3. Combine into a continuous portfolio return series
4. Compute metrics

PIT correctness
---------------
- For MomentumFactor, TrendFactor, DefensiveFactor: the only inputs are
  historical prices and they're correctly windowed via `as_of`.
- For ValueFactor, QualityFactor: yfinance's `Ticker.info` returns CURRENT
  fundamentals, not historical PIT. This is acknowledged in the result's
  `pit_warning` field. Treat numbers as proxy / upper bound.
- For PEAD, Insider: not currently backtestable on free data (need historical
  earnings dates and Form 4 filings indexed in time, both expensive feeds).

Survivorship
------------
- Universe = current S&P 500 / 100. Names delisted in the test period are
  absent. Survivorship bias inflates results — backtest is for *relative*
  ranking of sleeves, not absolute performance prediction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.portfolio.sleeves import Sleeve
from src.factors.data_provider import get_data_provider
from src.backtest.deflated_sharpe import deflated_sharpe_ratio

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    sleeve_name: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_rebalances: int
    n_observations: int
    daily_returns: pd.Series
    cumulative_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float
    turnover_annual: float
    deflated_sharpe: float
    pit_warning: bool
    weights_history: pd.DataFrame
    notes: str = ""

    def __repr__(self):
        return (
            f"<Backtest sleeve={self.sleeve_name!r} "
            f"sharpe={self.sharpe:.2f} dsr={self.deflated_sharpe:.2f} "
            f"dd={self.max_drawdown*100:.1f}% calmar={self.calmar:.2f} "
            f"turn={self.turnover_annual:.1f}x pit={'!!!' if self.pit_warning else 'ok'}>"
        )

    def summary(self) -> pd.Series:
        return pd.Series({
            "sleeve": self.sleeve_name,
            "start": str(self.start.date()),
            "end": str(self.end.date()),
            "n_rebalances": self.n_rebalances,
            "annualized_return_pct": round(self.annualized_return * 100, 2),
            "annualized_vol_pct": round(self.annualized_vol * 100, 2),
            "sharpe": round(self.sharpe, 3),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "calmar": round(self.calmar, 3),
            "turnover_annual_x": round(self.turnover_annual, 2),
            "deflated_sharpe": round(self.deflated_sharpe, 3),
            "pit_warning": self.pit_warning,
        })


# Sleeves that have PIT-correctness issues with free yfinance data
PIT_PROBLEMATIC_SLEEVES = {"value_quality_momentum", "pead", "insider_buying"}


class SleeveBacktester:
    """Walk a sleeve through history and report metrics."""

    def __init__(self, data_provider=None):
        self.dp = data_provider or get_data_provider()

    def backtest(
        self,
        sleeve: Sleeve,
        start: str,
        end: Optional[str] = None,
        n_trials_searched: int = 1,
    ) -> BacktestResult:
        """
        Run a sleeve backtest.

        Parameters
        ----------
        sleeve : Sleeve instance
        start, end : 'YYYY-MM-DD' (end defaults to today)
        n_trials_searched : number of hyperparameter configurations tried
            (for deflated Sharpe correction). Default 1 = "first attempt".
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.utcnow().normalize()
        if end_ts <= start_ts:
            raise ValueError("end must be after start")

        # 1) Build rebalance schedule
        reb_dates = self._rebalance_dates(start_ts, end_ts, sleeve.rebalance_freq)
        if len(reb_dates) < 2:
            raise ValueError(f"too few rebalance dates ({len(reb_dates)})")

        # 2) Pull a wide price frame for the universe (with buffer for factor lookbacks)
        universe = sleeve.universe()
        price_start = (start_ts - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
        price_end = end_ts.strftime("%Y-%m-%d")
        prices = self.dp.get_prices(universe, start=price_start, end=price_end)
        if prices is None or prices.empty:
            raise RuntimeError("no price data for universe")
        # Drop the lookback portion now that factors have been computed in get_prices
        bt_prices = prices.loc[prices.index >= start_ts]
        if bt_prices.empty:
            raise RuntimeError("no prices in backtest range")

        # 3) For each rebalance, compute weights using only PIT data
        all_weights: Dict[pd.Timestamp, Dict[str, float]] = {}
        for r in reb_dates:
            try:
                res = sleeve.weights(as_of=r.to_pydatetime())
                all_weights[r] = res.weights
            except Exception as e:
                logger.warning(f"sleeve weights failed at {r}: {e}")
                all_weights[r] = {}

        weights_df = self._weights_to_frame(all_weights, list(prices.columns))

        # 4) Compute daily portfolio returns
        daily_ret = self._portfolio_returns(weights_df, bt_prices)
        daily_ret = daily_ret.dropna()
        if len(daily_ret) < 30:
            raise RuntimeError(f"too few return observations: {len(daily_ret)}")

        # 5) Metrics
        cum_ret = float((1 + daily_ret).prod() - 1)
        n_years = (daily_ret.index[-1] - daily_ret.index[0]).days / 365.25
        ann_ret = float((1 + cum_ret) ** (1 / max(n_years, 1e-6)) - 1)
        ann_vol = float(daily_ret.std(ddof=1) * np.sqrt(252))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

        equity = (1 + daily_ret).cumprod()
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        max_dd = float(drawdown.min())
        calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0

        turnover = self._compute_turnover(weights_df, n_years)
        dsr = deflated_sharpe_ratio(daily_ret.values, n_trials=n_trials_searched)

        return BacktestResult(
            sleeve_name=sleeve.name,
            start=daily_ret.index[0],
            end=daily_ret.index[-1],
            n_rebalances=len(reb_dates),
            n_observations=len(daily_ret),
            daily_returns=daily_ret,
            cumulative_return=cum_ret,
            annualized_return=ann_ret,
            annualized_vol=ann_vol,
            sharpe=sharpe,
            max_drawdown=max_dd,
            calmar=calmar,
            turnover_annual=turnover,
            deflated_sharpe=dsr,
            pit_warning=(sleeve.name in PIT_PROBLEMATIC_SLEEVES),
            weights_history=weights_df,
            notes="",
        )

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _rebalance_dates(start: pd.Timestamp, end: pd.Timestamp, freq: str) -> pd.DatetimeIndex:
        if freq == "monthly":
            return pd.date_range(start, end, freq="MS")
        if freq == "weekly":
            return pd.date_range(start, end, freq="W-MON")
        if freq == "daily":
            return pd.date_range(start, end, freq="B")
        # 'event' = monthly proxy for backtest purposes
        return pd.date_range(start, end, freq="MS")

    @staticmethod
    def _weights_to_frame(
        weights_by_date: Dict[pd.Timestamp, Dict[str, float]],
        all_tickers: Sequence[str],
    ) -> pd.DataFrame:
        rows = []
        for d, w in weights_by_date.items():
            row = {t: w.get(t, 0.0) for t in all_tickers}
            row["__date"] = d
            rows.append(row)
        df = pd.DataFrame(rows).set_index("__date").sort_index()
        return df

    @staticmethod
    def _portfolio_returns(
        weights_df: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute the portfolio's daily return time series.

        Between rebalances, weights are held constant (no drift correction for
        simplicity — could refine with the multiplicative drift adjustment).
        """
        daily_ret = prices.pct_change(fill_method=None)
        # Forward-fill rebalance weights to each trading day
        # Align weights to the price index
        weights_aligned = weights_df.reindex(prices.index, method="ffill").fillna(0.0)
        # Portfolio return = Σ w_t * r_t
        # Use the previous day's weights for today's return (no look-ahead)
        weights_lagged = weights_aligned.shift(1).fillna(0.0)
        # Only consider tickers present in both
        common = weights_lagged.columns.intersection(daily_ret.columns)
        port_ret = (weights_lagged[common] * daily_ret[common]).sum(axis=1)
        return port_ret

    @staticmethod
    def _compute_turnover(weights_df: pd.DataFrame, n_years: float) -> float:
        """
        Annualized one-way turnover = mean(L1 distance between consecutive
        weight vectors) / 2 * (rebalances per year).
        """
        if len(weights_df) < 2 or n_years <= 0:
            return 0.0
        diffs = weights_df.diff().abs().sum(axis=1).dropna()
        rebalances_per_year = len(weights_df) / n_years
        return float(diffs.mean() / 2.0 * rebalances_per_year)
