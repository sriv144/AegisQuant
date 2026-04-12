"""
Monte Carlo Simulation Engine
==============================
Bootstrap-resamples a return series 10,000 times to produce a distribution
of outcomes rather than a single point estimate.

Reports:
    - 5th / 25th / 50th / 75th / 95th percentile Sharpe Ratios
    - Probability of ruin  : P(max_drawdown > ruin_threshold)
    - Strategy half-life   : median window where edge degrades by 50%
    - Cumulative return distribution over a fixed horizon

Usage:
    from src.backtest.monte_carlo import MonteCarloSimulator
    sim = MonteCarloSimulator()
    report = sim.run(oos_returns)
    sim.print_report(report)
"""

import math
import numpy as np
from typing import Dict, Any
from src.backtest.metrics import sharpe_ratio, max_drawdown, annualised_return

TRADING_DAYS = 252


class MonteCarloSimulator:
    """
    Bootstrap Monte Carlo over a 1-D array of daily returns.

    Args:
        n_simulations:   Number of bootstrap samples.
        horizon:         Length of each simulated path (trading days).
        ruin_threshold:  Drawdown level that counts as 'ruin' (e.g. 0.30 = 30%).
        seed:            Random seed for reproducibility.
    """

    def __init__(
        self,
        n_simulations: int = 10_000,
        horizon: int = TRADING_DAYS,
        ruin_threshold: float = 0.30,
        seed: int = 42,
    ):
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.ruin_threshold = ruin_threshold
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def run(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Run Monte Carlo bootstrap.

        Args:
            returns: 1-D array of daily returns from the OOS period.

        Returns:
            Dict with distribution statistics and per-simulation paths.
        """
        returns = np.asarray(returns, dtype=float)
        n = len(returns)
        if n < 10:
            raise ValueError(f"Need at least 10 return observations; got {n}")

        h = min(self.horizon, n)

        # Bootstrap: resample with replacement
        indices = self.rng.integers(0, n, size=(self.n_simulations, h))
        sim_paths = returns[indices]  # shape: (n_sim, h)

        # Metrics per simulation
        sharpes = np.array([sharpe_ratio(sim_paths[i]) for i in range(self.n_simulations)])
        ann_rets = np.array([annualised_return(sim_paths[i]) for i in range(self.n_simulations)])
        max_dds = np.array([max_drawdown(sim_paths[i]) for i in range(self.n_simulations)])

        # Cumulative returns (final value of $1 invested)
        cum_rets = (1 + sim_paths).prod(axis=1)

        # Probability of ruin
        p_ruin = float((np.abs(max_dds) > self.ruin_threshold).mean())

        # Strategy half-life: median number of days until rolling Sharpe
        # drops below 50% of the full-period Sharpe.
        full_sr = sharpe_ratio(returns)
        half_life = self._estimate_half_life(returns, target_sr=full_sr * 0.5)

        report: Dict[str, Any] = {
            # ── Sharpe distribution ──────────────────────────────────────
            "sharpe_p5": round(float(np.percentile(sharpes, 5)), 4),
            "sharpe_p25": round(float(np.percentile(sharpes, 25)), 4),
            "sharpe_p50": round(float(np.percentile(sharpes, 50)), 4),
            "sharpe_p75": round(float(np.percentile(sharpes, 75)), 4),
            "sharpe_p95": round(float(np.percentile(sharpes, 95)), 4),
            "sharpe_mean": round(float(sharpes.mean()), 4),
            "sharpe_std": round(float(sharpes.std()), 4),
            # ── Return distribution ──────────────────────────────────────
            "ann_return_p5": round(float(np.percentile(ann_rets, 5)), 4),
            "ann_return_p50": round(float(np.percentile(ann_rets, 50)), 4),
            "ann_return_p95": round(float(np.percentile(ann_rets, 95)), 4),
            # ── Drawdown / risk ──────────────────────────────────────────
            "max_dd_p5": round(float(np.percentile(max_dds, 5)), 4),
            "max_dd_p50": round(float(np.percentile(max_dds, 50)), 4),
            "max_dd_p95": round(float(np.percentile(max_dds, 95)), 4),
            "probability_of_ruin": round(p_ruin, 4),
            "ruin_threshold": self.ruin_threshold,
            # ── Cumulative return ────────────────────────────────────────
            "cum_return_p5": round(float(np.percentile(cum_rets, 5) - 1), 4),
            "cum_return_p50": round(float(np.percentile(cum_rets, 50) - 1), 4),
            "cum_return_p95": round(float(np.percentile(cum_rets, 95) - 1), 4),
            # ── Half-life ────────────────────────────────────────────────
            "strategy_half_life_days": half_life,
            # ── Meta ─────────────────────────────────────────────────────
            "n_simulations": self.n_simulations,
            "horizon_days": h,
            "n_input_returns": n,
            "full_period_sharpe": round(full_sr, 4),
        }
        return report

    # ------------------------------------------------------------------
    def print_report(self, report: Dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print("Monte Carlo Simulation Report")
        print(f"  {report['n_simulations']:,} bootstrap simulations "
              f"× {report['horizon_days']} days")
        print(f"  Input returns: {report['n_input_returns']} observations")
        print("=" * 60)
        print(f"\n  Sharpe Ratio Distribution:")
        print(f"    5th pct  : {report['sharpe_p5']:>8.3f}")
        print(f"    25th pct : {report['sharpe_p25']:>8.3f}")
        print(f"    Median   : {report['sharpe_p50']:>8.3f}  <- realistic case")
        print(f"    75th pct : {report['sharpe_p75']:>8.3f}")
        print(f"    95th pct : {report['sharpe_p95']:>8.3f}")
        print(f"\n  Annualised Return Distribution:")
        print(f"    5th pct  : {report['ann_return_p5']:>8.1%}")
        print(f"    Median   : {report['ann_return_p50']:>8.1%}")
        print(f"    95th pct : {report['ann_return_p95']:>8.1%}")
        print(f"\n  Max Drawdown Distribution:")
        print(f"    5th pct  : {report['max_dd_p5']:>8.1%}")
        print(f"    Median   : {report['max_dd_p50']:>8.1%}")
        print(f"    95th pct : {report['max_dd_p95']:>8.1%}")
        print(f"\n  Probability of Ruin (DD > {report['ruin_threshold']:.0%}): "
              f"{report['probability_of_ruin']:.1%}")
        print(f"  Strategy Half-Life: {report['strategy_half_life_days']} days")
        print("=" * 60)

    # ------------------------------------------------------------------
    def _estimate_half_life(self, returns: np.ndarray, target_sr: float) -> int:
        """
        Estimate the number of days after which the rolling (expanding)
        Sharpe drops below target_sr for the first time.
        Returns -1 if it never drops below target.
        """
        n = len(returns)
        min_window = 21  # need at least ~1 month to compute Sharpe
        for t in range(min_window, n):
            window_returns = returns[:t]
            sr = sharpe_ratio(window_returns)
            if sr < target_sr:
                return t
        return -1  # edge holds for the full period
