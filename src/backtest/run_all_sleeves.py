"""
Runner: backtest all 4 sleeves and report metrics.

Usage:
    python -m src.backtest.run_all_sleeves --start 2022-01-01 --end 2026-05-23

Output: a markdown-formatted table to stdout, plus per-sleeve detail prints.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.backtest.sleeve_backtester import SleeveBacktester, BacktestResult
from src.portfolio.sleeves import (
    ValueQualityMomentumSleeve,
    CrossSectionalMomentumSleeve,
    PEADSleeve,
    InsiderBuyingSleeve,
)

logger = logging.getLogger(__name__)


# The bar from the redesign plan: deflated Sharpe >= 0.4 to deploy live
DEPLOYMENT_GATE_DSR = 0.4


def main(argv=None):
    parser = argparse.ArgumentParser(description="Backtest all AegisQuant sleeves")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--universe", default="sp100",
                        help="sp100 (fast, default) or sp500 (slow)")
    parser.add_argument("--sleeves", default="all",
                        help="comma-separated subset, or 'all'")
    parser.add_argument("--n-trials", type=int, default=1,
                        help="number of hyperparameter configurations tried "
                             "(for deflated Sharpe correction)")
    parser.add_argument("--out", default=None,
                        help="optional path to write CSV summary")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    # Optionally narrow universe to speed things up
    from src.factors.universe import sp500_tickers, sp100_tickers
    universe_fn = {"sp100": sp100_tickers, "sp500": sp500_tickers}[args.universe]

    all_sleeves = {
        "value_quality_momentum": ValueQualityMomentumSleeve(),
        "xs_momentum": CrossSectionalMomentumSleeve(),
        "pead": PEADSleeve(),
        "insider_buying": InsiderBuyingSleeve(),
    }
    selected = (
        list(all_sleeves)
        if args.sleeves == "all"
        else [s.strip() for s in args.sleeves.split(",")]
    )

    bt = SleeveBacktester()
    results: list[BacktestResult] = []
    print(f"\n=== Backtest window: {args.start} -> {args.end or 'today'} | universe={args.universe} ===\n")

    for name in selected:
        sleeve = all_sleeves[name]
        # Narrow universe for speed
        sleeve.universe = lambda fn=universe_fn: fn()
        print(f"Running {name}...", flush=True)
        try:
            res = bt.backtest(sleeve, start=args.start, end=args.end,
                              n_trials_searched=args.n_trials)
            results.append(res)
            print(f"  -> {res}")
        except Exception as e:
            logger.error(f"{name} failed: {e}")
            print(f"  FAILED: {e}")

    if not results:
        print("No successful backtests.")
        sys.exit(1)

    # Summary table
    print("\n=== Summary ===\n")
    summary = pd.DataFrame([r.summary() for r in results])
    print(summary.to_string(index=False))

    # Deployment gate verdicts
    print("\n=== Deployment Gate (DSR >= 0.4) ===\n")
    for r in results:
        flag = "PASS" if r.deflated_sharpe >= DEPLOYMENT_GATE_DSR else "FAIL"
        warn = " (PIT-bias warning)" if r.pit_warning else ""
        print(f"  [{flag}] {r.sleeve_name}: DSR={r.deflated_sharpe:.3f}{warn}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False)
        print(f"\nWrote summary to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
