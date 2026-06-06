"""
US Sleeve-Based Trading Loop (v2)
==================================

Replaces the old per-ticker agent pipeline (main_us.py) with the factor-sleeve
architecture from the Phase 1-5 redesign.

Each cycle:
  1. Compute each ENABLED sleeve's target weights
  2. Get current macro regime (defensive overlay)
  3. Run Combiner (risk-parity inverse-vol)
  4. Run RiskOfficer (hard caps + drawdown gate)
  5. Compute deltas vs current Alpaca positions
  6. If KILL_SWITCH=true → log "would have done X" only
  7. Else → execute via broker

Env vars:
  MARKET=US
  ALPACA_API_KEY / ALPACA_SECRET_KEY / ALPACA_BASE_URL
  OPENAI_API_KEY                          (for text-feature LLM; falls back to heuristics)
  KILL_SWITCH=true                        (dry-run mode — no actual trades)
  ENABLED_SLEEVES=xs_momentum             (comma-separated; default xs_momentum only)
  UNIVERSE=sp100|sp500                    (default sp100 for speed)
  INITIAL_CAPITAL=100000

Usage:
    python main_us_v2.py                  # single cycle
    python main_us_v2.py --loop           # APScheduler every 60min during NYSE hours
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ.setdefault("MARKET", "US")

import pandas as pd

from src.portfolio.sleeves import (
    ValueQualityMomentumSleeve,
    CrossSectionalMomentumSleeve,
    PEADSleeve,
    InsiderBuyingSleeve,
    SleeveResult,
)
from src.portfolio.combiner import Combiner
from src.portfolio.risk_officer import RiskOfficer, RiskReview
from src.agents.text_features.macro_regime_agent import MacroRegimeAgent
from src.factors.universe import sp100_tickers, sp500_tickers

logger = logging.getLogger(__name__)

KILL_SWITCH = os.getenv("KILL_SWITCH", "false").lower() in ("true", "1", "yes")
ENABLED_SLEEVES = [
    s.strip() for s in os.getenv("ENABLED_SLEEVES", "xs_momentum").split(",") if s.strip()
]
UNIVERSE_NAME = os.getenv("UNIVERSE", "sp100").lower()
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000"))

ALL_SLEEVE_FACTORY = {
    "value_quality_momentum": ValueQualityMomentumSleeve,
    "xs_momentum": CrossSectionalMomentumSleeve,
    "pead": PEADSleeve,
    "insider_buying": InsiderBuyingSleeve,
}


# ── Persistence (read by dashboard) ─────────────────────────────────────────


SNAPSHOT_DIR = Path(__file__).parent / ".cache" / "sleeve_snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def _write_snapshot(payload: dict):
    """Write the latest cycle's output for the dashboard to consume."""
    path = SNAPSHOT_DIR / "latest.json"
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info(f"Wrote snapshot: {path}")
    except Exception as e:
        logger.error(f"Snapshot write failed: {e}")


# ── Core cycle ──────────────────────────────────────────────────────────────


def run_cycle(dry_run: Optional[bool] = None) -> dict:
    """
    Run one complete cycle: sleeves -> combiner -> risk officer -> (execute or log).
    Returns a dict suitable for JSON serialization.
    """
    if dry_run is None:
        dry_run = KILL_SWITCH

    cycle_start = datetime.now(timezone.utc)
    print(f"\n=== AegisQuant v2 cycle @ {cycle_start.isoformat()} ===")
    print(f"  KILL_SWITCH={KILL_SWITCH}  dry_run={dry_run}")
    print(f"  Enabled sleeves: {ENABLED_SLEEVES}")
    print(f"  Universe: {UNIVERSE_NAME}  Capital: ${INITIAL_CAPITAL:,.0f}")

    universe_fn = {"sp100": sp100_tickers, "sp500": sp500_tickers}[UNIVERSE_NAME]

    # 1) Compute each sleeve's weights
    sleeve_results: Dict[str, SleeveResult] = {}
    for name in ENABLED_SLEEVES:
        factory = ALL_SLEEVE_FACTORY.get(name)
        if factory is None:
            logger.warning(f"Unknown sleeve: {name}")
            continue
        sleeve = factory()
        sleeve.universe = lambda fn=universe_fn: fn()
        try:
            t0 = time.time()
            res = sleeve.weights()
            dt = time.time() - t0
            sleeve_results[name] = res
            print(f"  [{name}] {len(res.weights)} positions in {dt:.1f}s — {res.notes}")
        except Exception as e:
            logger.error(f"sleeve {name} failed: {e}")

    if not sleeve_results:
        print("No sleeve produced positions; aborting cycle.")
        return {"cycle_at": cycle_start.isoformat(), "status": "no_sleeves"}

    # 2) Macro regime
    macro = MacroRegimeAgent().compute()
    print(f"  Macro regime: {macro.score:+.2f} (conf {macro.confidence:.2f}) — {macro.rationale}")

    # 3) Combiner
    combiner = Combiner()
    target = combiner.combine(
        sleeve_results,
        macro_regime_score=macro.score,
        macro_regime_confidence=macro.confidence,
    )
    print(f"  Combiner: sleeve_weights={target.sleeve_weights}, "
          f"n_positions={target.n_positions}, cash={target.cash_weight:.1%}")

    # 4) RiskOfficer
    officer = RiskOfficer()
    current_dd = _current_drawdown_from_alpaca()
    review = officer.review(target, current_drawdown=current_dd)
    print(f"  RiskOfficer: {len(review.approved_weights)} approved positions, "
          f"{len(review.violations)} violations, dd_scale={review.drawdown_scaling}")
    for v in review.violations[:5]:
        print(f"    - {v}")
    if len(review.violations) > 5:
        print(f"    ... and {len(review.violations) - 5} more")

    # 5) Show top positions
    if review.approved_weights:
        print("  Top 10 approved positions:")
        for t, w in sorted(review.approved_weights.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {t}: {w*100:.2f}%  (${w * INITIAL_CAPITAL:,.0f})")

    # 6) Execute or dry-run
    if dry_run:
        print(f"  [DRY RUN] KILL_SWITCH active — no trades sent.")
        deltas = []
    else:
        deltas = _execute_deltas(review.approved_weights, INITIAL_CAPITAL)
        print(f"  Executed {len(deltas)} delta orders.")

    # 7) Persist snapshot for dashboard
    payload = {
        "cycle_at": cycle_start.isoformat(),
        "dry_run": dry_run,
        "kill_switch": KILL_SWITCH,
        "enabled_sleeves": ENABLED_SLEEVES,
        "macro_regime": {
            "score": macro.score, "confidence": macro.confidence,
            "rationale": macro.rationale, "metadata": macro.metadata,
        },
        "sleeve_results": {
            name: {
                "n_positions": len(r.weights),
                "weights": r.weights,
                "notes": r.notes,
            }
            for name, r in sleeve_results.items()
        },
        "combiner": {
            "sleeve_weights": target.sleeve_weights,
            "cash_weight": target.cash_weight,
            "rationale": target.rationale,
        },
        "risk_officer": {
            "approved_weights": review.approved_weights,
            "violations": review.violations,
            "drawdown_scaling": review.drawdown_scaling,
            "rationale": review.rationale,
        },
        "executed_deltas": deltas,
    }
    _write_snapshot(payload)
    return payload


# ── Alpaca integration (read DD, execute) ───────────────────────────────────


def _current_drawdown_from_alpaca() -> float:
    """Read current drawdown from Alpaca account. Returns 0.0 on failure."""
    try:
        import requests
        api_key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")
        base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
        if not api_key or not secret:
            return 0.0
        r = requests.get(
            f"{base}/v2/account",
            headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret},
            timeout=10,
        )
        if r.status_code != 200:
            return 0.0
        d = r.json()
        equity = float(d.get("equity", 0))
        last_equity = float(d.get("last_equity", equity))
        # Use cycle-over-cycle drawdown as a proxy; for proper peak-to-trough,
        # the dashboard's portfolio history endpoint already tracks it.
        if last_equity > 0:
            return min(0.0, (equity - last_equity) / last_equity)
        return 0.0
    except Exception as e:
        logger.warning(f"_current_drawdown_from_alpaca: {e}")
        return 0.0


def _execute_deltas(target_weights: Dict[str, float], nav: float) -> List[dict]:
    """
    Compute deltas vs current Alpaca positions and submit orders.
    Returns a list of order summaries.
    """
    try:
        import requests
        api_key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")
        base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
        if not api_key or not secret:
            logger.error("Alpaca credentials missing; cannot execute.")
            return []
        headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret}

        # Fetch current positions
        r = requests.get(f"{base}/v2/positions", headers=headers, timeout=10)
        if r.status_code != 200:
            logger.error(f"Alpaca positions fetch failed: {r.status_code}")
            return []
        current = {p["symbol"]: int(float(p["qty"])) for p in r.json()}

        # Fetch latest prices for sizing
        symbols = sorted(set(target_weights) | set(current))
        if not symbols:
            return []
        # Use yfinance instead of Alpaca data API to avoid subscription requirements
        import yfinance as yf
        prices_df = yf.download(symbols, period="1d", progress=False, auto_adjust=True)
        if "Close" in prices_df.columns:
            close = prices_df["Close"].iloc[-1]
            if hasattr(close, "to_dict"):
                last_prices = close.to_dict()
            else:
                last_prices = {symbols[0]: float(close)}
        else:
            last_prices = {}

        orders = []
        for sym in symbols:
            price = float(last_prices.get(sym, 0) or 0)
            if price <= 0:
                continue
            target_dollars = target_weights.get(sym, 0.0) * nav
            target_qty = int(target_dollars / price)
            current_qty = current.get(sym, 0)
            delta = target_qty - current_qty
            if delta == 0:
                continue
            side = "buy" if delta > 0 else "sell"
            qty = abs(delta)
            order = {
                "symbol": sym, "side": side, "qty": qty, "price": price,
                "current": current_qty, "target": target_qty,
            }
            try:
                resp = requests.post(
                    f"{base}/v2/orders",
                    headers=headers,
                    json={
                        "symbol": sym, "qty": qty, "side": side,
                        "type": "market", "time_in_force": "day",
                    },
                    timeout=10,
                )
                order["status"] = resp.status_code
                if resp.status_code >= 400:
                    order["error"] = resp.text[:200]
                else:
                    order["alpaca_id"] = resp.json().get("id")
            except Exception as e:
                order["status"] = "exception"
                order["error"] = str(e)[:200]
            orders.append(order)

        return orders
    except Exception as e:
        logger.error(f"_execute_deltas failed: {e}")
        return []


# ── Entry point ─────────────────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(description="AegisQuant v2 trading loop")
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously every 60min during NYSE hours")
    parser.add_argument("--dry-run", action="store_true",
                        help="Force dry-run even if KILL_SWITCH is off")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not args.loop:
        run_cycle(dry_run=args.dry_run or KILL_SWITCH)
        return 0

    # Loop mode: APScheduler
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("Install apscheduler for --loop mode: pip install apscheduler")
        return 1

    scheduler = BlockingScheduler()
    # NYSE hours: 9:30am-4:00pm ET, weekdays
    scheduler.add_job(
        run_cycle, CronTrigger(
            day_of_week="mon-fri", hour="9-15", minute="*/60",
            timezone="America/New_York",
        ),
        kwargs={"dry_run": args.dry_run or KILL_SWITCH},
        max_instances=1,
    )
    print("AegisQuant v2 loop started — Ctrl-C to stop.")
    scheduler.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
