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
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

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

NY_TZ = ZoneInfo("America/New_York")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.4f", name, raw, default)
        return default


KILL_SWITCH = _env_bool("KILL_SWITCH", False)
ENABLED_SLEEVES = [
    s.strip()
    for s in os.getenv("ENABLED_SLEEVES", "xs_momentum,value_quality_momentum").split(",")
    if s.strip()
]
UNIVERSE_NAME = os.getenv("UNIVERSE", "sp100").lower()
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000"))
V2_MAX_TOTAL_INVESTED = _env_float("V2_MAX_TOTAL_INVESTED", 0.65)
V2_MAX_SLEEVE_NAV = _env_float("V2_MAX_SLEEVE_NAV", 0.325)
V2_MAX_POSITION_NAV = _env_float("V2_MAX_POSITION_NAV", 0.05)
V2_MAX_SECTOR_NAV = _env_float("V2_MAX_SECTOR_NAV", 0.20)
V2_BETA_MIN = _env_float("V2_BETA_MIN", 0.0)
V2_BETA_MAX = _env_float("V2_BETA_MAX", 1.0)
V2_ENFORCE_BETA = _env_bool("V2_ENFORCE_BETA", True)
V2_MIN_TRADE_NAV_PCT = _env_float("V2_MIN_TRADE_NAV_PCT", 0.005)
V2_LIVE_START_DATE = os.getenv("V2_LIVE_START_DATE", "2026-06-08")
V2_REQUIRE_PRETRADE_REVIEW = _env_bool("V2_REQUIRE_PRETRADE_REVIEW", False)
BENCHMARK_SYMBOL = os.getenv("BENCHMARK_SYMBOL", "SPY")

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


def _live_start_reached(now_utc: Optional[datetime] = None) -> bool:
    """True once the configured New York trading date is reached."""
    if not V2_LIVE_START_DATE:
        return True
    try:
        start = date.fromisoformat(V2_LIVE_START_DATE)
    except ValueError:
        logger.warning("Invalid V2_LIVE_START_DATE=%r; live start gate disabled", V2_LIVE_START_DATE)
        return True
    now_utc = now_utc or datetime.now(timezone.utc)
    return now_utc.astimezone(NY_TZ).date() >= start


def _resolve_dry_run(requested_dry_run: Optional[bool], now_utc: Optional[datetime] = None) -> tuple[bool, str]:
    """Apply kill switch, live-start, and pre-trade review guards."""
    if requested_dry_run is True:
        return True, "requested"
    if KILL_SWITCH:
        return True, "KILL_SWITCH"
    if not _live_start_reached(now_utc):
        return True, f"before V2_LIVE_START_DATE={V2_LIVE_START_DATE}"
    if V2_REQUIRE_PRETRADE_REVIEW:
        return True, "V2_REQUIRE_PRETRADE_REVIEW"
    return False, "live_enabled"


def _loop_cron_minute() -> str:
    return "5,35"


# ── Core cycle ──────────────────────────────────────────────────────────────


def run_cycle(dry_run: Optional[bool] = None) -> dict:
    """
    Run one complete cycle: sleeves -> combiner -> risk officer -> (execute or log).
    Returns a dict suitable for JSON serialization.
    """
    cycle_start = datetime.now(timezone.utc)
    dry_run, dry_run_reason = _resolve_dry_run(dry_run, cycle_start)
    print(f"\n=== AegisQuant v2 cycle @ {cycle_start.isoformat()} ===")
    print(f"  KILL_SWITCH={KILL_SWITCH}  dry_run={dry_run} ({dry_run_reason})")
    print(f"  Enabled sleeves: {ENABLED_SLEEVES}")
    print(f"  Universe: {UNIVERSE_NAME}  Initial capital: ${INITIAL_CAPITAL:,.0f}")
    print(
        f"  Rollout caps: max_total={V2_MAX_TOTAL_INVESTED:.1%} "
        f"max_sleeve={V2_MAX_SLEEVE_NAV:.1%} max_position={V2_MAX_POSITION_NAV:.1%} "
        f"max_sector={V2_MAX_SECTOR_NAV:.1%} beta_cap={V2_BETA_MAX:.2f}"
    )

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
    combiner = Combiner(
        max_sleeve_nav=V2_MAX_SLEEVE_NAV,
        max_total_invested=V2_MAX_TOTAL_INVESTED,
    )
    target = combiner.combine(
        sleeve_results,
        macro_regime_score=macro.score,
        macro_regime_confidence=macro.confidence,
    )
    print(f"  Combiner: sleeve_weights={target.sleeve_weights}, "
          f"n_positions={target.n_positions}, cash={target.cash_weight:.1%}")

    # 4) RiskOfficer
    account_state = _alpaca_account_state()
    nav = float(account_state.get("equity") or INITIAL_CAPITAL)
    current_dd = float(account_state.get("drawdown") or 0.0)
    officer = RiskOfficer(
        max_position_nav=V2_MAX_POSITION_NAV,
        max_sector_nav=V2_MAX_SECTOR_NAV,
        max_sleeve_nav=V2_MAX_SLEEVE_NAV,
        beta_min=V2_BETA_MIN,
        beta_max=V2_BETA_MAX,
        enforce_beta=V2_ENFORCE_BETA,
    )
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
            print(f"    {t}: {w*100:.2f}%  (${w * nav:,.0f})")

    # 6) Execute or dry-run
    planned_deltas = _build_delta_orders(review.approved_weights, nav)
    _print_delta_preview(planned_deltas)
    if dry_run:
        print(f"  [DRY RUN] {dry_run_reason} - no trades sent.")
        deltas = planned_deltas
    else:
        deltas = _submit_delta_orders(planned_deltas)
        print(f"  Executed {len(deltas)} delta orders.")

    # 7) Persist snapshot for dashboard
    payload = {
        "cycle_at": cycle_start.isoformat(),
        "dry_run": dry_run,
        "dry_run_reason": dry_run_reason,
        "kill_switch": KILL_SWITCH,
        "enabled_sleeves": ENABLED_SLEEVES,
        "nav_used": nav,
        "account_state": account_state,
        "risk_limits": {
            "max_total_invested": V2_MAX_TOTAL_INVESTED,
            "max_sleeve_nav": V2_MAX_SLEEVE_NAV,
            "max_position_nav": V2_MAX_POSITION_NAV,
            "max_sector_nav": V2_MAX_SECTOR_NAV,
            "beta_min": V2_BETA_MIN,
            "beta_max": V2_BETA_MAX,
            "enforce_beta": V2_ENFORCE_BETA,
            "min_trade_nav_pct": V2_MIN_TRADE_NAV_PCT,
            "live_start_date": V2_LIVE_START_DATE,
        },
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
    _record_cycle_audit(cycle_start, payload, planned_deltas, deltas)
    return payload


def _record_cycle_audit(cycle_start: datetime, payload: dict, planned_deltas: List[dict], deltas: List[dict]) -> None:
    try:
        from src.engine.audit import audit_logger

        action = "NO_TRADE" if not planned_deltas else ("PLAN_ONLY" if payload.get("dry_run") else "EXECUTE")
        audit_logger.record_decision_cycle(
            run_id=f"us-v2-{cycle_start.strftime('%Y%m%d-%H%M%S')}",
            action=action,
            benchmark_symbol=BENCHMARK_SYMBOL,
            sleeve_weights=payload.get("combiner", {}).get("sleeve_weights", {}),
            approved_weights=payload.get("risk_officer", {}).get("approved_weights", {}),
            risk_violations=payload.get("risk_officer", {}).get("violations", []),
            planned_orders=planned_deltas,
            fills=[d for d in deltas if str(d.get("status", "")).startswith("2")],
            rejected_orders=[d for d in deltas if str(d.get("status", "")).upper() == "REJECTED"],
            portfolio_value=float(payload.get("nav_used") or 0.0),
            notes=payload.get("dry_run_reason", ""),
        )
    except Exception as e:
        logger.warning("decision cycle audit failed: %s", e)


# ── Alpaca integration (read DD, execute) ───────────────────────────────────


def _alpaca_config() -> tuple[str, dict]:
    api_key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
    headers = {}
    if api_key and secret:
        headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret}
    return base, headers


def _alpaca_account_state() -> dict:
    """Fetch live equity and peak-to-trough drawdown from Alpaca when available."""
    try:
        import requests
        base, headers = _alpaca_config()
        if not headers:
            return {"equity": INITIAL_CAPITAL, "drawdown": 0.0, "source": "initial_capital"}

        acct_resp = requests.get(f"{base}/v2/account", headers=headers, timeout=10)
        if acct_resp.status_code != 200:
            logger.warning("Alpaca account fetch failed: %s", acct_resp.status_code)
            return {"equity": INITIAL_CAPITAL, "drawdown": 0.0, "source": "initial_capital"}
        acct = acct_resp.json()
        equity = float(acct.get("equity") or acct.get("portfolio_value") or INITIAL_CAPITAL)
        last_equity = float(acct.get("last_equity") or equity)

        peak_equity = max(INITIAL_CAPITAL, equity, last_equity)
        try:
            hist_resp = requests.get(
                f"{base}/v2/account/portfolio/history",
                headers=headers,
                params={"period": "1M", "timeframe": "1D", "extended_hours": "false"},
                timeout=10,
            )
            if hist_resp.status_code == 200:
                hist = hist_resp.json()
                equities = [
                    float(x) for x in hist.get("equity", [])
                    if x is not None and float(x) > 0
                ]
                if equities:
                    peak_equity = max(peak_equity, max(equities))
        except Exception as e:
            logger.warning("Alpaca history fetch failed: %s", e)

        drawdown = min(0.0, equity / peak_equity - 1.0) if peak_equity > 0 else 0.0
        return {
            "equity": equity,
            "last_equity": last_equity,
            "peak_equity": peak_equity,
            "drawdown": drawdown,
            "cash": float(acct.get("cash") or 0.0),
            "buying_power": float(acct.get("buying_power") or 0.0),
            "source": "alpaca",
        }
    except Exception as e:
        logger.warning("_alpaca_account_state: %s", e)
        return {"equity": INITIAL_CAPITAL, "drawdown": 0.0, "source": "initial_capital"}


def _current_positions_from_alpaca() -> Dict[str, int]:
    try:
        import requests
        base, headers = _alpaca_config()
        if not headers:
            return {}
        resp = requests.get(f"{base}/v2/positions", headers=headers, timeout=10)
        if resp.status_code != 200:
            logger.error("Alpaca positions fetch failed: %s", resp.status_code)
            return {}
        return {p["symbol"]: int(float(p["qty"])) for p in resp.json()}
    except Exception as e:
        logger.error("Alpaca positions fetch failed: %s", e)
        return {}


def _latest_prices(symbols: List[str]) -> Dict[str, float]:
    if not symbols:
        return {}
    try:
        import yfinance as yf
        prices_df = yf.download(symbols, period="1d", progress=False, auto_adjust=True)
        if prices_df is None or prices_df.empty or "Close" not in prices_df.columns:
            return {}
        close = prices_df["Close"].iloc[-1]
        if hasattr(close, "to_dict"):
            return {k: float(v) for k, v in close.to_dict().items() if v and v > 0}
        return {symbols[0]: float(close)}
    except Exception as e:
        logger.warning("latest price fetch failed: %s", e)
        return {}


def _build_delta_orders(target_weights: Dict[str, float], nav: float) -> List[dict]:
    """Create a pre-trade order list without submitting anything."""
    current = _current_positions_from_alpaca()
    symbols = sorted(set(target_weights) | set(current))
    prices = _latest_prices(symbols)
    min_notional = max(0.0, V2_MIN_TRADE_NAV_PCT * nav)

    orders: List[dict] = []
    for sym in symbols:
        price = float(prices.get(sym, 0.0) or 0.0)
        if price <= 0:
            continue
        target_dollars = target_weights.get(sym, 0.0) * nav
        target_qty = int(target_dollars / price)
        current_qty = current.get(sym, 0)
        delta = target_qty - current_qty
        delta_notional = abs(delta * price)
        if delta == 0 or delta_notional < min_notional:
            continue
        side = "buy" if delta > 0 else "sell"
        orders.append({
            "symbol": sym,
            "side": side,
            "qty": abs(delta),
            "price": price,
            "current": current_qty,
            "target": target_qty,
            "target_weight": target_weights.get(sym, 0.0),
            "delta_notional": round(delta_notional, 2),
            "status": "planned",
        })
    return orders


def _print_delta_preview(orders: List[dict]) -> None:
    if not orders:
        print("  Pre-trade preview: no orders above threshold.")
        return
    gross = sum(float(o.get("delta_notional", 0.0)) for o in orders)
    buys = sum(1 for o in orders if o.get("side") == "buy")
    sells = sum(1 for o in orders if o.get("side") == "sell")
    print(f"  Pre-trade preview: {len(orders)} orders ({buys} buys / {sells} sells), gross delta ${gross:,.0f}")
    for o in orders[:15]:
        print(
            f"    {o['side'].upper()} {o['qty']} {o['symbol']} "
            f"target={o['target']} current={o['current']} est=${o['delta_notional']:,.0f}"
        )
    if len(orders) > 15:
        print(f"    ... and {len(orders) - 15} more")


def _submit_delta_orders(planned_orders: List[dict]) -> List[dict]:
    """Submit prebuilt delta orders to Alpaca."""
    try:
        import requests
        base, headers = _alpaca_config()
        if not headers:
            logger.error("Alpaca credentials missing; cannot execute.")
            return []

        submitted = []
        for order in planned_orders:
            out = dict(order)
            try:
                resp = requests.post(
                    f"{base}/v2/orders",
                    headers=headers,
                    json={
                        "symbol": order["symbol"],
                        "qty": order["qty"],
                        "side": order["side"],
                        "type": "market",
                        "time_in_force": "day",
                    },
                    timeout=10,
                )
                out["status"] = resp.status_code
                if resp.status_code >= 400:
                    out["error"] = resp.text[:200]
                else:
                    out["alpaca_id"] = resp.json().get("id")
            except Exception as e:
                out["status"] = "exception"
                out["error"] = str(e)[:200]
            submitted.append(out)
        return submitted
    except Exception as e:
        logger.error("_submit_delta_orders failed: %s", e)
        return []


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
    # NYSE hours: 30-minute paper-alpha decisions during regular session.
    scheduler.add_job(
        run_cycle, CronTrigger(
            day_of_week="mon-fri", hour="9-15", minute=_loop_cron_minute(),
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
