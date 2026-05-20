"""
US Live Trading Loop (Agent-Driven Autonomous Trader)
======================================================
Every 30-min cycle:
  1. Screen universe (~30 US stocks)
  2. Fetch technical indicators + sentiment for each
  3. Run 4 research agents per ticker (quant, fundamental, macro, sentiment)
  4. Run 9 strategies per ticker (momentum, mean reversion, trend, etc.)
  5. Trading Analyst (LLM) reasons about each ticker using ALL data,
     picks the right approach, and decides BUY / HOLD / EXIT with reasoning
  6. Weight engine: rank, diversify (max 10%/ticker, 15 positions), allocate
  7. Circuit breakers: long-only, max position, drawdown, market hours
  8. Delta execution: only trade the difference vs current Alpaca positions

The agents THINK. They don't follow rigid rules — they reason about each
stock's situation and explain WHY they're making each decision.

Set these env vars:
  MARKET=US  BROKER=alpaca
  ALPACA_API_KEY=...  ALPACA_SECRET_KEY=...
  ALPACA_BASE_URL=https://paper-api.alpaca.markets
  INITIAL_CAPITAL=100000
  OPENAI_API_KEY=...  (for LLM reasoning — falls back to heuristics without it)
"""

import os

# Force US market mode before any imports
os.environ.setdefault("MARKET", "US")

from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import logging

from src import config  # noqa: F401
from src.execution import get_broker
from src.execution.broker_base import BaseBroker
from src.data.us_market_data import us_market_data
from src.data.feature_engineering import feature_engineer
from src.data.alternative_data import alt_data as alt_data_collector
from src.data.us_universe_screener import us_universe_screener
from src.engine.circuit_breakers import ExecutionFailsafe
from src.engine.position_manager import position_manager, Position
from src.engine.capital_allocator import capital_allocator
from src.engine.consensus_weight_engine import ConsensusWeightEngine
from src.strategies import STRATEGY_REGISTRY
from src.agents.research.quant_agent import quant_agent
from src.agents.research.fundamental_agent import fundamental_agent
from src.agents.research.macro_agent import macro_agent
from src.agents.research.sentiment_agent import sentiment_agent
from src.agents.analyst.trading_analyst import trading_analyst
from src.db.models import db_manager

logger = logging.getLogger(__name__)

_failsafe = ExecutionFailsafe()
_weight_engine = ConsensusWeightEngine(max_positions=15, max_per_ticker=0.10)

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000"))


def _sync_positions_from_alpaca(broker: "BaseBroker") -> None:
    """
    Sync Alpaca live positions → open_positions DB table after each cycle.
    - Upserts every live Alpaca position (create if missing, update qty if changed)
    - Marks DB rows that are no longer in Alpaca as CLOSED (auto-exit detection)
    Bypasses the filled_qty poll gap that leaves the table empty.
    """
    from src.db.models import OpenPosition

    try:
        alpaca_positions = broker.get_positions()
    except Exception as e:
        logger.warning(f"[PositionSync] Could not fetch Alpaca positions: {e}")
        return

    if not alpaca_positions:
        logger.info("[PositionSync] No open positions in Alpaca — nothing to sync")
        return

    alpaca_map: dict = {p["ticker"]: p for p in alpaca_positions}
    today_str = datetime.now().strftime("%Y-%m-%d")
    now_str = datetime.now(timezone.utc).isoformat()

    try:
        with db_manager.SessionLocal() as session:
            # ── Upsert each live Alpaca position ────────────────────────
            for ticker, ap in alpaca_map.items():
                existing = (
                    session.query(OpenPosition)
                    .filter(OpenPosition.ticker == ticker, OpenPosition.status == "OPEN")
                    .first()
                )
                if existing:
                    # Refresh quantity in case of partial fills / additions
                    existing.quantity = int(ap["qty"])
                    existing.updated_at = now_str
                else:
                    # Brand-new position not previously tracked
                    session.add(OpenPosition(
                        ticker=ticker,
                        entry_price=float(ap["avg_price"]),
                        entry_date=today_str,
                        quantity=int(ap["qty"]),
                        trade_type="CNC",
                        strategy="alpaca_sync",
                        sector="US",
                        status="OPEN",
                        created_at=now_str,
                        updated_at=now_str,
                    ))
                    logger.info(f"[PositionSync] Inserted missing position: {ticker} qty={ap['qty']} avg=${ap['avg_price']:.2f}")

            # ── Auto-close DB rows no longer held in Alpaca ─────────────
            open_in_db = (
                session.query(OpenPosition).filter(OpenPosition.status == "OPEN").all()
            )
            closed_count = 0
            for pos in open_in_db:
                if pos.ticker not in alpaca_map:
                    pos.status = "CLOSED"
                    pos.exit_date = today_str
                    pos.exit_reason = "ALPACA_SYNC_EXIT"
                    pos.updated_at = now_str
                    closed_count += 1
                    logger.info(f"[PositionSync] Auto-closed {pos.ticker} (no longer in Alpaca)")

            session.commit()
            logger.info(
                f"[PositionSync] Done — {len(alpaca_map)} live positions, "
                f"{closed_count} auto-closed"
            )
    except Exception as e:
        logger.error(f"[PositionSync] DB write failed: {e}")


def _fetch_vix() -> float:
    """Fetch latest CBOE VIX from yfinance. Returns 20.0 on any failure."""
    return us_market_data.get_vix()


def _get_live_portfolio_state(broker: BaseBroker, tickers: list, current_prices: dict) -> dict:
    """
    Compute real portfolio state from DB-tracked positions and realized P&L.
    """
    vix = _fetch_vix()
    current_weights = np.zeros(len(tickers))

    pf = db_manager.compute_portfolio_value(INITIAL_CAPITAL, current_prices)
    portfolio_value = pf["portfolio_value"]
    drawdown = pf["current_drawdown"]

    mode_label = broker.__class__.__name__
    print(
        f"[LiveState] {mode_label} — ${portfolio_value:,.0f} "
        f"(realized=${pf['realized_pnl']:,.0f} unrealized=${pf['unrealized_pnl']:,.0f} "
        f"cash=${pf['cash_balance']:,.0f} positions={pf['open_position_count']})"
    )
    print(f"[LiveState] drawdown={drawdown:.4f}  VIX={vix:.2f}  peak=${pf['peak_equity']:,.0f}")

    return {
        "current_drawdown": drawdown,
        "vix_raw": vix,
        "current_weights": current_weights.tolist(),
        "portfolio_value": portfolio_value,
    }


def _get_held_tickers(broker: BaseBroker) -> set:
    """Query broker + DB for currently held positions."""
    held = set()
    try:
        for pos in broker.get_positions():
            sym = pos.get("ticker", pos.get("symbol", ""))
            if sym and int(pos.get("qty", 0)) > 0:
                held.add(sym)
    except Exception as e:
        print(f"[Positions] Broker query failed: {e}")

    try:
        for ticker in position_manager.get_open_positions():
            held.add(ticker)
    except Exception:
        pass

    return held


def _run_research_agents(ticker: str, indicators: dict, alt_data: dict, portfolio_state: dict) -> list:
    """Run all 4 research agents for a single ticker."""
    state = {
        "current_asset": ticker,
        "technical_indicators": indicators,
        "alternative_data": alt_data,
        "portfolio_state": portfolio_state,
        "research_signals": [],
    }

    signals = []
    for agent in [quant_agent, fundamental_agent, macro_agent, sentiment_agent]:
        try:
            result = agent.invoke(state)
            signals.extend(result.get("research_signals", []))
        except Exception as e:
            print(f"  [{ticker}] Agent {getattr(agent, 'name', '?')} failed: {e}")

    return signals


def _run_all_strategies(ticker: str, indicators: dict, portfolio_state: dict, alt_data: dict) -> list:
    """Run all 9 strategies for a single ticker."""
    signals = []
    for name, strategy in STRATEGY_REGISTRY.items():
        try:
            signal = strategy.generate_signal(
                ticker=ticker,
                indicators=indicators,
                portfolio_state=portfolio_state,
                alt_data=alt_data,
            )
            signals.append(signal)
        except Exception as e:
            print(f"  [{ticker}] Strategy {name} failed: {e}")
    return signals


def main_us_live_loop():
    print(f"\n{'='*60}")
    print(f"[{datetime.now()}] [AegisQuant] Agent-driven trading cycle starting...")
    print(f"{'='*60}")

    # ── Phase 1: Universe Screening ──────────────────────────────
    print("\n[Phase 1] Screening universe...")
    UNIVERSE = us_universe_screener.screen_universe()
    print(f"  Selected {len(UNIVERSE)} tickers: {', '.join(UNIVERSE[:10])}{'...' if len(UNIVERSE) > 10 else ''}")

    # ── Phase 2: Connect Broker ──────────────────────────────────
    broker = get_broker()
    broker.connect()

    # ── Phase 3: Position Management ─────────────────────────────
    print("\n[Phase 3] Checking position exits (SL/TP/aging)...")
    theo_prices = {}
    for tick in UNIVERSE:
        theo_prices[tick] = us_market_data.get_latest_quote(tick)

    exits = position_manager.daily_check(theo_prices)
    for ticker in exits:
        position_manager.close_position(ticker, theo_prices[ticker], reason="EXIT_SIGNAL")
        print(f"  Closed {ticker} (exit signal)")

    # ── Phase 4: Portfolio State ─────────────────────────────────
    portfolio_state = _get_live_portfolio_state(broker, UNIVERSE, theo_prices)
    intraday_budget, delivery_budget = capital_allocator.get_budgets(portfolio_state)

    # ── Phase 5: Data Collection ─────────────────────────────────
    print("\n[Phase 5] Computing indicators + sentiment for all tickers...")
    ticker_indicators = {}
    ticker_alt_data = {}
    hist_start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    hist_end = datetime.now().strftime("%Y-%m-%d")

    for ticker in UNIVERSE:
        hist = us_market_data.get_historical_data(ticker, start_date=hist_start, end_date=hist_end)
        if hist and len(hist) >= 20:
            df_feat = feature_engineer.compute_technical_indicators(hist)
            latest = df_feat.iloc[-1]
            ticker_indicators[ticker] = {
                k: float(v) for k, v in latest.items()
                if pd.notna(v) and isinstance(v, (int, float, np.number))
            }
        else:
            ticker_indicators[ticker] = {}

        news = alt_data_collector.get_recent_news(ticker)
        agg = feature_engineer.aggregate_sentiment(news)
        ticker_alt_data[ticker] = {
            "sentiment": agg.get("sentiment_score", 0.0),
            "sentiment_score": agg.get("sentiment_score", 0.0),
            "news_volume": agg.get("news_volume", 0),
        }

    held_tickers = _get_held_tickers(broker)
    print(f"  Currently holding {len(held_tickers)} positions: {sorted(held_tickers) if held_tickers else 'none'}")

    # ── Phase 6: Research Agents ─────────────────────────────────
    print("\n[Phase 6] Research agents analyzing all tickers...")
    all_agent_signals = {}
    for ticker in UNIVERSE:
        signals = _run_research_agents(
            ticker, ticker_indicators[ticker], ticker_alt_data[ticker], portfolio_state
        )
        all_agent_signals[ticker] = signals

    # ── Phase 7: Strategy Signals ────────────────────────────────
    print("\n[Phase 7] Running 9 strategies across all tickers...")
    all_strategy_signals = {}
    for ticker in UNIVERSE:
        signals = _run_all_strategies(
            ticker, ticker_indicators[ticker], portfolio_state, ticker_alt_data[ticker]
        )
        all_strategy_signals[ticker] = signals

    # ── Phase 8: Trading Analyst (THE CORE) ──────────────────────
    print("\n[Phase 8] Trading Analyst reasoning about each ticker...")
    llm_mode = "LLM" if trading_analyst.llm is not None else "FALLBACK (no OPENAI_API_KEY)"
    print(f"  Mode: {llm_mode}")

    analyst_decisions = {}
    for ticker in UNIVERSE:
        decision = trading_analyst.analyze_ticker(
            ticker=ticker,
            indicators=ticker_indicators[ticker],
            agent_signals=all_agent_signals[ticker],
            strategy_signals=all_strategy_signals[ticker],
            portfolio_state=portfolio_state,
            is_held=(ticker in held_tickers),
            current_price=theo_prices.get(ticker, 0.0),
        )
        analyst_decisions[ticker] = decision

    # ── Phase 9: Weight Allocation ───────────────────────────────
    print("\n[Phase 9] Computing portfolio weights...")
    target_weights, actions = _weight_engine.compute_target_weights(
        tickers=UNIVERSE,
        analyst_decisions=analyst_decisions,
    )

    # Build reasoning map for DB
    trade_reasoning_map = {}
    for i, ticker in enumerate(UNIVERSE):
        decision = analyst_decisions.get(ticker, {})
        trade_reasoning_map[ticker] = {
            "analyst_action": decision.get("action", "HOLD"),
            "analyst_confidence": decision.get("confidence", 0),
            "analyst_reasoning": decision.get("reasoning", ""),
            "strategy_used": decision.get("strategy_used", ""),
            "used_llm": decision.get("used_llm", False),
            "target_weight": round(float(target_weights[i]), 4),
            "agent_signals": [
                {"agent": s.get("agent_name", "?"), "action": s.get("action", ""), "confidence": s.get("confidence", 0)}
                for s in all_agent_signals.get(ticker, [])
            ],
            "strategy_signals": [
                {"strategy": s.get("strategy", "?"), "action": s.get("action", ""), "confidence": s.get("confidence", 0)}
                for s in all_strategy_signals.get(ticker, [])
            ],
        }

    print(f"  Target Weights -> {target_weights.round(3)}")

    # ── Phase 10: Circuit Breakers ───────────────────────────────
    trade_types = {t: "CNC" for t in UNIVERSE}
    cb_state = {
        "drawdown": portfolio_state["current_drawdown"],
        "vix_raw": portfolio_state["vix_raw"],
        "current_weights": np.array(portfolio_state["current_weights"]),
        "tickers": UNIVERSE,
        "trade_types": trade_types,
        "current_prices": theo_prices,
    }
    safe_weights, cb_reason = _failsafe.process_action(target_weights, cb_state)
    if cb_reason != "OK":
        print(f"  [CircuitBreaker] {cb_reason}")
    print(f"  Safe Weights -> {safe_weights.round(3)}")

    # ── Phase 11: Execution ──────────────────────────────────────
    print("\n[Phase 11] Executing trades (delta-based)...")
    results = broker.execute_target_weights(
        tickers=UNIVERSE,
        target_weights=safe_weights,
        theoretical_prices=theo_prices,
        portfolio_value=portfolio_state["portfolio_value"],
        trade_types=trade_types,
    )

    # Log new positions
    for i, ticker in enumerate(UNIVERSE):
        if actions.get(ticker) == "BUY" and safe_weights[i] > 0:
            result = results.get(ticker)
            if result and result.filled_qty > 0:
                fill_price = result.fill_price if result.fill_price > 0 else theo_prices[ticker]
                strategy = analyst_decisions.get(ticker, {}).get("strategy_used", "analyst")
                pos = Position.default_cnc(ticker, fill_price, result.filled_qty, strategy)
                position_manager.open_position(pos)

    # Execute EXIT signals
    for ticker in UNIVERSE:
        if actions.get(ticker) == "EXIT" and ticker in held_tickers:
            exit_price = theo_prices.get(ticker, 0.0)
            if exit_price > 0:
                reason = analyst_decisions.get(ticker, {}).get("reasoning", "Analyst EXIT")[:50]
                position_manager.close_position(ticker, exit_price, reason="ANALYST_EXIT")
                print(f"  EXIT {ticker} @ ${exit_price:.2f} — {reason}")

    # ── Phase 11b: Sync Alpaca → DB (fixes filled_qty gap) ───────
    print("\n[Phase 11b] Syncing Alpaca positions → DB...")
    _sync_positions_from_alpaca(broker)

    # ── Phase 12: Metrics & Logging ──────────────────────────────
    shortfall = broker.calculate_shortfall(UNIVERSE, safe_weights, theo_prices, results)
    total_commission = sum(r.commission for r in results.values())
    avg_slippage = sum(r.slippage_bps for r in results.values()) / max(len(results), 1)

    model_version = "analyst_llm_v1" if trading_analyst.llm else "analyst_fallback_v1"
    db_manager.log_decision_orm(
        timestamp=datetime.now(timezone.utc).isoformat(),
        ticker_universe=UNIVERSE,
        state_vector=[portfolio_state["current_drawdown"], portfolio_state["vix_raw"]],
        rl_output=target_weights,
        circuit_breaker_status=cb_reason,
        final_weights=safe_weights,
        transaction_costs=shortfall,
        model_version=model_version,
        trade_reasoning=trade_reasoning_map,
    )

    # Daily P&L
    try:
        daily_pnl = position_manager.get_daily_pnl()
        from src.db.models import DailyPnL
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        db_url = os.getenv("POSTGRES_URL", "sqlite:///aegisquant_live.db")
        engine = create_engine(db_url)
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()

        today = datetime.now().strftime("%Y-%m-%d")
        existing = session.query(DailyPnL).filter(DailyPnL.date == today).first()
        if existing:
            existing.total_portfolio_value = portfolio_state["portfolio_value"]
            existing.intraday_pnl = daily_pnl["intraday_pnl"]
            existing.delivery_pnl = daily_pnl["delivery_pnl"]
            existing.total_pnl = daily_pnl["total_pnl"]
            existing.drawdown = portfolio_state["current_drawdown"]
            existing.intraday_ratio_used = capital_allocator.current_intraday_ratio
        else:
            session.add(DailyPnL(
                date=today,
                total_portfolio_value=portfolio_state["portfolio_value"],
                intraday_pnl=daily_pnl["intraday_pnl"],
                delivery_pnl=daily_pnl["delivery_pnl"],
                total_pnl=daily_pnl["total_pnl"],
                drawdown=portfolio_state["current_drawdown"],
                intraday_ratio_used=capital_allocator.current_intraday_ratio,
            ))
        session.commit()
        session.close()
    except Exception as e:
        print(f"[DailyPnL] Failed: {e}")

    # ── Summary ──────────────────────────────────────────────────
    buy_count = sum(1 for a in actions.values() if a == "BUY")
    exit_count = sum(1 for a in actions.values() if a == "EXIT")
    hold_count = sum(1 for a in actions.values() if a == "HOLD")

    print(f"\n{'='*60}")
    print(f"[SUMMARY] {model_version} | cb={cb_reason}")
    print(f"  Decisions: BUY={buy_count}  HOLD={hold_count}  EXIT={exit_count}")
    print(f"  Orders executed: {len(results)}")
    print(f"  Shortfall: {shortfall:.2f} bps | Slippage: {avg_slippage:.1f} bps | Commission: ${total_commission:.2f}")
    if buy_count > 0:
        print(f"  Bought: {', '.join(t for t in UNIVERSE if actions.get(t) == 'BUY')}")
    if exit_count > 0:
        print(f"  Exited: {', '.join(t for t in UNIVERSE if actions.get(t) == 'EXIT')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="Execute immediately")
    args = parser.parse_args()

    if args.now:
        main_us_live_loop()
    else:
        print("Starting APScheduler Daemon (US Markets)...")
        print("AegisQuant armed. Pipeline runs every 30 minutes 9:35-15:50 ET (Mon-Fri).")
        from apscheduler.schedulers.blocking import BlockingScheduler

        scheduler = BlockingScheduler()
        scheduler.add_job(
            main_us_live_loop,
            'cron',
            day_of_week='mon-fri',
            hour='9-15',
            minute='5,35',
            timezone='US/Eastern',
            max_instances=1,
            coalesce=True,
        )

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            pass
