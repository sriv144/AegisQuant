"""
US Live Trading Loop (Agent-Consensus + Bollinger Band)
========================================================
Runs batch consensus scoring across all tickers every cycle (every 30 min
via GitHub Actions).

Architecture:
  Screen universe → fetch data → run 9 strategies + 4 research agents per ticker
  → ConsensusWeightEngine (0.6 * agent + 0.4 * strategy + BB filter)
  → rank + allocate (max 10%/ticker, max 15 positions, long-only)
  → delta vs Alpaca live positions → execute only the difference

Set these env vars:
  MARKET=US
  BROKER=alpaca  (or 'paper' for simulation)
  ALPACA_API_KEY=...
  ALPACA_SECRET_KEY=...
  ALPACA_BASE_URL=https://paper-api.alpaca.markets  (paper trading)
  INITIAL_CAPITAL=100000
"""

import os

# Force US market mode before any imports
os.environ.setdefault("MARKET", "US")

from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import logging

from src import config  # noqa: F401  # Ensures .env is loaded
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
from src.db.models import db_manager

logger = logging.getLogger(__name__)

_failsafe = ExecutionFailsafe()
_consensus_engine = ConsensusWeightEngine(
    agent_weight=0.6,
    strategy_weight=0.4,
    min_consensus=0.30,
    max_positions=15,
    max_per_ticker=0.10,
)

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000"))


def _fetch_vix() -> float:
    """Fetch latest CBOE VIX from yfinance. Returns 20.0 on any failure."""
    return us_market_data.get_vix()


def _get_live_portfolio_state(broker: BaseBroker, tickers: list, current_prices: dict) -> dict:
    """
    Compute real portfolio state from DB-tracked positions and realized P&L.
    No more hardcoded values — portfolio_value changes as trades win/lose.
    """
    vix = _fetch_vix()
    current_weights = np.zeros(len(tickers))

    pf = db_manager.compute_portfolio_value(INITIAL_CAPITAL, current_prices)
    portfolio_value = pf["portfolio_value"]
    drawdown = pf["current_drawdown"]

    mode_label = broker.__class__.__name__
    print(
        f"[LiveState] {mode_label} mode — portfolio_value=${portfolio_value:,.0f}  "
        f"realized=${pf['realized_pnl']:,.0f}  unrealized=${pf['unrealized_pnl']:,.0f}  "
        f"cash=${pf['cash_balance']:,.0f}  positions={pf['open_position_count']}"
    )
    print(f"[LiveState] drawdown={drawdown:.4f}  VIX={vix:.2f}  peak=${pf['peak_equity']:,.0f}")

    return {
        "current_drawdown": drawdown,
        "vix_raw": vix,
        "current_weights": current_weights.tolist(),
        "portfolio_value": portfolio_value,
    }


def _get_held_tickers(broker: BaseBroker) -> set:
    """Query broker for currently held positions."""
    held = set()
    try:
        positions = broker.get_positions()
        for pos in positions:
            sym = pos.get("ticker", pos.get("symbol", ""))
            qty = int(pos.get("qty", 0))
            if sym and qty > 0:
                held.add(sym)
    except Exception as e:
        print(f"[Positions] Failed to query broker positions: {e}")

    # Also check DB-tracked positions
    try:
        db_positions = position_manager.get_open_positions()
        for ticker in db_positions:
            held.add(ticker)
    except Exception:
        pass

    return held


def _run_research_agents(ticker: str, indicators: dict, alt_data: dict, portfolio_state: dict) -> list:
    """Run all 4 research agents for a single ticker. Returns list of signal dicts."""
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
            agent_signals = result.get("research_signals", [])
            signals.extend(agent_signals)
        except Exception as e:
            print(f"  [{ticker}] Agent {getattr(agent, 'name', '?')} failed: {e}")

    return signals


def _run_all_strategies(ticker: str, indicators: dict, portfolio_state: dict, alt_data: dict) -> list:
    """Run all 9 strategies for a single ticker. Returns list of signal dicts."""
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
    print(f"\n[{datetime.now()}] [US Pipeline] Waking up. Running agent-consensus cycle...")

    # 1. Refresh universe
    print("[Pipeline] Screening universe...")
    UNIVERSE = us_universe_screener.screen_universe()
    print(f"[Pipeline] Selected {len(UNIVERSE)} tickers from dynamic screening")

    # 2. Broker (auto-selects Alpaca/PaperBroker from env config)
    broker = get_broker()
    broker.connect()

    # 3. Position manager: close any SL/TP/aged positions FIRST
    print("[Pipeline] Checking position exits (SL/TP/aging)...")
    theo_prices = {}
    for tick in UNIVERSE:
        theo_prices[tick] = us_market_data.get_latest_quote(tick)

    exits = position_manager.daily_check(theo_prices)
    for ticker in exits:
        position_manager.close_position(ticker, theo_prices[ticker], reason="EXIT_SIGNAL")
        print(f"[PositionManager] Closed {ticker}")

    # 4. Build live portfolio state (uses DB-tracked P&L)
    portfolio_state = _get_live_portfolio_state(broker, UNIVERSE, theo_prices)

    # 5. Capital allocator: compute budgets
    intraday_budget, delivery_budget = capital_allocator.get_budgets(portfolio_state)
    print(f"[CapitalAllocator] Budgets: ${intraday_budget:,.0f} intraday, ${delivery_budget:,.0f} swing")

    # 6. Pre-compute technical indicators + sentiment for all tickers
    print("[Pipeline] Pre-computing technical indicators and sentiment signals...")
    ticker_indicators = {}
    ticker_alt_data = {}
    hist_start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    hist_end = datetime.now().strftime("%Y-%m-%d")

    for ticker in UNIVERSE:
        hist = us_market_data.get_historical_data(ticker, start_date=hist_start, end_date=hist_end)
        if hist and len(hist) >= 20:
            df_feat = feature_engineer.compute_technical_indicators(hist)
            latest = df_feat.iloc[-1]
            ticker_indicators[ticker] = {k: float(v) for k, v in latest.items() if pd.notna(v) and isinstance(v, (int, float, np.number))}
        else:
            ticker_indicators[ticker] = {}

        news = alt_data_collector.get_recent_news(ticker)
        agg = feature_engineer.aggregate_sentiment(news)
        ticker_alt_data[ticker] = {
            "sentiment": agg.get("sentiment_score", 0.0),
            "sentiment_score": agg.get("sentiment_score", 0.0),
            "news_volume": agg.get("news_volume", 0),
        }

    # 7. Get currently held tickers from broker + DB
    held_tickers = _get_held_tickers(broker)
    print(f"[Pipeline] Currently holding {len(held_tickers)} positions: {sorted(held_tickers)}")

    # 8. Run 4 research agents per ticker (batch)
    print("[Pipeline] Running 4 research agents across all tickers...")
    agent_signals = {}
    for ticker in UNIVERSE:
        signals = _run_research_agents(
            ticker, ticker_indicators[ticker], ticker_alt_data[ticker], portfolio_state
        )
        agent_signals[ticker] = signals

    # 9. Run 9 strategies per ticker (batch)
    print("[Pipeline] Running 9 strategies across all tickers...")
    strategy_signals = {}
    for ticker in UNIVERSE:
        signals = _run_all_strategies(
            ticker, ticker_indicators[ticker], portfolio_state, ticker_alt_data[ticker]
        )
        strategy_signals[ticker] = signals

    # 10. Consensus weight engine: score → BB filter → rank → allocate
    print("[Pipeline] Computing consensus weights with BB entry/exit filter...")
    target_weights, actions, scores = _consensus_engine.compute_target_weights(
        tickers=UNIVERSE,
        agent_signals=agent_signals,
        strategy_signals=strategy_signals,
        indicators=ticker_indicators,
        held_tickers=held_tickers,
    )

    # Log per-ticker decisions
    trade_reasoning_map = {}
    for i, ticker in enumerate(UNIVERSE):
        action = actions[ticker]
        score = scores[ticker]
        weight = target_weights[i]
        bb_pos = ticker_indicators.get(ticker, {}).get("BB_Position", 0.5)

        if action != "SKIP":
            print(f"  [{ticker}] {action} | consensus={score:.3f} | BB={bb_pos:.3f} | weight={weight:.3%}")

        trade_reasoning_map[ticker] = {
            "consensus_action": action,
            "consensus_score": round(score, 4),
            "bb_position": round(bb_pos, 4),
            "target_weight": round(float(weight), 4),
            "agent_signals": [
                {"agent": s.get("agent_name", "?"), "action": s.get("action", ""), "confidence": s.get("confidence", 0)}
                for s in agent_signals.get(ticker, [])
            ],
            "strategy_signals": [
                {"strategy": s.get("strategy", "?"), "action": s.get("action", ""), "confidence": s.get("confidence", 0)}
                for s in strategy_signals.get(ticker, [])
            ],
        }

    print(f"[Pipeline] Consensus Weights -> {target_weights.round(3)}")

    # 11. Run circuit breakers with live state
    trade_types = {t: "CNC" for t in UNIVERSE}  # All swing/delivery in consensus mode
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
        print(f"[CircuitBreaker] TRIGGERED: {cb_reason}. Weights adjusted.")
    print(f"[Pipeline] Safe Weights -> {safe_weights.round(3)}")

    # 12. Delta-based execution via broker
    results = broker.execute_target_weights(
        tickers=UNIVERSE,
        target_weights=safe_weights,
        theoretical_prices=theo_prices,
        portfolio_value=portfolio_state["portfolio_value"],
        trade_types=trade_types,
    )

    # 13. Log positions for new entries using actual fill prices
    for i, ticker in enumerate(UNIVERSE):
        if actions.get(ticker) == "ENTER" and safe_weights[i] > 0:
            result = results.get(ticker)
            if result and result.filled_qty > 0:
                fill_price = result.fill_price if result.fill_price > 0 else theo_prices[ticker]
                # Determine which strategy had highest confidence
                strat_sigs = strategy_signals.get(ticker, [])
                best_strat = "consensus"
                if strat_sigs:
                    long_strats = [s for s in strat_sigs if s.get("action") == "LONG"]
                    if long_strats:
                        best_strat = max(long_strats, key=lambda s: s.get("confidence", 0)).get("strategy", "consensus")
                pos = Position.default_cnc(ticker, fill_price, result.filled_qty, best_strat)
                position_manager.open_position(pos)

    # 14. Close positions for EXIT signals
    for ticker in UNIVERSE:
        if actions.get(ticker) == "EXIT" and ticker in held_tickers:
            exit_price = theo_prices.get(ticker, 0.0)
            if exit_price > 0:
                position_manager.close_position(ticker, exit_price, reason="BB_EXIT")
                print(f"[PositionManager] BB EXIT: {ticker} @ ${exit_price:.2f}")

    # 15. Metric computations
    shortfall = broker.calculate_shortfall(UNIVERSE, safe_weights, theo_prices, results)
    total_commission = sum(r.commission for r in results.values())
    total_slippage = sum(r.slippage_bps for r in results.values()) / max(len(results), 1)
    print(f"[Pipeline] Trade complete. Shortfall: {shortfall:.2f} bps, "
          f"Avg slippage: {total_slippage:.1f} bps, Commission: ${total_commission:.2f}")

    # 16. Log decision to database
    model_version = "consensus_bb_v1"
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

    # 17. Log daily P&L
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

        total_pnl = daily_pnl["total_pnl"]
        if existing:
            existing.total_portfolio_value = portfolio_state["portfolio_value"]
            existing.intraday_pnl = daily_pnl["intraday_pnl"]
            existing.delivery_pnl = daily_pnl["delivery_pnl"]
            existing.total_pnl = total_pnl
            existing.drawdown = portfolio_state["current_drawdown"]
            existing.intraday_ratio_used = capital_allocator.current_intraday_ratio
        else:
            pnl_record = DailyPnL(
                date=today,
                total_portfolio_value=portfolio_state["portfolio_value"],
                intraday_pnl=daily_pnl["intraday_pnl"],
                delivery_pnl=daily_pnl["delivery_pnl"],
                total_pnl=total_pnl,
                drawdown=portfolio_state["current_drawdown"],
                intraday_ratio_used=capital_allocator.current_intraday_ratio,
            )
            session.add(pnl_record)

        session.commit()
        session.close()
    except Exception as e:
        print(f"[DailyPnL] Failed to log: {e}")

    # Summary
    enter_count = sum(1 for a in actions.values() if a == "ENTER")
    exit_count = sum(1 for a in actions.values() if a == "EXIT")
    hold_count = sum(1 for a in actions.values() if a == "HOLD")
    print(f"[DB] Decision logged. model={model_version}  cb={cb_reason}")
    print(f"[Summary] ENTER={enter_count} EXIT={exit_count} HOLD={hold_count} | Orders={len(results)}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="Execute immediately instead of waiting for next interval")
    args = parser.parse_args()

    if args.now:
        main_us_live_loop()
    else:
        print("Starting APScheduler Daemon (US Markets)...")
        print("AegisQuant US is armed. Pipeline runs every 30 minutes from 09:35 to 15:50 ET (Mon-Fri).")
        from apscheduler.schedulers.blocking import BlockingScheduler

        scheduler = BlockingScheduler()

        # Run every 30 minutes during US market hours (9:35 AM – 3:50 PM ET)
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
