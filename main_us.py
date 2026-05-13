"""
US Live Trading Loop (Professional Multi-Mode Trader)
======================================================
Runs the dynamic multi-strategy RL pipeline on US markets with Alpaca.
- Dynamic universe screening (S&P 500 + growth stocks)
- Dual-mode trading: 80% swing (GTC) + 20% intraday (DAY orders)
- Position management with stop-loss, take-profit, and aging exits
- RL meta-learner for optimal intraday/swing split
Scheduled for 9:35 AM ET (market open + 5 min) on weekdays.

Set these env vars:
  MARKET=US
  BROKER=alpaca  (or 'paper' for simulation)
  ALPACA_API_KEY=...
  ALPACA_SECRET_KEY=...
  ALPACA_BASE_URL=https://paper-api.alpaca.markets  (paper trading)
  INITIAL_CAPITAL=100000  (default $100K for US paper)
"""

import os

# Force US market mode before any imports
os.environ.setdefault("MARKET", "US")

from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

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
from src.agents.orchestrator import orchestrator
from src.agents.state import AgentState
from src.agents.portfolio.pm_agent import pm_agent
from src.db.models import db_manager

_failsafe = ExecutionFailsafe()

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000"))  # $100K USD paper capital


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


def _fetch_weekly_strategy_scores() -> dict:
    """
    Compute per-strategy performance scores from realized trades in the last 30 days.
    """
    try:
        from sqlalchemy import text
        db_url = os.getenv("POSTGRES_URL", "sqlite:///aegisquant_live.db")
        from sqlalchemy import create_engine
        engine = create_engine(db_url)

        query = text("""
            SELECT strategy,
                   COUNT(*) as trade_count,
                   AVG(pnl_pct) as avg_pnl,
                   SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins
            FROM open_positions
            WHERE status = 'CLOSED'
              AND exit_date > datetime('now', '-30 days')
              AND pnl_pct IS NOT NULL
            GROUP BY strategy
            ORDER BY avg_pnl DESC
        """)

        with engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

        if not rows:
            return _default_strategy_scores()

        scores = {}
        for row in rows:
            strategy, trade_count, avg_pnl, wins = row
            win_rate = wins / trade_count if trade_count > 0 else 0.0
            scores[strategy] = round(avg_pnl * 100 + win_rate * 10, 2)

        print(f"[Strategy Scores] {len(scores)} strategies scored from {sum(r[1] for r in rows)} trades")
        return scores

    except Exception as e:
        print(f"[Strategy Scores] Query failed ({e}), using defaults")
        return _default_strategy_scores()


def _default_strategy_scores() -> dict:
    """Fallback scores — equal weight across core strategies."""
    return {
        "momentum": 0.0,
        "mean_reversion": 0.0,
        "trend_following": 0.0,
        "factor_investing": 0.0,
        "volatility_breakout": 0.0,
        "earnings_momentum": 0.0,
        "sector_rotation": 0.0,
        "gap_fill": 0.0,
        "pairs_trading": 0.0,
    }


def main_us_live_loop():
    print(f"\n[{datetime.now()}] [US Pipeline] Waking up. Initiating daily RL execution cycle...")

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

    # 5. Capital allocator: compute intraday vs swing budgets
    intraday_budget, delivery_budget = capital_allocator.get_budgets(portfolio_state)
    print(f"[CapitalAllocator] Budgets: ${intraday_budget:,.0f} intraday, ${delivery_budget:,.0f} swing")

    # 6. Pre-compute technical indicators + sentiment
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

    # 7. Fetch weekly strategy scores
    strategy_scores = _fetch_weekly_strategy_scores()

    # 8. Run the full LangGraph agent pipeline per ticker
    print("[Pipeline] Running LLM consensus + PPO inference via agent orchestrator...")
    target_weights = np.zeros(len(UNIVERSE))
    trade_types = {}
    trade_reasoning_map = {}
    model_version = "us_orchestrator_fallback"

    for i, ticker in enumerate(UNIVERSE):
        initial_state: AgentState = {
            "current_asset": ticker,
            "timestamp": datetime.now().isoformat(),
            "market_data": {"ticker": ticker, "price": theo_prices.get(ticker, 0.0)},
            "alternative_data": ticker_alt_data[ticker],
            "technical_indicators": ticker_indicators[ticker],
            "active_strategies": sorted(strategy_scores, key=strategy_scores.get, reverse=True)[:5],
            "strategy_scores": strategy_scores,
            "current_strategy": max(strategy_scores, key=strategy_scores.get) if strategy_scores else "momentum",
            "trade_type": "SKIP",
            "stop_loss_pct": 0.08,
            "take_profit_pct": 0.20,
            "intraday_budget": intraday_budget,
            "delivery_budget": delivery_budget,
            "research_signals": [],
            "committee_decision": {},
            "allocation_request": {},
            "risk_approval": {},
            "execution_result": {},
            "portfolio_state": portfolio_state,
        }
        final_state = orchestrator.run_cycle(initial_state)

        allocation = final_state.get("allocation_request", {})
        exposure = float(allocation.get("adjusted_exposure_pct") or allocation.get("target_exposure_pct") or 0.0)
        committee_dir = final_state.get("committee_decision", {}).get("direction", "LONG")
        target_weights[i] = exposure if committee_dir != "SHORT" else -exposure

        trade_type = final_state.get("trade_type", "SKIP")
        trade_types[ticker] = trade_type

        signals = final_state.get("research_signals", [])
        committee = final_state.get("committee_decision", {})
        risk = final_state.get("risk_approval", {})
        trade_reasoning_map[ticker] = {
            "research_signals": [
                {"agent": s.get("agent_name", "unknown"), "action": s.get("action", ""), "rationale": s.get("rationale", "")}
                for s in signals
            ],
            "committee": {"action": committee.get("action", ""), "direction": committee.get("direction", ""), "rationale": committee.get("rationale", "")},
            "allocation": {"exposure_pct": exposure, "rationale": allocation.get("rationale", "")},
            "risk": {"action": risk.get("action", ""), "rationale": risk.get("rationale", "")},
            "trade_type": trade_type,
        }

        if trade_type != "SKIP":
            print(f"  [{ticker}] {trade_type} @ {exposure*100:.1f}%")

    if pm_agent.rl_model is not None:
        model_version = "us_ppo_rl_live"

    # 9. Normalize to gross exposure constraint (1.5)
    gross = np.sum(np.abs(target_weights))
    if gross > 1.5:
        target_weights = target_weights * (1.5 / gross)

    print(f"[Pipeline] Raw RL Weights -> {target_weights.round(3)}")

    # 10. Run circuit breakers with live state
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

    # 11. Fire to execution via broker abstraction layer
    results = broker.execute_target_weights(
        tickers=UNIVERSE,
        target_weights=safe_weights,
        theoretical_prices=theo_prices,
        portfolio_value=portfolio_state["portfolio_value"],
        trade_types=trade_types,
    )

    # 12. Log positions for swing trades using actual fill prices
    for i, ticker in enumerate(UNIVERSE):
        if trade_types.get(ticker) == "CNC" and safe_weights[i] != 0:
            strategy = initial_state.get("current_strategy", "momentum")
            result = results.get(ticker)
            if result and result.filled_qty > 0:
                fill_price = result.fill_price if result.fill_price > 0 else theo_prices[ticker]
                pos = Position.default_cnc(ticker, fill_price, result.filled_qty, strategy)
                position_manager.open_position(pos)

    # 13. Metric computations
    shortfall = broker.calculate_shortfall(UNIVERSE, safe_weights, theo_prices, results)
    total_commission = sum(r.commission for r in results.values())
    total_slippage = sum(r.slippage_bps for r in results.values()) / max(len(results), 1)
    print(f"[Pipeline] Trade complete. Shortfall: {shortfall:.2f} bps, "
          f"Avg slippage: {total_slippage:.1f} bps, Commission: ${total_commission:.2f}")

    # 14. Log decision to database
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

    # 15. Log daily P&L
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

    print(f"[DB] Decision logged. model={model_version}  cb={cb_reason}")
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
        print("AegisQuant US is armed. Pipeline runs every 20 minutes from 09:35 to 15:50 ET (Mon-Fri).")
        from apscheduler.schedulers.blocking import BlockingScheduler

        scheduler = BlockingScheduler()

        # Run every 20 minutes during US market hours (9:35 AM – 3:40 PM ET)
        # Fires: :35, :55, :15 each hour from 9 to 14, then 15:35 as last run
        scheduler.add_job(
            main_us_live_loop,
            'cron',
            day_of_week='mon-fri',
            hour='9-14',
            minute='35,55,15',
            timezone='US/Eastern',
            max_instances=1,
            coalesce=True,
        )
        # Final fire at 15:35 — last decision before 4:00 PM close
        scheduler.add_job(
            main_us_live_loop,
            'cron',
            day_of_week='mon-fri',
            hour=15,
            minute=35,
            timezone='US/Eastern',
            max_instances=1,
            coalesce=True,
        )

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            pass
