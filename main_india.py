"""
India Live Trading Loop (Professional Multi-Mode Trader)
========================================================
Runs the dynamic multi-strategy RL pipeline on NSE/BSE with Angel One SmartAPI.
- Dynamic universe screening (all ~2000+ NSE stocks)
- Dual-mode trading: 80% delivery (CNC, 1-3 month) + 20% intraday (MIS, same-day)
- Position management with stop-loss, take-profit, and aging exits
- RL meta-learner for optimal intraday/delivery split
Scheduled for 9:15 AM IST (03:45 UTC) on weekdays.
"""

from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

from src import config  # noqa: F401  # Ensures .env is loaded
from src.execution.groww_executor import GrowwExecutor
from src.data.india_market_data import india_market_data
from src.data.feature_engineering import feature_engineer
from src.data.alternative_data import alt_data as alt_data_collector
from src.data.universe_screener import universe_screener
from src.engine.circuit_breakers import ExecutionFailsafe
from src.engine.position_manager import position_manager, Position
from src.engine.capital_allocator import capital_allocator
from src.agents.orchestrator import orchestrator
from src.agents.state import AgentState
from src.agents.portfolio.pm_agent import pm_agent
from src.db.models import db_manager

_failsafe = ExecutionFailsafe()

INITIAL_CAPITAL = 250_000.0  # ₹2.5 lakh paper capital — the single source of truth


def _fetch_india_vix() -> float:
    """Fetch latest India VIX from yfinance. Returns 20.0 on any failure."""
    return india_market_data.get_india_vix()


def _get_live_portfolio_state(executor: GrowwExecutor, tickers: list, current_prices: dict) -> dict:
    """
    Compute real portfolio state from DB-tracked positions and realized P&L.
    No more hardcoded values — portfolio_value changes as trades win/lose.
    """
    vix = _fetch_india_vix()
    current_weights = np.zeros(len(tickers))

    pf = db_manager.compute_portfolio_value(INITIAL_CAPITAL, current_prices)
    portfolio_value = pf["portfolio_value"]
    drawdown = pf["current_drawdown"]

    mode_label = "paper" if executor.mock_mode else "live"
    print(
        f"[LiveState] {mode_label} mode — portfolio_value=₹{portfolio_value:,.0f}  "
        f"realized=₹{pf['realized_pnl']:,.0f}  unrealized=₹{pf['unrealized_pnl']:,.0f}  "
        f"cash=₹{pf['cash_balance']:,.0f}  positions={pf['open_position_count']}"
    )
    print(f"[LiveState] drawdown={drawdown:.4f}  india_vix={vix:.2f}  peak=₹{pf['peak_equity']:,.0f}")

    return {
        "current_drawdown": drawdown,
        "vix_raw": vix,
        "current_weights": current_weights.tolist(),
        "portfolio_value": portfolio_value,
    }


def _fetch_weekly_strategy_scores() -> dict:
    """
    Compute per-strategy performance scores from realized trades in the last 30 days.
    Returns {strategy_name: score} where score is avg P&L % weighted by trade count.
    Strategies with no recent trades get a neutral score of 0.0.
    """
    try:
        from sqlalchemy import text
        import os

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
    """Fallback scores when no trade history exists — equal weight across core strategies."""
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


def main_india_live_loop():
    print(f"\n[{datetime.now()}] [India Pipeline] Waking up. Initiating daily RL execution cycle...")

    # 1. Refresh universe (max once per 7 days)
    print("[Pipeline] Screening universe...")
    UNIVERSE = universe_screener.screen_universe()
    print(f"[Pipeline] Selected {len(UNIVERSE)} tickers from dynamic screening")

    # 2. Groww Executor
    executor = GrowwExecutor(tickers=UNIVERSE, paper=True)

    # 3. Position manager: close any SL/TP/aged positions FIRST
    print("[Pipeline] Checking position exits (SL/TP/aging)...")
    theo_prices = {}
    for tick in UNIVERSE:
        theo_prices[tick] = india_market_data.get_latest_quote(tick)

    exits = position_manager.daily_check(theo_prices)
    for ticker in exits:
        position_manager.close_position(ticker, theo_prices[ticker], reason="EXIT_SIGNAL")
        print(f"[PositionManager] Closed {ticker}")

    # 4. Build live portfolio state (uses DB-tracked P&L, not hardcoded)
    portfolio_state = _get_live_portfolio_state(executor, UNIVERSE, theo_prices)

    # 5. Capital allocator: compute intraday vs delivery budgets
    intraday_budget, delivery_budget = capital_allocator.get_budgets(portfolio_state)
    print(f"[CapitalAllocator] Budgets: {intraday_budget:.0f} intraday, {delivery_budget:.0f} delivery")

    # 6. Pre-compute technical indicators + sentiment
    print("[Pipeline] Pre-computing technical indicators and sentiment signals...")
    ticker_indicators = {}
    ticker_alt_data = {}
    hist_start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    hist_end = datetime.now().strftime("%Y-%m-%d")

    for ticker in UNIVERSE:
        hist = india_market_data.get_historical_data(ticker, start_date=hist_start, end_date=hist_end)
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
    model_version = "india_orchestrator_fallback"

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
        # asset_allocation_agent writes adjusted_exposure_pct; PM writes target_exposure_pct
        exposure = float(allocation.get("adjusted_exposure_pct") or allocation.get("target_exposure_pct") or 0.0)
        # Use committee direction (set by LLM) — rl_direction from PM can be unreliable
        # if the RL model is degenerate (always outputs the same sign).
        committee_dir = final_state.get("committee_decision", {}).get("direction", "LONG")
        target_weights[i] = exposure if committee_dir != "SHORT" else -exposure

        # Extract trade_type from execution result
        trade_type = final_state.get("trade_type", "SKIP")
        trade_types[ticker] = trade_type

        # Capture trade reasoning from every agent in the pipeline
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
        model_version = "india_ppo_rl_live"

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

    # 11. Fire to execution
    fills = executor.execute_target_weights(safe_weights, theo_prices)

    # 12. Log positions for CNC trades
    for i, ticker in enumerate(UNIVERSE):
        if trade_types.get(ticker) == "CNC" and safe_weights[i] != 0:
            strategy = initial_state.get("current_strategy", "momentum")
            qty = int(safe_weights[i] * portfolio_state["portfolio_value"] / theo_prices[ticker])
            pos = Position.default_cnc(ticker, theo_prices[ticker], qty, strategy)
            position_manager.open_position(pos)

    # 13. Metric computations
    shortfall = executor.calculate_shortfall(safe_weights, theo_prices, fills)
    print(f"[Pipeline] Trade complete. Estimated Slippage: {shortfall:.2f} bps.")

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

    # 15. Log daily P&L (uses real tracked portfolio value, not hardcoded)
    try:
        daily_pnl = position_manager.get_daily_pnl()
        from src.db.models import DailyPnL
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        import os

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
        main_india_live_loop()
    else:
        print("Starting APScheduler Daemon (India)...")
        print("AegisQuant India is armed. Pipeline runs every 20 minutes from 09:15 to 15:25 IST (Mon-Fri).")
        from apscheduler.schedulers.blocking import BlockingScheduler

        scheduler = BlockingScheduler()

        # Run every 20 minutes during NSE market hours (09:15 – 15:05 IST)
        # Fires: :15, :35, :55 each hour from 9 to 14, then 15:05 as last run before close
        scheduler.add_job(
            main_india_live_loop,
            'cron',
            day_of_week='mon-fri',
            hour='9-14',
            minute='15,35,55',
            timezone='Asia/Kolkata',
            max_instances=1,        # prevent overlap if a run takes >20 min
            coalesce=True,          # skip missed fires instead of stacking them
        )
        # Final fire at 15:05 — last decision before 15:25 market close auction
        scheduler.add_job(
            main_india_live_loop,
            'cron',
            day_of_week='mon-fri',
            hour=15,
            minute=5,
            timezone='Asia/Kolkata',
            max_instances=1,
            coalesce=True,
        )

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            pass
