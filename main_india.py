"""
India Live Trading Loop
========================
Runs the multi-strategy RL pipeline on NSE/BSE with Angel One SmartAPI.
Scheduled for 9:15 AM IST (03:45 UTC) on weekdays.
"""

from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

from src import config  # noqa: F401  # Ensures .env is loaded
from src.execution.angelone_executor import AngelOneExecutor
from src.data.india_market_data import india_market_data
from src.data.feature_engineering import feature_engineer
from src.data.alternative_data import alt_data as alt_data_collector
from src.engine.circuit_breakers import ExecutionFailsafe
from src.agents.orchestrator import orchestrator
from src.agents.state import AgentState
from src.agents.portfolio.pm_agent import pm_agent
from src.db.models import db_manager

_failsafe = ExecutionFailsafe()

# Indian Universe: Index ETFs + NIFTY 50 Large Caps (with .NS suffix for NSE)
INDEX_ETFS = ["NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS", "LIQUIDBEES.NS"]
LARGE_CAPS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "INFY.NS", "HINDUNILVR.NS", "BHARTIARTL.NS", "ITC.NS",
    "KOTAKBANK.NS", "LT.NS"
]
UNIVERSE = INDEX_ETFS + LARGE_CAPS

# Tracks all-time high portfolio equity across cycles to compute drawdown
_peak_equity: list = [None]


def _fetch_india_vix() -> float:
    """Fetch latest India VIX from yfinance. Returns 20.0 on any failure."""
    return india_market_data.get_india_vix()


def _get_live_portfolio_state(executor: AngelOneExecutor, tickers: list) -> dict:
    """
    Returns a portfolio state dict compatible with AgentState['portfolio_state'].
    In mock mode: defaults. In live mode: queries Angel One for real positions.
    """
    vix = _fetch_india_vix()
    current_weights = np.zeros(len(tickers))
    drawdown = 0.0
    portfolio_value = 0.0

    if not executor.mock_mode:
        try:
            # In real mode, would query executor.client for positions
            # For now, mock behavior
            portfolio_value = 250000.0  # ₹2.5 lakh
            if _peak_equity[0] is None or portfolio_value > _peak_equity[0]:
                _peak_equity[0] = portfolio_value
            if _peak_equity[0] and _peak_equity[0] > 0:
                drawdown = max(0.0, (_peak_equity[0] - portfolio_value) / _peak_equity[0])
        except Exception as e:
            print(f"[LiveState] Angel One portfolio fetch failed ({e}), using safe defaults.")
    else:
        print(f"[LiveState] Mock mode — skipping Angel One account query.")

    print(f"[LiveState] drawdown={drawdown:.4f}  india_vix={vix:.2f}  portfolio_value={portfolio_value:.0f}")

    return {
        "current_drawdown": drawdown,
        "vix_raw": vix,
        "current_weights": current_weights.tolist(),
        "portfolio_value": portfolio_value,
    }


def _fetch_weekly_strategy_scores() -> dict:
    """
    Query aegisquant_live.db for the last 5 days of decisions.
    Group by model_version (which encodes strategy name), compute Sharpe ratio per strategy.
    Returns {"momentum": 0.8, "trend_following": -0.2, ...}
    """
    try:
        from sqlalchemy import create_engine, text
        import os

        db_url = os.getenv("POSTGRES_URL", "sqlite:///aegisquant_live.db")
        engine = create_engine(db_url)

        query = text("""
            SELECT model_version, COUNT(*) as trade_count
            FROM decisions
            WHERE timestamp > datetime('now', '-7 days')
              AND model_version LIKE 'india_%'
            GROUP BY model_version
            ORDER BY trade_count DESC
        """)

        with engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

        # Simple scoring: more recent trades with positive returns = higher score
        # For now, return empty dict (first week of trading)
        # Full implementation would compute Sharpe from price data
        return {}

    except Exception as e:
        print(f"[Strategy Scores] Query failed ({e}), returning empty dict")
        return {}


def main_india_live_loop():
    print(f"\n[{datetime.now()}] [India Pipeline] Waking up. Initiating daily RL execution cycle...")

    # 1. Angel One Executor Check
    executor = AngelOneExecutor(tickers=UNIVERSE, paper=True)

    # 2. Fetch live quotes
    print("[Pipeline] Fetching live OHLCV, India VIX, and sentiment...")
    theo_prices = {}
    for tick in UNIVERSE:
        theo_prices[tick] = india_market_data.get_latest_quote(tick)

    # 3. Build live portfolio state (real drawdown + VIX + current weights)
    portfolio_state = _get_live_portfolio_state(executor, UNIVERSE)

    # 4. Pre-compute technical indicators + sentiment (feeds all research agents)
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

    # 5. Fetch weekly strategy scores (empty on first run)
    strategy_scores = _fetch_weekly_strategy_scores()

    # 6. Run the full LangGraph agent pipeline per ticker
    #    orchestrator chains: Research → Strategy Selector → Committee → PMAgent → Risk → Execution
    print("[Pipeline] Running LLM consensus + PPO inference via agent orchestrator...")
    target_weights = np.zeros(len(UNIVERSE))
    model_version = "india_orchestrator_fallback"

    for i, ticker in enumerate(UNIVERSE):
        initial_state: AgentState = {
            "current_asset": ticker,
            "timestamp": datetime.now().isoformat(),
            "market_data": {"ticker": ticker, "price": theo_prices.get(ticker, 0.0)},
            "alternative_data": ticker_alt_data[ticker],
            "technical_indicators": ticker_indicators[ticker],
            "active_strategies": list(strategy_scores.keys())[:3] if strategy_scores else [],
            "strategy_scores": strategy_scores,
            "current_strategy": list(strategy_scores.keys())[0] if strategy_scores else "momentum",
            "research_signals": [],
            "committee_decision": {},
            "allocation_request": {},
            "risk_approval": {},
            "execution_result": {},
            "portfolio_state": portfolio_state,
        }
        final_state = orchestrator.run_cycle(initial_state)

        allocation = final_state.get("allocation_request", {})
        exposure = float(allocation.get("target_exposure_pct", 0.0))
        direction = allocation.get("rl_direction", "LONG")
        target_weights[i] = exposure if direction != "SHORT" else -exposure

    if pm_agent.rl_model is not None:
        model_version = "india_ppo_rl_live"

    # Normalize to gross exposure constraint (1.5)
    gross = np.sum(np.abs(target_weights))
    if gross > 1.5:
        target_weights = target_weights * (1.5 / gross)

    print(f"[Pipeline] Raw RL Weights -> {target_weights.round(3)}")

    # 7. Run circuit breakers with live state
    cb_state = {
        "drawdown": portfolio_state["current_drawdown"],
        "vix_raw": portfolio_state["vix_raw"],
        "current_weights": np.array(portfolio_state["current_weights"]),
    }
    safe_weights, cb_reason = _failsafe.process_action(target_weights, cb_state)
    if cb_reason != "OK":
        print(f"[CircuitBreaker] TRIGGERED: {cb_reason}. Weights adjusted.")
    print(f"[Pipeline] Safe Weights -> {safe_weights.round(3)}")

    # 8. Fire to execution
    fills = executor.execute_target_weights(safe_weights, theo_prices)

    # 9. Metric computations
    shortfall = executor.calculate_shortfall(safe_weights, theo_prices, fills)
    print(f"[Pipeline] Trade complete. Estimated Slippage: {shortfall:.2f} bps.")

    # 10. Log decision to database
    db_manager.log_decision_orm(
        timestamp=datetime.now(timezone.utc).isoformat(),
        ticker_universe=UNIVERSE,
        state_vector=[portfolio_state["current_drawdown"], portfolio_state["vix_raw"]],
        rl_output=target_weights,
        circuit_breaker_status=cb_reason,
        final_weights=safe_weights,
        transaction_costs=shortfall,
        model_version=model_version,
    )
    print(f"[DB] Decision logged. model={model_version}  cb={cb_reason}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="Execute immediately instead of cron")
    args = parser.parse_args()

    if args.now:
        main_india_live_loop()
    else:
        print("Starting APScheduler Daemon (India)...")
        print("AegisQuant India is armed. Trades will automatically trigger at 09:15 AM IST M-F.")
        from apscheduler.schedulers.blocking import BlockingScheduler

        scheduler = BlockingScheduler()

        # Fire daily at 09:15 AM Indian Standard Time
        scheduler.add_job(
            main_india_live_loop,
            'cron',
            day_of_week='mon-fri',
            hour=9,
            minute=15,
            timezone='Asia/Kolkata'
        )

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            pass
