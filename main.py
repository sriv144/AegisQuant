"""
Live Trading Loop
=================
Wraps the entire RL policy pipeline into a scheduler that runs autonomously.
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

from src import config  # noqa: F401  # Ensures .env is loaded before runtime setup.
from src.execution.alpaca_executor import AlpacaExecutor
from src.data.market_data import market_data
from src.data.feature_engineering import feature_engineer
from src.data.alternative_data import alt_data as alt_data_collector
from src.engine.circuit_breakers import ExecutionFailsafe
from src.agents.orchestrator import orchestrator
from src.agents.state import AgentState
from src.agents.portfolio.pm_agent import pm_agent
from src.db.models import db_manager

_failsafe = ExecutionFailsafe()

UNIVERSE = ["SPY", "QQQ", "TLT", "GLD"]

# Tracks all-time high portfolio equity across cycles to compute drawdown.
_peak_equity: list = [None]


def _fetch_live_vix() -> float:
    """Fetch the latest VIX close from yfinance. Returns 20.0 on any failure."""
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
        return float(vix)
    except Exception as e:
        print(f"[LiveState] VIX fetch failed ({e}), defaulting to 20.0")
        return 20.0


def _get_live_portfolio_state(executor: AlpacaExecutor, tickers: list) -> dict:
    """
    Returns a portfolio state dict compatible with AgentState['portfolio_state']
    and the circuit-breaker cb_state.  Falls back gracefully in mock mode.
    """
    vix = _fetch_live_vix()
    current_weights = np.zeros(len(tickers))
    drawdown = 0.0
    portfolio_value = 0.0

    if not executor.mock_mode:
        try:
            account = executor.client.get_account()
            portfolio_value = float(account.portfolio_value)

            # Update running peak to compute drawdown
            if _peak_equity[0] is None or portfolio_value > _peak_equity[0]:
                _peak_equity[0] = portfolio_value
            if _peak_equity[0] and _peak_equity[0] > 0:
                drawdown = max(0.0, (_peak_equity[0] - portfolio_value) / _peak_equity[0])

            # Per-ticker weights from live positions
            positions = {
                p.symbol: float(p.market_value)
                for p in executor.client.get_all_positions()
            }
            for i, ticker in enumerate(tickers):
                current_weights[i] = positions.get(ticker, 0.0) / portfolio_value if portfolio_value > 0 else 0.0

        except Exception as e:
            print(f"[LiveState] Alpaca portfolio fetch failed ({e}), using safe defaults.")
    else:
        print(f"[LiveState] Mock mode — skipping Alpaca account query.")

    print(f"[LiveState] drawdown={drawdown:.4f}  vix={vix:.2f}  portfolio_value={portfolio_value:.0f}")

    return {
        "current_drawdown": drawdown,
        "vix_raw": vix,
        "current_weights": current_weights.tolist(),
        "portfolio_value": portfolio_value,
    }


def main_live_loop():
    print(f"\n[{datetime.now()}] Waking up. Initiating daily RL execution cycle...")

    # 1. Alpaca Executor Check
    executor = AlpacaExecutor(tickers=UNIVERSE, paper=True)

    # 2. Fetch live OHLCV quotes
    print("[Pipeline] Fetching live OHLCV, VIX, and Yield Curves...")
    theo_prices = {}
    for tick in UNIVERSE:
        theo_prices[tick] = market_data.get_latest_quote(tick)

    # 3. Build live portfolio state (Gap 2: real drawdown + VIX + current weights)
    portfolio_state = _get_live_portfolio_state(executor, UNIVERSE)

    # 4. Pre-compute technical indicators + sentiment (feeds all research agents)
    print("[Pipeline] Pre-computing technical indicators and sentiment signals...")
    ticker_indicators = {}
    ticker_alt_data = {}
    hist_start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    hist_end = datetime.now().strftime("%Y-%m-%d")

    for ticker in UNIVERSE:
        hist = market_data.get_historical_data(ticker, start_date=hist_start, end_date=hist_end)
        if hist and len(hist) >= 20:
            df_feat = feature_engineer.compute_technical_indicators(hist)
            latest = df_feat.iloc[-1]
            ticker_indicators[ticker] = {k: float(v) for k, v in latest.items() if pd.notna(v) and isinstance(v, (int, float, np.number))}
        else:
            ticker_indicators[ticker] = {}

        news = alt_data_collector.get_recent_news(ticker)
        agg = feature_engineer.aggregate_sentiment(news)
        # Sentiment agent reads key "sentiment"; we expose both for completeness
        ticker_alt_data[ticker] = {
            "sentiment": agg.get("sentiment_score", 0.0),
            "sentiment_score": agg.get("sentiment_score", 0.0),
            "news_volume": agg.get("news_volume", 0),
        }

    # 5. Run the full LangGraph agent pipeline per ticker (Gap 1 + Gap 3)
    #    orchestrator chains: Research → Committee → PMAgent (PPO) → Risk → Execution
    print("[Pipeline] Running LLM consensus + PPO inference via agent orchestrator...")
    target_weights = np.zeros(len(UNIVERSE))
    model_version = "orchestrator_fallback"

    for i, ticker in enumerate(UNIVERSE):
        initial_state: AgentState = {
            "current_asset": ticker,
            "timestamp": datetime.now().isoformat(),
            "market_data": {"ticker": ticker, "price": theo_prices.get(ticker, 0.0)},
            "alternative_data": ticker_alt_data[ticker],
            "technical_indicators": ticker_indicators[ticker],
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
        model_version = "ppo_rl_live"

    # Normalize to gross exposure constraint (1.5)
    gross = np.sum(np.abs(target_weights))
    if gross > 1.5:
        target_weights = target_weights * (1.5 / gross)

    print(f"[Pipeline] Raw RL Weights -> {target_weights.round(3)}")

    # 6. Run circuit breakers with live state (Gap 2)
    cb_state = {
        "drawdown": portfolio_state["current_drawdown"],
        "vix_raw": portfolio_state["vix_raw"],
        "current_weights": np.array(portfolio_state["current_weights"]),
    }
    safe_weights, cb_reason = _failsafe.process_action(target_weights, cb_state)
    if cb_reason != "OK":
        print(f"[CircuitBreaker] TRIGGERED: {cb_reason}. Weights adjusted.")
    print(f"[Pipeline] Safe Weights -> {safe_weights.round(3)}")

    # 7. Fire to execution
    fills = executor.execute_target_weights(safe_weights, theo_prices)

    # 8. Metric computations
    shortfall = executor.calculate_shortfall(safe_weights, theo_prices, fills)
    print(f"[Pipeline] Trade complete. Estimated Slippage: {shortfall:.2f} bps.")

    # 9. Log decision to database (Gap 4)
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
        main_live_loop()
    else:
        print("Starting APScheduler Daemon...")
        print("AegisQuant is armed. Trades will automatically trigger at 09:35 AM ET M-F.")
        scheduler = BlockingScheduler()

        # Fire daily at 09:35 AM New York time
        scheduler.add_job(
            main_live_loop,
            'cron',
            day_of_week='mon-fri',
            hour=9,
            minute=35,
            timezone='America/New_York'
        )

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            pass
