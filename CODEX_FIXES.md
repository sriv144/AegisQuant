# AegisQuant — Codex Fix Specification

> **Purpose:** This document is a complete, self-contained specification for fixing AegisQuant's critical problems.  
> **Order matters.** Fixes are numbered by dependency — Fix 1 must land before Fix 2, etc.  
> **Rule:** Do NOT delete existing files. Modify in place. Preserve all existing tests.  
> **Rule:** After each fix, run `pytest tests/` and ensure all existing tests still pass.  
> **Rule:** Add new tests for every fix as specified in each section's "Verification" block.

---

## Fix 1: Replace Random-Noise RL Environment With Real Market Data

### Problem
`src/engine/rl_env.py` generates observations via `np_random.uniform(-1, 1)` (lines 58-62). The PPO models trained on this learn nothing about markets. The market return in `step()` is also synthetic random noise (line 82: `actual_return = (consensus * 0.05) + noise`).

A proper environment already exists: `src/backtest/historical_env.py` (14D state vector, real OHLCV replay, regime-adaptive reward). But the PM agent (`src/agents/portfolio/pm_agent.py`) loads models that were trained against the broken 6D random env.

### What To Do

**Step 1a: Create a training script** at `src/models/train_india.py`:
```python
"""
Train PPO on Indian market data using HistoricalHedgeFundEnv.
Trains on NIFTY50 constituents, saves to model_registry via ModelRegistry.
"""
```
- Fetch 3+ years of OHLCV for a basket of 10-15 NSE large-caps (e.g. RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS, BHARTIARTL.NS, ITC.NS, SBIN.NS, LT.NS, HCLTECH.NS) using yfinance
- Create a `HistoricalHedgeFundEnv` (from `src/backtest/historical_env.py`) for each ticker
- Train PPO from `stable_baselines3` with `total_timesteps=50_000` per ticker
- Use the existing `RegimeDetector` from `src/engine/regime_detector.py` (fit it on the training data)
- After training, evaluate out-of-sample on the last 6 months of data
- Register the best model via `ModelRegistry.register_model()` from `src/models/registry.py`
- Promote it to production via `ModelRegistry.promote_model(model_id, "production")`
- The script should be runnable as: `python -m src.models.train_india`

**Step 1b: Update PM Agent to use 14D observation space:**

File: `src/agents/portfolio/pm_agent.py`

Currently the PM agent constructs a 6D state (line 70-100) and loads models with a 6D observation space (line 42). Update:

1. Change `observation_space` in `_load_rl_model()` (line 42) from `shape=(6,)` to `shape=(14,)` and `low=-2.0, high=2.0` to match `HistoricalHedgeFundEnv.OBS_DIM = 14`
2. Rewrite `_extract_rl_state()` to construct a 14D vector matching `HistoricalHedgeFundEnv._get_obs()`:
   - `[0]` Volatility_20_Z — get from `state["technical_indicators"].get("Volatility_20_Z", 0.0)`
   - `[1]` RSI_14_Z — from `state["technical_indicators"].get("RSI_14_Z", 0.0)`
   - `[2]` MACD_Z — from `state["technical_indicators"].get("MACD_Z", 0.0)`
   - `[3]` BB_Position_Z — from `state["technical_indicators"].get("BB_Position_Z", 0.0)`
   - `[4]` mom_12m_Z — from `state["technical_indicators"].get("mom_12m_Z", 0.0)`
   - `[5]` current_weight — 0.0 (no position tracking across tickers yet)
   - `[6]` drawdown — from `state["portfolio_state"].get("current_drawdown", 0.0)`
   - `[7-10]` regime one-hot — default `[1,0,0,0]` (Bull Quiet) or derive from VIX: VIX<15 → [1,0,0,0], VIX 15-25 → [0,1,0,0], VIX 25-35 → [0,0,1,0], VIX>35 → [0,0,0,1]
   - `[11]` portfolio_return_5d — 0.0 (not available in single-pass)
   - `[12]` vix_z — from `state["portfolio_state"].get("vix_raw", 20.0)`, normalize as `(vix - 20) / 10` clipped to [-2, 2]
   - `[13]` yield_curve_slope — 0.0 (not available in live state currently)
   - Clip all values to [-2.0, 2.0] and return as `np.float32`

**Step 1c: Keep the old `rl_env.py` but add a deprecation docstring** at the top:
```python
# DEPRECATED: This environment trains on random noise. 
# Use src/backtest/historical_env.py for real training.
```

### Files To Modify
- `src/agents/portfolio/pm_agent.py` — update `_extract_rl_state()` and `_load_rl_model()`
- `src/engine/rl_env.py` — add deprecation notice only

### Files To Create
- `src/models/train_india.py` — new training script

### Verification
- Create `tests/test_train_india.py` that:
  1. Imports `HistoricalHedgeFundEnv`, creates it with a small synthetic DataFrame (50 rows of OHLCV)
  2. Verifies `env.observation_space.shape == (14,)` and `env.action_space.shape == (1,)`
  3. Runs `env.reset()` and `env.step(env.action_space.sample())` without error
  4. Verifies returned observation has shape (14,)
- Add a test to `tests/test_pm_agent.py` (new file) that:
  1. Creates a mock AgentState with `technical_indicators` containing RSI_14_Z, MACD_Z, etc.
  2. Calls `pm_agent._extract_rl_state(state)` 
  3. Asserts returned array has shape (14,) and all values are within [-2.0, 2.0]

---

## Fix 2: Close the Portfolio Feedback Loop

### Problem
`main_india.py` line 52 hardcodes `portfolio_value = 250000.0`. It never changes. Fills from paper trading vanish. Drawdown is always 0. The system has no memory of its own P&L.

### What To Do

**Step 2a: Create a persistent portfolio state manager:**

Create `src/engine/portfolio_state.py`:
```python
"""
Persistent portfolio state across trading cycles.
Reads/writes to the daily_pnl and open_positions tables.
Computes real portfolio value from: cash + sum(position_value for all open positions).
"""
```

This class should:
1. On initialization, read the latest `daily_pnl` row from the database (using the existing SQLAlchemy models from `src/db/models.py`)
2. If no row exists (first run), start with `INITIAL_CAPITAL = 250_000.0` (from env var `AEGIS_INITIAL_CAPITAL` or default)
3. Compute current portfolio value as:
   ```
   portfolio_value = cash_remaining + sum(quantity * current_price for each OPEN position in open_positions table)
   ```
4. Track `peak_equity` across all runs (store in DB or compute from max of `daily_pnl.total_portfolio_value`)
5. Compute real `drawdown = (peak - current) / peak`
6. Provide method `get_portfolio_state(current_prices: dict) -> dict` returning:
   ```python
   {
       "portfolio_value": float,
       "cash_remaining": float,
       "current_drawdown": float,
       "peak_equity": float,
       "current_weights": list,  # weight of each ticker in portfolio
       "vix_raw": float,
   }
   ```
7. Provide method `update_after_fills(fills: dict, prices: dict)` that:
   - Calculates new cash: `cash -= sum(fill_quantity * fill_price)` for buys, `cash += ...` for sells
   - Writes updated `daily_pnl` row

**Step 2b: Update `main_india.py` to use `PortfolioState`:**

Replace the `_get_live_portfolio_state()` function (lines 41-70) to use the new `PortfolioState` class instead of hardcoding 250000.

Specifically:
1. Import `PortfolioState` from `src.engine.portfolio_state`
2. At the start of `main_india_live_loop()`, create/get the portfolio state by querying current prices for all open positions
3. After fills (line 233: `fills = executor.execute_target_weights(...)`), call `portfolio_state.update_after_fills(fills, theo_prices)` 
4. In the daily_pnl logging section (lines 260-284), use real computed values instead of the hardcoded ones

### Files To Modify
- `main_india.py` — replace `_get_live_portfolio_state()`, update the fill and P&L logging sections

### Files To Create
- `src/engine/portfolio_state.py`

### Verification
- Create `tests/test_portfolio_state.py` that:
  1. Creates an in-memory SQLite database
  2. Inserts 3 open positions with known entry prices
  3. Calls `get_portfolio_state()` with current prices = entry_price * 1.05 (5% gain)
  4. Asserts `portfolio_value > initial_capital`
  5. Asserts `current_drawdown == 0` (since portfolio is at all-time-high)
  6. Then simulates a loss: current_prices = entry_price * 0.90
  7. Asserts `current_drawdown > 0`
  8. Asserts `portfolio_value < initial_capital`

---

## Fix 3: Wire the 9 Trading Strategies Into the Live Pipeline

### Problem
9 strategy classes exist in `src/strategies/` with `generate_signal()` methods but are never called. The orchestrator (`src/agents/orchestrator.py`) only uses the 4 research agents. The strategy_selector_agent selects strategy names but doesn't invoke the strategy classes.

### What To Do

**Step 3a: Create a strategy runner agent** at `src/agents/research/strategy_runner_agent.py`:
```python
"""
Invokes the appropriate trading strategy's generate_signal() based on
the active_strategies selected by strategy_selector_agent.
Returns strategy signals in the same format as other research agents.
"""
```

This agent should:
1. Import all 9 strategy classes:
   ```python
   from src.strategies.momentum import MomentumStrategy
   from src.strategies.mean_reversion import MeanReversionStrategy
   from src.strategies.trend_following import TrendFollowingStrategy
   from src.strategies.factor_investing import FactorInvestingStrategy
   from src.strategies.pairs_trading import PairsTradingStrategy
   from src.strategies.gap_fill import GapFillStrategy
   from src.strategies.volatility_breakout import VolatilityBreakoutStrategy
   from src.strategies.earnings_momentum import EarningsMomentumStrategy
   from src.strategies.sector_rotation import SectorRotationStrategy
   ```
2. Maintain a `STRATEGY_MAP = {"momentum": MomentumStrategy(), "mean_reversion": MeanReversionStrategy(), ...}`
3. In its `invoke(state)` method:
   - Read `state["active_strategies"]` (list of strategy names, set by strategy_selector_agent)
   - Read `state["current_strategy"]` (primary strategy name)
   - For each active strategy, call `strategy.generate_signal(ticker, indicators, portfolio_state, alt_data)`
   - Convert each strategy's output to the standard research signal format:
     ```python
     {
         "agent_name": f"Strategy_{strategy.name}",
         "action": signal["action"].replace("LONG", "PROPOSE_LONG").replace("SHORT", "PROPOSE_SHORT"),
         "confidence": signal["confidence"],
         "rationale": signal["rationale"],
     }
     ```
   - Return `{"research_signals": [list of converted signals]}`

**Step 3b: Add strategy_runner to the orchestrator:**

File: `src/agents/orchestrator.py`

1. Import the new agent: `from .research.strategy_runner_agent import strategy_runner_agent`
2. In `_research_node()` (line 59), add after the existing 4 agent invocations:
   ```python
   strat_res = strategy_runner_agent.invoke(state)
   ```
3. Append its signals to the accumulated signals list:
   ```python
   signals = q_res['research_signals'] + f_res['research_signals'] + m_res['research_signals'] + s_res['research_signals'] + strat_res['research_signals']
   ```

**Step 3c: Update the committee agent** to handle variable number of voters.

File: `src/agents/executive/strategy_committee_agent.py`

Currently the committee expects exactly 4 research signals. After this fix, it will receive 4 base signals + N strategy signals (where N = number of active strategies, typically 2-3). 

Review the committee voting logic — it should already work with variable-length `research_signals` since it iterates and counts votes. Verify this is the case. If it hardcodes "4 voters" anywhere, fix it to use `len(signals)`.

### Files To Modify
- `src/agents/orchestrator.py` — add strategy_runner to the research node
- `src/agents/executive/strategy_committee_agent.py` — verify works with N signals (no hardcoded count)

### Files To Create
- `src/agents/research/strategy_runner_agent.py`

### Verification
- Create `tests/test_strategy_runner.py` that:
  1. Creates a mock state with `active_strategies: ["momentum", "mean_reversion"]`, `current_strategy: "momentum"`, and realistic `technical_indicators`
  2. Calls `strategy_runner_agent.invoke(state)`
  3. Asserts returned dict has `"research_signals"` key
  4. Asserts each signal has `agent_name`, `action`, `confidence`, `rationale`
  5. Asserts `action` is one of `"PROPOSE_LONG"`, `"PROPOSE_SHORT"`, `"HOLD"`

---

## Fix 4: Add Nifty50 Benchmark Tracking to the Live Pipeline

### Problem
The project's thesis is "beats Nifty50." There is no live benchmark tracking. The DB already has `BenchmarkDaily` and `PerformanceDaily` tables (in `src/db/models.py`, lines 89-133) but they are never populated by the live pipeline.

### What To Do

**Step 4a: Create a benchmark tracker** at `src/engine/benchmark_tracker.py`:
```python
"""
Tracks NIFTYBEES.NS (Nifty50 ETF) daily returns alongside AegisQuant portfolio.
Populates benchmark_daily and performance_daily tables.
"""
```

This class should:
1. Fetch NIFTYBEES.NS daily close from yfinance (1d period, or last available close)
2. Compute daily return: `(today_close - yesterday_close) / yesterday_close`
3. Compute cumulative return from inception (first `daily_pnl` entry date)
4. Save to `BenchmarkDaily` table (from `src/db/models.py` line 89)
5. Compare against AegisQuant's daily return from `DailyPnL` table
6. Compute:
   - `excess_return = aegis_daily_return - benchmark_daily_return`
   - `cumulative_aegis_return` and `cumulative_benchmark_return`
   - `rolling_sharpe_7` and `rolling_sharpe_30` for both (annualized: `mean / std * sqrt(252)`)
   - `hit_rate_30`: fraction of last 30 days where AegisQuant beat the benchmark
   - `max_drawdown` for both
   - `verdict`: "OUTPERFORMING" if 30-day rolling Sharpe > benchmark Sharpe AND excess_return > 0, else "UNDERPERFORMING", or "INSUFFICIENT_DATA" if < 30 days
7. Save to `PerformanceDaily` table (from `src/db/models.py` line 106)
8. Provide method `get_latest() -> dict` and `get_history(days=90) -> list[dict]`

**Step 4b: Call the benchmark tracker from `main_india.py`:**

After the daily_pnl logging section (around line 284), add:
```python
from src.engine.benchmark_tracker import benchmark_tracker
benchmark_tracker.update_daily(
    portfolio_value=portfolio_state["portfolio_value"],
    date=datetime.now().strftime("%Y-%m-%d"),
)
```

**Step 4c: Add API endpoints to the FastAPI server:**

File: `src/webapp/server.py`

Add two endpoints:
1. `GET /api/performance` — returns latest PerformanceDaily row + last 90 days of history
2. `GET /api/benchmark` — returns last 90 days of BenchmarkDaily rows

NOTE: The frontend `app.js` already fetches `/api/performance` (line 53) and renders a benchmark chart (lines 117-142). The endpoints just need to exist and return the right data.

### Files To Modify
- `main_india.py` — add benchmark_tracker call after daily P&L logging
- `src/webapp/server.py` — add `/api/performance` and `/api/benchmark` endpoints

### Files To Create
- `src/engine/benchmark_tracker.py`

### Verification
- Create `tests/test_benchmark_tracker.py` that:
  1. Creates an in-memory DB with 10 days of `DailyPnL` entries (known values)
  2. Mocks yfinance to return known NIFTYBEES prices
  3. Calls `benchmark_tracker.update_daily()` for each day
  4. Asserts `BenchmarkDaily` table has 10 rows
  5. Asserts `PerformanceDaily` table has 10 rows
  6. Verifies `excess_return = aegis_return - benchmark_return` for each day
  7. Verifies verdict is "INSUFFICIENT_DATA" for first 29 days

---

## Fix 5: Log and Display Agent Reasoning Per Trade

### Problem
The user wants to see WHY each trade was made. The DB already has an `AgentReasoning` table (`src/db/models.py` line 153) but the live pipeline doesn't populate it. The frontend `app.js` already has `openDecisionDetail()` (line 290) that fetches `/api/decision-detail/{run_id}` and renders agent reasoning rows — but this endpoint doesn't exist.

### What To Do

**Step 5a: Generate a `run_id` per cycle and log agent reasoning:**

File: `main_india.py`

1. At the top of `main_india_live_loop()`, generate a unique run_id:
   ```python
   import uuid
   run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M')}_{uuid.uuid4().hex[:6]}"
   ```

2. Pass `run_id` to `db_manager.log_decision_orm()` (currently called at line 248):
   ```python
   db_manager.log_decision_orm(
       run_id=run_id,
       ...existing params...
   )
   ```

3. After each ticker's orchestrator run (inside the for loop at line 169), log each research signal's reasoning to the `AgentReasoning` table:
   ```python
   from src.db.models import AgentReasoning
   
   for signal in final_state.get("research_signals", []):
       reasoning = AgentReasoning(
           run_id=run_id,
           ticker=ticker,
           agent_name=signal.get("agent_name", "unknown"),
           action=signal.get("action", "UNKNOWN"),
           confidence=float(signal.get("confidence", 0.0)),
           rationale=signal.get("rationale", ""),
       )
       # save to DB (use a session)
   ```
   Also log the committee decision and allocation decision as separate reasoning rows.

**Step 5b: Add the decision-detail API endpoint:**

File: `src/webapp/server.py`

Add endpoint `GET /api/decision-detail/{run_id}` that:
1. Queries `AgentReasoning` table filtered by `run_id` — returns all reasoning rows
2. Queries `MarketObservation` table filtered by `run_id` — returns observation context
3. Queries `PaperOrder` and `PaperFill` tables filtered by `run_id`
4. Queries `RLModelEvaluation` table filtered by `run_id`
5. Returns JSON matching the structure expected by `app.js` `openDecisionDetail()`:
   ```json
   {
       "reasoning": [{"ticker": "...", "agent_name": "...", "action": "...", "confidence": 0.7, "rationale": "..."}],
       "observations": [{"vix": 18.5, "universe_size": 35, "data_quality_status": "OK"}],
       "orders": [...],
       "fills": [...],
       "rl": {"reward": 0.01, "readiness_status": "BLOCKED"},
       "beginner_explanation": {
           "headline": "Today's decision summary",
           "risk_note": "...",
           "ticker_explanations": ["RELIANCE: Bought because momentum + macro signals aligned (3/4 agents voted LONG)"]
       }
   }
   ```

**Step 5c: Generate beginner-friendly explanations:**

For the `beginner_explanation` field, create a simple function in `src/webapp/server.py` or a separate module that:
1. Groups reasoning rows by ticker
2. For each ticker that got a trade (weight != 0):
   - Count LONG vs SHORT votes
   - Take the highest-confidence agent's rationale
   - Generate: `"{TICKER}: {direction} because {top_rationale} ({long_count}/{total} agents agreed)"`
3. Headline: "Traded {N} stocks: {long_count} long, {short_count} short"
4. Risk note: "Max drawdown: {drawdown}%. Circuit breaker: {status}"

### Files To Modify
- `main_india.py` — add run_id, log agent reasoning after each ticker
- `src/webapp/server.py` — add `/api/decision-detail/{run_id}` endpoint

### Files To Create
- None (use existing tables)

### Verification
- Create `tests/test_reasoning_logging.py` that:
  1. Creates an in-memory DB
  2. Inserts 5 `AgentReasoning` rows with a known run_id
  3. Queries them back grouped by ticker
  4. Asserts correct counts and fields
- Manually verify: run `python main_india.py --now`, then check that `AgentReasoning` table has rows, then hit `GET /api/decision-detail/{run_id}` and verify JSON structure

---

## Fix 6: Add Missing API Endpoints the Dashboard Already Expects

### Problem
The frontend `app.js` line 48 fetches 9 API endpoints:
```javascript
const [portfolio, latestRun, perf, dataQuality, decisionRows, watch, orderRows, tradeRows, rl] = await Promise.all([
    fetchJson('/api/portfolio', {}),
    fetchJson('/api/latest-run', {}),
    fetchJson('/api/performance', {}),      // MISSING — added in Fix 4
    fetchJson('/api/data-quality', {}),      // MISSING
    fetchJson('/api/decisions', []),
    fetchJson('/api/watchlist', []),          // MISSING
    fetchJson('/api/orders', []),             // MISSING
    fetchJson('/api/trades', []),             // MISSING
    fetchJson('/api/rl', {}),                // MISSING
]);
```

Only 4 endpoints exist (`/api/portfolio`, `/api/latest-run`, `/api/decisions`, `/`). The remaining 5 need to be created.

### What To Do

File: `src/webapp/server.py`

Add these endpoints using the existing DB tables:

1. **`GET /api/data-quality`** — query latest `DataQualitySnapshot` row (table exists in models.py line 137)
   ```python
   @app.get("/api/data-quality")
   def get_data_quality():
       # Return {"latest": {score, status, notes, missing_quote_count, ...}}
   ```

2. **`GET /api/watchlist`** — query `AgentReasoning` rows from the latest run_id, grouped by ticker, computing an "attention score" (average confidence across agents)
   ```python
   @app.get("/api/watchlist")
   def get_watchlist():
       # Return list of {ticker, attention_score, top_rationale, beginner_reason, monitored_signals}
   ```

3. **`GET /api/orders`** — query `PaperOrder` table (exists in models.py line 188), ordered by timestamp DESC, limit 100
   ```python
   @app.get("/api/orders")
   def get_orders():
       # Return list of {timestamp, ticker, side, product_type, quantity, status, rejection_reason, strategy}
   ```

4. **`GET /api/trades`** — query `PaperFill` table (exists in models.py line 211), ordered by timestamp DESC, limit 100
   ```python
   @app.get("/api/trades")
   def get_trades():
       # Return list of {timestamp, ticker, side, quantity, price, slippage_bps, fees}
   ```

5. **`GET /api/rl`** — query latest `CapitalAllocatorState` + last 20 `RLModelEvaluation` rows
   ```python
   @app.get("/api/rl")
   def get_rl():
       # Return {allocator: {current_intraday_ratio, rl_enabled, weeks_of_data}, evaluations: [...]}
   ```

### Files To Modify
- `src/webapp/server.py` — add 5 new endpoints

### Verification
- Start the server: `uvicorn src.webapp.server:app --port 8000`
- Hit each endpoint and verify it returns valid JSON (even if empty arrays/objects when no data exists)
- No 500 errors on empty database

---

## Fix 7: Log Paper Orders and Fills to the Database

### Problem  
The `PaperOrder` and `PaperFill` tables exist in `src/db/models.py` (lines 188-228) but are never populated. The `GrowwExecutor.execute_target_weights()` returns fills but they're not persisted.

### What To Do

File: `main_india.py`

After the execution step (line 233: `fills = executor.execute_target_weights(safe_weights, theo_prices)`), persist each fill:

```python
from src.db.models import PaperOrder, PaperFill
import uuid

for i, ticker in enumerate(UNIVERSE):
    if abs(safe_weights[i]) < 0.001:
        continue
    
    order_id = f"ord_{run_id}_{ticker}"
    fill_id = f"fill_{run_id}_{ticker}"
    side = "BUY" if safe_weights[i] > 0 else "SELL"
    quantity = int(abs(safe_weights[i]) * portfolio_state["portfolio_value"] / theo_prices.get(ticker, 1))
    
    order = PaperOrder(
        run_id=run_id,
        order_id=order_id,
        ticker=ticker,
        side=side,
        product_type=trade_types.get(ticker, "CNC"),
        quantity=quantity,
        target_weight=float(safe_weights[i]),
        notional=abs(safe_weights[i]) * portfolio_state["portfolio_value"],
        status="FILLED",
        strategy=initial_state.get("current_strategy", "momentum"),
    )
    
    fill = PaperFill(
        run_id=run_id,
        order_id=order_id,
        fill_id=fill_id,
        ticker=ticker,
        side=side,
        product_type=trade_types.get(ticker, "CNC"),
        quantity=quantity,
        price=theo_prices.get(ticker, 0),
        slippage_bps=shortfall if isinstance(shortfall, (int, float)) else 0,
        fees=0.0,  # Could compute from cost_model
    )
    
    # save both to DB via session
```

### Files To Modify
- `main_india.py` — add order/fill logging after execution

### Verification
- Run `python main_india.py --now`
- Query `paper_orders` and `paper_fills` tables
- Verify at least 1 order and 1 fill exist per traded ticker

---

## Fix 8: Log Market Observations per Run

### Problem
The `MarketObservation` table exists (`src/db/models.py` line 172) but is never populated. The decision-detail endpoint needs this data.

### What To Do

File: `main_india.py`

After fetching India VIX and before the per-ticker loop, log a market observation:

```python
from src.db.models import MarketObservation

market_obs = MarketObservation(
    run_id=run_id,
    vix=portfolio_state["vix_raw"],
    universe_size=len(UNIVERSE),
    data_quality_status="OK",  # or from data quality checks
    notes=f"Intraday budget: {intraday_budget:.0f}, Delivery budget: {delivery_budget:.0f}",
)
# save to DB
```

### Files To Modify
- `main_india.py` — add market observation logging

### Verification
- Run `python main_india.py --now`
- Query `market_observations` table
- Verify 1 row exists with correct VIX and universe_size

---

## Fix 9: Add Basic Authentication to the Dashboard

### Problem
The dashboard has zero authentication. Anyone with network access can see all data.

### What To Do

File: `src/webapp/server.py`

Add simple API key authentication:

1. Read `AEGIS_API_KEY` from environment (default to a generated UUID if not set — print it on startup)
2. Create a dependency that checks the `Authorization: Bearer <key>` header OR a `?key=<key>` query parameter
3. Apply to all `/api/*` routes
4. The static file serving (`/`, `/static/*`) should remain open (the HTML/JS/CSS is not sensitive)
5. Update `app.js` to read the key from a `<meta>` tag or localStorage and include it in fetch headers

File: `src/webapp/static/app.js`

1. On page load, check if `localStorage.getItem('aegis_api_key')` exists
2. If not, prompt the user (simple `prompt()` dialog or a login form overlay)
3. Include the key in all fetch calls: `headers: { 'Authorization': 'Bearer ' + key }`
4. If any fetch returns 401, clear the key and re-prompt

### Files To Modify
- `src/webapp/server.py` — add API key auth middleware
- `src/webapp/static/app.js` — add auth header to fetches
- `.env.example` — add `AEGIS_API_KEY=`

### Verification
- Start server without setting `AEGIS_API_KEY` — it should generate and print one
- Hit `/api/portfolio` without auth header — should return 401
- Hit `/api/portfolio` with correct Bearer token — should return 200
- Load dashboard in browser — should prompt for key, then work after entering it

---

## Fix 10: Parallelize Per-Ticker Processing

### Problem
`main_india.py` processes 40 tickers sequentially (line 169). Each ticker runs the full orchestrator. This is slow.

### What To Do

File: `main_india.py`

Replace the sequential for loop with `concurrent.futures.ThreadPoolExecutor`:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_ticker(i, ticker, ...):
    """Process a single ticker through the orchestrator. Returns (i, exposure, trade_type)."""
    initial_state = { ... }  # same as current
    final_state = orchestrator.run_cycle(initial_state)
    # extract exposure and trade_type
    return i, exposure, trade_type, final_state

with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(process_ticker, i, ticker, ...): ticker for i, ticker in enumerate(UNIVERSE)}
    for future in as_completed(futures):
        i, exposure, trade_type, final_state = future.result()
        target_weights[i] = exposure
        trade_types[UNIVERSE[i]] = trade_type
```

**Important:** The orchestrator and agents must be thread-safe. Since each invocation creates its own state dict and yfinance calls are independent, this should work. But verify that:
- `pm_agent.rl_model.predict()` is thread-safe (SB3 models are safe for inference)
- SQLAlchemy sessions are not shared (each thread should create its own session)
- yfinance calls are independent

### Files To Modify
- `main_india.py` — parallelize the ticker loop

### Verification
- Run `python main_india.py --now` with `UNIVERSE` of 10+ tickers
- Verify it completes in < 3 minutes (vs potentially 10+ minutes sequential)
- Verify all tickers produce results (no missing weights)
- Verify database entries are correct

---

## Summary — Execution Order

| # | Fix | Creates | Modifies | Dependencies |
|---|-----|---------|----------|-------------|
| 1 | RL Environment + Training | `src/models/train_india.py` | `pm_agent.py`, `rl_env.py` | None |
| 2 | Portfolio Feedback Loop | `src/engine/portfolio_state.py` | `main_india.py` | None |
| 3 | Wire Strategies | `src/agents/research/strategy_runner_agent.py` | `orchestrator.py` | None |
| 4 | Benchmark Tracking | `src/engine/benchmark_tracker.py` | `main_india.py`, `server.py` | Fix 2 |
| 5 | Agent Reasoning Logging | None | `main_india.py`, `server.py` | None |
| 6 | Missing API Endpoints | None | `server.py` | Fix 5 |
| 7 | Paper Order/Fill Logging | None | `main_india.py` | Fix 5 (needs run_id) |
| 8 | Market Observations | None | `main_india.py` | Fix 5 (needs run_id) |
| 9 | Dashboard Auth | None | `server.py`, `app.js`, `.env.example` | None |
| 10 | Parallel Processing | None | `main_india.py` | None |

**Fixes 1, 2, 3, 5, 9, 10 can be done in parallel.**  
**Fix 4 depends on Fix 2. Fixes 6, 7, 8 depend on Fix 5.**

---

## Post-Fix Verification Checklist

After ALL fixes are applied, run this end-to-end test:

1. `pytest tests/` — all tests pass (old + new)
2. `python -m src.models.train_india` — trains a model, saves to registry, prints Sharpe > 0 on out-of-sample
3. `python main_india.py --now` — completes in < 5 minutes, logs to all DB tables
4. Query `daily_pnl` table — portfolio_value != 250000 after second run
5. Query `agent_reasoning` table — has rows with rationales
6. Query `paper_orders` / `paper_fills` — has rows
7. Query `benchmark_daily` — has NIFTYBEES data
8. Query `performance_daily` — has comparison metrics
9. `uvicorn src.webapp.server:app --port 8000` — start server
10. Hit all 9 `/api/*` endpoints — all return valid JSON
11. Open dashboard in browser — charts render, positions show reasoning, benchmark comparison visible
12. Hit `/api/portfolio` without auth — returns 401
