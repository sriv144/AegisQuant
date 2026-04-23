# AegisQuant - System Architecture

This document is the interview-ready reference for how AegisQuant composes
its multi-agent trading pipeline. The README covers *what* each phase does;
this doc covers *how the pieces talk to each other* and *where the
production safety rails live*.

## 1. High-Level Data Flow

```
          +---------------------+
          |  Universe Screener  |  (weekly; NSE ~2000 tickers -> top 30-50)
          +----------+----------+
                     |
                     v
          +---------------------+
          |   Macro Agent       |  (VIX, yield curve, regime signal)
          +----------+----------+
                     |
                     v
          +---------------------+
          |  Research Agent     |  (per-ticker features: z-score vol,
          |                     |   momentum, value, regime tag)
          +----------+----------+
                     |
                     v
          +---------------------+
          | Strategy Selector   |  (VIX-regime + weekly perf scores pick
          |                     |   top 2-3 of 9 strategies per cycle)
          +----------+----------+
                     |
                     v
          +---------------------+
          |  Committee Vote     |  (ensemble LLM consensus scoring +
          |                     |   direction: LONG / SHORT / FLAT)
          +----------+----------+
                     |
                     v
          +---------------------+
          | Asset Allocator RL  |  (PPO policy maps state -> exposure %;
          |                     |   turnover + covariance penalties)
          +----------+----------+
                     |
                     v
          +---------------------+
          |  Execution Agent    |  (tags CNC/MIS, sizes lots, submits to
          |                     |   Groww/Angel One; falls back to mock)
          +----------+----------+
                     |
                     v
          +---------------------+
          |  Risk + Circuit     |  (time window, drawdown, per-position SL/TP,
          |   Breakers          |   MIS auto-close before 15:25 IST)
          +----------+----------+
                     |
                     v
          +---------------------+
          |  Audit + Dashboard  |  (SQLAlchemy log, Streamlit + SPA
          |                     |   Positions / Decisions drill-down)
          +---------------------+
```

## 2. Agent Responsibilities

| Agent | Role | Key output field |
| --- | --- | --- |
| `universe_selector_agent` | Screen 2000+ NSE tickers on liquidity, quality, opportunity, diversification | `universe_snapshot` rows |
| `macro_agent` | IST-aware VIX + yield-curve snapshot | `regime`, `vix_level` |
| `research_agent` | Per-ticker features, fundamentals, momentum | `ticker_features` |
| `strategy_selector` | Nominate 2-3 active strategies for the cycle | `active_strategies`, `strategy_scores` |
| `committee` | LLM consensus scoring across strategies | `committee_decision.direction`, `conviction` |
| `asset_allocator` | RL PPO policy -> `adjusted_exposure_pct` | `allocation_request.adjusted_exposure_pct` |
| `execution_agent` | Tag trade type (MIS vs CNC), size in whole lots, route to broker | orders + `trade_type` |
| `circuit_breakers` | Block or kill orders violating time / drawdown / PnL bounds | veto with reason |
| `position_manager` | Track open CNC positions, enforce SL / TP / aging exits | `OpenPosition` rows |
| `capital_allocator` | Weekly RL meta-learner adjusting intraday / delivery split | `budgets` |

## 3. Trade-Mode Contract

AegisQuant runs two product types simultaneously:

- **CNC (delivery)** - default 80% of capital, hold 1-3 months, governed
  by `PositionStopLossRule` (-8% SL, +20% TP, aging exit).
- **MIS (intraday)** - default 20%, must close before 15:25 IST or the
  `MISAutoCloseRule` fires and forces exit.

The 80/20 split is not static - the `capital_allocator` RL meta-learner
adjusts the ratio weekly based on realised performance of each bucket.

## 4. Safety Rails

1. **Broker execution is gated** behind `ENABLE_BROKER_EXECUTION=True`.
   With the flag off, orders round-trip to a mock executor while still
   exercising every upstream agent. This is the default in dev / CI.
2. **TOTP login fallback**: if Groww / Angel One credentials fail, the
   executor silently falls back to mock mode rather than raising, so a
   broker outage cannot take the whole pipeline down.
3. **Time window**: `TimeWindowRule` enforces 09:15-15:25 IST with
   timezone-aware comparisons (tested against cloud runners which
   default to UTC).
4. **Drawdown circuit breaker**: hard-stop on portfolio drawdown
   threshold breach.
5. **Zero-weight protection**: if the RL policy collapses to a
   degenerate sign (historically seen at `-0.1842` for every ticker),
   the pipeline uses `committee_decision.direction` instead of the raw
   RL sign (see `main_india.py`).

## 5. Persistence

| Table | Purpose |
| --- | --- |
| `OpenPosition` | CNC inventory with entry price, entry time, SL/TP thresholds |
| `DailyPnL` | Per-day realised and unrealised PnL |
| `UniverseSnapshot` | Weekly screener output, for reproducibility and audit |
| Decisions log | Every committee vote with per-ticker rationale (drives dashboard drill-down) |

## 6. Observability

- **Streamlit Command Center** (`src/ui/dashboard.py`) - live paper-trading
  PnL, regime shifts, SHAP global feature attribution.
- **Web dashboard SPA** (`server.py` + `frontend/index.html`) - Dashboard /
  Positions / Decisions pages. Positions page pulls live prices from
  `yfinance` to estimate shares; Decisions page expands per-ticker
  breakdown on row click.
- **Alerting** - SQLAlchemy audit rows plus SMTP / Slack alerting hooks
  for circuit-breaker vetoes and drawdown events.

## 7. Where to Look Next

- `src/agents/orchestrator.py` - the LangGraph-style wiring of the
  agents above.
- `src/agents/state.py` - the shared state schema each agent mutates.
- `src/engine/circuit_breakers.py` - all safety rails in one module.
- `src/engine/position_manager.py` - CNC inventory + SL/TP logic.
- `src/engine/capital_allocator.py` - RL meta-learner for the CNC/MIS
  split.
- `main_india.py` - the 20-minute heartbeat that drives everything in
  IST market hours.
