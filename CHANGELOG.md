# Changelog

All notable changes to AegisQuant are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MIT `LICENSE`, `CONTRIBUTING.md`, GitHub issue + PR templates.
- `Makefile` with install / test / lint / format / docker-up / train / dashboard / clean targets.
- `.pre-commit-config.yaml` running `ruff` and standard hygiene hooks.
- Hardened `.gitignore` to prevent pip version-spec stub files (`=0.2.36`, etc.),
  zipped model artifacts, and pytest log dumps from re-entering the repo.
- `RESEARCH_LOG.md` seeded for the auto-researcher cycle.

## 2026-05-16

### Added
- **LLM Trading Analyst agent**: one Anthropic Claude call per ticker that reasons over
  indicators, all 4 research-agent signals, and all 9 strategy signals to emit
  BUY/HOLD/EXIT with confidence and allocation. Soft-consensus fallback when no API key.

### Changed
- `ConsensusWeightEngine` simplified to allocation math (rank by confidence, top 15, normalize).
- Removed the rigid `BB_Position > 1.0 / < 0.0` entry/exit gates that blocked all candidates in testing.
- `OpenRouter` calls now cap at `max_tokens=1024` to stay inside free-tier budget.
- New `SKIP_TIME_CHECK` env var for off-hours testing.

## 2026-05-14

### Added
- GitHub Actions trading workflow with 19 cron triggers (9:30 AM – 3:30 PM ET, Mon–Fri).
- `repository_dispatch` trigger fired every 20 min by cron-job.org for reliable scheduling.
- Docker + docker-compose setup, Oracle Cloud Free Tier one-shot setup script.
- Holiday circuit breaker covering NYSE 2025–2027 + weekend guard, DST-aware `zoneinfo`.
- `/health` liveness probe.

### Changed
- Migrated broker SDK from `alpaca-trade-api` to `alpaca-py` to resolve the `websockets<11`
  vs `yfinance>=1.2` (`websockets>=13`) conflict.

## 2026-05-13

### Added
- **US market support** with Alpaca paper-trading adapter (`alpaca_broker.py`), US universe
  screener (S&P 500 + 110-stock seed), US market-data collector with CBOE VIX, `main_us.py`
  daemon, and `MARKET=US/IN` env switching.
- **Strategy rewrite**: all 9 strategies switched from single-threshold to multi-indicator
  confluence scoring (momentum, mean-reversion, factor model, trend-following, pairs trading,
  volatility breakout, gap fill, earnings momentum, sector rotation).
- **Trade reasoning pipeline** captured per-ticker JSON: research signals, committee verdict,
  allocation sizing, risk approval. Surfaced via `/api/decisions/{id}/reasoning`,
  `/api/positions/detailed`, `/api/trades/closed`.
- **Dashboard upgrades**: WebSocket real-time push (`/ws/live`), Nifty50 benchmark overlay,
  optional JWT auth, trade-reasoning drill-down modal, Trade History page.
- **Execution layer**: `BaseBroker` abstraction, realistic paper trading with slippage models
  (none / fixed / realistic) + NSE/US cost models + partial fills. Zerodha + Angel One adapters.
- **PostgreSQL support** with connection pooling, SQLite fallback retained.

### Fixed
- Closed the portfolio feedback loop: real DB-tracked portfolio value replaces the hard-coded
  baseline; drawdown is now computed from live state.
- Strategy scoring no longer returns `{}` — now ranks by avg P&L % and win rate from closed trades.
- RL training defaults to walk-forward curriculum on real Nifty50 data (`--synthetic` opt-in).
- 15-minute TTL cache on `yfinance` and 30-minute cache on news with 429 rate-limit detection.

## 2026-04-17 — Initial dashboard

### Added
- Multi-page SPA (Dashboard / Positions / Decisions) with KPI cards, exposure donut, holdings bar,
  ticker drill-downs.
- `/api/latest-run` endpoint with live yfinance prices for share estimates.
