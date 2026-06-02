# Research Log

This log tracks autonomous-research / auto-improvement passes over the
repository. Each entry records what was scored, what was implemented,
and what was deliberately skipped, so future runs can avoid re-doing
work that has already shipped.

## 2026-06-02 — Auto-Researcher v4

### Resume-worthiness score at start of run

`83 / 100` — ranked #2 of 6.

Breakdown:

- Tech stack prestige: 24 / 25 — PPO / SAC RL agents, LLM consensus
  scoring, SHAP attribution, Gaussian HMM regime detection,
  Monte Carlo walk-forward, Alpaca live trading. Top-tier quant ML.
- Commit recency: 20 / 25.
- Feature completeness: 18 / 20 — Multi-asset Gym env, walk-forward
  backtester, Streamlit command center, scheduler-driven live daemon,
  Markdown + JSON audit reporting that is deliberately honest about
  underperformance.
- Stars / visibility: 8 / 15.
- README quality: 13 / 15 — Clear phase breakdown, run-the-matrix
  commands, install path; could use screenshots.

### Implemented on branch `claude/lucid-darwin-MN9a4`

- **docs: seed this `RESEARCH_LOG.md`.** No code or CI changes this
  pass.

### Evaluated and skipped

- Adding a `pytest` CI workflow. Skipped this run because the test
  suite very likely transitively imports `stable-baselines3`,
  `gymnasium`, `alpaca-py`, `shap`, `hmmlearn`, and `torch`. Standing
  up that environment in GitHub Actions is doable but slow, and
  test hermeticity needs to be verified first (`.env` reads,
  `yfinance` network calls, model registry filesystem assumptions)
  before turning the run into a required gate. Queued as a dedicated
  follow-up.
- Touching the existing `.github/workflows/trade.yml`. That workflow
  appears to be wired to the live trading schedule and is intentional
  infra; auto-researcher should not modify it without explicit
  instruction.
- Migrating the LLM consensus path from any current provider to
  Claude. Skipped — changing the runtime LLM in a quant pipeline
  alters signal generation and must be A/B-tested with audit reports.
- Any change to `src/`, `model_registry/`, or backtest outputs.
  Skipped.

### Candidates for next run

1. Add a `pytest` CI workflow after verifying the test suite is
   hermetic (no live `yfinance`, no Alpaca calls, no `.env`
   requirement at import time).
2. Add a minimal smoke job that imports `src.backtest.reporting` and
   regenerates an audit report from a checked-in fixture JSON.
3. Refactor the LLM consensus call to use the Anthropic Claude SDK
   with prompt caching for the regime-context prompts.
4. Add screenshots of the Streamlit Command Center (live PnL, SHAP
   feature importances, regime shifts) to the README.
5. Publish a versioned `BENCHMARKS.md` so walk-forward results have a
   stable, dated reference instead of living only in JSON artifacts.
