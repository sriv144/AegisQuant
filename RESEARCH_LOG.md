# Research Log

This file tracks autonomous research and improvement runs against this repo.
Each run lists what was implemented, what was evaluated and skipped, and the
next-run candidate list.

## 2026-05-15 — Auto-Researcher v4

**Resume-worthiness score at start of run: 90 / 100**

Signal breakdown:
- Tech stack prestige: 25/25 (RL PPO/SAC + LLM consensus + SHAP + Gaussian HMM regime detection + Alpaca live execution)
- Commit recency: 25/25 (last push 2026-05-14, day before this run)
- Feature completeness: 18/20 (walk-forward backtester, Streamlit command center, live trading daemon, audit trail)
- Stars + visibility: 8/15 (1 star, 1 fork)
- README quality: 14/15 (clear phases, install + run instructions)

### Implemented this run

Branch: `claude/lucid-darwin-dYRjh`

- `ci(tests)`: added `.github/workflows/tests.yml` that runs the existing pytest suite (circuit breakers, cost model, monte carlo, multi-asset env, regime detector, runtime safety, walk-forward, benchmarks) on every push and PR to `main`. The repo already has `trade.yml` running the live trading cycle, but no test CI — this closes the gap and is independent of the production schedule.
- `chore(gitignore)`: extended `.gitignore` to cover three sources of repo clutter currently visible at root:
  - pip / shell-redirect artifacts whose names start with `=` (the repo currently has `=0.2.36`, `=0.29.1`, and `=2.3.0` files, which are pip install transcripts captured by accident when running e.g. `pip install yfinance >=0.2.36` in a shell that interpreted `>=` as a redirect).
  - pytest output dumps (`pytest_clean.txt`, `pytest_output.txt`, `pytest_run.txt`).
  - trained-model `.zip` snapshots (`ppo_curriculum_SPY.zip`, `ppo_portfolio_manager.zip`) — these are large binaries that belong in a model registry or as a release artifact, not in git history.

The stray files already tracked in the repo are intentionally **not** deleted in this commit. Removing tracked files would rewrite the working set, which crosses the 'changes cannot break existing functionality' bar for a code that ships to a live broker. The `.gitignore` change ensures no new ones leak; cleanup of existing ones is left to the maintainer.

### Why this was prioritized

A quant trading repo with `trade.yml` running in production but no test CI on top of it is a credibility risk. Wiring a separate `tests.yml` is the highest-impact change available without touching either the live execution path or the training pipeline. The `.gitignore` tightening was bundled in the same commit because the underlying root cause (shell redirect interpretation) is recurring and the gitignore is the only safe place to fix it.

### Evaluated and skipped this run

- Delete the `=0.2.36`, `=0.29.1`, `=2.3.0`, `pytest_*.txt`, `pcr.txt`, and `*.zip` files from git history — skipped: requires a deletion commit and the user may still want the trained-model zips around for reproducibility.
- Refactor `langchain-openai` consensus scoring to Anthropic Claude — skipped: production trading code path; needs offline benchmark of signal agreement before swapping providers.
- Add Postgres + Redis as CI services for the `alert_history` and `trading_history` paths — skipped: the suite is unit-style today and does not need them.
- Pin Python 3.11 explicitly in `pyproject.toml` (currently inferred from requirements.txt + trade.yml) — skipped: low impact, will bundle with the next config sweep.

### Next-run candidates

1. Delete the stray `=*`, `pytest_*.txt`, `pcr.txt`, and trained-model zip files from the working tree in a single 'chore: prune unused root artifacts' commit (low risk once `.gitignore` has shipped).
2. Add Anthropic Claude as an alternative LLM consensus provider behind `LLM_PROVIDER=anthropic`.
3. Add a `pytest --cov` step and publish coverage.
4. Add a `ruff` lint step (the project already follows roughly PEP 8 — a one-time formatting pass would unlock the gate).
5. Promote the `tests.yml` pytest step out of `continue-on-error` once the GHA environment is verified to mirror local results in `pytest_clean.txt`.
