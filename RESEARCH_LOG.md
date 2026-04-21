# Research Log

Automated improvement log maintained by Auto-Researcher.
Each run appends a dated entry describing what was implemented, what was skipped, and why.

---

## 2026-04-21 — Auto-Researcher v4

**Resume score at the start of this run:** 78/100 (top-3: RL/PPO + SHAP + Alpaca + live-trading cron, recent commits).

**Implemented (branch `claude/focused-newton-TXG01`):**
- Added `.github/workflows/ci.yml`: runs `pytest tests/` on Python 3.11 for pushes and PRs against `main`. CI uses placeholder keys and forces `ENABLE_BROKER_EXECUTION=False` + `ENABLE_MOCK_DATA=True` so tests run offline. Independent from the existing `trade.yml` cron (which executes live trading on a schedule).
- Seeded this `RESEARCH_LOG.md`.

**Why this was prioritized:**
Repo already has a `trade.yml` cron that runs the live pipeline, but nothing gates code changes. README documents `python -m pytest tests/` as the safety verification before pushing to staging/production, and committed `pytest_clean.txt` / `pytest_run.txt` artifacts confirm tests exist and run. A PR-gating CI turns that manual step into an automatic safety net on a live-trading repo — highest-leverage, lowest-risk change.

**Evaluated and skipped this run:**
- Cleaning up stray `=0.2.36`, `=0.29.1`, `=2.3.0` files in the repo root (likely created by `pip install foo =0.2.36` with a stray space). They're harmless but noisy — skipped to keep this PR narrowly scoped to CI. Queued for next run.
- Removing `pytest_output.txt` / `pytest_run.txt` / `pytest_clean.txt` from VCS (CI output belongs in Actions artifacts, not git). Same reason — scope.
- Building a separate backtest CI job (`python src/backtest/walk_forward.py --mc-sims 100`): too slow for PR gating, fits better as nightly.
- Linting with `ruff`: not configured in `pyproject.toml` yet; adding it would surface a flood of findings outside the CI scope of this PR.

**Next-run candidates:**
- Delete the stray `=<version>` root files and the committed `pytest_*.txt` logs.
- Add a nightly `backtest.yml` workflow running a short walk-forward on a single ticker and asserting Sharpe > threshold.
- Introduce `ruff` with an initial allow-list, then tighten over successive runs.
- Extract the India pipeline env vars into a single `configs/ci.env` reference to keep `trade.yml` and `ci.yml` in sync.
