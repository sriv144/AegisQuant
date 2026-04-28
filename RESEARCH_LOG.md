# Research Log

A running log of automated improvement runs against this repo.

## 2026-04-28 — Auto-Researcher v4

**Resume score (start of run):** 87 / 100

- Tech stack prestige: 24 (RL with PPO/SAC, LLM consensus scoring, SHAP, HMM regimes, Alpaca live trading)
- Commit recency: 25 (pushed 2026-04-19)
- Feature completeness: 17 (multi-phase, Streamlit dashboard, audit trails, tested locally)
- Stars / visibility: 8 (1 star + 1 fork)
- README quality: 13 (technical and clear, but no badges, no automated test signal)

### Implemented on `claude/lucid-darwin-jnSqo`

1. **`.github/workflows/tests.yml`** — a pytest CI workflow that:
   - Runs on push and PR to `main`, plus manual `workflow_dispatch`.
   - Sets up Python 3.11, caches pip, installs `requirements.txt`, then runs `pytest tests/`.
   - Provides dummy values for `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` / `ANTHROPIC_API_KEY` so any module that reads env vars at import time doesn't blow up in CI.
   - Uses a concurrency group so superseded pushes auto-cancel.
2. **This `RESEARCH_LOG.md`**.

### Why these were prioritized

- AegisQuant already had `.github/workflows/trade.yml` (a scheduled trading workflow), but no test workflow. Continuous test signal is the highest-leverage change for a project with an existing pytest suite (`pytest_clean.txt`, `pytest_run.txt`, etc., show tests are exercised locally).
- A green CI badge on the README is a real recruiter signal at very low risk.
- The workflow does not add any new code paths to runtime, only installs and tests. Zero risk to live trading or training.

### Evaluated and skipped this run

- **Cleaning up the stray `=0.2.36`, `=0.29.1`, `=2.3.0` files at repo root.** These look like artifacts from a malformed `pip install "pkg>=X"` command that escaped quoting. Tempting to delete, but the user may be intentionally tracking them. Skipped as ambiguous.
- **Adding model-card / backtest-results pages to the README.** Higher value but requires reading model output and writing accurate numbers. Logged for next run.
- **Pinning `requirements.txt`.** Real value, but pinning across PPO/SAC/SHAP/Alpaca touches a lot of moving versions. Too risky without local verification.
- **README badges** (CI status, Python version). Worth doing once the workflow has run at least once and produced a stable badge URL. Logged.

### Next-run candidates

- Add a CI status badge + Python-version badge to `README.md` once `tests.yml` has produced at least one run.
- Add a `MODEL_CARD.md` summarising the PPO curriculum results and walk-forward backtest numbers.
- Pin `requirements.txt` to known-good versions and add a `requirements-dev.txt` with pytest / ruff / mypy.
- Add a tiny `make` or `pyproject` script registry so common entrypoints (train, backtest, dashboard, live) are discoverable.
- Investigate the stray `=X.Y.Z` files at repo root and either delete or document them.
