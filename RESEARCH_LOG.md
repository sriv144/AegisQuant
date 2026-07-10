# Research Log

This file tracks autonomous improvements made by the Auto-Researcher agent.
Each entry records what was implemented, what was evaluated and skipped,
and candidates queued for the next run, so we never repeat work.

## 2026-06-09 — Auto-Researcher v4

**Resume-worthiness score at start of run:** 86/100 — highest of the six-repo portfolio (RL + LLM consensus + institutional risk overlay, most recent commit activity, most stars).

**Branch:** `claude/lucid-darwin-rj6hm9`

### Implemented this run
- **Bug fix (cleanup):** removed three pip-stderr artifacts that were accidentally committed at the repo root — `=0.2.36`, `=0.29.1`, `=2.3.0`. These almost certainly came from a `pip install foo=0.2.36` (missing the second `=`) where the shell redirected `pip`'s usage error into files named after the version string. They made the repo root look broken on first impression — the single highest-leverage fix on the highest-scoring repo in the portfolio.
- **New CI workflow** at `.github/workflows/ci.yml` — first push/PR CI for the repo (the existing `trade.yml` is the production trading cron, not a CI gate). Installs `requirements.txt`, sets `ENABLE_MOCK_DATA=True` so no real Alpaca / broker calls fire, and runs `pytest tests/` over the deterministic unit suite (`test_circuit_breakers`, `test_monte_carlo`, `test_cost_model`, `test_walk_forward`, etc.). Root-level scripts like `test_groww_connection.py` that need real broker credentials are deliberately excluded.
- `RESEARCH_LOG.md` — seeded so future agent runs have memory of what's been tried.

### Why these were prioritized
- The garbage files were the single most visible defect on the highest-scoring repo. Free win, zero risk, every recruiter who lands on the repo root benefits.
- CI on top of an already-mature pytest suite turns the test suite into a guard rail and adds a green badge to back the README's institutional-grade claims.

### Evaluated and skipped
- README rewrite — already strong and accurate (architecture phases, exact run commands, audit-report transparency note). Diff risk outweighs improvement.
- Gitignoring large committed artifacts (`run_log.txt`, `pcr.txt`, `ppo_*.zip`, `pytest_*.txt`) — worth doing but `git rm` against committed history requires an explicit maintainer call (do they want LFS, do they want them archived?). Queued for next run.
- Moving `.github/workflows/trade.yml` secrets to environment-scoped secrets — governance change, needs maintainer confirmation.
- `INDIA_PIPELINE_SUMMARY.md` and `PROFESSIONAL_TRADER_UPGRADE.md` reorganization into `docs/` — cosmetic, defer.
- `CODEX_FIXES.md` at repo root (29.9 KB) — looks like a working-notes file; same story, defer to a focused cleanup pass.

### Next-run candidates (priority order)
1. **Untrack heavy artifacts**: add `run_log.txt`, `pytest_*.txt`, `*.zip` model checkpoints, `model_registry/`, `backtest_results/`, `pcr.txt` to `.gitignore` and `git rm --cached` them. Cuts clone size dramatically.
2. **Move working-notes markdowns** (`CODEX_FIXES.md`, `INDIA_PIPELINE_SUMMARY.md`, `PROFESSIONAL_TRADER_UPGRADE.md`, `plan.md`) into a `docs/` subdir to clean the root.
3. **Add a Streamlit dashboard screenshot or short GIF** to the README — the project deserves it.
4. **Add a `LICENSE`** (MIT or Apache-2.0) — currently absent.
5. **Add a `pyproject.toml` lint section** + ruff job in CI once a style is agreed.
6. **Anthropic / Claude LLM-consensus provider**: README mentions Anthropic keys in `.env.example` — surface a `--llm-provider claude` option and document it (resume signal for AI consensus scoring).
