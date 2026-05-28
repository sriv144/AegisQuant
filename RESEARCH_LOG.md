# Research Log

This file records autonomous improvement runs performed by Auto-Researcher.
Each entry captures what was evaluated, what was implemented, and what was
skipped, so that future runs do not repeat the same work.

## Prior runs (on unmerged `claude/lucid-darwin-*` branches)

- **2026-04-24** added `.github/workflows/tests.yml` (pytest on push/PR with
  `ENABLE_MOCK_DATA=True`, `ENABLE_BROKER_EXECUTION=False`) and an MIT
  `LICENSE`. That work is not merged to `main`; this run does not duplicate it.

## 2026-05-28 — Auto-Researcher v4

**Resume-worthiness score at start of run:** ~90 / 100 — the strongest project
on the account (PPO/SAC RL + LLM consensus + Gaussian-HMM regime detection +
SHAP attribution + Alpaca/Groww execution), 2 stars, updated within the week.

**Branch:** `claude/lucid-darwin-0QUHr` (from `main`).

### Implemented

- **`.github/workflows/codeql.yml`** — CodeQL security + quality scanning on
  push/PR to `main` and a weekly schedule. This is the exact surface CodeQL
  catches well: a Python codebase that handles broker and LLM API credentials.
  Pure additive wiring; `security-events: write` scoped, 20-minute timeout,
  cancel-in-progress concurrency. Complements (does not replace) the existing
  `trade.yml` production heartbeat.
- **Hardened `.gitignore`** to stop re-committing pip-stderr redirect files
  (`=0.2.36`, `=0.29.1`, `=2.3.0`), `*.log`, `run_log.txt`, and `pytest_*.txt`.
- **Removed the existing stray artifacts** from the repo root in follow-up
  commits on this branch (see below).
- **Seeded this `RESEARCH_LOG.md`.**

### Why this was prioritized

The repo root was the first thing a recruiter sees and it was cluttered with
broken-looking `=X.Y.Z` files and large captured logs (`run_log.txt` was
~217 KB). Cleaning that up plus adding security scanning is high resume signal,
clean to implement, and zero risk to the trading logic — nothing in `src/`
imports these artifacts.

### Evaluated and skipped

- **Deleting `pcr.txt`.** Ambiguous — could be put/call-ratio *data* rather than
  a log. Left in place pending confirmation.
- **Deleting root model archives** (`ppo_curriculum_SPY.zip`,
  `ppo_portfolio_manager.zip`). Large binaries that belong in `model_registry/`,
  but code may reference them by path; left in place to stay zero-risk.
- **Lint job (ruff/black).** Would surface pre-existing style findings as
  spurious CI red; needs a formatting-baseline commit first.
- **Consolidating `PROFESSIONAL_TRADER_UPGRADE.md` / `INDIA_PIPELINE_SUMMARY.md`
  / `plan.md` / `CODEX_FIXES.md` into a `docs/` tree.** Worthwhile docs cleanup,
  but larger than this run's scope.

### Next-run candidates

1. Move the root model archives into `model_registry/` (after verifying no
   code references the root paths) and confirm/clean `pcr.txt`.
2. Add a `ruff format` baseline commit, then a lint CI job.
3. Consolidate the scattered top-level `*.md` design docs into `docs/`.
4. Split `trade.yml` into `india-trade.yml` / `us-trade.yml`.
5. Merge the unmerged `tests.yml` + `LICENSE` to `main`.
