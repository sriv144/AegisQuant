# Research Log

This file records autonomous improvement runs performed by Auto-Researcher.
Each entry captures what was evaluated, what was implemented, and what was
skipped, so that future runs do not repeat the same work.

## 2026-05-21 — Auto-Researcher v4

**Resume-worthiness score at start of run:** ~83 / 100 (top 3 of 6).

**Branch:** `claude/lucid-darwin-InUHR` (from `main`).

### Implemented

- **`.github/workflows/tests.yml`** — runs `pytest tests/ -q` on every push
  and PR to `main`, Python 3.11. Forces `ENABLE_MOCK_DATA=True` and
  `ENABLE_BROKER_EXECUTION=False` so the suite never touches `yfinance` or
  live broker APIs. Pip-cached, cancel-in-progress concurrency, 20-minute
  timeout. The only pre-existing workflow (`trade.yml`) is a live trading
  heartbeat — it should not double as test infrastructure, so a dedicated
  `tests.yml` is pure upside.
- **MIT `LICENSE`** — the repo was previously unlicensed.
- **`.gitignore` hygiene** — added rules for `=*` (pip-install stderr
  accidentally redirected into files such as `=0.2.36` / `=0.29.1` /
  `=2.3.0`), `pytest_*.txt`, `pcr.txt`, and `*.log`, so this junk stops
  re-landing in the repo root.
- **README status badges** + a CI note in the testing section.
- **Seeded this `RESEARCH_LOG.md`.**

### Why this was prioritized

Strong technical positioning (PPO/SAC RL + LLM consensus + HMM regime
detection + SHAP attribution + Alpaca execution) but `main` had no CI
correctness signal. A dedicated `tests.yml` creates the green-badge story
recruiters look for with zero risk to the live trading loop. The `.gitignore`
hygiene addresses a visible cleanliness problem — stray files in the repo
root are the first thing a recruiter browsing the project sees.

### Evaluated and skipped

- **Deleting the existing stray root files** (`=0.2.36`, `=0.29.1`, `=2.3.0`,
  `pcr.txt`, `pytest_clean.txt`, `pytest_output.txt`, `pytest_run.txt`).
  Removal needs per-file delete commits rather than the atomic additive
  `push_files` path used here; `.gitignore` now prevents recurrence and the
  cleanup is queued as a dedicated `chore:` commit for the next run.
- **Lint job.** No style config pinned; would surface pre-existing findings
  as spurious CI red. Defer to a `ruff format` baseline run.
- **Paper-trading smoke test.** Requires mocked Alpaca/Groww shims — bigger
  than a CI wiring commit.

### Next-run candidates

1. `chore:` remove the stray `=X.Y.Z` / `pytest_*.txt` / `pcr.txt` files
   from the repo root via per-file deletes.
2. Add a `ruff` job starting from a `ruff format` baseline commit.
3. Split `trade.yml` into `india-trade.yml` / `us-trade.yml`.
4. Consolidate `PROFESSIONAL_TRADER_UPGRADE.md` + `INDIA_PIPELINE_SUMMARY.md`
   + `plan.md` into a `docs/` tree linked from the README.
5. Add a CodeQL security-scanning workflow.
