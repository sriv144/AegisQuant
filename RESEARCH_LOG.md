# Research Log

This file tracks autonomous codebase-improvement runs. Each entry records
what was implemented, why it was prioritized, and what was deferred.

## 2026-05-20 — Auto-Researcher v4

**Resume-worthiness score at start of run: 85 / 100**
(High tech-stack prestige: RL-based quant trading, PPO/SAC, walk-forward
backtesting, SHAP attribution, regime detection. Most recently updated of
the tracked repos, real test suite, working CI.)

### Implemented (branch `claude/lucid-darwin-sFMHv`)
- **Repository hygiene** — hardened `.gitignore` to exclude stray build and
  test artifacts, and removed accidentally committed files from the repo
  root: `=0.2.36`, `=0.29.1`, and `=2.3.0`. Those files are byproducts of
  shell redirects (e.g. `pip install 'pkg>=0.2.36' > =0.2.36`) and have no
  function in the project.

### Why this was prioritized
AegisQuant is feature-complete with a solid README and working CI, so the
highest-value safe change was presentation. A recruiter browsing the repo
root currently sees garbage files named like version numbers, which reads
as careless. Removing them and preventing recurrence is zero-risk and
directly improves how the project presents.

### Evaluated and skipped
- **README rewrite** — skipped; the existing README already describes the
  architecture, install, and run modes clearly.
- **Removing `pytest_clean.txt`, `pytest_output.txt`, `pytest_run.txt`,
  `pcr.txt`** — deferred; these are also likely stray output captures, but
  were left untouched this run to avoid removing anything potentially
  referenced. They are now covered by `.gitignore`. Flagged below.
- **CI changes** — skipped; `.github/workflows/trade.yml` already exists.
- **Feature work** — skipped this run; substantive trading-logic changes
  carry real breakage risk and were out of scope for a safe, fast pass.

### Next-run candidates
- Remove the remaining stray captures (`pytest_*.txt`, `pcr.txt`) after
  confirming nothing references them.
- Move the committed model archives (`ppo_*.zip`) into `model_registry/`
  or Git LFS / releases.
- Verify README run commands match actual `src/` entry points.
