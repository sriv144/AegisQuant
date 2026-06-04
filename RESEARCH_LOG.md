# Research Log

Long-running notebook for the auto-researcher: what was evaluated, what was
shipped, and what is queued for next runs.

## 2026-06-04 - Auto-Researcher v4

**Resume score going in:** 85 / 100

Strongest repo in the portfolio on tech-stack prestige (RL + LLM consensus
for multi-asset trading) and recency. Two cosmetic issues were dragging the
first-impression score down disproportionately:

1. Three empty marker files at the repo root (`=0.2.36`, `=0.29.1`,
   `=2.3.0`) from accidental `pip install <pkg> =<ver>` invocations.
   These render as broken files in the GitHub file browser and are the
   first thing a recruiter sees.
2. The only CI workflow (`trade.yml`) is a scheduled live-trading runner.
   There was no per-push correctness signal.

### Implemented (branch: claude/lucid-darwin-ahxvL)

- **chore: remove `=0.2.36`, `=0.29.1`, `=2.3.0`** from the repo root
  via three `delete_file` commits.
- **chore: harden `.gitignore`** with `/=*` so the same typo cannot
  re-introduce these files.
- **feat: add `.github/workflows/ci.yml`**, a deliberately lightweight
  CI that does (a) `python -m compileall src tests`, (b) flake8 with
  `--select=E9,F63,F7,F82` (runtime / undefined-name errors only).
  Skips the heavy `pip install -r requirements.txt` step on purpose:
  stable-baselines3 + shap + hmmlearn make the install slow and
  noisy, and the existing `trade.yml` already exercises the full
  runtime on its own schedule with secrets attached.

### Why this was prioritized

- The cruft removal is the highest impact-per-byte change available on
  this repo. It is purely cosmetic in terms of behavior, and it has zero
  risk of breaking trading logic.
- A correctness-signal CI on every push catches the easy class of
  mistakes (syntax errors, broken imports introduced during a rebase)
  that would otherwise only be found at trading-cycle execution time on
  `trade.yml`, possibly with money on the line.

### Evaluated and skipped

- **Full pytest CI.** The `tests/` suite is exercised in README
  instructions, but running it in CI requires installing the heavy
  RL stack (sb3, shap, hmmlearn, alpaca-py). That's a 5+ minute job
  per push and is more failure-prone than valuable on a learning repo.
  Deferred until tests are partitioned into a "fast" subset.
- **Anthropic Claude consensus path.** README mentions LLM consensus
  scoring; auditing whether Claude is wired in alongside the existing
  LLM client requires reading `src/` deeply and is out of scope for a
  low-risk run. Queued.
- **Trim top-level Markdown drift** (`CODEX_FIXES.md`,
  `INDIA_PIPELINE_SUMMARY.md`, `PROFESSIONAL_TRADER_UPGRADE.md`,
  `plan.md`, `pcr.txt`, `run_log.txt`, `pytest_*.txt`). These are
  internal notes and should probably move to `docs/` or be removed,
  but the call belongs to the author rather than the auto-researcher.

### Next-run candidates

1. Audit `src/` for a Claude / Anthropic backend in the LLM consensus
   layer and wire one in if missing (the README advertises it).
2. Split `tests/` into a `fast` subset and run that in CI.
3. Move the top-level `*.md` design notes and `*.txt` run logs out of
   the repo root.
4. Add a `README` badge for the new `ci` workflow.
