# Research Log

Autonomous research and improvement history for AegisQuant.

## 2026-04-29 — Auto-Researcher v4

**Resume score at start of run:** 82 / 100.

**Branch:** `claude/lucid-darwin-dI8Uk`

### What was implemented
- Added an MIT `LICENSE` so the project is unambiguously open-source.
- Hardened `.gitignore` to cover three categories of accidental commits that
  are visible in the current `main` tree:
  1. `pytest_run.txt`, `pytest_output.txt`, `pytest_clean.txt` — local pytest
     captures that should not live in version control.
  2. `=*` glob — stray empty files left behind when a previous shell ran
     `pip install <pkg> =0.2.36` (a typo that captures the version constraint
     into a file named `=0.2.36`). Three such files are still present at the
     repo root: `=0.2.36`, `=0.29.1`, `=2.3.0`.
  3. Standard Python build/cache directories.
- Followed up with delete commits to remove the three `=*` artifact files
  from the working tree on this branch (kept as separate commits so the
  cleanup is auditable in `git log`).
- Seeded this `RESEARCH_LOG.md`.

### Why this was prioritized
The codebase already has substantial RL/PPO + SHAP + HMM regime detection +
Alpaca execution depth (resume score 82). Visible polish issues, not new
features, were the cheapest resume-relevant lift:
- A reviewer skimming the repo root sees `=0.2.36`, `=2.3.0`, `=0.29.1`,
  `pcr.txt`, `pytest_run.txt`, three `*.zip` model artifacts, and immediately
  downgrades their estimate of code hygiene. Cleaning that up is high impact,
  low risk.
- No license file is a hard blocker for reuse.
- The existing `.github/workflows/trade.yml` is a deploy / live-trade hook,
  not a unit-test workflow. Adding a generic `pytest` workflow was considered
  but skipped because the test suite imports `alpaca-py`, `stable-baselines3`,
  `shap`, and `hmmlearn`, which combine to slow CI dramatically and need
  careful environment pinning.

### Evaluated and skipped
- **README rewrite:** the current README has some flowery phrasing
  (“transmuting AI weights”, “punch trades natively”) that is mildly off in a
  finance context, but it is also accurate about phase structure. A rewrite
  was deferred to avoid losing technical content.
- **Removing the `*.zip` model artifacts** from the repo root: these are
  pre-trained PPO checkpoints that scripts may load directly. Removing them
  could break `walk_forward.py` or `main.py` for anyone cloning fresh, so
  they were left in place pending a `model_registry/` migration.
- **Adding a unit-test CI workflow:** deferred for the dependency-weight
  reasons above.
- **Removing `pcr.txt` and `plan.md` from root:** unclear whether they are
  intentional notes or accidental commits. Left as-is.

### Next-run candidates
- Migrate `ppo_curriculum_SPY.zip` and `ppo_portfolio_manager.zip` into
  `model_registry/` and update loaders.
- Add a slim `pytest` CI workflow that pins SB3 + gymnasium versions to
  match `requirements.txt`.
- README pass: keep the technical claims, rewrite the marketing-flavored
  phrasing to match an institutional-quant tone.
- Consider splitting `INDIA_PIPELINE_SUMMARY.md` and
  `PROFESSIONAL_TRADER_UPGRADE.md` into a `docs/` directory.
