# Research Log

This file tracks autonomous research and improvement runs against this
repository.

## 2026-04-27 — Auto-Researcher v4

**Resume score at start of run:** 79 / 100 — top 2 of 6 across the portfolio.

**Branch:** `claude/lucid-darwin-snPHW`.

### Implemented

- **Cleanup:** removed three accidentally-committed files at the repo root —
  `=0.2.36`, `=0.29.1`, `=2.3.0`. These were artifacts from a malformed
  `pip install package =version` invocation (the shell created files named
  after the bare version specifiers). `=0.29.1` had captured the full pip
  install log; the other two were empty. None had any functional value, and
  they made the project look unfinished on the GitHub file browser.
- **CI:** added `.github/workflows/tests.yml` that installs `requirements.txt`
  and runs the existing pytest suite (walk-forward, regime detector,
  multi-asset env, cost model, circuit breakers, runtime safety) on every push
  and pull request. The existing `trade.yml` workflow is unchanged.
- **README polish:** rewrote the README to drop unnecessary purple prose
  ("transmuting", "punch trades natively", "deep Streamlit projecting
  continuous SHAP permutations") and added an architecture table mapping each
  phase to its source path. The actual command-line surface is preserved
  verbatim so muscle memory still works.
- **Seeded this `RESEARCH_LOG.md`.**

### Why this was prioritized

AegisQuant has the strongest technical depth of the financial-stack repos
(RL + HMM + SHAP + walk-forward + execution layer). Three high-leverage,
low-risk wins were available in one pass: garbage cleanup, CI, and a tighter
README. The cleanup matters most — broken filenames at the repo root are the
first thing a recruiter sees on the file listing.

### Evaluated and skipped

- **Bumping `langgraph==0.0.26`:** very old pinned version; upgrading is risky
  without runtime testing and was deferred.
- **Replacing committed model zips (`ppo_curriculum_SPY.zip`,
  `ppo_portfolio_manager.zip`):** higher-risk repo surgery; should move to
  `model_registry/` releases or git-LFS in a dedicated PR.
- **Removing committed `pytest_*.txt` log files:** worth doing, but skipped to
  keep this diff focused on the highest-impact wins.

### Next-run candidates

- Move committed model zips to a release artifact or git-LFS and gitignore
  them.
- Remove committed `pytest_clean.txt`, `pytest_run.txt`, `pytest_output.txt`;
  have CI publish equivalent artifacts instead.
- Bump pinned versions of `langgraph`, `langchain-openai`, `pydantic` to
  current minors with a smoke-test pass.
- Add a SHAP plot to the README under "Explainability" so the
  attribution story is visible without running the dashboard.
