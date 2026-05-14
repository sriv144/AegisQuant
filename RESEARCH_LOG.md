# Research Log

This file tracks autonomous research and improvement runs against this
repository.

## 2026-04-27 — Auto-Researcher v4

**Resume score at start of run:** 79 / 100 — top 2 of 6 across the portfolio.

**Branch:** `claude/lucid-darwin-snPHW`.

### Implemented

- **Cleanup:** removed three accidentally-committed files at the repo root —
  `=0.2.36`, `=0.29.1`, `=2.3.0`. These were artifacts from a malformed
  `pip install package =version` invocation.
- **CI:** added `.github/workflows/tests.yml` that installs `requirements.txt`
  and runs the existing pytest suite on every push and pull request.
- **README polish:** rewrote the README to drop unnecessary purple prose and
  added an architecture table mapping each phase to its source path.
- Seeded this `RESEARCH_LOG.md`.

### Next-run candidates

- Move committed model zips to a release artifact or git-LFS and gitignore
  them.
- Remove committed `pytest_clean.txt`, `pytest_run.txt`, `pytest_output.txt`.
- Bump pinned versions of `langgraph`, `langchain-openai`, `pydantic`.
- Add a SHAP plot to the README under "Explainability".

## 2026-05-14 — Auto-Researcher v4

**Resume score at start of run:** 79 / 100 — still top 2 of 6. Note that the
`main` branch SHA was updated today (2026-05-14), so the maintainer is
actively iterating; this is the wrong moment to push a large parallel
refactor on a new claude branch.

**Branch:** `claude/lucid-darwin-5vVWf`.

### Implemented

No code changes this run. This commit only updates the research log to
preserve memory continuity.

### Why no implementation this run

The prior `claude/lucid-darwin-snPHW` branch already shipped the three
highest-leverage wins (`=...` file cleanup, CI workflow, README polish) and
is still waiting to merge into `main`. The remaining 2026-04-27 next-run
candidates are not safe one-shots from this agent:

- **Removing `pytest_clean.txt` / `pytest_run.txt` / `pytest_output.txt`:**
  trivially safe in principle, but doing it on a fresh branch off `main`
  would conflict with the prior unmerged cleanup branch and create a
  three-way merge headache for the maintainer. Better to either rebase
  `snPHW` first or batch all cleanup in a single follow-up branch from a
  freshly-merged `main`.
- **Model zip migration to git-LFS or release artifacts:** changes how the
  training pipeline loads weights at runtime; a one-shot autonomous push
  without a smoke-test pass risks breaking inference. Needs a focused PR
  with explicit testing.
- **Bumping `langgraph` / `langchain-openai` / `pydantic`:** very old pinned
  versions; upgrading without runtime testing is risky.
- **SHAP plot in README:** requires running the existing dashboard against
  a saved model to capture a real plot. Cannot be generated inside this
  autonomous run.

### Next-run candidates

After `claude/lucid-darwin-snPHW` merges to `main`:

1. Single "repo cleanup" branch that removes `pytest_clean.txt`,
   `pytest_run.txt`, `pytest_output.txt`, and adds matching ignores to
   `.gitignore` so CI logs don't get committed again.
2. Move `ppo_curriculum_SPY.zip` and `ppo_portfolio_manager.zip` into a
   GitHub release tagged `v0.1.0-models` (or git-LFS), update the
   model-loading code to download on first use, and gitignore the local
   paths.
3. Capture a SHAP plot from the existing dashboard, commit under
   `docs/explainability/`, and embed in the README.
4. Smoke-test version bumps for `langgraph`, `langchain-openai`,
   `pydantic` on a throwaway branch before pinning the bumped versions.
