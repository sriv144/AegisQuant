# Research Log

Maintained by the Auto-Researcher passes. Each entry records what was looked
at, what was implemented, and what was deferred so future runs do not redo
the same work.

## 2026-06-08 — Auto-Researcher v4

**Resume-worthiness score (start of run):** ~78 / 100
- Tech stack prestige (25): 25 — RL (PPO/SAC) + LLM consensus + multi-asset
  portfolio optimization is top-tier for an ML/quant resume.
- Commit recency (25): 24 — last push 2026-05-22, within the 30-day window.
- Feature completeness (20): 16 — walk-forward backtester, Streamlit
  dashboard, Alpaca live adapter, audit reporting all present.
- Stars + visibility (15): 4 — 2 stars; visibility is the main weak point.
- README quality (15): 9 — covers install + run paths but lacks an
  architecture diagram and badges; clutter at repo root distracts.

**Implemented on branch `claude/lucid-darwin-fX5uP`:**
- `.github/workflows/ci.yml` — ruff lint (advisory) + py_compile on tracked
  `.py` files. Gives the repo a visible CI signal without installing the heavy
  RL/ML dependency tree.
- `SECURITY.md` — vulnerability reporting policy plus broker-specific
  trading-safety guidance (paper-trading default, drawdown breaker, key
  rotation).
- `docs/architecture.md` — Mermaid diagrams of the 6-phase pipeline and the
  risk-gate decision tree. Mermaid renders inline on GitHub, so this gives a
  reviewer a one-glance view of the system.
- `RESEARCH_LOG.md` — this file.

**Why prioritized:** AegisQuant is the highest-scoring repo in the portfolio.
The weak spots are visibility and "first-30-seconds" comprehension, not
feature work. Hygiene + a diagram pay off disproportionately for a recruiter
landing on the repo. Functional code was not touched, so blast radius is zero.

**Evaluated and skipped this run:**
- Deleting repo-root clutter (`=0.2.36`, `=0.29.1`, `=2.3.0`,
  `pytest_*.txt`, `run_log.txt`, `pcr.txt`, multiple `*_SUMMARY.md` files).
  These look like stale dev artifacts and removing them would clean the
  landing page significantly, but doing so safely requires confirming each is
  not referenced by tooling. Queued for the next pass with explicit owner
  confirmation.
- Removing committed `.zip` model weights from history. High resume value but
  history-rewriting; needs the owner's call.
- Migrating any remaining OpenAI integration to Anthropic. Needs a code audit
  to confirm no behaviour regression in the LLM consensus scorer.

**Next-run candidates:**
- Add a Streamlit dashboard screenshot to the README (currently text-only).
- Promote `docs/architecture.md` content into a `## Architecture` section of
  the README so the diagrams render on the landing page.
- Add `pip-audit` to the CI workflow once dependency pinning is reviewed.
- Wire a `pytest -q tests/` step into CI behind a `[ci-tests]` opt-in marker
  once the test suite is confirmed green in a clean environment.
