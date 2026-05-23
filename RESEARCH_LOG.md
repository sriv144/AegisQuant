# Research Log — AegisQuant

Durable memory for the auto-researcher agent. Each run appends an entry
documenting what was implemented, what was deliberately skipped, and what
the next viable improvement looks like. Do not delete prior entries.

---

## 2026-05-23 — Auto-Researcher v4

**Resume score at start of run:** 90 / 100 (top of the portfolio)

**Implemented on branch `claude/lucid-darwin-dmZJS`:**
- Added `.github/workflows/ci.yml` — runs ruff lint (advisory) + pytest on
  every push to `main` and every `claude/**` branch, plus on PRs. The
  existing `trade.yml` only runs live trading cycles on cron, so there
  was no fast PR feedback loop. CI installs `requirements.txt` with
  Python 3.11 and runs the full `tests/` suite with mock-data env vars
  so the broker and LLM clients never reach real services.
- Seeded this `RESEARCH_LOG.md`.

**Why prioritized:** AegisQuant scored highest of the six portfolio repos
on resume-worthiness (RL + LLM consensus + risk-aware quant, freshest
commits, most stars). It already has 18+ pytest files and a documented
audit workflow, but lacked a lint+test CI gate. Adding one is the
highest-value, lowest-risk improvement: it visibly demonstrates testing
discipline to recruiters without touching any trading code paths.

**Evaluated and skipped this run:**
- *Cleanup of stray repo-root garbage files* (`=0.2.36`, `=0.29.1`,
  `=2.3.0`, several `pytest_*.txt` and `run_log.txt`). These look like
  shell artifacts from `pip install pkg=ver` typos. Skipped because
  `push_files` is additive-only — deletions need a separate tool path
  and risk losing referenced log data. Logged for the next run.
- *Refactoring `main_us.py` / `main_india.py`* (~17k+23k lines) to share
  a common runner. Skipped: too large, too easy to break the live
  trading path, no test coverage on the entry points themselves.
- *Removing committed model artifacts* (`ppo_curriculum_SPY.zip`,
  `ppo_portfolio_manager.zip`). Skipped: these may be referenced by
  documentation or example notebooks. Needs owner confirmation before
  rewriting history.
- *Adding Anthropic Claude as a first-class LLM option* (currently only
  OpenAI is wired). Skipped: would be a real feature, but requires deep
  reading of the consensus-scoring module — out of scope for a single
  safe-by-default pass.

**Next-run candidates (in priority order):**
1. Delete the stray `=0.2.36` / `=0.29.1` / `=2.3.0` files and the
   committed `pytest_*.txt` / `run_log.txt` logs (one cleanup commit).
2. Add a `Makefile` with `make test`, `make lint`, `make backtest`,
   `make audit-report` targets so the README's three different commands
   become one-liners.
3. Add a Claude-based alternative to the consensus LLM scorer behind a
   feature flag, with mocked-LLM unit tests.
4. Add an architecture diagram (`docs/architecture.png` or a Mermaid
   block in the README) showing the Phase 0–6 pipeline.
