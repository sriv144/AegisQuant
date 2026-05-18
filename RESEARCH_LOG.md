# Research Log

A running log of autonomous research-and-development cycles on AegisQuant.
Each entry summarizes the resume-impact score at the start of the run, what
was shipped on the listed branch, what was evaluated and skipped, and the
candidate ideas left for the next pass.

---

## 2026-05-18 — Auto-Researcher v4

**Resume score at start of run:** 90/100 (highest of the 6 target repos)
**Branch:** `claude/lucid-darwin-9dHiZ`

### Implemented (docs / polish bundle)
- `LICENSE` (MIT, with a research-software disclaimer for trading risk).
- `CONTRIBUTING.md` with dev-loop, test, lint, and broker-secrets guidance.
- `CHANGELOG.md` backfilled from main's merge history through May 2026 (LLM analyst,
  GH Actions trading, Alpaca migration, dashboard rebuild, US market support, strategy
  rewrite, broker abstraction, dashboard polish).
- `Makefile` with `install / install-dev / test / lint / format / docker-up / docker-down /
  train / backtest / dashboard / live / clean` targets.
- `.pre-commit-config.yaml`: ruff + ruff-format + standard hygiene hooks.
- `.gitignore` hardened so future commits stop carrying pip version-spec stub files
  (`=0.2.36`, `=0.29.1`, `=2.3.0`), zipped model artifacts (`ppo_*.zip`), pytest log
  dumps (`pytest_*.txt`), and PCR debug dumps that currently live at the repo root.
- GitHub `ISSUE_TEMPLATE/bug_report.md`, `ISSUE_TEMPLATE/feature_request.md`,
  `PULL_REQUEST_TEMPLATE.md` (PR template includes a live-trading safety checklist).

### Why this was prioritized
- AegisQuant is the most feature-complete + most recently active repo (merged through
  2026-05-16). Adding code carries real blast risk; adding **professional polish** is
  pure upside and is exactly what "showcase enhancement mode" calls for.
- The repo root currently carries a handful of files that look unprofessional in a
  recruiter screenshot (pip side-effect stubs, model `.zip` blobs, pytest log dumps).
  The expanded `.gitignore` makes sure they stop coming back.
- Every file in this commit is additive or a clearly safer rewrite of `.gitignore`.
  No runtime code paths change, so the existing tests still apply unchanged.

### Evaluated and skipped (with reasons)
- **Deleting the existing root-level junk** (`=0.2.36`, `pcr.txt`, `pytest_*.txt`,
  `ppo_*.zip`) — the GitHub MCP `push_files` tool only adds/updates; it cannot remove
  files. Cleanup will need a follow-up commit pushed via `git rm` from a worktree.
- **README rewrite** — the current README is marketing-heavy ("transmuting AI weights",
  "punch trades natively"). Worth cleaning up, but a big rewrite needs a careful review
  pass on technical accuracy. Deferred to a focused docs cycle.
- **GitHub Actions CI workflow for the test suite** — the repo already runs a scheduled
  trading workflow; layering a separate pytest workflow needs care to not double-trigger
  on the trading branch. Deferred.
- **New strategies / new broker adapters** — too large for a single cycle, and the
  recent work already added a lot of strategy and broker code.

### Next-run candidates
1. `git rm` the root-level pip side-effect stubs (`=0.2.36`, `=0.29.1`, `=2.3.0`),
   `pcr.txt`, `pytest_*.txt`, and the `ppo_*.zip` binaries; move any kept model
   weights into `model_registry/` (now ignored except for `.gitkeep`).
2. Add `.github/workflows/test.yml` running `make test` on push/PR (non-trading branches).
3. Rewrite `README.md` with sober technical language and a real architecture diagram.
4. Add a `docs/` page with screenshots of the Streamlit dashboard.
5. Add `docs/safety.md` explaining the circuit breakers and risk gates.
6. Add `pyproject.toml` with proper metadata + tool config (ruff, pytest, mypy).
7. Add an OANDA or Interactive Brokers adapter for multi-broker depth.
