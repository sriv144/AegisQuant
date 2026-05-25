# Research Log

A running record of auto-researcher passes against this repo.

## 2026-05-25 -- Auto-Researcher v4

**Resume score at start:** 87/100. Strong RL + LLM-consensus trading
pipeline, recent commits, real walk-forward + audit reporting tooling,
helm chart, Dockerfile. Loses points on repo hygiene: garbage files in
the root (`=0.2.36`, `=0.29.1`, `=2.3.0` from a misformatted
`pip install pkg=version` redirect), multiple `pytest_*.txt` and
`run_log.txt` snapshots committed alongside source, no MIT license, and
the only CI workflow runs the live trading cycle (too heavy for PR
validation, depends on broker secrets).

**Implemented on branch `claude/lucid-darwin-FVAfQ`:**

- Added `.github/workflows/tests.yml`: dedicated pytest workflow on push
  + PR. Sets `ENABLE_MOCK_DATA=True` and dummy broker keys so the
  existing 17 test files run hermetically without hitting Alpaca.
- Expanded `.gitignore` to cover the recurring leaks: `pytest_*.txt`,
  `run_log.txt`, `logs/`, large `*.zip` checkpoints, and the `=*` glob
  that catches files named like `=0.2.36` from `pip install pkg=ver`
  shell typos.
- Added MIT `LICENSE`.
- Seeded this `RESEARCH_LOG.md`.

**Why this was prioritized:** Repo hygiene was the only thing dragging
an otherwise strong project's resume score below the 90 line. A CI badge
on the README + a proper license is what most reviewers look for in the
first 10 seconds of scanning a GitHub profile.

**Evaluated and skipped:**

- Deleting the existing junk files (`=0.2.36`, `=0.29.1`, `=2.3.0`,
  `pcr.txt`, `pytest_*.txt`, `run_log.txt`). Each deletion via the
  GitHub MCP delete_file tool is a separate commit and would balloon the
  PR. Will land in a dedicated follow-up pass with one commit per
  deletion. The new `.gitignore` rules prevent recurrence.
- Refactoring `main_us.py` / `main_india.py` / `main.py` into a single
  market-parameterised entrypoint. The existing trade.yml workflow
  invokes `main_us.py` directly, so refactoring touches deployment as
  well. Out of scope for a low-risk pass.
- Adding ruff/mypy: would generate many findings across the 50+ source
  files and is better staged as its own audit.

**Next-run candidates:**

- One-commit-per-file cleanup pass to remove the root-level junk.
- README upgrade: add CI badge, architecture diagram (Mermaid), and
  performance numbers from the latest walk-forward audit report.
- Optional: ruff + black pre-commit hooks gated by CI.
