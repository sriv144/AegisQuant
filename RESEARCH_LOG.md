# Research Log

This file records autonomous improvement runs performed by Auto-Researcher.
Each entry captures what was evaluated, what was implemented, and what was
skipped, so that future runs do not repeat the same work.

## 2026-04-24 — Auto-Researcher v4

**Resume-worthiness score at start of run:** ~81 / 100

Strong technical positioning (PPO/SAC RL + LLM consensus + regime detection
+ SHAP attribution + Alpaca/Groww execution), solid README, live India
pipeline wired up through `.github/workflows/trade.yml`, and a populated
`tests/` directory. The visible gap was that there was no CI signal proving
those tests pass — the only workflow on the repo schedules a live trading
job, not correctness checks.

### Implemented on branch `claude/lucid-darwin-wCHAT`

- **`.github/workflows/tests.yml`.** Runs `pytest tests/ -q` on every push
  and PR to `main`, Python 3.11, ubuntu-latest.
  - Forces `ENABLE_MOCK_DATA=True` and `ENABLE_BROKER_EXECUTION=False` so
    the suite never reaches for `yfinance` / broker APIs mid-CI.
  - Pip cache keyed on `requirements.txt`.
  - Cancel-in-progress concurrency and a 20-minute timeout guard.
- **MIT LICENSE** added — the repo was previously unlicensed.
- **Seeded this `RESEARCH_LOG.md`.**

### Why prioritized over alternatives

The existing `trade.yml` is a production heartbeat and should not double as
test infrastructure; they have different triggers, different secrets, and
different failure semantics. Splitting off a dedicated `tests.yml` is pure
upside and creates the green-badge story recruiters look for. Zero risk to
the live India trading loop.

### Evaluated and skipped

- **Root cleanup of stray files** (`=0.2.36`, `=0.29.1`, `=2.3.0`, plus the
  `pytest_*.txt` logs and `pcr.txt`). These look like stderr redirected to
  files during `pip install -r requirements.txt '=X.Y'` — noise in the
  repo root. Deferred this run because it belongs in its own `chore:` commit
  and needs `git rm`, which is outside the additive `push_files` path.
  Queued for the next run via the `delete_file` tool.
- **README architecture diagram.** The current README is already detailed
  and readable. A diagram is a nice-to-have, not a friction fix.
- **Lint job.** Same reasoning as embodied-skill-composer: would surface
  pre-existing findings as spurious CI red. Defer to a targeted run.
- **Paper-trading smoke test.** Requires network to Alpaca / Groww; would
  need mocked broker shims. Bigger than a CI wiring commit.

### Next-run candidates

1. `chore: remove stray \"=X.Y.Z\" pip-stderr files and pytest_*.txt logs
   from the repo root` — use `delete_file` for each, then amend `.gitignore`.
2. Add a `ruff` job, starting from a `ruff format` baseline commit.
3. Split `trade.yml` into a `india-trade.yml` / `us-trade.yml` pair so the
   two pipelines can evolve independently.
4. Collapse `PROFESSIONAL_TRADER_UPGRADE.md` + `INDIA_PIPELINE_SUMMARY.md`
   + `plan.md` into a coherent `docs/` tree and link them from the README.
5. Add a CodeQL workflow for security scanning (Python RL + broker APIs
   handling credentials is the exact surface CodeQL catches well).
