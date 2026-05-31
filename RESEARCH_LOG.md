# Research Log

Automated improvement log maintained by the auto-researcher agent.

---

## 2026-05-31 — Auto-Researcher v4

**Resume score at start of run:** 89 / 100 (1st of 6 target repos)

### Implemented (branch: `claude/lucid-darwin-D2Gmr`)

- **`.github/workflows/ci.yml`** — dedicated pytest CI workflow that runs on
  every push and PR against `main`. Matrix runs on Python 3.11 and 3.12 and
  forces `ENABLE_MOCK_DATA=True` so brokers are never hit from CI.
- **`LICENSE`** — MIT, dated 2026. The repo previously had no license file,
  which technically made it "all rights reserved" — a blocker for any
  recruiter / collaborator who wanted to use the code.
- **`CONTRIBUTING.md`** — short contributor guide covering safety rules (no
  live broker calls in tests, no real keys), local setup, test commands, and a
  PR checklist.

### Why this was prioritized

- AegisQuant was the highest-scoring repo (89/100) but was missing the
  basic OSS hygiene that recruiters scan for: LICENSE, CONTRIBUTING, and a
  visible test-CI badge target.
- The existing `.github/workflows/trade.yml` is a **cron-driven live trading
  runner**, not a test workflow. Without a separate CI job, a regression in
  the test suite could ship unnoticed and only break at 09:35 ET.
- All three additions are additive and cannot break the existing trading
  pipeline.

### Evaluated and skipped

- **Cleaning up root junk files** (`=0.2.36`, `=0.29.1`, `=2.3.0`, several
  `pytest_*.txt` logs, `pcr.txt`, `run_log.txt`). These look like accidental
  artifacts from `pip install foo>=...` shell redirection. They are clear
  cleanup wins but require `delete_file` per file rather than the atomic
  `push_files` flow, so they are tracked for the next run.
- **Adding a Streamlit screenshot to the README.** Would require launching the
  dashboard in a sandbox to capture an image — too heavy for this run.
- **Migrating the LangChain `langchain-openai` dep to `langchain-anthropic`.**
  The README already mentions "Anthropic keys"; need to verify which code
  paths actually use OpenAI vs Claude before swapping. Deferred.

### Next-run candidates

1. Delete the root junk files (`=0.2.36`, `=0.29.1`, `=2.3.0`, the
   `pytest_*.txt` logs, `pcr.txt`, `run_log.txt`).
2. Add a `.github/workflows/lint.yml` running `ruff` + `black --check`.
3. Audit `langchain-openai` usage and migrate any agent calls to Claude.
4. Add a `docs/` directory with the existing `CODEX_FIXES.md` /
   `PROFESSIONAL_TRADER_UPGRADE.md` / `INDIA_PIPELINE_SUMMARY.md` moved out of
   the root so the README isn't drowned.
5. Embed a sample `audit_*_report.md` screenshot in the README so visitors
   immediately see the honest-reporting feature.
