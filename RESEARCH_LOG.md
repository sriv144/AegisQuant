# Research Log

This log tracks automated research and improvement runs by the
auto-researcher agent. Each entry captures the project's resume-worthiness
score at run start, what was implemented (with branch), why it was
prioritized, what was evaluated and skipped, and the next-run candidates.

---

## 2026-05-13 — Auto-Researcher v4

**Resume score at start of run:** 90 / 100
- Tech stack prestige: 24/25 (PPO + SAC + LLM consensus + Alpaca/Groww)
- Commit recency: 25/25 (active within the last 48 hours)
- Feature completeness: 19/20 (US + India pipelines, walk-forward, dashboard)
- Stars + visibility: 7/15 (1 star, 1 fork)
- README quality: 15/15 (architecture phases, install, run, test)

**Implemented (branch: `claude/lucid-darwin-HHSYf`):**
- `.github/workflows/ci.yml` — GitHub Actions workflow that runs the
  existing pytest suite on Python 3.11 with `ENABLE_MOCK_DATA=True` and
  `ENABLE_BROKER_EXECUTION=False` so no broker credentials are required.
  This complements the existing `trade.yml` scheduler workflow and gives
  PR-level signal on the 8 test files under `tests/`.
- `RESEARCH_LOG.md` — this file (seeded for future runs).

**Why this was prioritized:**
- The repo already ships a scheduled trader workflow (`trade.yml`) but no
  pre-merge test workflow, so PRs land on `main` without any automated
  signal that `tests/test_walk_forward.py`, `test_circuit_breakers.py`,
  `test_monte_carlo.py`, etc. still pass.
- Tests are designed for offline mock data, so adding CI is low-risk and
  needs zero new secrets.
- A green CI badge on a quant trading repo is one of the highest
  recruiter-trust signals available.

**Evaluated and skipped this run:**
- Deleting the stray `=0.2.36`, `=0.29.1`, `=2.3.0` files at the repo
  root (artifacts from `pip install >=X.Y.Z` shell-quoting bugs).
  Skipped because `push_files` cannot delete; needs a follow-up commit
  via the contents API or local `git rm`.
- Unifying `main.py` and `main_india.py` behind a `--market {us,india}`
  flag. Higher impact but non-trivial; deferred to avoid risk of
  breaking the live India scheduler that fires daily via `trade.yml`.
- Anthropic Claude consensus path — `.env.example` already reserves
  `ANTHROPIC_API_KEY`; wiring an actual `langchain-anthropic` agent is a
  good next-run feature.
- Cleaning the committed `pytest_*.txt` log files. Not destructive but
  noisy; bundle with the stray-file cleanup commit later.

**Next-run candidates (ranked):**
1. Delete `=0.2.36` / `=0.29.1` / `=2.3.0` / `pytest_*.txt` repo-root
   artifacts in one cleanup commit.
2. Add an Anthropic Claude consensus voter alongside the OpenAI path,
   gated on `ANTHROPIC_API_KEY`.
3. Add a SHAP feature-importance screenshot to the README.
4. Unify `main.py` + `main_india.py` behind a single CLI entry with
   a `--market` flag.
5. Wire a nightly `walk_forward.py --mc-sims 1000` smoke job that
   uploads the model artifact, gated by `workflow_dispatch`.
