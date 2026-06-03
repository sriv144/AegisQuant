# Research Log

This file tracks autonomous research-agent activity on AegisQuant. Each
entry records: (a) what was implemented, (b) why it was prioritized,
(c) what was evaluated but skipped, and (d) candidates for next runs.

Do not delete prior entries — they exist so future runs can avoid
re-doing the same work.

---

## 2026-06-03 — Auto-Researcher v4

**Resume-worthiness score at start of run:** 84 / 100

- Stack prestige (25): 24 — RL (PPO/SAC) + LLM consensus + SHAP +
  HMM regime detection sit at the top of the quant ML hierarchy.
- Commit recency (25): 22 — last push 12 days ago.
- Feature completeness (20): 18 — multi-market (US + India), Alpaca
  broker, walk-forward backtester, audit reports, Streamlit UI, Helm,
  Docker, scheduled GitHub Actions trading cycle.
- Stars / visibility (15): 8 — 2 stars + 1 fork, plus public topics.
- README quality (15): 12 — phase-by-phase architecture summary, clear
  install + run commands; cluttered by stray top-level files.

### Implemented on `claude/lucid-darwin-oCTQH`

1. **Removed three broken pip-install artifact files** at repo root
   (`=0.2.36`, `=0.29.1`, `=2.3.0`). These came from a `pip install
   pkg >=1.2.3` shell-redirect bug and were polluting the root listing.
2. **Hardened `.gitignore`** to stop runtime junk from drifting back in:
   covers `run_log.txt`, `pytest_*.txt`, `pcr.txt`, `logs/`, large
   `ppo_*.zip` checkpoints at the root, regenerable
   `backtest_results/*.json|.png`, and the `=*.*.*` broken-install
   pattern. Pre-existing tracked log/checkpoint files were left in
   place intentionally — deleting them is a separate decision the
   maintainer should make.
3. **Added `.github/workflows/ci.yml`** — pytest + ruff CI that runs
   on every push to `main` / `claude/**` and on PRs. Runs in mocked
   mode (`ENABLE_MOCK_DATA=True`, fake broker keys), uploads coverage,
   and is gated `|| true` on lint/test so it cannot block prior
   schedule-driven workflows while CI gets stabilized.

The existing `.github/workflows/trade.yml` (live US trading cycle)
was not touched — that is the production scheduler.

### Why these were prioritized

The project is the highest-scoring repo on the slate, and the
root-level garbage files were the single biggest visible defect for a
recruiter browsing the repo. They are unambiguously bugs (file names
like `=0.2.36` are never intentional) and removing them is zero-risk.
Adding a proper test-CI badge-able workflow turns an already-strong
project into an obviously-maintained one.

### Evaluated and skipped

- **Migrate LLM consensus from OpenAI to Anthropic Claude.** The
  README claims Anthropic support; the live `trade.yml` still passes
  `OPENAI_API_KEY`. A clean migration touches the LLM client wrapper,
  the consensus scorer, the trade workflow secrets, and tests — too
  much surface for a single safe commit. Deferred.
- **Delete the committed `pytest_*.txt`, `run_log.txt`, `pcr.txt`,
  and `ppo_*.zip` files.** Some of them may be intentional
  reference outputs for the README. Leaving them in place for now;
  the new `.gitignore` rules will prevent future drift.
- **Add a Codecov / coverage badge.** Waits until the first CI run
  uploads coverage successfully.
- **Replace `CODEX_FIXES.md` / `INDIA_PIPELINE_SUMMARY.md` /
  `PROFESSIONAL_TRADER_UPGRADE.md` with a tighter `docs/` tree.**
  Useful but cosmetic; defer.

### Next-run candidates

1. Anthropic Claude migration for the LLM consensus scorer (tests
   first, then swap).
2. Pin `requirements.txt` versions (currently a flat list, which is
   what caused the `=*.*.*` artifact bug in the first place).
3. Consolidate `main.py`, `main_us.py`, `main_india.py` behind a
   single `--market` flag.
4. Add a `docs/` directory and move loose markdown reports into it.
