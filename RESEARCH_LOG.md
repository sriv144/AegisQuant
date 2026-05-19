# Research Log

A running ledger of autonomous-improvement passes against this repository.
Each entry records the resume-worthiness score at the start of the run,
what was implemented, what was evaluated and skipped, and what the next
pass should look at.

## 2026-05-19 — Auto-Researcher v4

**Resume-worthiness score at start of run: 83 / 100** (rank 2 of 6).

| Signal | Score |
| --- | --- |
| Tech stack prestige (RL PPO/SAC + LLM consensus + SHAP + HMM regime + live broker) | 24 / 25 |
| Commit recency (updated 2026-05-16) | 25 / 25 |
| Feature completeness (9 strategies, dual-mode trading, dashboard SPA, audit trail) | 18 / 20 |
| Stars + visibility (1 star + 1 fork) | 5 / 15 |
| README quality (Phase 0–6 architecture, install, run, test) | 11 / 15 |

### Implemented this run (branch: `claude/lucid-darwin-lpd1a`)

No code or config changes. This commit only seeds `RESEARCH_LOG.md` on
the pre-assigned branch so the next run has continuity.

### Why no implementation this run

AegisQuant has an unusually deep stack of unmerged auto-researcher
branches that already covers every safe additive change available:

- `wCHAT` (2026-04-24) — `.github/workflows/tests.yml` + MIT LICENSE.
- `QFlkH` — ruff + pre-commit + hardened `.gitignore`.
- `ONkrp` (2026-04-23) — `docs/ARCHITECTURE.md`.
- `lOIj4` (2026-04-25) — `docs/BACKTEST_REPORT_TEMPLATE.md`.
- `T7V0c` (2026-04-26) — CITATION.cff + CONTRIBUTING.md + SECURITY.md
  + `.gitattributes` + further `.gitignore` hardening.

The two remaining classes of work both fall outside a safe-by-default
autonomous run:

1. **Destructive cleanup.** Removing the stray `=0.2.36`, `=0.29.1`,
   `=2.3.0`, `pcr.txt`, and `pytest_*.txt` files from the repo root
   requires `delete_file` calls, not the additive `push_files` path,
   and warrants its own `chore: remove pip-stderr artefacts` PR with
   the changes laid out clearly for review.
2. **Code-touching changes against a live-trading repo.** Anything that
   modifies `src/` carries reproducibility risk against the model
   checkpoints already in `model_registry/` and the live-trading
   `trade.yml` cron — not appropriate for an unattended commit.

Token budget this run was spent on the three repos with clear,
unblocked next-run candidates:

- `FinLens` — Anthropic Claude provider option behind `LLM_PROVIDER`
  (`claude/admiring-davinci-lpd1a`).
- `Autonomous-SRE-Agent` — helm lint + docker compose validation
  workflow (`claude/fervent-edison-lpd1a`).
- `salesnuero` — first CI workflow on the repo, backend compile +
  frontend build (`claude/compassionate-keller-lpd1a`).

### Evaluated and skipped

- **`chore:` cleanup of `=0.2.36`, `=0.29.1`, `=2.3.0`, `pcr.txt`,
  `pytest_*.txt`.** Highest visible repository-quality lift available,
  but needs `delete_file` per path — deferred to a dedicated PR.
- **README badges row + Streamlit Command Center screenshot.** The
  badges need the `wCHAT` CI workflow to be on `main` first (otherwise
  they link to a workflow that does not exist), and the screenshot
  needs a live dashboard run.
- **Top-level `Makefile` wrapping `train` / `backtest` / `dashboard`.**
  Real value, but `wCHAT` already wired CI and `QFlkH` already wired
  ruff + pre-commit — a Makefile is best landed alongside the
  documentation reorg, not in isolation.
- **Promoting `.zip` model archives in `model_registry/` to Git LFS.**
  Would change clone semantics for every existing checkout; needs a
  coordinated migration commit and owner sign-off.
- **Moving `INDIA_PIPELINE_SUMMARY.md`, `PROFESSIONAL_TRADER_UPGRADE.md`,
  and `plan.md` under `docs/`.** Atomic file moves require the same
  `delete_file` path as the cleanup above; queue with that PR.
- **CodeQL workflow.** Strong-fit for a Python live-trading repo handling
  broker credentials, but adding a security-scanning workflow that
  surfaces existing findings on the first run is best done with the
  owner's awareness rather than as a silent autonomous commit.

### Next-run candidates (priority order)

1. Dedicated `chore: remove pip-typo and pytest-log artefacts` PR using
   `delete_file` for each of `=0.2.36`, `=0.29.1`, `=2.3.0`, `pcr.txt`,
   `pytest_*.txt`; pair with a `.gitignore` reinforcement.
2. After `wCHAT` (CI) lands on `main`, add CI / license / Python / last
   commit badges to the top of README, plus a screenshot row of the
   Streamlit Command Center.
3. Top-level `Makefile` wrapping `train` / `backtest` / `dashboard`
   commands so the README quick-start collapses to `make backtest` /
   `make dashboard`.
4. Reorganise `INDIA_PIPELINE_SUMMARY.md`, `PROFESSIONAL_TRADER_UPGRADE.md`,
   `plan.md` under `docs/` so the repo root reads cleanly; link them
   from the README + `docs/ARCHITECTURE.md`.
5. Add a CodeQL workflow for security scanning (Python + broker APIs
   handling credentials — the exact surface CodeQL catches well).

### Prior research-log context

Previous runs on unmerged `claude/lucid-darwin-*` /
`claude/focused-newton-*` branches (most recent first, none merged to
`main`):

- `T7V0c` (2026-04-26) — CITATION + CONTRIBUTING + SECURITY +
  `.gitignore` / `.gitattributes` hardening.
- `lOIj4` (2026-04-25) — `docs/BACKTEST_REPORT_TEMPLATE.md`.
- `ONkrp` (2026-04-23) — `docs/ARCHITECTURE.md`.
- `wCHAT` (2026-04-24) — `.github/workflows/tests.yml` + MIT LICENSE.
- `QFlkH` — ruff + pre-commit + hardened `.gitignore`.
- `focused-newton-KSTnM`, `focused-newton-TXG01` — earlier rounds.
