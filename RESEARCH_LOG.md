# Research Log

A running ledger of autonomous-improvement passes against this repository.
Each entry records the resume-worthiness score at the start of the run, what
was implemented, what was evaluated and skipped, and what the next pass
should look at.

## 2026-04-26 — Auto-Researcher v4

- Branch: `claude/lucid-darwin-T7V0c`
- Resume score at start of run: **80 / 100**
  - Tech stack prestige: 23 (PPO/SAC + SHAP + HMM + multi-broker)
  - Commit recency: 25 (last meaningful commit 2026-04-17)
  - Feature completeness: 16 (live + paper trade, dashboard, audit trail)
  - Stars + visibility: 5
  - README quality: 11 (good prose, no badges/screens, stray repo artefacts)

### Implemented this run
- `CITATION.cff` so AegisQuant is machine-citable from research papers and
  GitHub's "Cite this repository" widget.
- `CONTRIBUTING.md` describing the three reproducibility / safety /
  auditability invariants and the Conventional Commits workflow.
- `SECURITY.md` describing the private-disclosure path for live-trading,
  circuit-breaker bypass, or credential leakage issues.
- Hardened `.gitignore` so future `pip install foo =1.2.3` typos cannot
  commit literal `=1.2.3` files into the repo root, and so stray pytest
  capture files / large rendered backtest artefacts stop sneaking in.
- `.gitattributes` for cross-OS line-ending normalisation and binary diff
  hints for `.zip` model archives and rendered images.

### Why this was prioritized
A prior `claude/lucid-darwin-wCHAT` pass already shipped LICENSE, CI, and a
research log seed. Re-doing those would duplicate work. The repo's most
visible remaining issues — the `=0.2.36`, `=0.29.1`, `=2.3.0` files in the
root, the lack of a citation file, and the missing contributor / security
docs — are all repository-quality gaps that hurt the GitHub landing page
without requiring any change to the live trading path.

### Evaluated and skipped
- **Deleting the `=0.2.36`, `=0.29.1`, `=2.3.0` pip-typo files** — `push_files`
  cannot remove files atomically with additions and a destructive cleanup
  warrants a dedicated PR-sized commit with a clear `chore: remove pip-typo
  artefacts` subject. Hardening `.gitignore` (this run) prevents recurrence;
  the deletion is queued for next run.
- **Promoting `.zip` model archives to Git LFS** — would change clone
  semantics for everyone with the repo already cloned; needs a coordinated
  migration commit, not a one-shot.
- **README polish (badges, screenshots, table-of-contents)** — the README is
  intentionally heavy on prose; restructuring it should happen alongside the
  CI workflow that already lives on a sibling claude branch so the badge row
  links to a real workflow.
- **Refactoring `pytest_*.txt` reports out of the repo root** — they are
  effectively logs and now ignored going forward; cleanup deferred to the
  same `chore:` PR as the pip-typo files.

### Next-run candidates
1. Dedicated `chore:` commit removing the pip-typo files and the four
   `pytest_*.txt` capture files from history.
2. Add badges (CI status, license, Python version, last-commit) to the top
   of README and a screenshot row of the Streamlit Command Center.
3. Add a `Makefile` wrapping the train / backtest / dashboard commands so
   the README quick-start collapses to `make backtest` / `make dashboard`.
4. Promote `*.zip` model archives in `model_registry/` to Git LFS in a
   coordinated migration commit.
5. Move `INDIA_PIPELINE_SUMMARY.md`, `PROFESSIONAL_TRADER_UPGRADE.md`, and
   `plan.md` under `docs/` so the repo root reads cleanly.
