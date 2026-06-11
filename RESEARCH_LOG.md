# Research Log

Living record of automated improvements made by the Auto-Researcher agent.

## 2026-06-11 — Auto-Researcher v4

**Resume score at start of run:** 82 / 100
- Tech prestige (RL + finance + Streamlit + Anthropic): 24/25
- Recency (updated 2026-05-22, within 30 days): 22/25
- Feature completeness (multi-asset PPO + walk-forward + UI + live trading): 17/20
- Stars (2): 5/15
- README quality (strong architecture pitch, no badges, garbage files visible in root): 14/15

### Implemented on `claude/lucid-darwin-07c8b9`
- **docs: README badges + tagline** — added Python / RL / Streamlit / pytest / license shields and a one-line tagline so the repo page reads as production-grade at first glance.
- **chore: harden .gitignore** — added patterns for `=*` (misquoted-pip-install artifacts), root `ppo_*.zip` (committed model bundles), `pytest_*.txt`, `run_log.txt`, `pcr.txt`, and `*.jsonl` history files.
- **chore: delete vestigial root files** — separate commits remove `=0.2.36`, `=0.29.1`, `=2.3.0` which were accidentally created when someone ran `pip install <pkg> '=0.x.y'` in a shell that interpreted the `>` redirection.

### Why prioritized
- AegisQuant scored highest on tech prestige and recency. The three `=0.x.y` files visible in the GitHub file listing materially hurt recruiter impression of an otherwise strong RL+finance project.
- README already describes architecture clearly; lightweight polish (badges + headline) is the highest impact-per-risk move.

### Evaluated and skipped
- **Add `ci.yml` workflow** — repo already has `.github/workflows/trade.yml`; adding a CI matrix without coordinating with the existing scheduled workflow risked overlap. Logged for next run.
- **Anthropic LLM consensus scoring wiring** — README mentions LLM consensus but `src/` does not show the integration. Touching live-trading code without integration tests is unsafe for a one-shot.
- **Removing committed `ppo_*.zip` model bundles** — these are referenced by README quick-start and may be the only artifact a reader can run inference against. Left in place; gitignore now blocks future commits.
- **Compacting top-level docs (`CODEX_FIXES.md`, `INDIA_PIPELINE_SUMMARY.md`, `PROFESSIONAL_TRADER_UPGRADE.md`, `plan.md`)** — content overlap suggests cleanup is warranted, but each may have audit value. Punted to next run.

### Next-run candidates
- Add `.github/workflows/ci.yml` (ruff + pytest collect-only) alongside the existing trade workflow.
- Wire Anthropic Claude into the SHAP consensus scoring path with offline fixtures.
- Move `ppo_*.zip` model bundles into `model_registry/` and update README quick-start.
- Add a Streamlit dashboard screenshot to the README.
- Compact the top-level loose markdown docs into a single `docs/` tree.
