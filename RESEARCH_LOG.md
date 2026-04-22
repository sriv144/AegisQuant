# Research Log

Automated improvement log maintained by Auto-Researcher.
Each run appends a dated entry describing what was implemented, what was skipped, and why.

---

## 2026-04-22 — Auto-Researcher v4

**Resume score at the start of this run:** 85/100 (top-2: PPO/SAC + LLM consensus + SHAP + Alpaca live-trading daemon; HMM regime detection and drawdown circuit breakers are unusually strong resume signals for a single-author quant repo).

**Prior open auto-researcher branch on this repo:** `claude/focused-newton-TXG01` (2026-04-21) added `.github/workflows/ci.yml` and seeded a RESEARCH_LOG. That branch has not merged yet, so the main branch is still missing both the CI workflow and a RESEARCH_LOG — this run only touches concerns that do not collide with that open PR.

**Implemented (branch `claude/lucid-darwin-QFlkH`):**
- `.ruff.toml`: minimal, non-disruptive ruff config (E/F/W/I/UP rules, `line-length = 120`, per-file ignores for `tests/` and `scripts/`, excludes binary model + report dirs, `=*`, and the `pytest_*.txt` snapshot files). Targets py3.11 to match the CI job on the prior branch.
- `.pre-commit-config.yaml`: ruff + ruff-format + core hygiene hooks (trailing-whitespace, end-of-file-fixer, check-yaml, check-toml, large-file guard at 512 KB).
- Tightened `.gitignore`: now blocks accidental pip-install artifacts matching `=*` (the `=0.2.36`, `=0.29.1`, `=2.3.0` files already in the repo are from a stray space in a `pip install foo =1.2.3` command), the committed `pytest_clean.txt` / `pytest_output.txt` / `pytest_run.txt` snapshots, `.ruff_cache/`, `.mypy_cache/`, and common IDE/OS clutter.
- Seeded this `RESEARCH_LOG.md` on main so memory persists after the other claude/* branches merge.

**Why this was prioritized:**
The prior auto-researcher branch queued three tasks: delete the stray `=<version>` files, delete the committed `pytest_*.txt` snapshots, and introduce ruff. Hard-deleting files via the MCP `push_files` tool cannot be done atomically alongside additive changes, so this run addressed the recurrence-prevention layer (`.gitignore` + ruff + pre-commit) in a single clean commit. A follow-up run can safely hard-delete the stray files knowing the gitignore will keep them from coming back.

**Evaluated and skipped this run:**
- Hard-deleting `=0.2.36`, `=0.29.1`, `=2.3.0`, `pytest_clean.txt`, `pytest_output.txt`, `pytest_run.txt` from VCS: requires six separate `delete_file` commits (the MCP `push_files` atomic tool cannot remove files), which conflicts with the preferred atomic-commit style for this run. Re-queued with the hardened `.gitignore` in place.
- Adding a `pyproject.toml`: would collide with the open `claude/focused-newton-TXG01` CI branch if it later adds one. Standalone `.ruff.toml` keeps both branches independently mergeable.
- Nightly `backtest.yml` workflow: too slow + live-market dependent for a PR gate, and the prior branch already owns the `.github/workflows/` directory — kept out of this commit to avoid merge conflicts.
- Switching `.env.example` away from `OPENAI_API_KEY` toward Anthropic-only: touches runtime behavior (`ANTHROPIC_API_KEY` already listed, `OPENAI_API_KEY` is still used by the live LangChain agent workflow). Queued as a scoped follow-up so any provider switch is reviewed in its own PR.

**Next-run candidates:**
- Hard-delete the six stray files above (the new `.gitignore` now prevents recurrence).
- Introduce a `pyproject.toml` unifying build metadata, ruff pointer, and mypy config once the prior CI branch lands.
- Add a nightly `backtest.yml` that runs a short walk-forward on a single ticker and asserts `sharpe > 0` + `max_drawdown < threshold`.
- Wire `.pre-commit-config.yaml` into CI via a dedicated `pre-commit-ci.yml` job so style drift is caught on PRs, not just locally.
- Centralize environment variables shared by `trade.yml` and the new `ci.yml` into `configs/ci.env` (called out on the previous branch's next-run list).
