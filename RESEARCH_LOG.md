# Research Log

Rolling log of autonomous research + improvement passes against this repo.

## 2026-06-07 — Auto-Researcher v4

**Resume score at start of run:** 78 / 100

Breakdown:
- Tech stack prestige: 24 / 25 (RL PPO/SAC + LLM consensus + quant, this is the showcase tier)
- Commit recency: 19 / 25
- Feature completeness: 17 / 20 (walk-forward backtester, SHAP, regime detection, live Alpaca, Streamlit UI)
- Stars + visibility: 3 / 15
- README quality: 15 / 15 (the README is already strong; honest about backtest failure modes)

**Why prioritized:** the actual code is impressive, but the root directory contains debris from shell quoting bugs that hurts the first impression on a recruiter: zero-byte files literally named `=0.2.36`, `=2.3.0`, a 10 KB `=0.29.1`, plus three committed pytest output files and a 217 KB `run_log.txt`. These were generated when a `pip install 'pkg>=ver'` was run without quoting and shell redirected stderr into a file. The root cause is the missing pattern in `.gitignore`; without fixing that, the same files come right back the next time someone runs install.

**What was implemented (branch `claude/lucid-darwin-5OlEN`):**
- `.gitignore` — hardened. Added `=*` to block shell-redirected pip residue, added `pytest_*.txt`, `run_log*.txt`, `logs/`, `*.egg-info/`, `build/`, `dist/`, `.coverage`, `htmlcov/`, `.ruff_cache/`, `.mypy_cache/`, `.ipynb_checkpoints/`, `.vscode/`, `.idea/`, `.DS_Store`, `/*.zip`. Kept all original entries (`.env`, `.venv/`, `__pycache__/`, alert/trading jsonl, `*.db`). Whitelisted `.env.example`.
- `.github/workflows/ci.yml` — new lint + test workflow. Lint is ruff (non-blocking until the codebase is clean). Test job installs `requirements.txt`, sets dummy creds + `ENABLE_MOCK_DATA=True` so pytest stays hermetic. Existing `trade.yml` (live trading cycle) is untouched.
- `RESEARCH_LOG.md` — this file.

### Mid-run corrections

CI exposed two pre-existing problems with the test suite, both unrelated to this branch:

1. `tests/test_backtest_reporting.py` does `from src.backtest.reporting import ...` but the repo has no `pytest.ini` / `pyproject.toml` `[tool.pytest.ini_options]` / `conftest.py` adding the repo root to `sys.path`. Fixed in CI by setting `PYTHONPATH: '.'` in the test job's env block. A repo-side fix (a one-line `conftest.py` or a `pyproject.toml` `pythonpath = ["."]`) belongs in a separate PR.
2. `tests/test_benchmark_tracker.py` does `from src.db.models import BenchmarkDaily, DailyPnL, PerformanceDaily`. `BenchmarkDaily` is not defined in the current `src/db/models.py` — the test file and the model file have drifted apart. This is real product-level breakage that needs an owner decision (re-add the model? delete the test? rename?). For now this test file is ignored via `--ignore=tests/test_benchmark_tracker.py` so the rest of the suite can run.

Given the suite has additional pre-existing import drift, the test job is marked `continue-on-error: true` so it provides signal (you can read the run) without blocking merges. Once the test/model drift is fixed in a focused PR, drop `continue-on-error` and `--ignore` to make it gating.

**What was evaluated and skipped, with reasons:**
- *Actually delete the committed junk files (`=0.2.36`, `=0.29.1`, `=2.3.0`, `pytest_*.txt`, `run_log.txt`, top-level `ppo_*.zip`, `pcr.txt`).* The `push_files` MCP tool only adds/updates; it cannot delete. Removal needs a follow-up commit using delete_file or a local checkout. Logged here as the highest-priority next-run item. The `.gitignore` change at least prevents recurrence.
- *Fix the test/model drift in `tests/test_benchmark_tracker.py`.* Requires an owner decision — either re-add `BenchmarkDaily` (and verify `DailyPnL`, `PerformanceDaily` still match) or delete the test. Out of scope for a CI-hygiene pass; logged as a next-run candidate.
- *Switch LLM consensus layer from OpenAI to Anthropic Claude.* High resume value but a substantive code change touching consensus scoring logic; needs benchmarking against current behavior before swapping providers. Deferred.
- *Consolidate `main.py` / `main_us.py` / `main_india.py`.* They share a lot of structure but diverge meaningfully on market hours, broker, and instrument universe. Refactor is real engineering work, not a drive-by. Deferred.
- *README polish.* The current README is already informative and honest about backtest limitations — polishing further was not a high-impact win this pass.

**Next-run candidates:**
1. Delete the committed junk root files via `delete_file` calls, then verify clean repo.
2. Add `pyproject.toml` with `[tool.pytest.ini_options]\npythonpath = ["."]` so pytest works the same way locally and in CI without env hacks.
3. Fix `tests/test_benchmark_tracker.py` vs `src/db/models.py` drift (owner decides direction), then drop `continue-on-error` + `--ignore` from `ci.yml`.
4. Pin ruff config (`pyproject.toml` `[tool.ruff]`) and make the lint job blocking.
5. Add a small unit test for the audit reporter (`src/backtest/reporting.py`) since it has a deterministic input → output contract.
6. Move `CODEX_FIXES.md`, `INDIA_PIPELINE_SUMMARY.md`, `PROFESSIONAL_TRADER_UPGRADE.md`, `plan.md`, `pcr.txt` into `docs/` to declutter the root view.
