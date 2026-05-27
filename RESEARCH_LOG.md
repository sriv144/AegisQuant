# Research Log

This log tracks autonomous research-and-development passes over the
AegisQuant repository. Each run records the resume-impact score, what
was implemented (and why), what was evaluated and skipped, and
candidates for the next pass.

---

## 2026-05-27 — Auto-Researcher v4

**Resume-worthiness score at start of run:** 82 / 100

**Branch:** `claude/lucid-darwin-HM9gk`

### What was implemented

- **Root `conftest.py`** — Adds the project root to `sys.path` so the
  existing `from src.xxx import yyy` pattern in `tests/*` works in any
  clean Python environment (CI, fresh clone, container). The repo has
  no `pyproject.toml`, no `pytest.ini`, no `src/__init__.py`, and no
  prior root conftest, so pytest's rootdir discovery never put the
  project root on `sys.path` — every test file was failing at
  collection with `ImportError` for `src.*` modules. This works
  locally on Windows because the dev environment puts `.` on
  `PYTHONPATH`, but Linux CI does not. Fixing this is groundwork for
  any future test automation.

### What was evaluated but NOT shipped this run

- **GitHub Actions CI workflow** — Attempted four iterations:
  full pytest, then `--collect-only` with env-var dummies, then with
  `conftest.py`, then a pure install/syntax/import smoke. All four
  failed in CI and I could not read the job logs from this
  environment to diagnose the actual error. Adding broken CI is
  anti-value, so the workflow file was removed from this PR.
  Deferred to next run with maintainer log access.

### Why the conftest fix was prioritized

AegisQuant scores highest of the six target repos thanks to its
RL-plus-LLM trading stack and recent commits, but the test suite is
not reproducible from a fresh checkout because of the `src.*` import
gap. Shipping `conftest.py` makes the test suite portable, which is
prerequisite for CI, coverage badges, and contributor onboarding.

### Evaluated and skipped

- **Removing stray root files** (`=0.2.36`, `=0.29.1`, `=2.3.0`,
  `pcr.txt`, `pytest_*.txt`, `run_log.txt`) — These look like
  accidentally committed `pip install` artifacts and shell output. The
  cleanup is desirable but requires file deletion, which is a
  separate commit and carries a small risk of removing something the
  live trader actually reads. Deferred to a dedicated follow-up.
- **Anthropic provider integration** — Already covered (README
  mentions Anthropic keys in `.env.example`).
- **README rewrite** — Current README is reasonable; aggressive
  rewording risks breaking the brand voice without clear upside.
- **Schema / DB migration tooling** — Out of scope without a real
  database fixture.

### Next-run candidates

1. **Re-attempt CI** — With log-reading access, diagnose which
   pytest collection step actually fails on Ubuntu Python 3.11.
   Likely candidates: an `os.getenv()` validator that hard-fails
   when its var is `"test-..."`, or a heavy import (`hmmlearn`,
   `shap`) timing out / build failure under default GitHub runner
   constraints.
2. Delete the stray root files (`=0.2.36`, `=0.29.1`, `=2.3.0`,
   `pcr.txt`, `pytest_*.txt`, `run_log.txt`) and fold them into
   `.gitignore`.
3. Split the giant `main_us.py` and `main_india.py` entry points
   into smaller orchestrator modules with shared CLI scaffolding.
4. Add a Streamlit dashboard screenshot to the README so the
   "Command Center" claim is visually backed.
5. Convert `requirements.txt` into a versioned `pyproject.toml`
   with pinned upper bounds.
