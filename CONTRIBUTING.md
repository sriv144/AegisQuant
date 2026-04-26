# Contributing to AegisQuant

Thanks for taking the time to look at AegisQuant. This is a research-grade
trading pipeline, so contributions need to keep three properties intact at all
times:

1. **Reproducibility** — every backtest must be deterministic given a fixed
   seed and the pinned `requirements.txt`.
2. **Safety** — the live trading paths (`src/execution/`, `main.py`,
   `main_india.py`) must respect every circuit breaker. Any change that
   touches order routing needs a paper-mode regression.
3. **Auditability** — features must be SHAP-attributable and decisions must
   leave a trail in the SQLAlchemy audit store.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env         # then fill in keys
```

Run the full test suite before opening a PR:

```bash
python -m pytest tests/ -q
```

## Workflow

1. Open or claim an issue first — most non-trivial changes (env shape, reward,
   feature pipeline, broker adapter) need a short design discussion.
2. Branch off `main` using a descriptive name, e.g. `feat/regime-hmm-bull-tail`
   or `fix/circuit-breaker-ist-tz`.
3. Keep commits focused. Use Conventional Commit prefixes:
   - `feat:` new capability
   - `fix:` bug fix
   - `perf:` measurable speedup or cost reduction
   - `refactor:` no behaviour change
   - `docs:` docs / README only
   - `test:` tests only
4. Update or add tests for any new code path.
5. Open a PR using the template; fill in the risk + paper-trade sections.

## Code style

- Python 3.10+. Type-hint public functions.
- Keep RL environment shapes documented in a docstring at the top of the file.
- Don't commit broker secrets, even in tests — use `monkeypatch` and the
  `.env.example` placeholders.

## Reporting a security issue

See `SECURITY.md` for the private-disclosure path. Do not open public issues
for live-trading or credential-handling vulnerabilities.
