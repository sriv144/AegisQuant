# Contributing to AegisQuant

Thanks for your interest! AegisQuant is research software that touches live
broker APIs, so we keep the development loop tight and the safety bar high.

## Development setup

```bash
git clone https://github.com/sriv144/AegisQuant.git
cd AegisQuant
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your keys (Alpaca + Anthropic)
```

Or, with the bundled Makefile:

```bash
make install
```

## Running tests

```bash
make test
# or:
python -m pytest tests/ -q
```

> Tests are fully mocked and require **no** broker or LLM credentials.
> If you add a test that needs a live API, gate it behind an `IT_LIVE=1`
> environment variable and skip when unset.

## Style + linting

We use `ruff` for both linting and formatting, wired through `pre-commit`:

```bash
pip install pre-commit
pre-commit install
```

Then every commit auto-runs `ruff check --fix` and `ruff format`. To run
them manually:

```bash
make lint     # check only
make format   # auto-fix
```

## Branching and commits

- Develop on a feature branch off `main`, e.g. `feat/regime-detector`.
- Conventional Commits prefixes: `feat:`, `fix:`, `perf:`, `refactor:`,
  `docs:`, `test:`, `chore:`, `ci:`.
- Reference the affected subsystem in parentheses where useful:
  `feat(broker): add OANDA adapter`.
- Small, atomic PRs land faster than mega-changes.

## Secrets and live trading safety

- **Never** commit a real broker key. `.env` is git-ignored; double-check
  `git status` before committing anything in the repo root.
- New broker adapters must subclass `BaseBroker` and respect the
  `LongOnlyRule` and `MaxPositionRule` circuit breakers.
- Any change to position sizing, order routing, or risk gates requires a
  paper-trading dry run plus a test that exercises the new path.

## Adding a new strategy

1. Add the module under `src/strategies/`.
2. Register it in `STRATEGY_REGISTRY` so the orchestrator picks it up.
3. Add a unit test under `tests/strategies/` covering the entry/exit logic.
4. Update `CHANGELOG.md` under `[Unreleased]`.

## Opening a PR

- Fill out the PR template (it auto-populates from `.github/PULL_REQUEST_TEMPLATE.md`).
- Note any new env vars added to `.env.example`.
- Confirm `make test` and `make lint` are both green locally.
