# Contributing to AegisQuant

Thanks for taking an interest in AegisQuant. This is a research-grade
algorithmic trading codebase; please read the safety notes below before
opening a PR.

## Ground rules

- **No live trading code paths in tests.** Tests must run with
  `ENABLE_MOCK_DATA=True` and must not reach Alpaca, Groww, or any other
  broker.
- **Reproducibility:** seed every stochastic component (`numpy`, `torch`,
  `gymnasium`, `stable_baselines3`) so backtests are deterministic where it
  matters.
- **No real API keys** in code, fixtures, or commit messages. Use
  `.env.example` as the source of truth for required env vars.
- **Risk gates first.** Changes to circuit breakers, drawdown limits, or
  position sizing need an explanation in the PR description and a passing
  walk-forward audit report.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env         # leave keys blank for offline development
```

## Running the test suite

```bash
ENABLE_MOCK_DATA=True python -m pytest tests/ -q
```

CI (`.github/workflows/ci.yml`) runs the same command on Python 3.11 and
3.12 for every push and pull request.

## Pull-request checklist

- [ ] `pytest` passes locally
- [ ] New behavior has at least one regression test
- [ ] No `print(...)` debug statements left behind
- [ ] README / docstrings updated if user-facing behavior changed
- [ ] If touching the RL stack, attach a short note on training
      configuration and expected Sharpe/return delta

## Reporting issues

Use GitHub Issues. For anything involving a broker connection, scrub the
logs of keys, account IDs, and order IDs before pasting them.
