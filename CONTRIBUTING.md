# Contributing to AegisQuant

Thanks for your interest in helping AegisQuant grow. This guide documents how to work on the project safely. AegisQuant is a multi-asset RL trading system — changes that touch the live trading path, broker integration, or model registry deserve extra scrutiny.

## Project setup

```bash
git clone https://github.com/sriv144/AegisQuant.git
cd AegisQuant
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with Alpaca, Anthropic, and Groww credentials for the surface area you actually plan to test. Live trading paths must never be enabled in unit tests.

## Branching

- `main` is the only deployable branch.
- Feature work goes on `feat/<short-description>`.
- Bug fixes go on `fix/<short-description>`.
- Auto-researcher commits land on `claude/*` branches.

Never push directly to `main` — open a pull request and let CI run first.

## Tests

```bash
python -m pytest tests/ -v
```

The full suite covers walk-forward backtests, the Monte Carlo bootstrap, regime detection, circuit breakers, the paper-portfolio ledger, reasoning logs, and the dashboard auth layer. Please add a focused test alongside any new behaviour. Property-style tests for risk gates are particularly welcome.

## Style

- Python 3.11+.
- Format and lint with [ruff](https://docs.astral.sh/ruff/) — `ruff check src tests` and `ruff format src tests`.
- Prefer explicit names: `position_weight`, not `pw`.
- Keep functions side-effect free where you can; the RL training path already has enough randomness.
- Never log API keys, account ids, or raw order payloads.

## Pull requests

A good PR description includes:

1. The problem you are solving and why it matters for AegisQuant.
2. A short note on backtest impact, if the change touches features, the env, or the reward function.
3. The exact command you used to validate it locally.
4. Whether the change is safe to enable in live trading immediately, or whether it should be gated behind an env flag.

## Security & risk

- Never commit `.env`, broker credentials, or model artifacts that contain proprietary signals.
- Run `pip-audit -r requirements.txt` before bumping dependencies.
- Treat the live trading daemon as production: drawdown circuit breakers, regime gates, and the implementation-shortfall tracker must remain wired up at all times.
- If you discover a security issue, follow `SECURITY.md` instead of opening a public issue.

## License

By contributing you agree your contributions are licensed under the same terms as the rest of the repository.
