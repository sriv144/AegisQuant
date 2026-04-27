# AegisQuant

AegisQuant is a multi-asset algorithmic trading research platform that combines
reinforcement learning (PPO/SAC via Stable-Baselines3) with explicit risk
management: Z-score feature normalization, Gaussian HMM regime detection, SHAP
attribution for the policy, drawdown circuit breakers, and
implementation-shortfall cost tracking.

The goal is to bridge ML research and execution. Train a policy in a
continuous-weight portfolio Gym environment, validate it through Monte-Carlo
walk-forward backtesting, and route weights to Alpaca paper or live orders
behind safety checks.

## Highlights

- **Data + features:** synchronous yfinance ingestion, Z-score normalized
  features, Gaussian HMM regime labels.
- **Backtesting:** chronological walk-forward folds with Monte-Carlo bootstrap.
- **RL environment:** continuous `[-1.0, 1.0]` portfolio weights with turnover
  and covariance penalties.
- **Risk:** drawdown circuit breakers, implementation-shortfall accounting,
  cost model with friction.
- **Explainability:** SHAP global feature attribution exported as
  `shap_feature_importance.json`.
- **Execution:** Alpaca broker wrapper that converts continuous weights into
  integer-lot orders.
- **Operations:** Streamlit dashboard for live PnL / regime / SHAP, SQLAlchemy
  audit trail, Slack and SMTP alerts.

## Architecture

| Phase | Responsibility | Entry points |
| --- | --- | --- |
| 0–1 | Data ingestion + walk-forward backtester | `src/data/`, `src/backtest/walk_forward.py` |
| 2–3 | RL portfolio environment + training | `src/envs/`, `train_rl.py` |
| 4   | Alpaca execution and order routing | `src/broker/` |
| 5   | SHAP attribution + Streamlit dashboard | `src/ui/dashboard.py` |
| 6   | SQLAlchemy audit trail + Slack/SMTP alerts | `src/observability/` |

Core libraries: `stable-baselines3`, `gymnasium`, `shap`, `alpaca-py`,
`streamlit`, `hmmlearn`.

## Installation

```bash
git clone https://github.com/sriv144/AegisQuant.git
cd AegisQuant
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and supply the keys listed there (Alpaca, Anthropic, etc.).
```

## Usage

### 1. Walk-forward backtest

Train PPO across chronological folds with Monte-Carlo bootstrap:

```bash
python src/backtest/walk_forward.py --algo PPO --mc-sims 10000
```

Outputs:

- model checkpoints under `model_registry/`
- per-fold metrics
- `shap_feature_importance.json` for the trained policy

### 2. Streamlit dashboard

```bash
streamlit run src/ui/dashboard.py
```

Shows live paper-trading PnL, HMM regime shifts, and SHAP global feature
attribution.

### 3. Live trading daemon

```bash
python main.py            # waits for the configured 09:35 ET schedule
python main.py --now      # force a single immediate cycle off-schedule
```

Pulls live state, runs inference, and submits Alpaca orders behind the
configured drawdown circuit breakers.

## Tests

```bash
python -m pytest tests/
```

Coverage includes the walk-forward engine, regime detector, multi-asset Gym
environment, cost model, circuit breakers, and runtime safety checks. The same
suite runs in CI on every push.

## Repository layout

```
src/
  backtest/        walk-forward and Monte-Carlo
  envs/            multi-asset Gym environments
  broker/          Alpaca order routing
  observability/   audit trail, alerts
  ui/              Streamlit dashboard
tests/             pytest test suite
plan.md            research notes and roadmap
.env.example       required environment variables
```

## License

Research project; portfolio use only. Backtesting and live-trading code is
provided as-is and is not investment advice.
