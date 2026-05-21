# AegisQuant

[![tests](https://github.com/sriv144/AegisQuant/actions/workflows/tests.yml/badge.svg)](https://github.com/sriv144/AegisQuant/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

AegisQuant is a production-grade, multi-asset algorithmic trading pipeline utilizing Reinforcement Learning (PPO/SAC) combined with Large Language Model consensus scoring to actively generate systematic alpha.

The system structurally bridges the gap between pure ML research and financial deployment by embedding institutional risk-management techniques (Continuous Feature Normalization, Gaussian HMM Regime Detection, SHAP Agent Attribution, Drawdown Circuit Breakers, and Implementation Shortfall tracking).

## Core Architecture
- **Phase 0–1**: Synchronous multi-asset `yfinance` pipelines computing Z-score normalized volatility curves, fed into a Monte Carlo bootstrap walk-forward testing engine.
- **Phase 2–3**: Continuous `[-1.0, 1.0]` Gym environments optimizing portfolios natively against turnover friction and covariance/correlation penalties.
- **Phase 4**: Alpaca Broker wrappers transmuting AI weights into discrete integer lot orders execution tracking.
- **Phase 5**: Deep Streamlit UI projecting continuous SHAP permutations mapping exactly *why* the AI generated its signals.
- **Phase 6**: SQLAlchemy audit trails and active SLACK/SMTP alerting loops.

## Installation

```bash
# 1. Clone & Enter Directory
cd AegisQuant

# 2. Install core library requirements
pip install -r requirements.txt
# (Includes stable-baselines3, gymnasium, shap, alpaca-py, streamlit, hmmlearn)

# 3. Secure Env Vars
cp .env.example .env
# Edit .env and supply your Alpaca or Anthropic keys.
```

## Running the Matrix

### 1. The Walk-Forward Backtester
To train the PPO model from scratch across chronological cross-fold validations mapping 10 years of OHLCV:
```bash
python src/backtest/walk_forward.py --algo PPO --mc-sims 10000
```
*Outputs will dump `model.zip` into `model_registry/` while generating exact `shap_feature_importance.json` traces.*

### 2. The Command Center (UI)
Streamlit hosts the Phase 5 interactive metrics:
```bash
streamlit run src/ui/dashboard.py
```
*Visualizes live paper-trading PnL, Regime shifts, and SHAP Global Feature Attributions.*

### 3. The Live Trading Daemon
Launch the APScheduler heartbeat. Armed with `.env` keys, it will automatically extract live states, run inference, and punch trades natively via Alpaca at exactly **09:35 AM ET** every weekday:
```bash
python main.py
```
*To force immediate execution off schedule, run `python main.py --now`.*

## System Testing
The codebase is mapped heavily against `pytest`. Execute safety verifications before pushing model states up to Staging/Production:
```bash
python -m pytest tests/
```

Every push and pull request to `main` also runs the suite automatically via
GitHub Actions ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)).
CI sets `ENABLE_MOCK_DATA=True` and `ENABLE_BROKER_EXECUTION=False` so the
correctness checks never reach for `yfinance` or live broker APIs — this is a
dedicated test signal, kept separate from the live `trade.yml` heartbeat.

## Authors
_Built originally to merge Modern Portfolio Theory with Autonomous AI frameworks._
