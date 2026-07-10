# AegisQuant

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Reinforcement Learning](https://img.shields.io/badge/RL-PPO%20%7C%20SAC-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-orange?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

> **Reinforcement-learning portfolio manager with HMM regime detection, SHAP attribution, and live Alpaca execution — built for honest, auditable backtests.**

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

### 0. Backtest Audit Report
To turn an existing walk-forward JSON artifact into an honest Markdown + JSON audit report:
```bash
python -m src.backtest.reporting --input backtest_results/walk_forward_multi_SPY_QQQ_TLT_GLD.json
```
This writes:
- `backtest_results/audit_multi_SPY_QQQ_TLT_GLD_report.md`
- `backtest_results/audit_multi_SPY_QQQ_TLT_GLD_summary.json`

The audit report is deliberately transparent. If the RL strategy fails against benchmarks or risk gates, the report says so plainly instead of presenting AegisQuant as a profitable trading bot.

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

## Authors
_Built originally to merge Modern Portfolio Theory with Autonomous AI frameworks._
