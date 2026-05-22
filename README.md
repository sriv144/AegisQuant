# AegisQuant

> Production-grade, multi-asset algorithmic trading pipeline that pairs **Reinforcement Learning (PPO/SAC)** with **LLM consensus scoring** to research systematic alpha — with institutional risk management built in.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![Reinforcement Learning](https://img.shields.io/badge/RL-PPO%20%7C%20SAC-orange.svg)
![Framework](https://img.shields.io/badge/framework-stable--baselines3-green.svg)
![UI](https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg)
![Status](https://img.shields.io/badge/status-research-yellow.svg)

AegisQuant structurally bridges the gap between pure ML research and financial deployment by embedding institutional risk-management techniques (Continuous Feature Normalization, Gaussian HMM Regime Detection, SHAP Agent Attribution, Drawdown Circuit Breakers, and Implementation Shortfall tracking).

## Highlights

- **Reinforcement-learning portfolio agent** — continuous `[-1.0, 1.0]` Gymnasium environments trained with PPO/SAC, optimized natively against turnover friction and covariance/correlation penalties.
- **Walk-forward backtesting** — Monte Carlo bootstrap with chronological cross-fold validation over ~10 years of OHLCV data.
- **Regime awareness** — Gaussian HMM regime detection feeding risk gates and drawdown circuit breakers.
- **Explainability first** — SHAP global and per-decision feature attribution surfaced directly in the UI.
- **Multi-market pipelines** — separate US and India entrypoints (`main_us.py`, `main_india.py`).
- **Live execution** — Alpaca broker integration converting AI weights into discrete integer lot orders on an APScheduler heartbeat.
- **Honest reporting** — the audit reporter states plainly when the strategy fails its benchmarks instead of overselling results.

## Core Architecture
- **Phase 0–1**: Synchronous multi-asset `yfinance` pipelines computing Z-score normalized volatility curves, fed into a Monte Carlo bootstrap walk-forward testing engine.
- **Phase 2–3**: Continuous `[-1.0, 1.0]` Gym environments optimizing portfolios natively against turnover friction and covariance/correlation penalties.
- **Phase 4**: Alpaca Broker wrappers transmuting AI weights into discrete integer lot orders execution tracking.
- **Phase 5**: Deep Streamlit UI projecting continuous SHAP permutations mapping exactly *why* the AI generated its signals.
- **Phase 6**: SQLAlchemy audit trails and active SLACK/SMTP alerting loops.

## Tech Stack

| Layer | Technology |
|-------|------------|
| RL & training | stable-baselines3 (PPO/SAC), Gymnasium |
| Market data | yfinance, pandas, numpy, scipy, statsmodels |
| Regime detection | hmmlearn (Gaussian HMM) |
| Explainability | SHAP |
| Execution | alpaca-py |
| Orchestration | APScheduler, LangGraph |
| Persistence | SQLAlchemy, PostgreSQL, Redis |
| Interface & API | Streamlit, FastAPI, uvicorn |
| Alerting | slack-sdk, SMTP |

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

## Project Layout

```
AegisQuant/
├── src/                 # Core library: backtest, environments, agents, risk, UI, broker
├── main.py              # Live trading daemon entrypoint
├── main_us.py           # US-market research pipeline
├── main_india.py        # India-market research pipeline
├── train_rl.py          # PPO/SAC training entrypoint
├── weekly_review.py     # Scheduled performance-review job
├── backtest_results/    # Walk-forward JSON artifacts and audit reports
├── model_registry/      # Saved RL model checkpoints
├── tests/               # pytest suite
├── deploy/              # Deployment manifests
├── Dockerfile           # Container build
├── docker-compose.yml   # Local stack
└── .env.example         # Environment variable template
```

## Disclaimer

AegisQuant is a **research and educational project**. Nothing in this repository
constitutes financial advice. Reinforcement-learning trading strategies can and
do lose money; backtested performance does not predict live results. Run the
live trading daemon only with capital you can afford to lose, and only after
reviewing the audit reports. The authors accept no liability for financial loss.

## Authors
_Built originally to merge Modern Portfolio Theory with Autonomous AI frameworks._
