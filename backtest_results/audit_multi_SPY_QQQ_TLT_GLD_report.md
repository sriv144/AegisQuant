# AegisQuant Backtest Audit Report

**Source artifact:** `backtest_results/walk_forward_multi_SPY_QQQ_TLT_GLD.json`
**Generated:** 2026-05-12T01:13:39.472788+00:00
**Universe:** SPY, QQQ, TLT, GLD
**Verdict:** `FAILED_RISK_GATE`

## Executive Readout

This report is an audit artifact, not a profitability claim. It preserves the current walk-forward result honestly so the project can be evaluated as a quant research system with benchmark comparison, risk gates, and failure analysis.

## Aggregate OOS Metrics

| Metric | Value |
|---|---:|
| Annualised return | -63.47% |
| Annualised volatility | 17.66% |
| Sharpe ratio | -5.6909 |
| Sortino ratio | -7.3889 |
| Max drawdown | -99.97% |
| Calmar ratio | -0.6348 |
| Win rate | 32.77% |
| Profit factor | 0.3528 |
| Deflated Sharpe ratio | 0.0000 |

## Benchmark Comparison

| Benchmark | Sharpe | Ann. Return | Max DD | RL Sharpe Delta |
|---|---:|---:|---:|---:|
| 12-1 Month Momentum | 5.3178 | 20.90% | -30.35% | -11.0087 |
| Equal-Weight Monthly (SPY,QQQ,TLT,GLD) | 1.0141 | 13.05% | -25.60% | -6.7050 |
| 60/40 SPY+AGG | 0.8063 | 10.34% | -21.72% | -6.4972 |
| Buy & Hold SPY | 0.7781 | 16.34% | -33.72% | -6.4690 |
| Random Policy (same env) | -2.1941 | -18.14% | -80.44% | -3.4968 |
| AegisQuant RL Strategy | -5.6909 | -63.47% | -99.97% | 0.0000 |

## Monte Carlo Downside

- Probability of ruin: 99.99%
- Sharpe p5/p50/p95: -7.5566 / -5.7492 / -3.9943
- Annualised return p5/p50/p95: -72.95% / -63.51% / -51.25%

## Walk-Forward Integrity

- Windows evaluated: 16
- Failed windows: 0
- OOS return observations: 1999

## Top Feature Attributions

- `GLD_macd`: 0.0447
- `TLT_rsi`: 0.0429
- `QQQ_rsi`: 0.0425
- `QQQ_macd`: 0.0417
- `GLD_mom`: 0.0416
- `GLD_vol`: 0.0403
- `TLT_macd`: 0.0393
- `SPY_rsi`: 0.0383
- `TLT_vol`: 0.0377
- `SPY_vol`: 0.0364

## Interpretation

The current RL strategy fails the risk gate: drawdown is extreme and benchmark-relative Sharpe is strongly negative. That is still a useful engineering result because the platform exposes the failure instead of hiding it. The next research step is to debug reward design, position constraints, transaction-cost assumptions, and benchmark leakage before presenting any alpha claim.

## Next Research Step

Run a constrained baseline before retraining PPO: equal-weight, momentum, and volatility-targeted allocations should become the minimum acceptance bar. Only reintroduce RL after the environment can reproduce sane benchmark behavior.
