# Backtest Report Template

Every new strategy, RL checkpoint, or LLM-consensus rule that targets
`model_registry/` MUST ship with a filled-in copy of this template at
`docs/backtests/<strategy>-<YYYY-MM-DD>.md`. The promotion checklist
at the bottom of this file maps to the safety rails documented in
`docs/ARCHITECTURE.md` (drawdown circuit breaker, time-window rule,
broker execution gating).

## 1. Setup

| Field | Value |
| --- | --- |
| Strategy / model name |  |
| Author |  |
| Report date |  |
| Code commit SHA |  |
| Git branch |  |
| Linked PR |  |
| Pipeline (US / India / Both) |  |
| Trade mode (CNC / MIS / Mixed) |  |
| Capital (notional) |  |
| Universe definition |  |
| Universe size at start of test |  |
| Backtest period (start, end) |  |
| Bar interval (1d / 1h / etc.) |  |
| Data source(s) |  |
| Transaction cost assumption (bps) |  |
| Slippage model |  |
| Borrow / shorting cost (if any) |  |
| Random seed(s) |  |

## 2. Walk-forward folds

Walk-forward validation is required. Document fold boundaries verbatim
so a reviewer can rerun any fold deterministically.

| Fold | Train start | Train end | Test start | Test end | Train bars | Test bars |
| --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| ... |  |  |  |  |  |  |

## 3. Headline metrics (out-of-sample, aggregated across folds)

Report mean and 95 percent bootstrap CI from `--mc-sims` Monte Carlo
resampling at the trade level. Single point estimates without a CI are
not acceptable.

| Metric | Mean | 95% CI low | 95% CI high | Benchmark | Delta vs benchmark |
| --- | --- | --- | --- | --- | --- |
| CAGR (net) |  |  |  |  |  |
| Volatility (ann.) |  |  |  |  |  |
| Sharpe (ann.) |  |  |  |  |  |
| Sortino (ann.) |  |  |  |  |  |
| Calmar |  |  |  |  |  |
| Max drawdown |  |  |  |  |  |
| Time underwater (max, days) |  |  |  |  |  |
| Hit rate |  |  |  |  |  |
| Avg win / avg loss |  |  |  |  |  |
| Turnover (ann.) |  |  |  |  |  |
| Avg holding period |  |  |  |  |  |
| Capacity estimate (USD or INR) |  |  |  |  |  |
| Tail beta to market |  |  |  |  |  |

Benchmark must match the pipeline (US: SPY total return; India: NIFTY 50
total return).

## 4. Per-fold table

| Fold | Net return | Sharpe | Max DD | Hit rate | Turnover | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| ... |  |  |  |  |  |  |

Dispersion across folds is the single best out-of-sample stability
signal. Report standard deviation of fold Sharpe and the worst fold
explicitly.

## 5. Regime-conditional performance (HMM)

Use the same Gaussian HMM regime labels documented in
`docs/ARCHITECTURE.md`. The strategy must not be a single-regime
outlier.

| Regime | Bars in regime | Net return | Sharpe | Max DD | Hit rate |
| --- | --- | --- | --- | --- | --- |
| Low-vol bull |  |  |  |  |  |
| High-vol bull |  |  |  |  |  |
| Low-vol bear |  |  |  |  |  |
| High-vol bear |  |  |  |  |  |

Flag any regime where the lower 95% CI of Sharpe sits below zero.

## 6. SHAP attribution snapshot

- Top 10 features by mean absolute SHAP value (table).
- One concrete trade with its full SHAP waterfall (chart link or PNG).
- Confirmation that no single feature dominates by more than 50 percent
  of attribution variance, or, if it does, justification.

## 7. Risk-of-overfit checks

- Parameter sensitivity: report metric range across at least three
  perturbations of the headline hyperparameter.
- Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014). Report value
  and the assumed number of independent trials.
- Train vs test gap: |train Sharpe - test Sharpe| should be reported
  per fold; flag folds where the gap exceeds 1.0.
- Look-ahead audit: confirm no future bar information leaks into
  features (label, comment which lookback windows were used).
- Survivorship: confirm the universe is built from a point-in-time
  membership snapshot, not the latest index members.

## 8. Live-paper sanity check

Before promotion the strategy must run on paper for at least 5 trading
days in the matching pipeline. Report:

- Realised vs simulated daily PnL correlation.
- Realised vs simulated turnover.
- Number of broker rejections / partial fills.
- Any time-window-rule, drawdown-breaker, or RL-sign guardrail trips.

## 9. Failure mode analysis

Write at least one paragraph describing the largest drawdown event in
the out-of-sample period: what regime, what feature flipped, what
would have stopped the strategy from holding. This is the section
where the model fails interview-grade scrutiny if it is missing.

## 10. Promotion checklist (go / no-go)

The strategy may be promoted to `model_registry/` only when every
box is checked.

- [ ] Walk-forward folds defined with explicit dates and committed.
- [ ] All headline metrics include 95% CI from MC bootstrap.
- [ ] Per-fold dispersion reported; worst fold not catastrophic.
- [ ] Regime-conditional table populated for all four HMM regimes
      that appear in the test window.
- [ ] SHAP top-10 + one full waterfall attached.
- [ ] Deflated Sharpe ratio reported and exceeds 0.
- [ ] No look-ahead: feature-construction code reviewed.
- [ ] Universe is point-in-time, not survivorship-biased.
- [ ] Live-paper run >=5 trading days completed without circuit-
      breaker trips outside the documented expectation.
- [ ] Failure mode analysis written.
- [ ] Linked PR has at least one reviewer approval from outside the
      author's working branch.
- [ ] Risk parameters in `.env.example` updated if the strategy
      requires new bounds.
- [ ] Dashboard panels (Decisions / Positions) verified to render the
      new strategy's metadata.

## 11. Sign-off

| Role | Name | Date | Signature / commit |
| --- | --- | --- | --- |
| Strategy author |  |  |  |
| Independent reviewer |  |  |  |
| Risk owner |  |  |  |
