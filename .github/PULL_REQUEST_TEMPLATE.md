## Summary

<!-- One or two sentences on what changed and why. -->

## Risk surface

- [ ] Backtest / RL training only — no live-broker reachability
- [ ] Touches risk gates, position sizing, or order construction
- [ ] Touches Alpaca broker layer or live trade daemon
- [ ] Docs / CI / scaffolding only

## Test plan

- [ ] `pytest tests/` passes locally with `ENABLE_MOCK_DATA=True`
- [ ] Audit report regenerated (if backtest pipeline changed)
- [ ] Manual smoke test against paper-trading account (if broker path changed)

## Trading safety checklist (skip if N/A)

- [ ] No new code path can place a live order without a risk gate
- [ ] Drawdown circuit breaker thresholds were not weakened
- [ ] No secrets, API keys, or position data in diff
