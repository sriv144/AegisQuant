# AegisQuant v3

AegisQuant v3 is a benchmark-aware, long-only research and Alpaca paper-trading
system. Its champion strategy is a 69% SPY core, 30% point-in-time
cross-sectional-momentum satellite, and 1% operational cash.

The supported runtime is deliberately narrow:

- `shadow` is the default and cannot submit broker orders.
- `paper` requires protected, explicit enablement and the exact Alpaca paper endpoint.
- There is no `live` CLI mode or live workflow.
- PPO/RL, VQM, PEAD, insider, macro-timing, shorts, leverage, and options do not
  affect v3 targets.
- Legacy `main.py`, `main_us.py`, and `main_us_v2.py` commands are disabled shims.

## Safe local start

Install the reviewed Python 3.11 lockfile and run a read-only health probe:

```bash
python -m pip install -r requirements.lock
python main_us_v3.py --mode shadow --purpose health
```

Bootstrap a local shadow database, then run a one-shot shadow rebalance with a
frozen input bundle:

```bash
set DATABASE_URL=sqlite:///data/aegisquant_v3.db
python main_us_v3.py --mode shadow --purpose bootstrap
python main_us_v3.py --mode shadow --purpose rebalance --input-bundle data/v3_runtime_input.json
```

The equivalent Docker path is opt-in and one-shot:

```bash
docker compose --profile manual-shadow run --rm manual-shadow
```

## Safety and promotion

Paper execution requires durable PostgreSQL, fresh successful broker reads, a
database lease, no unresolved orders, promotable point-in-time data, exact
config/data/commit-bound research evidence, a valid market window, the kill
switch off, and explicit execution enablement. A failed gate is a block; it
never falls back to shadow.

Every run writes a complete redacted audit bundle below `artifacts/<run_id>/`.
PostgreSQL remains the source of truth.

See [the v3 operations and verification runbook](docs/aegisquant-v3-runbook.md)
for protected-environment setup, frozen-data requirements, rollout gates, and
how to measure SPY-relative improvement.

## Tests

Run the same broker-POST guard used by CI:

```bash
set RUN_NETWORK_TESTS=0
set PYTHONPATH=tests
python -m pytest -p v3_workflow.no_network_order_post
```

Research results are not a guarantee of future returns. Paper fills, market
gaps, data defects, and regime changes can materially change realized results.
