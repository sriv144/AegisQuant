# AegisQuant v3 operations and verification runbook

## What v3 changes

The v3 runtime replaces the legacy agent-driven loops with one deterministic
portfolio constructor shared by research, shadow, and paper execution. It uses
69% SPY, up to 30% in 30 point-in-time S&P 500 momentum names, and 1% cash.
Macro and RL output are attribution-only or quarantined.

Paper mode is fail-closed. It cannot run without protected enablement, the
exact paper host, PostgreSQL, a lease, fresh Alpaca reads, reconciled prior
orders, passing promotion evidence, fresh quotes, and an eligible session.

## Required protected environments

Create these GitHub environments manually; repository code cannot create or
approve them:

- `alpaca-paper-shadow`
- `alpaca-paper-execution` with required reviewers

Configure environment secrets/variables:

- `POSTGRES_URL`: durable non-loopback PostgreSQL URL.
- `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`: paper credentials only.
- `AEGISQUANT_ACCOUNT_KEY`: the 16-character account fingerprint reported by
  the first paper bootstrap; never use the raw Alpaca account ID.
- `V3_INPUT_BUNDLE_URL`: credential-free HTTPS URL for the frozen inline JSON bundle.
- `V3_INPUT_BUNDLE_SHA256`: exact lowercase SHA-256 of that JSON file.
- `AEGISQUANT_KILL_SWITCH`: keep `true` until a reviewed manual paper run.

Do not add an OpenAI key to the trading environment.

## Database and account bootstrap

1. Dispatch `bootstrap` in `shadow` mode to apply Alembic migrations.
2. Dispatch `bootstrap` in `paper` mode under the protected environment. This
   requires the frozen input bundle, performs broker reads only, fingerprints
   up to 500 historical orders, records the hashed account identity, marks
   existing holdings `LEGACY_UNATTRIBUTED`, and starts the isolated shadow
   account cash-only at the audited NAV. It also writes the migration delta,
   open-order adjustments, ADV flags, and a 5-bp cost estimate to the audit
   artifact; `broker_post_count` remains zero.
3. Copy the reported fingerprint into `AEGISQUANT_ACCOUNT_KEY`.
4. Do not reset, liquidate, or relabel the Alpaca account automatically.

## Frozen input contract

Rebalance and EOD probes consume a content-addressed JSON bundle. Scheduled
runs do not download mutable market data. The bundle must contain:

- prior-month signal timestamp and total-return price history;
- point-in-time constituent, issuer, share-class, sector, and ADV metadata;
- current quotes no older than 60 seconds for execution;
- manifests with source, availability/freeze timestamps, row count, coverage,
  tier, warnings, and SHA-256;
- the first three eligible NYSE sessions for the month;
- an EOD SPY total-return mark for EOD runs.

Open/public snapshots remain `RESEARCH_ONLY`. Only independently validated,
survivorship-safe snapshots may generate promotable paper targets.

## Research evidence

Pre-register the 2005–2014 discovery, 2015–2019 validation, and untouched
2020–2026-06-30 holdout study before recording the final holdout. Keep every
attempted trial in `experiment_runs`. Only `TrustedStudyRunner` may mint final
paper-promotion evidence; normal registry calls cannot self-assert a trusted
attestation. Paper binds that attestation to the stable research-data hash and
the current commit. Call `TrustedStudyRunner.preregister(...)` before
`TrustedStudyRunner.run(...)`; `TrustedStudyResult.record(...)` refuses to
create a preregistration after the holdout has been evaluated.

Paper promotion remains blocked unless exact config/data/commit-bound evidence
passes every gate, including:

- net OOS annualized excess return versus SPY at least 1.5 percentage points;
- information ratio at least 0.40;
- beta 0.90–1.10 and tracking error no greater than 6%;
- positive 15 bp one-way cost-stress excess return;
- maximum drawdown no greater than 25% and no more than two points worse than SPY;
- turnover, rolling-window consistency, fold concentration, PSR, DSR, PBO,
  parameter-neighborhood, PIT, coverage, and hash-parity gates.

## Rollout and verification

You do not need to wait weeks to verify engineering correctness. Before market
observation, CI verifies deterministic hashes, accounting, corporate actions,
cost monotonicity, idempotency, crash recovery, no-network shadow behavior,
workflow contracts, migrations, and container configuration.

Economic and operational evidence does require elapsed sessions:

1. **Shadow gate:** five consecutive NYSE sessions with complete artifacts,
   same-date NAV/SPY rows, repeated-input target parity, and zero POSTs.
2. **Manual paper gate:** at least 30 NYSE sessions and two monthly rebalances,
   complete lifecycle reconciliation, no duplicate IDs, median arrival
   slippage at most 15 bps, p95 at most 40 bps, drift at most 50 bps, and no
   unresolved incidents.
3. **Preliminary economics:** 60–90 sessions. Compare cumulative net return,
   rolling excess return, tracking error, information ratio, drawdown,
   turnover, and costs with SPY—not raw dollar profit alone.
4. **Automatic paper:** enable only after both research and manual-paper gates.
   Live trading remains out of scope and has no CLI/workflow path.

The dashboard v3 endpoints expose current status, run details, nonterminal
orders, last reconciliation, gate failures, strategy identity, and SPY-relative
NAV. Audit artifacts are copies; investigate PostgreSQL for authoritative state.
Accepted paper orders are polled under the renewed database lease. If an order
has not reached a terminal state after 15 minutes, v3 requests cancellation,
halts the remaining batch, and requires reconciliation; acceptance is never
treated as a fill.

## Drawdown behavior

- At 10% drawdown, treat the state as a warning and conduct a risk review.
- At 15%, fresh broker equity and durable peak activate a daily idempotent
  containment decision: cancel open buys, submit only risk-reducing satellite
  sells, retain the 69% SPY core, and hold approximately 31% cash.
- A gap can exceed 15%; this is a trigger, not a guaranteed loss cap.
- Satellite re-entry requires durable de-risk state, drawdown below 10%, the
  next monthly window, a manually approved bundle, and protected paper approval.

## Return interpretation

The active sleeve is only 30% of NAV. Before costs, approximate portfolio
excess return is:

```text
0.30 × (satellite return − SPY return)
+ 0.01 × (cash return − SPY return)
− transaction costs and slippage
```

Consequently, a 1.5-point annual net portfolio excess gate generally requires
the satellite to beat SPY by materially more than five points after allowing
for cash drag and implementation costs. This is a hurdle, not a forecast.

Moving from the legacy 65%-invested cap to roughly 99% invested also increases
market participation in both directions. In an illustrative +10% market year,
34 points of additional beta-like exposure could add roughly 3.4% gross; in an
illustrative -20% year it could subtract roughly 6.8%, before satellite alpha,
costs, and risk controls. Do not interpret higher exposure as free alpha.
