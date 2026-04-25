# Research Log

Persistent memory used by the auto-researcher agent. Each run appends a dated
section so future runs can see what was evaluated, what shipped, and what was
deliberately skipped.

## 2026-04-23 - Auto-Researcher v4

**Resume-worthiness score at start of run: 86 / 100**
- Tech stack prestige: 24/25 (RL PPO/SAC + LLM consensus + SHAP + HMM regime + live broker)
- Commit recency: 25/25 (last push 2026-04-19)
- Feature completeness: 18/20 (9 strategies, dynamic universe, dual-mode trading, position mgr, capital allocator, dashboard SPA)
- Stars / visibility: 5/15 (1 star, 1 fork)
- README quality: 14/15 (Phase 0-6 architecture, install, run, test)

### Branch
`claude/lucid-darwin-ONkrp`

### Implemented
- `docs/ARCHITECTURE.md` - dedicated system-design doc covering the full
  multi-agent data flow (universe screener -> macro -> research -> strategy
  selector -> committee -> asset allocator -> execution -> risk/circuit
  breakers -> dashboard), the CNC/MIS trade-mode contract, safety rails
  (broker execution gating, TOTP fallback, time-window rule, drawdown
  breaker, degenerate-RL-sign guard), persistence schema, observability,
  and a "where to look next" file map.

### Why this was prioritized
AegisQuant has exceptional technical depth (RL + multi-agent + live broker +
dashboard) but the internal wiring is only implicit in the codebase. A
dedicated `ARCHITECTURE.md` is the single highest resume lever this run:
zero breakage risk, makes the repo cite-able in interviews, and does not
collide with any open branch.

### Prior claude/* branches observed (unmerged on main)
- `claude/lucid-darwin-QFlkH` - added ruff + pre-commit + hardened
  `.gitignore` + previous RESEARCH_LOG seed.
- `claude/focused-newton-KSTnM`, `claude/focused-newton-TXG01` -
  earlier auto-researcher rounds; content not re-inspected.

### Evaluated and skipped
- **Dedicated CI workflow (pytest + ruff)**: deferred until QFlkH merges.
- **Dependabot**: would risk silently breaking paper-trading reproducibility.
- **New trading strategy module**: requires backtesting before shipping.

### Next-run candidates
1. Merge QFlkH first (ruff + pre-commit), then add a CI workflow here
   that runs ruff + pytest.
2. Add `docs/BACKTEST_REPORT_TEMPLATE.md` so every new strategy must ship
   with a standardised walk-forward summary.
3. Add a minimal `notebooks/shap_walkthrough.ipynb` that renders a SHAP
   waterfall for one real signal.
4. Extract dashboard SPA fetch logic into a typed client.
5. Record a short 60s screen capture of the dashboard.

## 2026-04-25 - Auto-Researcher v4

**Resume-worthiness score at start of run: 86 / 100** (no main-branch
changes since 2026-04-19; cumulative claude/* backlog grew by one entry).

### Branch
`claude/lucid-darwin-lOIj4`

### Implemented
- `docs/BACKTEST_REPORT_TEMPLATE.md` - the next-run candidate #2 from
  the 2026-04-23 entry, shipped as-is. It captures setup
  (universe / period / costs / slippage), the walk-forward fold
  definition, headline metrics with 95% bootstrap CI, per-fold
  dispersion, regime-conditional performance against the existing
  Gaussian HMM regimes, SHAP top-feature attribution, risk-of-overfit
  checks (parameter sensitivity, deflated Sharpe, look-ahead audit,
  survivorship audit), live-paper sanity check, failure-mode analysis,
  and a go/no-go promotion checklist that maps to the safety rails
  already in `docs/ARCHITECTURE.md` (drawdown breaker, time-window
  rule, broker execution gating). Pure docs; zero runtime risk.

### Why this was prioritized
AegisQuant is a live-trading repo. Any code-touching change carries
reproducibility risk against existing model checkpoints in
`model_registry/`. The 2026-04-23 entry already explicitly listed the
backtest template as a documented next-run candidate, and shipping it
turns implicit promotion criteria into a written rubric - direct
resume value ("every new strategy ships with a deflated-Sharpe-aware
walk-forward report") with zero runtime risk.

It is also orthogonal to every existing claude/* branch on this repo:
- `claude/lucid-darwin-QFlkH` adds ruff + pre-commit (no overlap).
- `claude/lucid-darwin-ONkrp` adds `docs/ARCHITECTURE.md` (referenced
  by this template; complementary, not duplicate).

### Prior claude/* branches observed (unmerged on main)
- `claude/lucid-darwin-ONkrp` (2026-04-23) - ARCHITECTURE.md
- `claude/lucid-darwin-QFlkH` - ruff + pre-commit + .gitignore harden
- `claude/focused-newton-KSTnM`, `claude/focused-newton-TXG01` -
  earlier rounds; not re-inspected.

### Evaluated and skipped
- **CI workflow**: still deferred to after QFlkH merges so ruff config
  doesn't fight a parallel branch.
- **SHAP walkthrough notebook**: needs a real signal recorded from
  paper-trading; not safe to fabricate sample data into a doc that
  reads as authoritative.
- **Dashboard typed client extraction**: requires actual JS/TS
  refactor on `index.html` + `app.js` and end-to-end testing of the
  SPA. Out of scope for a safe-by-default run.
- **Dependabot**: same reproducibility argument as 2026-04-23.
- **Generic README polish**: README is already strong; the marginal
  resume signal is lower than a backtest rubric.

### Next-run candidates
1. After QFlkH merges, ship the CI workflow (ruff + pytest) referenced
   in the 2026-04-23 entry.
2. Fill in the first concrete `docs/backtests/<strategy>-<date>.md` with
   real numbers from one of the existing PPO checkpoints in
   `model_registry/` so the template gets validated end-to-end.
3. Add `notebooks/shap_walkthrough.ipynb` once a recorded paper-trading
   signal exists (linked from the backtest template's section 6).
4. Clean up accidental top-level files (`=0.2.36`, `=0.29.1`, `=2.3.0`,
   `pcr.txt`, `pytest_*.txt`) that look like a malformed pip install
   side effect; needs explicit owner approval before deletion.
5. Extract dashboard SPA fetch logic into a typed client and add
   regression tests for `/api/latest-run`.
