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
  and a "where to look next" file map. The README documents phases
  0-6 at a high level; this doc gives an interview-ready deep dive.

### Why this was prioritized
AegisQuant has exceptional technical depth (RL + multi-agent + live broker +
dashboard) but the internal wiring is only implicit in the codebase. A
dedicated `ARCHITECTURE.md` is the single highest resume lever this run:
zero breakage risk, makes the repo cite-able in interviews, and does not
collide with any open branch.

### Prior claude/* branches observed (unmerged on main)
- `claude/lucid-darwin-QFlkH` - added ruff + pre-commit + hardened
  `.gitignore` + previous RESEARCH_LOG seed. This new ONkrp branch
  intentionally does NOT touch those files; a three-way merge with
  QFlkH will only conflict on RESEARCH_LOG.md which is trivial to
  resolve (concat entries).
- `claude/focused-newton-KSTnM`, `claude/focused-newton-TXG01` -
  earlier auto-researcher rounds; content not re-inspected this run.

### Evaluated and skipped
- **Dedicated CI workflow (pytest + ruff)**: the QFlkH branch already
  has ruff + pre-commit config, and `.github/workflows/trade.yml` (live
  trading cron) is already active. Adding a second workflow on this branch
  before QFlkH merges would fight QFlkH's ruff config. Defer until
  QFlkH is merged.
- **Dependabot**: `yfinance`, `alpaca-py`, `stable-baselines3` version pins
  are tightly coupled with the model checkpoints in `model_registry/`.
  Mass auto-bumps would risk silently breaking paper-trading reproducibility.
- **New trading strategy module**: high potential but requires backtesting
  + walk-forward validation before shipping. Out of scope for a safe
  single-run improvement.

### Next-run candidates
1. Merge QFlkH first (ruff + pre-commit), then add a CI workflow here
   that runs ruff + pytest.
2. Add `docs/BACKTEST_REPORT_TEMPLATE.md` so every new strategy must ship
   with a standardised walk-forward summary.
3. Add a minimal `notebooks/shap_walkthrough.ipynb` that renders a SHAP
   waterfall for one real signal - huge resume artifact.
4. Extract dashboard SPA fetch logic into a typed client (`src/ui/api_client.py`)
   so `index.html` + `app.js` can be regression-tested.
5. Record a short 60s screen capture of the dashboard (Dashboard ->
   Positions -> Decisions drill-down) and link it from the README -
   "great project with a bad README gets no stars" applies to live demos too.
