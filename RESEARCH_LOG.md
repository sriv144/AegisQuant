# RESEARCH_LOG.md

Persistent memory for the auto-researcher agent. Read top-to-bottom before deciding what to ship on the next pass.

---

## 2026-06-12 — Auto-Researcher v4

**Resume score at start of run:** 85 / 100 (top of the 6-repo portfolio)

**Score breakdown:**
- Tech stack prestige: 25/25 — PPO/SAC + LLM consensus + HMM regime + SHAP attribution + Alpaca live integration is a top-tier quant stack.
- Commit recency: 25/25 — updated 2026-05-22, most recent of all 6 target repos.
- Feature completeness: 17/20 — walk-forward backtester, audit reports, Streamlit dashboard, scheduled trade workflow already shipped.
- Stars / visibility: 5/15 — 2 stars (highest in the portfolio) but still discovery-limited.
- README quality: 13/15 — strong narrative + commands, no architecture diagram inline.

### What was implemented this pass (branch `claude/lucid-darwin-qiarsa`)

Pure additive scaffolding — zero source-code, CI, or config touched:

- `.github/ISSUE_TEMPLATE/bug_report.yml` — structured, with a surface-area dropdown and a "do not paste broker keys" reminder.
- `.github/ISSUE_TEMPLATE/feature_request.yml`
- `.github/ISSUE_TEMPLATE/config.yml` — disables blank issues.
- `.github/PULL_REQUEST_TEMPLATE.md` — includes a **risk surface** matrix and a trading-safety checklist (no live order without a risk gate, drawdown thresholds intact, no secrets/positions in diff).
- `CHANGELOG.md` — Keep-a-Changelog format.

### Why these and not something bigger?

The agent's open-PR inventory for this repo (PRs #8–#23) already covers CI workflows, LICENSE, badges, CONTRIBUTING.md, SECURITY.md, CodeQL, root-cleanup of `=*` pip artifacts, and Mermaid architecture docs — none of them merged. Stacking more of the same is anti-impact at this point. Issue/PR templates and a CHANGELOG are the most-visible-yet-unaddressed maintenance signals left.

### Evaluated and skipped

- **Deleting the three stray `=0.2.36` / `=0.29.1` / `=2.3.0` files at repo root.** The `push_files` MCP tool cannot delete; would need a `delete_file` call sequence. Already queued in multiple prior PRs.
- **Adding a dashboard screenshot to README.** Requires actually running Streamlit; out of scope for a non-runtime pass.
- **Promoting one of the existing open `claude/*` PRs to ready-for-review.** That's a maintainer decision, not an auto-researcher decision.
- **A demo Jupyter notebook reading the included `backtest_results/*.json`.** Higher-impact but riskier — wants confirmation of artifact paths and Python deps not in `requirements.txt`.

### Next-run candidates (priority order)

1. **Delete the three `=0.x.y` repo-root pip-typo artifacts** via `delete_file`, then add `=*` to `.gitignore` in the same branch.
2. **Demo notebook**: `notebooks/00_audit_report_demo.ipynb` that loads `backtest_results/walk_forward_multi_SPY_QQQ_TLT_GLD.json` (if present) and renders the audit cards — would double as a screenshot source for the README.
3. **Merge the lowest-risk of the open scaffolding PRs** (issue templates + CHANGELOG + LICENSE) into one clean PR.
4. **`docs/RISK_FRAMEWORK.md`** — extract the risk-gate logic narrative from README into its own document so the audit report can link to it.
5. **Anthropic Claude consensus scorer**: surface an env-toggled Claude path for the LLM consensus signal (currently OpenAI-only based on README).
