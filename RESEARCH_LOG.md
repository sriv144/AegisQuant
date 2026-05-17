# Research Log

A running log of automated research-and-development passes against this repository.

## 2026-05-17 — Auto-Researcher v4

**Resume-worthiness score at start of run: 87 / 100** (rank 1 of 6).

| Signal | Score |
| --- | --- |
| Tech stack prestige (RL + LLM consensus + live broker + finance) | 25 / 25 |
| Commit recency (updated 2026-05-16, yesterday) | 25 / 25 |
| Feature completeness (PPO/SAC, walk-forward, Streamlit, Alpaca, alerts, SHAP) | 18 / 20 |
| Stars + visibility (1 star, 1 fork) | 5 / 15 |
| README quality (architecture text + 3 runnable entry points, no diagrams) | 14 / 15 |

### Implemented this run (branch: `claude/lucid-darwin-Quaj7`)

- **feat(agents): opt-in Claude consensus scorer.** Added
  `src/agents/research/claude_consensus_scorer.py`. It scores the existing
  fundamental / macro / quant / sentiment proposal slate against Anthropic's
  `claude-sonnet-4-6` model and returns a structured
  `{consensus, confidence, rationale, model}` dict. Returns a fresh `ABSTAIN`
  on every disabled / failure path (no key, no SDK, empty proposals, network
  failure, malformed JSON) so it is safe to call unconditionally from the
  executive agent. Defaults to `claude-sonnet-4-6`, overridable via the
  `model=` kwarg or `ANTHROPIC_MODEL` env variable.
- **feat(deps): require anthropic SDK.** Added `anthropic>=0.40.0` to
  `requirements.txt`. The dependency is only imported lazily inside
  `score_consensus`, so import failures still degrade gracefully if the wheel
  ever fails to install.
- **docs(readme): Mermaid architecture diagram + Claude-consensus section.**
  Added a `mermaid flowchart LR` block under a new "Architecture Diagram"
  heading covering the Phase 0-6 data flow plus the opt-in Claude path; added
  a documented usage block for the new scorer.

### Why this was prioritized

The top-ranked open candidate from the previous run
(`claude/lucid-darwin-Ow84F`, 2026-05-16) was the Anthropic-backed LLM
consensus layer. The portfolio is Claude / Anthropic-only by policy, so
promoting Claude into the research committee is the single most on-brand
improvement still available. Doing it as an additive `score_consensus`
function (instead of refactoring `base_agent.py`) keeps the existing
OpenAI / OpenRouter flow working unchanged and removes the breakage risk that
called out this work as deserving its own branch.

The Mermaid diagram is the lowest-cost win in the previous next-run list and
renders inline on GitHub, so the architecture story is visible without
clicking through to source.

### Evaluated and skipped

- **Refactoring `base_agent.py` to dispatch between OpenAI / Anthropic at
  init time.** Touches every research agent and would require a real
  test-run loop to confirm the slate still parses identically. Out of scope
  for a one-shot pass.
- **Deleting stray root files (`=*`, `pytest_*.txt`, `pcr.txt`).** Open
  research notes already flag these but warn they may be referenced by
  scripts. Without a local test run, the safest path is still to leave them
  and rely on the `.gitignore` tightening already on `claude/lucid-darwin-Ow84F`.
- **Moving committed `*.zip` PPO checkpoints to a GitHub release.** Same
  reasoning — needs a verification loop to confirm no script greps for them.
- **Splitting `tests/` into `tests/unit/` and `tests/integration/`.** Worth
  doing, but the diff is large and risks renaming files that CI greps for.

### Next-run candidates

1. Wire `score_consensus` into the executive agent so its output is logged
   alongside the OpenAI committee verdict (still gated on `ANTHROPIC_API_KEY`).
2. Add a `tests/test_claude_consensus_scorer.py` that mocks `anthropic` and
   asserts the ABSTAIN paths (no key, no SDK, bad JSON) plus the happy path.
3. Move committed `*.zip` PPO checkpoints out of git into a GitHub release.
4. Delete `=*` / `pytest_*.txt` / `pcr.txt` stray root files once a script
   audit confirms they are unreferenced.
5. Bump pinned `langgraph==0.0.26` and `langchain-openai==0.0.6` to current
   minors with a smoke-test pass.

### Prior research-log context

Previous runs (most recent first, none merged to `main`):

- `claude/lucid-darwin-Ow84F` (2026-05-16) — CI workflow + `.gitignore`
  tightening for stray-file categories.
- `claude/lucid-darwin-snPHW` (2026-04-27) — Removed three malformed pip
  artifact files, added pytest CI, rewrote README, seeded research log.
