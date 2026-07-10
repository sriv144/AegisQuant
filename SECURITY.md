# Security Policy

## Supported Versions

Only the latest commit on `main` is actively maintained. Older branches and
tags are not patched.

## Reporting a Vulnerability

If you discover a security issue, **please do not open a public GitHub issue.**
Instead, contact the maintainer directly so the fix can be coordinated before
disclosure. A response can be expected within a reasonable window depending on
severity.

When reporting, please include:

- A description of the issue and its impact
- A minimal reproduction (script, request, or sequence of steps)
- Any logs, stack traces, or affected file paths
- Your suggested fix or mitigation, if you have one

## Secrets Hygiene

AegisQuant integrates with broker APIs (Alpaca, Groww) and LLM providers
(Anthropic). To keep secrets safe:

- Never commit a populated `.env` file. The repository contract is
  `.env.example` only.
- Never paste live API keys into issues, PRs, or commit messages.
- Rotate any key that has been pushed to a public commit, even briefly,
  including force-pushed branches.
- Treat `model_registry/` and trained `.zip` weights as potentially sensitive
  if they were trained on private data.

## Trading Safety

This project executes real trades when configured against a live broker. To
prevent accidental order placement:

- Default to Alpaca **paper trading** endpoints unless you have explicitly
  flipped to live.
- Keep the drawdown circuit breaker enabled in production runs.
- Audit `backtest_results/` and the SHAP attribution output before promoting
  a model from `model_registry/` to live execution.

## Dependencies

Dependencies are pinned in `requirements.txt`. Re-pin before each release
rather than letting transitive versions float. Run `pip-audit` (or equivalent)
periodically to surface known CVEs in the dependency graph.
