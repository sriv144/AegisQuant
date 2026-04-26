# Security Policy

AegisQuant interacts with live brokerage APIs (Alpaca, Groww, Angel One),
holds API keys at runtime, and can place real orders when
`ENABLE_BROKER_EXECUTION=true`. Treat any vulnerability with that blast
radius as critical.

## Supported versions

Only the latest commit on `main` receives security fixes.

## Reporting a vulnerability

**Do not** open a public GitHub issue for any of the following:

- Credential leakage (broker keys, OpenAI / Anthropic keys, OAuth tokens).
- Bypass of `circuit_breakers.py` (time window, drawdown, kill switch).
- Order-routing path that places orders outside the configured paper / live
  mode flag.
- SQL or command injection in any execution or audit-trail path.
- Path traversal or arbitrary-write in the model-registry loader.

Instead, email the maintainer privately with a clear reproduction, the
broker / mode it affects, and your suggested fix or mitigation. You can
expect an acknowledgement within 7 days and a coordinated disclosure plan
before any public mention.

## Hardening checklist for self-hosted deployments

- Keep `ENABLE_BROKER_EXECUTION` set to `false` until paper-mode regressions
  pass on a fresh clone.
- Run the live daemon under a dedicated non-root user with no kubeconfig.
- Rotate broker API keys every 90 days and immediately if the host's
  filesystem is shared with untrusted code.
- Mount `.env` as a read-only secret; never bake keys into the Docker image.
