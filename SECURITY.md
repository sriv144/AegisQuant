# Security Policy

AegisQuant moves real capital when wired to a broker. We take security seriously.

## Reporting a vulnerability

Please do **not** open a public GitHub issue for security-sensitive reports. Instead, email the maintainer privately with:

- a clear description of the problem,
- a minimal proof of concept (without dumping real keys, account numbers, or order traces),
- the affected component (broker layer, RL env, reasoning logs, dashboard auth, etc.),
- the impact you expect (data leakage, unauthorized trade, denial of service, etc.).

We will acknowledge receipt within a few business days and work with you on a coordinated fix.

## In scope

- The Alpaca and Groww broker wrappers.
- The Streamlit dashboard authentication path.
- Persistence layers (SQLAlchemy audit trail, paper portfolio ledger).
- The reasoning log and SHAP attribution surfaces — they must never leak account-level data.
- Dependency vulnerabilities surfaced by `pip-audit` against `requirements.txt`.

## Out of scope

- Findings produced only against an obviously misconfigured local deployment (e.g. `.env` checked into your fork).
- Theoretical model bias or drawdown criticisms that do not constitute a security issue — please open a normal issue for those.

## Hardening checklist

When deploying AegisQuant for live or paper trading:

- [ ] `.env` is excluded from version control and stored in a secrets manager.
- [ ] Broker keys are scoped to the smallest permission set that lets the daemon trade.
- [ ] Dashboard auth is enabled and uses a strong password / SSO.
- [ ] Drawdown and regime circuit breakers are enabled in production config.
- [ ] Logs scrub broker account ids before leaving the host.
