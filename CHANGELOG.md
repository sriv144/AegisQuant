# Changelog

All notable changes to AegisQuant are recorded in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows semantic-ish versioning (the public surface is the CLI entry points + `src/` packages, not a published library).

## [Unreleased]

### Added
- Repository scaffolding: GitHub issue templates (bug / feature), PR template with a trading-safety checklist, and this changelog. (`claude/lucid-darwin-qiarsa`)
- `RESEARCH_LOG.md` — persistent memory for the auto-researcher agent.

### Notes
- See `RESEARCH_LOG.md` for the running record of automated improvement passes, scoring, and queued next-run candidates.
- Existing open `claude/*` branches contain prior, unmerged proposals (CI workflow, badges, LICENSE, CONTRIBUTING/SECURITY, CodeQL, repo-root cleanup). This changelog will start tracking them once they merge.
