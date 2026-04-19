# Research Log

Tracks autonomous improvements made by Auto-Researcher v4.

## 2026-04-19 — Cruft cleanup
- Branch: `claude/focused-newton-KSTnM`
- Removed accidental pip-install artifacts (`=0.2.36`, `=0.29.1`, `=2.3.0`) created by `pip install pkg=ver` typos
- Removed committed pytest output dumps (`pytest_clean.txt`, `pytest_output.txt`, `pytest_run.txt`) and scratch file `pcr.txt`
- Hardened `.gitignore` to prevent recurrence (patterns: `=*`, `pytest_*.txt`, `pcr.txt`, `/*.zip`)
- Resume signal: cleaner root for recruiters cloning the repo; no unexplained junk files
- Skipped this run: README polish (already strong), CI workflow (needs test-time discussion), removing `.zip` model checkpoints from root (would destroy existing work — owner should decide)
