"""
Agent Journal - persistent markdown memory.

Every cron fire:
  * reads `memory/strategy.md` + `memory/learnings.md` + tail of `memory/trade_log.md`
  * seeds them into AgentState so the LLM committee sees its own prior context
  * appends a 3-line entry to `memory/trade_log.md` at the end of the run

The weekly review job appends a dated H2 block to `memory/learnings.md`.

Stateless, filesystem-only. No DB dependency.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

MEMORY_DIR = Path("memory")
STRATEGY_FILE = MEMORY_DIR / "strategy.md"
TRADE_LOG_FILE = MEMORY_DIR / "trade_log.md"
LEARNINGS_FILE = MEMORY_DIR / "learnings.md"


def _ensure_dir() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    except Exception:
        return ""


# Readers

def read_strategy() -> str:
    return _read(STRATEGY_FILE)


def read_learnings(tail_chars: int = 4000) -> str:
    txt = _read(LEARNINGS_FILE)
    return txt[-tail_chars:] if len(txt) > tail_chars else txt


def read_trade_log(tail_lines: int = 50) -> str:
    txt = _read(TRADE_LOG_FILE)
    if not txt:
        return ""
    lines = txt.splitlines()
    return "\n".join(lines[-tail_lines:])


# Writers

def append_run(
    ts: str,
    cb_reason: str,
    weights: Dict[str, float],
    trade_types: Dict[str, str],
    truth_summary: Dict[str, object] | None = None,
) -> None:
    """Append a 3-line block summarising this run."""
    _ensure_dir()
    active = {t: w for t, w in weights.items() if abs(w) >= 1e-3}
    trades = {t: tt for t, tt in trade_types.items() if tt != "SKIP"}

    block = [f"### {ts}"]
    block.append(
        f"CB={cb_reason} - active={len(active)} - "
        f"CNC={sum(1 for v in trades.values() if v=='CNC')} - "
        f"MIS={sum(1 for v in trades.values() if v=='MIS')}"
    )
    if active:
        # Top 6 by magnitude, round to 3dp
        top = sorted(active.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
        w_str = " - ".join(f"{t}={w:+.2f}" for t, w in top)
        block.append(f"weights: {w_str}")
    else:
        block.append("weights: (none)")

    if truth_summary:
        aegis = float(truth_summary.get("aegis_return", 0.0) or 0.0) * 100
        bench = float(truth_summary.get("benchmark_return", 0.0) or 0.0) * 100
        excess = float(truth_summary.get("excess_return", 0.0) or 0.0) * 100
        verdict = truth_summary.get("verdict", "INSUFFICIENT_DATA")
        block.append(
            f"truth: Aegis {aegis:+.2f}% vs NIFTY {bench:+.2f}% "
            f"(excess {excess:+.2f}%) - {verdict}"
        )

    entry = "\n".join(block) + "\n\n"
    try:
        with TRADE_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(entry)
    except Exception as e:
        print(f"[Journal] append_run failed: {e}")


def write_weekly_review(summary_md: str) -> None:
    """Append a dated review block to learnings.md."""
    _ensure_dir()
    stamp = datetime.utcnow().strftime("%Y-%m-%d")
    block = f"## Weekly Review - {stamp}\n\n{summary_md.strip()}\n\n"
    try:
        with LEARNINGS_FILE.open("a", encoding="utf-8") as f:
            f.write(block)
    except Exception as e:
        print(f"[Journal] write_weekly_review failed: {e}")
