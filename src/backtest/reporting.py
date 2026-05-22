"""Generate audit-style reports from AegisQuant backtest artifacts.

This module intentionally reports weak or failed strategy results plainly. Its
job is to turn raw walk-forward JSON into a decision-quality research artifact,
not to market a trading bot.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_KEYS = [
    "annualised_return",
    "annualised_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "win_rate",
    "profit_factor",
    "deflated_sharpe_ratio",
]


def _metric(metrics: dict[str, Any], key: str) -> Any:
    value = metrics.get(key)
    return "not_available" if value is None else value


def _pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{value * 100:.2f}%"


def _num(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{value:.4f}"


def _strategy_verdict(aggregate: dict[str, Any], benchmarks: dict[str, Any]) -> str:
    sharpe = aggregate.get("sharpe_ratio")
    max_drawdown = aggregate.get("max_drawdown")
    benchmark_sharpes = [
        b.get("sharpe_ratio")
        for name, b in benchmarks.items()
        if name != "rl_strategy" and isinstance(b.get("sharpe_ratio"), (int, float))
    ]
    best_benchmark = max(benchmark_sharpes) if benchmark_sharpes else None

    if isinstance(max_drawdown, (int, float)) and max_drawdown <= -0.8:
        return "FAILED_RISK_GATE"
    if isinstance(sharpe, (int, float)) and best_benchmark is not None and sharpe < best_benchmark:
        return "UNDERPERFORMED_BENCHMARKS"
    if isinstance(sharpe, (int, float)) and sharpe > 1:
        return "PROMISING_RESEARCH_SIGNAL"
    return "INCONCLUSIVE"


def summarize_backtest(raw: dict[str, Any]) -> dict[str, Any]:
    aggregate = raw.get("aggregate", {}) or {}
    benchmarks = raw.get("benchmarks", {}) or {}
    windows = raw.get("windows", []) or []
    failed_windows = [w for w in windows if w.get("error")]
    benchmark_rows = []

    rl_sharpe = aggregate.get("sharpe_ratio")
    for name, metrics in benchmarks.items():
        benchmark_sharpe = metrics.get("sharpe_ratio")
        delta = (
            round(rl_sharpe - benchmark_sharpe, 4)
            if isinstance(rl_sharpe, (int, float)) and isinstance(benchmark_sharpe, (int, float))
            else "not_available"
        )
        benchmark_rows.append(
            {
                "name": name,
                "label": metrics.get("label", name),
                "sharpe_ratio": _metric(metrics, "sharpe_ratio"),
                "annualised_return": _metric(metrics, "annualised_return"),
                "max_drawdown": _metric(metrics, "max_drawdown"),
                "rl_sharpe_delta": delta,
            }
        )

    benchmark_rows.sort(
        key=lambda row: row["sharpe_ratio"] if isinstance(row["sharpe_ratio"], (int, float)) else -999,
        reverse=True,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tickers": raw.get("tickers", []),
        "verdict": _strategy_verdict(aggregate, benchmarks),
        "aggregate": {key: _metric(aggregate, key) for key in METRIC_KEYS},
        "oos_returns_count": raw.get("oos_returns_count", "not_available"),
        "window_count": len(windows),
        "failed_window_count": len(failed_windows),
        "benchmarks": benchmark_rows,
        "monte_carlo": raw.get("monte_carlo", {}) or {},
        "top_features": list((raw.get("feature_importance", {}) or {}).items())[:10],
    }


def render_markdown(summary: dict[str, Any], source_path: Path) -> str:
    aggregate = summary["aggregate"]
    mc = summary["monte_carlo"]
    lines = [
        "# AegisQuant Backtest Audit Report",
        "",
        f"**Source artifact:** `{source_path.as_posix()}`",
        f"**Generated:** {summary['generated_at']}",
        f"**Universe:** {', '.join(summary['tickers']) if summary['tickers'] else 'not_available'}",
        f"**Verdict:** `{summary['verdict']}`",
        "",
        "## Executive Readout",
        "",
        "This report is an audit artifact, not a profitability claim. It preserves the current walk-forward result honestly so the project can be evaluated as a quant research system with benchmark comparison, risk gates, and failure analysis.",
        "",
        "## Aggregate OOS Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Annualised return | {_pct(aggregate['annualised_return'])} |",
        f"| Annualised volatility | {_pct(aggregate['annualised_volatility'])} |",
        f"| Sharpe ratio | {_num(aggregate['sharpe_ratio'])} |",
        f"| Sortino ratio | {_num(aggregate['sortino_ratio'])} |",
        f"| Max drawdown | {_pct(aggregate['max_drawdown'])} |",
        f"| Calmar ratio | {_num(aggregate['calmar_ratio'])} |",
        f"| Win rate | {_pct(aggregate['win_rate'])} |",
        f"| Profit factor | {_num(aggregate['profit_factor'])} |",
        f"| Deflated Sharpe ratio | {_num(aggregate['deflated_sharpe_ratio'])} |",
        "",
        "## Benchmark Comparison",
        "",
        "| Benchmark | Sharpe | Ann. Return | Max DD | RL Sharpe Delta |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in summary["benchmarks"]:
        lines.append(
            f"| {row['label']} | {_num(row['sharpe_ratio'])} | {_pct(row['annualised_return'])} | {_pct(row['max_drawdown'])} | {_num(row['rl_sharpe_delta'])} |"
        )

    lines.extend(
        [
            "",
            "## Monte Carlo Downside",
            "",
            f"- Probability of ruin: {_pct(mc.get('probability_of_ruin', 'not_available'))}",
            f"- Sharpe p5/p50/p95: {_num(mc.get('sharpe_p5', 'not_available'))} / {_num(mc.get('sharpe_p50', 'not_available'))} / {_num(mc.get('sharpe_p95', 'not_available'))}",
            f"- Annualised return p5/p50/p95: {_pct(mc.get('ann_return_p5', 'not_available'))} / {_pct(mc.get('ann_return_p50', 'not_available'))} / {_pct(mc.get('ann_return_p95', 'not_available'))}",
            "",
            "## Walk-Forward Integrity",
            "",
            f"- Windows evaluated: {summary['window_count']}",
            f"- Failed windows: {summary['failed_window_count']}",
            f"- OOS return observations: {summary['oos_returns_count']}",
            "",
            "## Top Feature Attributions",
            "",
        ]
    )

    if summary["top_features"]:
        for name, value in summary["top_features"]:
            lines.append(f"- `{name}`: {_num(value)}")
    else:
        lines.append("- not_available")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The current RL strategy fails the risk gate: drawdown is extreme and benchmark-relative Sharpe is strongly negative. That is still a useful engineering result because the platform exposes the failure instead of hiding it. The next research step is to debug reward design, position constraints, transaction-cost assumptions, and benchmark leakage before presenting any alpha claim.",
            "",
            "## Next Research Step",
            "",
            "Run a constrained baseline before retraining PPO: equal-weight, momentum, and volatility-targeted allocations should become the minimum acceptance bar. Only reintroduce RL after the environment can reproduce sane benchmark behavior.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_report(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    summary = summarize_backtest(raw)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem.replace("walk_forward_", "audit_")
    json_path = output_dir / f"{stem}_summary.json"
    md_path = output_dir / f"{stem}_report.md"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(summary, input_path), encoding="utf-8")
    return md_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an audit report from a walk-forward JSON artifact.")
    parser.add_argument(
        "--input",
        default="backtest_results/walk_forward_multi_SPY_QQQ_TLT_GLD.json",
        help="Path to a walk-forward JSON result artifact.",
    )
    parser.add_argument(
        "--output-dir",
        default="backtest_results",
        help="Directory where Markdown and JSON audit reports will be written.",
    )
    args = parser.parse_args()

    md_path, json_path = generate_report(Path(args.input), Path(args.output_dir))
    print(f"Markdown report: {md_path}")
    print(f"JSON summary: {json_path}")


if __name__ == "__main__":
    main()
