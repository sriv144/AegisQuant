"""
AegisQuant Executive Dashboard — Phase 5
=========================================
Panels:
  1. Aggregate KPIs (OOS Sharpe, Max DD, DSR, Win Rate)
  2. IS vs OOS Sharpe per walk-forward window (bar chart)
  3. Benchmark comparison bar chart
  4. Monte Carlo percentile distribution
  5. Feature importance (permutation-based, from walk-forward JSON)
  6. Agent attribution (quant / macro / sentiment frequency from trading log)
  7. Drawdown alert configuration

Run:
    streamlit run src/ui/dashboard.py
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config  # noqa: F401

st.set_page_config(page_title="AegisQuant Control", layout="wide")
st.title("AegisQuant Executive Dashboard")
st.caption("Phase 5 — Explainability & Monitoring")

RESULTS_DIR = Path("backtest_results")
TRADING_LOG = Path("trading_history.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# Load latest walk-forward JSON
# ─────────────────────────────────────────────────────────────────────────────
if not RESULTS_DIR.exists():
    st.warning("No backtest results found. Run walk-forward first.")
    st.stop()

files = sorted(RESULTS_DIR.glob("walk_forward*.json"), key=lambda f: f.stat().st_mtime)
if not files:
    st.warning("No walk-forward JSON files found in backtest_results/")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Select backtest run", [f.name for f in reversed(files)], index=0
)
with open(RESULTS_DIR / selected_file) as fh:
    data = json.load(fh)

tickers = data.get("tickers", [])
agg = data.get("aggregate", {})
windows = data.get("windows", [])
benchmarks = data.get("benchmarks", {})
mc = data.get("monte_carlo", {})
fi = data.get("feature_importance", {})

# ─────────────────────────────────────────────────────────────────────────────
# Panel 1: Aggregate KPIs
# ─────────────────────────────────────────────────────────────────────────────
st.subheader(f"Aggregate OOS Metrics — {', '.join(tickers)}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("OOS Sharpe", f"{agg.get('sharpe_ratio', 0):.3f}")
c2.metric("OOS Sortino", f"{agg.get('sortino_ratio', 0):.3f}")
c3.metric("Max Drawdown", f"{agg.get('max_drawdown', 0):.1%}")
c4.metric("Deflated Sharpe (DSR)", f"{agg.get('deflated_sharpe_ratio', 0):.4f}")
c5.metric("Win Rate", f"{agg.get('win_rate', 0):.1%}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Panel 2: IS vs OOS Sharpe per window
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Walk-Forward Windows — In-Sample vs Out-of-Sample Sharpe")

if windows:
    win_ids = [w["window_id"] for w in windows]
    is_sharpes = [w.get("train_metrics", {}).get("sharpe_ratio", 0) for w in windows]
    oos_sharpes = [w.get("val_metrics", {}).get("sharpe_ratio", 0) for w in windows]

    x = np.arange(len(win_ids))
    width = 0.4

    fig1, ax1 = plt.subplots(figsize=(12, 4))
    bars_is  = ax1.bar(x - width/2, is_sharpes,  width, label="In-Sample Sharpe",  color="#4C72B0", alpha=0.85)
    bars_oos = ax1.bar(x + width/2, oos_sharpes, width, label="OOS Sharpe",         color="#DD8452", alpha=0.85)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"W{i}" for i in win_ids], fontsize=9)
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title("IS Sharpe vs OOS Sharpe per Walk-Forward Window")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)
else:
    st.info("No window-level data available.")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Panel 3: Benchmark comparison + Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────
col_bench, col_mc = st.columns(2)

with col_bench:
    st.subheader("Benchmark Comparison")
    if benchmarks:
        bench_rows = []
        for key, m in benchmarks.items():
            bench_rows.append({
                "Strategy": m.get("label", key),
                "Ann. Return": f"{m.get('annualised_return', 0):.1%}",
                "Sharpe": f"{m.get('sharpe_ratio', 0):.3f}",
                "Max DD": f"{m.get('max_drawdown', 0):.1%}",
                "Win Rate": f"{m.get('win_rate', 0):.1%}",
            })
        st.dataframe(pd.DataFrame(bench_rows).set_index("Strategy"), use_container_width=True)

        # Bar chart for Sharpe comparison
        bench_labels = [m.get("label", k)[:25] for k, m in benchmarks.items()]
        bench_sharpes = [m.get("sharpe_ratio", 0) for m in benchmarks.values()]
        colors = ["#2ca02c" if s > 0 else "#d62728" for s in bench_sharpes]

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.barh(bench_labels, bench_sharpes, color=colors, alpha=0.85)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Sharpe Ratio")
        ax2.set_title("Strategy vs Benchmarks (Sharpe)")
        ax2.grid(axis="x", alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("No benchmark data in this run. Re-run with updated walk_forward.py.")

with col_mc:
    st.subheader("Monte Carlo Bootstrap (10,000 simulations)")
    if mc:
        mc_rows = [
            {"Metric": "Sharpe (5th pct)",     "Value": f"{mc.get('sharpe_p5', 0):.3f}"},
            {"Metric": "Sharpe (median)",       "Value": f"{mc.get('sharpe_p50', 0):.3f}"},
            {"Metric": "Sharpe (95th pct)",     "Value": f"{mc.get('sharpe_p95', 0):.3f}"},
            {"Metric": "Ann. Return (median)",  "Value": f"{mc.get('ann_return_p50', 0):.1%}"},
            {"Metric": "Max DD (median)",       "Value": f"{mc.get('max_dd_p50', 0):.1%}"},
            {"Metric": "Probability of Ruin",   "Value": f"{mc.get('probability_of_ruin', 0):.1%}"},
            {"Metric": "Strategy Half-Life",    "Value": f"{mc.get('strategy_half_life_days', -1)} days"},
        ]
        st.dataframe(pd.DataFrame(mc_rows).set_index("Metric"), use_container_width=True)

        # Fan chart: Sharpe percentile bars
        pct_labels = ["p5", "p25", "p50", "p75", "p95"]
        pct_values = [mc.get(f"sharpe_{p}", 0) for p in pct_labels]
        colors_mc = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]

        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.bar(pct_labels, pct_values, color=colors_mc, alpha=0.85)
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.set_ylabel("Sharpe Ratio")
        ax3.set_title("Sharpe Distribution Across Simulations")
        ax3.grid(axis="y", alpha=0.3)
        st.pyplot(fig3)
        plt.close(fig3)
    else:
        st.info("No Monte Carlo data. Re-run walk_forward.py.")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Panel 4: Feature Importance (permutation-based)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Feature Importance (Permutation-Based, Averaged Across OOS Windows)")
st.caption(
    "Each bar = mean absolute change in agent action when that feature is perturbed by ±1 std. "
    "Higher = policy relies on this feature more."
)

if fi:
    top_n = st.slider("Show top N features", min_value=5, max_value=min(30, len(fi)), value=15)
    fi_sorted = sorted(fi.items(), key=lambda x: -x[1])[:top_n]
    feat_names, feat_vals = zip(*fi_sorted)

    fig4, ax4 = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    bars = ax4.barh(feat_names, feat_vals, color="royalblue", alpha=0.85)
    ax4.invert_yaxis()
    ax4.set_xlabel("Normalised Importance")
    ax4.set_title(f"Top {top_n} Features by Permutation Importance")
    ax4.grid(axis="x", alpha=0.3)
    # Annotate values
    for bar, val in zip(bars, feat_vals):
        ax4.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=8)
    st.pyplot(fig4)
    plt.close(fig4)
else:
    st.info(
        "No feature importance data found. Run walk-forward with the updated walk_forward.py "
        "(includes permutation importance computation per window)."
    )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Panel 5: Agent Attribution (from trading_history.jsonl)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Agent Attribution — Decision Frequency from Trading Log")

if TRADING_LOG.exists():
    records = []
    with open(TRADING_LOG) as fh:
        for line in fh:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    if records:
        df_log = pd.DataFrame(records)

        # Count how often each agent's signal matched the final committee decision
        agent_cols = [c for c in df_log.columns if "agent" in c.lower() and "signal" in c.lower()]
        if agent_cols:
            attribution = {col: int(df_log[col].notna().sum()) for col in agent_cols}
            fig5, ax5 = plt.subplots(figsize=(6, 4))
            ax5.barh(list(attribution.keys()), list(attribution.values()), color="#9467bd", alpha=0.85)
            ax5.set_xlabel("Number of Active Signals")
            ax5.set_title("Agent Signal Frequency")
            ax5.grid(axis="x", alpha=0.3)
            st.pyplot(fig5)
            plt.close(fig5)
        else:
            # Show raw log tail
            st.dataframe(df_log.tail(20), use_container_width=True)
    else:
        st.info("Trading log is empty.")
else:
    st.info(
        f"No trading log found at `{TRADING_LOG}`. "
        "Run `python main.py` to generate live agent decisions."
    )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Panel 6: Drawdown Alert Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Drawdown Alert Configuration")
st.caption("Configure thresholds for automated Slack/email alerts (requires alerting.py + webhook URL in .env).")

col_al1, col_al2 = st.columns(2)
with col_al1:
    dd_threshold = st.slider(
        "Max Drawdown Threshold (%)", min_value=5, max_value=50, value=10, step=1
    )
    sharpe_threshold = st.slider(
        "Minimum Acceptable Sharpe (rolling 20d)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1
    )
    webhook_url = st.text_input(
        "Slack Webhook URL", value=os.environ.get("SLACK_WEBHOOK_URL", ""), type="password"
    )

with col_al2:
    # Display current portfolio state from trading log if available
    if TRADING_LOG.exists() and records:
        df_log = pd.DataFrame(records)
        if "portfolio_value" in df_log.columns:
            initial_val = float(df_log["portfolio_value"].iloc[0])
            current_val = float(df_log["portfolio_value"].iloc[-1])
            peak_val = float(df_log["portfolio_value"].max())
            current_dd = (peak_val - current_val) / peak_val if peak_val > 0 else 0.0

            st.metric("Current Portfolio Value", f"${current_val:,.0f}")
            st.metric(
                "Current Drawdown",
                f"{current_dd:.1%}",
                delta=f"{'ALERT' if current_dd * 100 > dd_threshold else 'OK'}",
                delta_color="inverse",
            )
            if current_dd * 100 > dd_threshold:
                st.error(
                    f"DRAWDOWN ALERT: Current DD {current_dd:.1%} exceeds threshold "
                    f"{dd_threshold}%. Configure alerting.py to auto-send notifications."
                )
        else:
            st.info("Trading log exists but has no portfolio_value column yet.")
    else:
        st.info("Portfolio state unavailable — no live trading log.")

if st.button("Save Alert Config to .env"):
    env_path = Path(".env")
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    keys_to_set = {
        "MAX_DRAWDOWN_THRESHOLD": str(dd_threshold / 100),
        "MIN_SHARPE_THRESHOLD": str(sharpe_threshold),
        "SLACK_WEBHOOK_URL": webhook_url,
    }
    existing = {ln.split("=")[0]: i for i, ln in enumerate(lines) if "=" in ln}
    for key, val in keys_to_set.items():
        entry = f"{key}={val}"
        if key in existing:
            lines[existing[key]] = entry
        else:
            lines.append(entry)
    env_path.write_text("\n".join(lines) + "\n")
    st.success(".env updated. Restart the alerting daemon to apply.")

st.markdown("---")
st.caption("AegisQuant v0.5 | Phase 5 Complete")
