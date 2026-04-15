"""
AegisQuant Executive Dashboard — Phase 5 + Live Trading
=========================================================
Tabs:
  Live Trading  — real-time paper trading decisions from aegisquant_live.db
  Backtest      — walk-forward IS/OOS, benchmarks, Monte Carlo, feature importance

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

RESULTS_DIR = Path("backtest_results")
TRADING_LOG = Path("trading_history.jsonl")
DB_PATH = Path("aegisquant_live.db")

tab_live, tab_backtest = st.tabs(["Live Paper Trading", "Backtest Results"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — LIVE PAPER TRADING
# ─────────────────────────────────────────────────────────────────────────────
with tab_live:
    st.subheader("Live Paper Trading — Decision Log")
    st.caption("Reads from `aegisquant_live.db`. Refresh the page to see the latest cycle.")

    UNIVERSE = ["SPY", "QQQ", "TLT", "GLD"]

    def _load_live_decisions(limit: int = 50) -> pd.DataFrame:
        """Load the most recent decisions from the SQLite audit DB."""
        try:
            from sqlalchemy import create_engine, text
            db_url = os.getenv("POSTGRES_URL", f"sqlite:///{DB_PATH}")
            engine = create_engine(db_url)
            query = text(
                "SELECT id, timestamp, circuit_breaker_status, model_version, "
                "final_weights, state_vector, transaction_costs "
                "FROM decisions ORDER BY id DESC LIMIT :lim"
            )
            with engine.connect() as conn:
                df = pd.read_sql(query, conn, params={"lim": limit})
            return df
        except Exception as e:
            st.warning(f"Could not load decisions: {e}")
            return pd.DataFrame()

    df_decisions = _load_live_decisions()

    if df_decisions.empty:
        st.info(
            "No live decisions found yet. Run `python main.py --now` to generate a cycle, "
            "then refresh this page."
        )
    else:
        # ── KPI row ───────────────────────────────────────────────────────────
        latest = df_decisions.iloc[0]

        try:
            final_weights = json.loads(latest["final_weights"])
        except Exception:
            final_weights = [0.0] * len(UNIVERSE)

        try:
            state_vec = json.loads(latest["state_vector"])
            live_drawdown = float(state_vec[0]) if len(state_vec) > 0 else 0.0
            live_vix = float(state_vec[1]) if len(state_vec) > 1 else 0.0
        except Exception:
            live_drawdown, live_vix = 0.0, 0.0

        cb_status = latest.get("circuit_breaker_status", "OK")
        model_ver = latest.get("model_version", "unknown")
        slippage = float(latest.get("transaction_costs") or 0.0)
        total_cycles = len(df_decisions)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Cycles Logged", total_cycles)
        k2.metric("Live VIX", f"{live_vix:.1f}")
        k3.metric("Portfolio Drawdown", f"{live_drawdown:.2%}")
        k4.metric("Circuit Breaker", cb_status, delta=None)
        k5.metric("Last Slippage", f"{slippage:.2f} bps")

        st.markdown("---")

        # ── Current weights bar chart ─────────────────────────────────────────
        col_chart, col_meta = st.columns([1.4, 1])

        with col_chart:
            st.subheader("Latest Target Weights")
            weights_arr = np.array(final_weights[: len(UNIVERSE)], dtype=float)
            colors = ["#2ca02c" if w >= 0 else "#d62728" for w in weights_arr]
            fig_w, ax_w = plt.subplots(figsize=(6, 3))
            ax_w.bar(UNIVERSE[: len(weights_arr)], weights_arr, color=colors, alpha=0.85)
            ax_w.axhline(0, color="black", linewidth=0.8)
            ax_w.set_ylabel("Target Weight")
            ax_w.set_title(f"Cycle #{latest['id']}  |  {latest['timestamp'][:19]}")
            ax_w.set_ylim(-1.1, 1.1)
            ax_w.grid(axis="y", alpha=0.3)
            for j, (tick, w) in enumerate(zip(UNIVERSE[: len(weights_arr)], weights_arr)):
                ax_w.text(j, w + (0.03 if w >= 0 else -0.06), f"{w:.2f}", ha="center", fontsize=9)
            st.pyplot(fig_w)
            plt.close(fig_w)

        with col_meta:
            st.subheader("Cycle Metadata")
            st.table(
                pd.DataFrame(
                    {
                        "Field": ["Model", "CB Status", "VIX", "Drawdown", "Slippage"],
                        "Value": [
                            model_ver,
                            cb_status,
                            f"{live_vix:.2f}",
                            f"{live_drawdown:.2%}",
                            f"{slippage:.2f} bps",
                        ],
                    }
                ).set_index("Field")
            )

        st.markdown("---")

        # ── Weight history heatmap ────────────────────────────────────────────
        st.subheader("Weight History (Last 20 Cycles)")

        weight_rows = []
        for _, row in df_decisions.head(20).iterrows():
            try:
                ww = json.loads(row["final_weights"])
                weight_rows.append([row["timestamp"][:16]] + [round(float(w), 3) for w in ww[: len(UNIVERSE)]])
            except Exception:
                continue

        if weight_rows:
            df_wh = pd.DataFrame(weight_rows, columns=["Timestamp"] + UNIVERSE[: len(weight_rows[0]) - 1])
            df_wh = df_wh.set_index("Timestamp")

            fig_h, ax_h = plt.subplots(figsize=(10, max(3, len(df_wh) * 0.4)))
            im = ax_h.imshow(
                df_wh.values.astype(float),
                aspect="auto",
                cmap="RdYlGn",
                vmin=-1,
                vmax=1,
            )
            ax_h.set_xticks(range(len(df_wh.columns)))
            ax_h.set_xticklabels(df_wh.columns, fontsize=10)
            ax_h.set_yticks(range(len(df_wh.index)))
            ax_h.set_yticklabels(df_wh.index, fontsize=8)
            plt.colorbar(im, ax=ax_h, label="Weight")
            ax_h.set_title("Target Weights Across Cycles (green=long, red=short)")
            st.pyplot(fig_h)
            plt.close(fig_h)

        st.markdown("---")

        # ── Raw decision log table ────────────────────────────────────────────
        st.subheader("Decision Log (Last 20 Rows)")
        display_cols = ["id", "timestamp", "model_version", "circuit_breaker_status", "transaction_costs"]
        st.dataframe(
            df_decisions[display_cols].head(20).rename(
                columns={
                    "circuit_breaker_status": "CB Status",
                    "transaction_costs": "Slippage (bps)",
                    "model_version": "Model",
                }
            ),
            use_container_width=True,
        )

        # ── SQLite Viewer tip ─────────────────────────────────────────────────
        st.info(
            "Tip: You have the **SQLite Viewer** extension installed in VSCode. "
            "Open `aegisquant_live.db` directly in the editor to browse all rows — "
            "no terminal needed."
        )

    st.markdown("---")

    # ── Drawdown Alert Config ─────────────────────────────────────────────────
    st.subheader("Drawdown Alert Configuration")
    col_al1, col_al2 = st.columns(2)
    with col_al1:
        dd_threshold = st.slider("Max Drawdown Threshold (%)", 5, 50, 10, 1)
        sharpe_threshold = st.slider("Min Acceptable Sharpe (rolling 20d)", -2.0, 2.0, 0.0, 0.1)
        webhook_url = st.text_input(
            "Slack Webhook URL", value=os.environ.get("SLACK_WEBHOOK_URL", ""), type="password"
        )
    with col_al2:
        if not df_decisions.empty:
            st.metric("Current Drawdown", f"{live_drawdown:.1%}")
            if live_drawdown * 100 > dd_threshold:
                st.error(
                    f"DRAWDOWN ALERT: {live_drawdown:.1%} exceeds threshold {dd_threshold}%."
                )
            else:
                st.success(f"Drawdown within threshold ({dd_threshold}%)")
        else:
            st.info("Run a cycle first to see live drawdown.")

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
        st.success(".env updated.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — BACKTEST RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_backtest:
    st.subheader("Backtest Results")
    st.caption("Phase 5 — Explainability & Monitoring")

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

    # ── Panel 1: Aggregate KPIs ───────────────────────────────────────────────
    st.subheader(f"Aggregate OOS Metrics — {', '.join(tickers)}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("OOS Sharpe", f"{agg.get('sharpe_ratio', 0):.3f}")
    c2.metric("OOS Sortino", f"{agg.get('sortino_ratio', 0):.3f}")
    c3.metric("Max Drawdown", f"{agg.get('max_drawdown', 0):.1%}")
    c4.metric("Deflated Sharpe (DSR)", f"{agg.get('deflated_sharpe_ratio', 0):.4f}")
    c5.metric("Win Rate", f"{agg.get('win_rate', 0):.1%}")
    st.markdown("---")

    # ── Panel 2: IS vs OOS Sharpe ─────────────────────────────────────────────
    st.subheader("Walk-Forward Windows — In-Sample vs Out-of-Sample Sharpe")
    if windows:
        win_ids = [w["window_id"] for w in windows]
        is_sharpes = [w.get("train_metrics", {}).get("sharpe_ratio", 0) for w in windows]
        oos_sharpes = [w.get("val_metrics", {}).get("sharpe_ratio", 0) for w in windows]
        x = np.arange(len(win_ids))
        width = 0.4
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.bar(x - width / 2, is_sharpes, width, label="In-Sample Sharpe", color="#4C72B0", alpha=0.85)
        ax1.bar(x + width / 2, oos_sharpes, width, label="OOS Sharpe", color="#DD8452", alpha=0.85)
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

    # ── Panel 3: Benchmark + Monte Carlo ──────────────────────────────────────
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
            st.info("No benchmark data in this run.")

    with col_mc:
        st.subheader("Monte Carlo Bootstrap (10,000 simulations)")
        if mc:
            mc_rows = [
                {"Metric": "Sharpe (5th pct)", "Value": f"{mc.get('sharpe_p5', 0):.3f}"},
                {"Metric": "Sharpe (median)", "Value": f"{mc.get('sharpe_p50', 0):.3f}"},
                {"Metric": "Sharpe (95th pct)", "Value": f"{mc.get('sharpe_p95', 0):.3f}"},
                {"Metric": "Ann. Return (median)", "Value": f"{mc.get('ann_return_p50', 0):.1%}"},
                {"Metric": "Max DD (median)", "Value": f"{mc.get('max_dd_p50', 0):.1%}"},
                {"Metric": "Probability of Ruin", "Value": f"{mc.get('probability_of_ruin', 0):.1%}"},
                {"Metric": "Strategy Half-Life", "Value": f"{mc.get('strategy_half_life_days', -1)} days"},
            ]
            st.dataframe(pd.DataFrame(mc_rows).set_index("Metric"), use_container_width=True)
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

    # ── Panel 4: Feature Importance ───────────────────────────────────────────
    st.subheader("Feature Importance (Permutation-Based)")
    st.caption(
        "Each bar = mean absolute change in agent action when that feature is perturbed by ±1 std."
    )
    if fi:
        top_n = st.slider("Show top N features", 5, min(30, len(fi)), 15)
        fi_sorted = sorted(fi.items(), key=lambda x: -x[1])[:top_n]
        feat_names, feat_vals = zip(*fi_sorted)
        fig4, ax4 = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
        bars = ax4.barh(feat_names, feat_vals, color="royalblue", alpha=0.85)
        ax4.invert_yaxis()
        ax4.set_xlabel("Normalised Importance")
        ax4.set_title(f"Top {top_n} Features by Permutation Importance")
        ax4.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, feat_vals):
            ax4.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=8)
        st.pyplot(fig4)
        plt.close(fig4)
    else:
        st.info("No feature importance data found. Run walk-forward with updated walk_forward.py.")
    st.markdown("---")

st.caption("AegisQuant v0.6 | Live Trading + Phase 5 Complete")
