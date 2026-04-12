"""
Walk-Forward Optimization Engine (Multi-Asset)
==============================================
Trains a fresh PPO agent on each rolling training window and evaluates it
out-of-sample on the following validation window across N parallel assets.

Usage:
    from src.backtest.walk_forward import WalkForwardEngine
    engine = WalkForwardEngine(tickers=["SPY", "TLT", "GLD"])
    results = engine.run()
    engine.print_summary(results)
"""

import json
import logging
import math
import random
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_all_metrics
from src.backtest.multi_asset_env import MultiAssetEnv
from src.backtest.benchmarks import BenchmarkSuite
from src.backtest.monte_carlo import MonteCarloSimulator
from src.engine.regime_detector import RegimeDetector
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

@dataclass
class WindowResult:
    window_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    train_metrics: Dict[str, Any] = field(default_factory=dict)
    val_metrics: Dict[str, Any] = field(default_factory=dict)
    val_returns: List[float] = field(default_factory=list)
    val_weights: List[List[float]] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    n_train_steps: int = 0
    error: Optional[str] = None

@dataclass
class WalkForwardResults:
    tickers: List[str]
    windows: List[WindowResult] = field(default_factory=list)
    aggregate: Dict[str, Any] = field(default_factory=dict)
    oos_all_returns: List[float] = field(default_factory=list)
    benchmarks: Dict[str, Any] = field(default_factory=dict)
    monte_carlo: Dict[str, Any] = field(default_factory=dict)

class WalkForwardEngine:
    def __init__(
        self,
        tickers: List[str] = ["SPY", "TLT", "GLD"],
        train_years: int = 3,
        val_months: int = 6,
        step_months: int = 6,
        history_start: str = "2015-01-01",
        train_timesteps: int = 20_000,
        seed: int = 42,
        results_dir: str = "backtest_results",
        algo: str = "PPO"
    ):
        self.tickers = tickers
        self.train_years = train_years
        self.val_months = val_months
        self.step_months = step_months
        self.history_start = history_start
        self.train_timesteps = train_timesteps
        self.seed = seed
        self.algo = algo.upper()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.regime_detectors = {}

    def run(self, n_trials: int = 1) -> WalkForwardResults:
        self._set_seeds(self.seed)
        df_dict = self._fetch_data()

        windows = self._build_windows(df_dict)
        num_assets = len(self.tickers)
        logger.info("Running %d walk-forward windows for %d assets", len(windows), num_assets)
        print(f"\n[WalkForward] {self.tickers} | {len(windows)} windows | "
              f"{self.train_years}y train / {self.val_months}m val\n")

        results = WalkForwardResults(tickers=self.tickers)
        registry = ModelRegistry()

        for wid, (train_df_dict, val_df_dict, dates) in enumerate(windows, start=1):
            length = len(train_df_dict[self.tickers[0]])
            print(f"  Window {wid}/{len(windows)} | "
                  f"train {dates['train_start']}->{dates['train_end']} | "
                  f"val {dates['val_start']}->{dates['val_end']} "
                  f"({length} training bars)")

            result = WindowResult(
                window_id=wid,
                train_start=dates["train_start"],
                train_end=dates["train_end"],
                val_start=dates["val_start"],
                val_end=dates["val_end"],
            )

            try:
                model, train_returns, train_weights, model_zip_path = self._train(train_df_dict, seed=self.seed + wid, wid=wid)
                result.n_train_steps = self.train_timesteps
                result.train_metrics = compute_all_metrics(train_returns, train_weights, n_trials=n_trials, label=f"train_w{wid}")

                val_returns, val_weights, feat_imp = self._evaluate(model, val_df_dict)
                result.val_returns = val_returns.tolist()
                result.val_weights = val_weights.tolist()
                result.feature_importance = feat_imp
                result.val_metrics = compute_all_metrics(val_returns, val_weights, n_trials=n_trials, label=f"val_w{wid}")

                try:
                    registry.register_model(
                        model_zip_source=model_zip_path,
                        algorithm=self.algo,
                        oos_metrics=result.val_metrics,
                        hyperparams={"lr": 3e-4, "timesteps": self.train_timesteps, "seed": self.seed + wid},
                    )
                except Exception as reg_exc:
                    logger.warning("Model registry failed for window %d: %s", wid, reg_exc)

                results.oos_all_returns.extend(val_returns.tolist())

                sr = result.val_metrics.get("sharpe_ratio", 0)
                mdd = result.val_metrics.get("max_drawdown", 0)
                print(f"    -> OOS Sharpe: {sr:.3f}  MaxDD: {mdd:.1%}")

            except Exception as exc:
                result.error = str(exc)
                logger.warning("Window %d failed: %s", wid, exc)
                print(f"    -> ERROR: {exc}")

            results.windows.append(result)

        if results.oos_all_returns:
            oos = np.array(results.oos_all_returns)
            results.aggregate = compute_all_metrics(oos, n_trials=n_trials, label="aggregate_oos")
            print(f"\n[WalkForward] Aggregate OOS | "
                  f"Sharpe={results.aggregate.get('sharpe_ratio', 0):.3f} | "
                  f"MaxDD={results.aggregate.get('max_drawdown', 0):.1%} | "
                  f"DSR={results.aggregate.get('deflated_sharpe_ratio', 0):.3f}")

        # Run benchmark comparison over the full OOS date span
        val_starts = [w.val_start for w in results.windows if not w.error]
        val_ends   = [w.val_end   for w in results.windows if not w.error]
        if val_starts and val_ends:
            bench_start = min(val_starts)
            bench_end   = max(val_ends)
            print(f"\n[Benchmarks] Running comparison {bench_start} -> {bench_end} ...")
            try:
                suite = BenchmarkSuite(universe=self.tickers, seed=self.seed)
                rl_arr = np.array(results.oos_all_returns) if results.oos_all_returns else None
                results.benchmarks = suite.run(start=bench_start, end=bench_end, rl_returns=rl_arr)
                suite.print_comparison(results.benchmarks)
            except Exception as exc:
                logger.warning("Benchmark run failed: %s", exc)
                print(f"  [Benchmarks] WARNING: {exc}")

        if results.oos_all_returns:
            print("\n[Monte Carlo] Running 10,000 bootstrap simulations on OOS returns...")
            try:
                sim = MonteCarloSimulator(n_simulations=10000, seed=self.seed)
                mc_report = sim.run(np.array(results.oos_all_returns))
                results.monte_carlo = mc_report
                sim.print_report(mc_report)
            except Exception as e:
                print(f"  [Monte Carlo] ERROR: {e}")
                logger.warning("Monte Carlo failed: %s", e)

        # Tranche 2: SHAP Agent Attribution
        try:
            print("\n[Attribution] Generating SHAP values for the most recent model...")
            from src.engine.agent_attribution import AgentAttributionEngine
            
            # Use the latest val environment as background states
            last_val_env = MultiAssetEnv(val_df_dict, self.tickers, regime_detectors=self.regime_detectors)
            background_states = np.array([last_val_env._get_obs() for _ in range(last_val_env.n_steps)])
            
            attr_engine = AgentAttributionEngine(model_path=None, save_dir=self.results_dir)
            attr_engine.model = model # inject directly
            attr_engine.compute_shap_importance(background_states, num_samples=100)
            
            # Reload feature importance for the UI to read from JSON
            import json
            shap_json = self.results_dir / "shap_feature_importance.json"
            if shap_json.exists():
                with open(shap_json, "r") as f:
                    results.aggregate["feature_importance"] = json.load(f)
        except Exception as e:
            logger.warning(f"Attribution generation failed: {e}")

        self._save_results(results)
        return results

    def print_summary(self, results: WalkForwardResults) -> None:
        print(f"\n{'='*80}")
        print(f"Walk-Forward Summary — {results.tickers}")
        print(f"{'='*80}")
        header = f"{'Win':>3}  {'Val Period':>22}  {'Sharpe':>7}  {'Sortino':>8}  "
        header += f"{'MaxDD':>7}  {'WinRate':>8}  {'PF':>6}  {'p-val':>7}"
        print(header)
        print("-" * 80)
        for w in results.windows:
            if w.error:
                print(f"{w.window_id:>3}  {w.val_start} -> {w.val_end}  ERROR: {w.error}")
                continue
            m = w.val_metrics
            print(
                f"{w.window_id:>3}  {w.val_start} -> {w.val_end}  "
                f"{m.get('sharpe_ratio',0):>7.3f}  "
                f"{m.get('sortino_ratio',0):>8.3f}  "
                f"{m.get('max_drawdown',0):>7.1%}  "
                f"{m.get('win_rate',0):>8.1%}  "
                f"{m.get('profit_factor',0):>6.2f}  "
                f"{m.get('p_value',1):>7.4f}"
            )
        print("-" * 80)
        if results.aggregate:
            a = results.aggregate
            print(
                f"{'AGG':>3}  {'(all OOS windows)':>22}  "
                f"{a.get('sharpe_ratio',0):>7.3f}  "
                f"{a.get('sortino_ratio',0):>8.3f}  "
                f"{a.get('max_drawdown',0):>7.1%}  "
                f"{a.get('win_rate',0):>8.1%}  "
                f"{a.get('profit_factor',0):>6.2f}  "
                f"{a.get('p_value',1):>7.4f}"
            )
            print(f"\n  Deflated Sharpe Ratio (DSR): {a.get('deflated_sharpe_ratio',0):.4f}")
        print("=" * 80)

    def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        import yfinance as yf
        from src.data.feature_engineering import feature_engineer
        from src.data.market_data import market_data
        
        # Fetch the baseline macro data once.
        macro_df = market_data.get_macro_data(start_date=self.history_start)
        
        df_dict = {}
        for tick in self.tickers:
            df = yf.download(tick, start=self.history_start, auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError(f"No data for {tick} from {self.history_start}")
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
                
            # Compute technical indicators upfront to ensure MultiAssetEnv has them
            records = []
            for ts, row in df.iterrows():
                records.append({
                    "timestamp": str(ts), "ticker": tick,
                    "open": float(row.get("open", 0)), "high": float(row.get("high", 0)),
                    "low": float(row.get("low", 0)), "close": float(row.get("close", 0)),
                    "volume": float(row.get("volume", 0)),
                })
            features = feature_engineer.compute_technical_indicators(records)
            features = features.dropna(subset=["Volatility_20"])
            features.index = pd.to_datetime(features.index)
            
            # Join macro data
            features = features.join(macro_df, how='left')
            features = features.ffill().bfill()
            
            features["Daily_Return"] = features["close"].pct_change().fillna(0)
            features["mom_12m"] = features["close"].pct_change(252).fillna(0)
            features["mom_12m_Z"] = feature_engineer._rolling_zscore(features["mom_12m"], window=63).fillna(0)
            
            df_dict[tick] = features
            
            # Train and fit Regime Detector
            det = RegimeDetector(n_states=4, random_state=self.seed)
            det.fit(features)
            if det._is_fitted:
                self.regime_detectors[tick] = det
                
        return df_dict

    def _build_windows(self, df_dict: Dict[str, pd.DataFrame]):
        idx = df_dict[self.tickers[0]].index
        for t in self.tickers[1:]:
            idx = idx.intersection(df_dict[t].index)
        idx = idx.sort_values()

        first_date = idx.min().date()
        last_date = idx.max().date()

        windows = []
        val_start = first_date + relativedelta(years=self.train_years)

        while True:
            val_end = val_start + relativedelta(months=self.val_months)
            if val_end > last_date:
                break

            train_start = val_start - relativedelta(years=self.train_years)
            train_end = val_start - timedelta(days=1)

            train_df_dict = {t: df_dict[t].loc[str(train_start): str(train_end)] for t in self.tickers}
            val_df_dict = {t: df_dict[t].loc[str(val_start): str(val_end)] for t in self.tickers}

            # Safety check length of intersection
            if len(train_df_dict[self.tickers[0]]) < 100 or len(val_df_dict[self.tickers[0]]) < 20:
                val_start += relativedelta(months=self.step_months)
                continue

            windows.append((
                train_df_dict,
                val_df_dict,
                {
                    "train_start": str(train_start),
                    "train_end": str(train_end),
                    "val_start": str(val_start),
                    "val_end": str(val_end),
                }
            ))
            val_start += relativedelta(months=self.step_months)

        return windows

    def _train(self, train_df_dict: Dict[str, pd.DataFrame], seed: int, wid: int = 0):
        from stable_baselines3 import PPO, SAC, TD3

        self._set_seeds(seed)
        env = MultiAssetEnv(train_df_dict, self.tickers, regime_detectors=self.regime_detectors)
        n_steps = min(2048, max(64, env.n_steps // 2))

        if self.algo == "SAC":
            model = SAC("MlpPolicy", env, verbose=0, learning_rate=3e-4, seed=seed)
        elif self.algo == "TD3":
            model = TD3("MlpPolicy", env, verbose=0, learning_rate=3e-4, seed=seed)
        else:
            model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, n_steps=n_steps, seed=seed)

        model.learn(total_timesteps=self.train_timesteps)

        # Save model weights to disk for registry
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_zip_path = str(models_dir / f"wf_w{wid}_{self.algo.lower()}")
        model.save(model_zip_path)
        model_zip_path = model_zip_path + ".zip"  # SB3 appends .zip automatically

        env.reset(seed=seed)
        train_returns, train_weights = self._rollout(model, env)
        return model, train_returns, train_weights, model_zip_path

    def _evaluate(self, model, val_df_dict: Dict[str, pd.DataFrame]):
        env = MultiAssetEnv(val_df_dict, self.tickers, regime_detectors=self.regime_detectors)
        env.reset()
        returns, weights = self._rollout(model, env)
        feat_imp = self._compute_feature_importance(model, env, self.tickers)
        return returns, weights, feat_imp

    @staticmethod
    def _compute_feature_importance(model, env: MultiAssetEnv, tickers: List[str]) -> Dict[str, float]:
        """
        Permutation feature importance: for each feature, perturb it by ±1 std across
        a sample of observations and measure the mean absolute change in the predicted action.
        Higher = the policy relies on this feature more.
        """
        FEATURE_NAMES = ["vol", "rsi", "macd", "bb", "mom",
                         "curr_weight", "drawdown",
                         "regime_0", "regime_1", "regime_2", "regime_3",
                         "port_ret_5d"]
        n_assets = len(tickers)
        obs_dim = MultiAssetEnv.FEATURES_PER_ASSET * n_assets

        # Collect a sample of observations by running the env
        sample_obs = []
        obs, _ = env.reset()
        done = False
        while not done and len(sample_obs) < 100:
            sample_obs.append(obs.copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)

        if len(sample_obs) < 5:
            return {}

        sample_obs = np.array(sample_obs)  # (N, obs_dim)
        baseline_actions, _ = model.predict(sample_obs, deterministic=True)

        importance: Dict[str, float] = {}
        for feat_idx in range(obs_dim):
            asset_idx = feat_idx // MultiAssetEnv.FEATURES_PER_ASSET
            feat_local = feat_idx % MultiAssetEnv.FEATURES_PER_ASSET
            feat_name = f"{tickers[asset_idx]}_{FEATURE_NAMES[feat_local]}"

            perturbed = sample_obs.copy()
            std = float(np.std(perturbed[:, feat_idx])) or 1.0
            perturbed[:, feat_idx] += std

            perturbed_actions, _ = model.predict(perturbed, deterministic=True)
            delta = float(np.mean(np.abs(perturbed_actions - baseline_actions)))
            importance[feat_name] = round(delta, 6)

        # Normalise so values sum to 1
        total = sum(importance.values()) or 1.0
        return {k: round(v / total, 6) for k, v in sorted(importance.items(), key=lambda x: -x[1])}


    @staticmethod
    def _rollout(model, env: MultiAssetEnv):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
        return np.array(env._daily_returns), np.array(env._weights_log if hasattr(env, '_weights_log') else env.current_weights)

    def _save_results(self, results: WalkForwardResults) -> None:
        tick_str = "_".join(self.tickers)
        path = self.results_dir / f"walk_forward_multi_{tick_str}.json"
        
        # Benchmarks: strip non-serialisable objects before saving
        bench_serialisable = {}
        for k, v in results.benchmarks.items():
            bench_serialisable[k] = {mk: (float(mv) if isinstance(mv, (np.floating, np.integer)) else mv)
                                      for mk, mv in v.items() if not isinstance(mv, np.ndarray)}

        # Aggregate feature importance: average normalised importance across all valid windows
        all_fi: Dict[str, List[float]] = {}
        for w in results.windows:
            for fname, fval in w.feature_importance.items():
                all_fi.setdefault(fname, []).append(fval)
        agg_fi = {k: round(float(np.mean(v)), 6) for k, v in all_fi.items()} if all_fi else {}
        # Re-normalise
        fi_total = sum(agg_fi.values()) or 1.0
        agg_fi = {k: round(v / fi_total, 6) for k, v in sorted(agg_fi.items(), key=lambda x: -x[1])}

        data = {
            "tickers": results.tickers,
            "aggregate": results.aggregate,
            "oos_returns_count": len(results.oos_all_returns),
            "feature_importance": agg_fi,
            "benchmarks": bench_serialisable,
            "monte_carlo": results.monte_carlo,
            "windows": [
                {
                    "window_id": w.window_id,
                    "train_start": w.train_start,
                    "val_start": w.val_start,
                    "val_end": w.val_end,
                    "train_metrics": w.train_metrics,
                    "val_metrics": w.val_metrics,
                    "feature_importance": w.feature_importance,
                }
                for w in results.windows
            ],
        }
        path.write_text(json.dumps(data, indent=2))
        print(f"Results saved to {path}")

    @staticmethod
    def _set_seeds(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch; torch.manual_seed(seed)
        except ImportError:
            pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Asset Walk-Forward Backtester")
    # Full 13-asset diversified universe (equities, fixed income, commodities, FX)
    FULL_UNIVERSE = ["SPY", "QQQ", "IWM", "EEM", "EFA", "TLT", "HYG", "LQD", "GLD", "USO", "DX-Y.NYB"]
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "TLT", "GLD", "IWM", "EEM"])
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--val-months", type=int, default=6)
    parser.add_argument("--step-months", type=int, default=6)
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC", "TD3"])
    parser.add_argument("--report", action="store_true", help="Generate unified backtest report")
    args = parser.parse_args()

    engine = WalkForwardEngine(
        tickers=args.tickers, train_years=args.train_years,
        val_months=args.val_months, step_months=args.step_months,
        train_timesteps=args.timesteps, algo=args.algo
    )
    res = engine.run()
    engine.print_summary(res)
    
    if args.report:
        report_path = engine.results_dir / f"unified_report_{'_'.join(args.tickers)}.md"
        lines = [
            f"# AegisQuant Unified Report: {', '.join(args.tickers)}",
            f"\n## Algorithm: {args.algo}",
            f"**Aggregate Sharpe:** {res.aggregate.get('sharpe_ratio', 0):.3f}",
            f"**Max Drawdown:** {res.aggregate.get('max_drawdown', 0):.1%}",
            f"**Statistically Significant (DSR):** {res.aggregate.get('deflated_sharpe_ratio', 0):.3f}",
            "\n## Benchmarks:"
        ]
        if res.benchmarks:
            for b_name, b_metrics in res.benchmarks.items():
                lines.append(f"- **{b_name}**: Sharpe {b_metrics.get('sharpe_ratio',0):.3f}")
                
        if res.monte_carlo:
            lines.append("\n## Monte Carlo Downside (5th Percentile):")
            lines.append(f"- **Sharpe:** {res.monte_carlo.get('sharpe_p5', 0)}")
            lines.append(f"- **Probability of Ruin:** {res.monte_carlo.get('probability_of_ruin', 0)}")
            
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nUnified Report generated at {report_path}")
