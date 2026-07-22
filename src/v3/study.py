"""Trusted, pre-registered v3 research runner.

The runner derives every monthly target from a point-in-time provider and the
same :class:`PortfolioConstructor` used by shadow/paper.  Callers supply frozen
event/accounting tables, never performance metrics.  Metrics, promotion
evidence and the attestation hash are calculated from the resulting ledgers.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import pandas as pd

from .backtest import EventBacktestResult, EventDrivenBacktester
from .config import StrategyConfig, load_strategy_config
from .data import (
    REQUIRED_RESEARCH_DATASETS,
    PointInTimeDataProvider,
    research_data_sha256,
    snapshot_sha256,
)
from .metrics import SpyRelativeMetrics, annualized_return, compute_spy_relative_metrics
from .portfolio import PortfolioConstructor, PortfolioInputs, PortfolioPlan, SecurityMetadata
from .research import ExperimentEvidence, PreRegisteredStudy, PromotionDecision, evaluate_promotion


RUNNER_VERSION = "aegisquant-trusted-study-v1"


class StudyRunError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FrozenStudyMarketData:
    execution_prices: pd.DataFrame
    dividends: pd.DataFrame | None = None
    splits: pd.DataFrame | None = None
    delisting_recovery_prices: pd.DataFrame | None = None
    delisting_returns: pd.DataFrame | None = None
    symbol_changes: pd.DataFrame | None = None
    fractionable: Mapping[str, bool] | None = None


@dataclass(frozen=True, slots=True)
class TrustedStudyResult:
    study: PreRegisteredStudy
    research_data_sha256: str
    plans: tuple[PortfolioPlan, ...]
    champion_base: EventBacktestResult
    champion_stress: EventBacktestResult
    reference_results: Mapping[str, EventBacktestResult]
    metrics: SpyRelativeMetrics
    evidence: ExperimentEvidence
    promotion: PromotionDecision
    promotion_metrics: Mapping[str, Any]
    study_attestation_sha256: str

    def record(
        self,
        registry: Any,
        *,
        config: StrategyConfig,
        commit_sha: str,
        trial_family: str = "spy_xsmom_core_satellite_v3",
    ) -> Any:
        """Record every attempted reference and the final holdout exactly once.

        ``registry`` is intentionally duck-typed to keep this broker-free
        module independent from SQLAlchemy imports.  The production
        ``ExperimentRegistry`` implements this surface.
        """

        if config.sha256 != self.study.config_sha256:
            raise StudyRunError("study/config identity mismatch")
        parameters = {
            "runner_version": RUNNER_VERSION,
            "registration_sha256": self.study.registration_sha256,
            "champion": self.study.champion.name,
            "holdout_end": self.study.holdout.end.date().isoformat(),
        }
        registry.require_preregistration(
            config=config,
            data_manifest_sha256=self.research_data_sha256,
            commit_sha=commit_sha,
            trial_family=trial_family,
            parameters=parameters,
        )
        support_metrics = {
            key: value
            for key, value in self.promotion_metrics.items()
            if key
            not in {
                "trusted_study_runner",
                "study_attestation_sha256",
                "runner_version",
            }
        }
        for name in sorted(self.reference_results):
            registry.record_trial(
                config=config,
                data_manifest_sha256=self.research_data_sha256,
                commit_sha=commit_sha,
                trial_family=trial_family,
                split_name=f"reference:{name}",
                parameters={"reference": name},
                metrics=support_metrics,
                warnings=("reference_or_neighbor_trial",),
            )
        registry.record_trial(
            config=config,
            data_manifest_sha256=self.research_data_sha256,
            commit_sha=commit_sha,
            trial_family=trial_family,
            split_name="stress_15bps",
            parameters={"cost_bps_one_way": self.study.stress_cost_bps_one_way},
            metrics=support_metrics,
            warnings=("stress_trial",),
        )
        return registry.record_trusted_final_holdout(
            config=config,
            commit_sha=commit_sha,
            trial_family=trial_family,
            result=self,
        )


class TrustedStudyRunner:
    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or load_strategy_config()
        self.study = PreRegisteredStudy.from_config(self.config)
        self.constructor = PortfolioConstructor(self.config)

    def preregister(
        self,
        provider: PointInTimeDataProvider,
        registry: Any,
        *,
        commit_sha: str,
        trial_family: str = "spy_xsmom_core_satellite_v3",
    ) -> Any:
        """Freeze the exact study/data/commit identity before holdout evaluation."""

        readiness = provider.readiness(REQUIRED_RESEARCH_DATASETS)
        if not readiness.promotable:
            raise StudyRunError(f"provider is not promotable: {list(readiness.blockers)}")
        manifests = tuple(provider.manifest(name) for name in REQUIRED_RESEARCH_DATASETS)
        if any(manifest.warnings for manifest in manifests):
            raise StudyRunError("validated study data contains provenance warnings")
        return registry.register_preregistration(
            config=self.config,
            data_manifest_sha256=research_data_sha256(manifests),
            commit_sha=commit_sha,
            trial_family=trial_family,
            parameters=self._registration_parameters(),
        )

    def _registration_parameters(self) -> dict[str, str]:
        return {
            "runner_version": RUNNER_VERSION,
            "registration_sha256": self.study.registration_sha256,
            "champion": self.study.champion.name,
            "holdout_end": self.study.holdout.end.date().isoformat(),
        }

    @staticmethod
    def monthly_signal_schedule(
        sessions: Sequence[pd.Timestamp],
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
        index = pd.DatetimeIndex(pd.to_datetime(list(sessions)))
        if index.tz is not None:
            index = index.tz_convert("America/New_York").normalize().tz_localize(None)
        else:
            index = index.normalize()
        index = index.drop_duplicates().sort_values()
        first_by_month: dict[pd.Period, pd.Timestamp] = {}
        for session in index:
            if pd.Timestamp(start).normalize() <= session <= pd.Timestamp(end).normalize():
                first_by_month.setdefault(session.to_period("M"), session)
        schedule: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for execution_session in first_by_month.values():
            position = int(index.searchsorted(execution_session, side="left"))
            if position == 0:
                raise StudyRunError(
                    f"no prior completed signal session before {execution_session.date()}"
                )
            schedule.append((index[position - 1], execution_session))
        return tuple(schedule)

    def run(
        self,
        provider: PointInTimeDataProvider,
        market_data: FrozenStudyMarketData,
        *,
        initial_cash: float = 100_000.0,
        prior_trial_returns: pd.DataFrame | None = None,
    ) -> TrustedStudyResult:
        readiness = provider.readiness(REQUIRED_RESEARCH_DATASETS)
        if not readiness.promotable:
            raise StudyRunError(f"provider is not promotable: {list(readiness.blockers)}")
        manifests = tuple(provider.manifest(name) for name in REQUIRED_RESEARCH_DATASETS)
        if any(manifest.warnings for manifest in manifests):
            raise StudyRunError("validated study data contains provenance warnings")
        data_identity = research_data_sha256(manifests)

        execution_prices = self._session_frame(market_data.execution_prices)
        if execution_prices.index.min() > self.study.discovery.start:
            raise StudyRunError("execution history must include pre-discovery signal history")
        if execution_prices.index.max() < self.study.holdout.end:
            raise StudyRunError("execution history does not reach the locked holdout end")
        schedule = self.monthly_signal_schedule(
            execution_prices.index,
            start=self.study.discovery.start,
            end=self.study.holdout.end,
        )
        plans = self._build_plans(provider, execution_prices.index, schedule, manifests)
        targets = {plan.signal_date: plan for plan in plans}
        run_kwargs = {
            "dividends": market_data.dividends,
            "splits": market_data.splits,
            "delistings": market_data.delisting_recovery_prices,
            "delisting_returns": market_data.delisting_returns,
            "symbol_changes": market_data.symbol_changes,
            "fractionable": market_data.fractionable,
        }
        base_engine = EventDrivenBacktester(
            initial_cash=initial_cash,
            transaction_cost_bps=self.study.base_cost_bps_one_way,
        )
        stress_engine = EventDrivenBacktester(
            initial_cash=initial_cash,
            transaction_cost_bps=self.study.stress_cost_bps_one_way,
        )
        champion_base = base_engine.run(execution_prices, targets, **run_kwargs)
        champion_stress = stress_engine.run(execution_prices, targets, **run_kwargs)

        references = (*self.study.references, *self.study.neighboring_allocations)
        reference_results: dict[str, EventBacktestResult] = {}
        for reference in references:
            scaled = {
                plan.signal_date: self._reference_weights(plan, reference)
                for plan in plans
            }
            reference_results[reference.name] = base_engine.run(
                execution_prices, scaled, **run_kwargs
            )

        benchmark_levels = provider.get_benchmark_total_return(
            execution_prices.index.min(),
            self.study.holdout.end,
            pd.Timestamp(self.study.holdout.end).tz_localize("America/New_York")
            + pd.Timedelta(hours=16),
        )
        benchmark_levels = self._session_series(benchmark_levels, "SPY total return")
        spy_returns = benchmark_levels.pct_change(fill_method=None).fillna(0.0)
        oos_start = self.study.validation.start
        oos_end = self.study.holdout.end
        champion_oos, spy_oos = self._aligned_period(
            champion_base.daily_returns, spy_returns, oos_start, oos_end
        )
        stress_oos, _ = self._aligned_period(
            champion_stress.daily_returns, spy_returns, oos_start, oos_end
        )
        trial_series = [champion_oos, stress_oos]
        for result in reference_results.values():
            returns, _ = self._aligned_period(result.daily_returns, spy_returns, oos_start, oos_end)
            trial_series.append(returns)
        aligned_trials = pd.concat(trial_series, axis=1, join="inner").dropna()
        if prior_trial_returns is not None:
            prior = prior_trial_returns.copy(deep=True)
            if not isinstance(prior.index, pd.DatetimeIndex) or prior.empty:
                raise StudyRunError("prior trial returns require a non-empty DatetimeIndex")
            aligned_trials = pd.concat([aligned_trials, prior], axis=1, join="inner").dropna()
        n_trials = aligned_trials.shape[1]
        metrics = compute_spy_relative_metrics(
            champion_oos,
            spy_oos,
            n_trials=n_trials,
            trial_returns=aligned_trials.to_numpy(dtype=float),
        )
        stress_excess = annualized_return(stress_oos) - annualized_return(spy_oos)

        folds = self.study.anchored_folds(execution_prices.index)
        fold_excess = tuple(
            self._period_excess(champion_base.daily_returns, spy_returns, fold.test)
            for fold in folds
            if fold.test.start >= self.study.validation.start
        )
        neighbors = tuple(
            self._period_excess(
                reference_results[reference.name].daily_returns,
                spy_returns,
                self._combined_oos_period(),
            )
            for reference in self.study.neighboring_allocations
        )
        expected_hashes = {plan.signal_date: plan.weight_sha256 for plan in plans}
        parity = all(
            expected_hashes.get(signal) == target_hash
            for signal, target_hash in champion_base.executed_target_hashes
        ) and len(champion_base.executed_target_hashes) == len(plans)
        evidence = ExperimentEvidence(
            metrics=metrics,
            point_in_time_validated=True,
            survivorship_safe=True,
            leakage_checks_passed=True,
            pit_warnings=(),
            required_price_coverage=min(manifest.coverage for manifest in manifests),
            target_hash_parity=parity,
            fold_net_excess_returns=fold_excess,
            fold_alpha_contributions=fold_excess,
            annual_one_way_turnover=champion_base.annualized_one_way_turnover,
            stress_net_annualized_excess_return=stress_excess,
            neighboring_net_excess_returns=neighbors,
            attempted_trials=n_trials,
            recorded_trials=n_trials,
            holdout_evaluations=1,
        )
        promotion = evaluate_promotion(evidence)
        promotion_metrics = self._promotion_mapping(evidence)
        attestation = self._attestation(
            data_identity,
            plans,
            champion_base,
            champion_stress,
            promotion_metrics,
        )
        promotion_metrics = MappingProxyType(
            {
                **promotion_metrics,
                "runner_version": RUNNER_VERSION,
                "trusted_study_runner": True,
                "study_attestation_sha256": attestation,
            }
        )
        return TrustedStudyResult(
            study=self.study,
            research_data_sha256=data_identity,
            plans=plans,
            champion_base=champion_base,
            champion_stress=champion_stress,
            reference_results=MappingProxyType(reference_results),
            metrics=metrics,
            evidence=evidence,
            promotion=promotion,
            promotion_metrics=promotion_metrics,
            study_attestation_sha256=attestation,
        )

    def _build_plans(
        self,
        provider: PointInTimeDataProvider,
        sessions: pd.DatetimeIndex,
        schedule: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
        manifests: tuple[Any, ...],
    ) -> tuple[PortfolioPlan, ...]:
        plans: list[PortfolioPlan] = []
        current_holdings: frozenset[str] = frozenset()
        required_history = (
            self.config.momentum.lookback_sessions + self.config.momentum.skip_sessions
        )
        for signal_session, _execution_session in schedule:
            signal_position = int(sessions.get_loc(signal_session))
            if signal_position < required_history:
                raise StudyRunError(
                    f"insufficient signal history before {signal_session.date()}"
                )
            signal_at = signal_session.tz_localize("America/New_York") + pd.Timedelta(hours=16)
            constituents = provider.get_constituents(signal_at)
            if "symbol" not in constituents.columns:
                raise StudyRunError("constituent data requires symbol")
            symbols = tuple(sorted(set(constituents["symbol"].astype(str).str.upper())))
            master = provider.get_security_master(symbols, signal_at)
            sectors = provider.get_sector_history(symbols, signal_at)
            adv = provider.get_adv(symbols, signal_at)
            sector_weights = provider.get_benchmark_sector_weights(signal_at)
            securities = self._security_metadata(symbols, master, sectors, adv)
            history_start = sessions[signal_position - required_history]
            total_return_prices = provider.get_total_return_prices(
                (self.config.benchmark, *symbols),
                history_start,
                signal_session,
                signal_at,
            )
            plan = self.constructor.construct(
                PortfolioInputs(
                    signal_date=signal_at,
                    total_return_prices=total_return_prices,
                    securities=securities,
                    benchmark_sector_weights=sector_weights.to_dict(),
                    current_holdings=current_holdings,
                    manifests=manifests,
                )
            )
            plans.append(plan)
            current_holdings = frozenset(plan.selected_symbols)
        return tuple(plans)

    @staticmethod
    def _security_metadata(
        symbols: Sequence[str],
        master: pd.DataFrame,
        sectors: pd.DataFrame,
        adv: pd.Series,
    ) -> tuple[SecurityMetadata, ...]:
        if "symbol" not in master.columns or "issuer_id" not in master.columns:
            raise StudyRunError("security master requires symbol and issuer_id")
        if "symbol" not in sectors.columns or "sector" not in sectors.columns:
            raise StudyRunError("sector history requires symbol and sector")
        master_rows = master.drop_duplicates("symbol", keep="last").set_index("symbol")
        sector_rows = sectors.drop_duplicates("symbol", keep="last").set_index("symbol")
        result: list[SecurityMetadata] = []
        for symbol in symbols:
            try:
                result.append(
                    SecurityMetadata(
                        symbol=symbol,
                        issuer_id=str(master_rows.at[symbol, "issuer_id"]),
                        sector=str(sector_rows.at[symbol, "sector"]),
                        adv_30d=float(adv.at[symbol]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise StudyRunError(f"incomplete PIT metadata for {symbol}") from exc
        return tuple(result)

    @staticmethod
    def _reference_weights(plan: PortfolioPlan, reference: Any) -> Mapping[str, float]:
        direct = {
            symbol: weight
            for symbol, weight in plan.target_weights
            if symbol != plan.benchmark_symbol
        }
        direct_total = sum(direct.values())
        weights = {plan.benchmark_symbol: float(reference.core_weight)}
        if direct_total > 0 and reference.satellite_weight > 0:
            for symbol, weight in direct.items():
                weights[symbol] = float(reference.satellite_weight) * weight / direct_total
        else:
            weights[plan.benchmark_symbol] += float(reference.satellite_weight)
        return dict(sorted(weights.items()))

    @staticmethod
    def _session_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
            raise StudyRunError("execution prices require a non-empty DatetimeIndex")
        out = frame.copy(deep=True)
        if out.index.tz is not None:
            out.index = out.index.tz_convert("America/New_York").normalize().tz_localize(None)
        else:
            out.index = out.index.normalize()
        if out.index.has_duplicates:
            raise StudyRunError("execution prices contain duplicate NY sessions")
        return out.sort_index()

    @staticmethod
    def _session_series(series: pd.Series, name: str) -> pd.Series:
        if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
            raise StudyRunError(f"{name} requires a DatetimeIndex")
        out = series.astype(float).copy()
        if out.index.tz is not None:
            out.index = out.index.tz_convert("America/New_York").normalize().tz_localize(None)
        else:
            out.index = out.index.normalize()
        if out.index.has_duplicates or out.isna().any() or (out <= 0).any():
            raise StudyRunError(f"{name} is missing, duplicated, or non-positive")
        return out.sort_index()

    @staticmethod
    def _aligned_period(
        portfolio: pd.Series,
        spy: pd.Series,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> tuple[pd.Series, pd.Series]:
        aligned = pd.concat(
            [portfolio.rename("portfolio"), spy.rename("spy")], axis=1, join="inner"
        ).loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna()
        if len(aligned) < 32:
            raise StudyRunError("study period has fewer than 32 aligned sessions")
        return aligned["portfolio"], aligned["spy"]

    def _combined_oos_period(self) -> Any:
        from .research import StudyPeriod

        return StudyPeriod(self.study.validation.start, self.study.holdout.end)

    @staticmethod
    def _period_excess(portfolio: pd.Series, spy: pd.Series, period: Any) -> float:
        aligned = pd.concat([portfolio, spy], axis=1, join="inner").loc[
            period.start : period.end
        ].dropna()
        if len(aligned) < 2:
            raise StudyRunError(f"insufficient observations in period {period}")
        return annualized_return(aligned.iloc[:, 0]) - annualized_return(aligned.iloc[:, 1])

    @staticmethod
    def _promotion_mapping(evidence: ExperimentEvidence) -> dict[str, Any]:
        metrics = evidence.metrics
        positive_fold_fraction = sum(value > 0 for value in evidence.fold_net_excess_returns) / len(
            evidence.fold_net_excess_returns
        )
        positives = [max(0.0, value) for value in evidence.fold_alpha_contributions]
        max_contribution = max(positives) / sum(positives) if sum(positives) > 0 else math.inf
        return {
            "pit_leakage_survivorship_warnings": list(evidence.pit_warnings),
            "required_price_coverage": evidence.required_price_coverage,
            "target_hash_parity": evidence.target_hash_parity,
            "oos_annualized_excess_return": metrics.net_annualized_excess_return,
            "information_ratio": metrics.information_ratio,
            "beta": metrics.beta,
            "tracking_error": metrics.tracking_error,
            "max_drawdown": metrics.portfolio_max_drawdown,
            "spy_max_drawdown": metrics.spy_max_drawdown,
            "positive_rolling_12m_fraction": metrics.positive_rolling_12m_fraction,
            "positive_oos_fold_fraction": positive_fold_fraction,
            "annual_one_way_turnover": evidence.annual_one_way_turnover,
            "stress_15bps_excess_return": evidence.stress_net_annualized_excess_return,
            "max_fold_alpha_contribution": max_contribution,
            "psr_probability": metrics.psr,
            "dsr_probability": metrics.dsr,
            "pbo": metrics.pbo,
            "neighboring_parameters_preserve_excess_sign": all(
                value > 0 for value in evidence.neighboring_net_excess_returns
            ),
            "pre_registered_study": True,
            "holdout_end": "2026-06-30",
            "related_trial_count": evidence.attempted_trials,
        }

    def _attestation(
        self,
        data_identity: str,
        plans: Sequence[PortfolioPlan],
        base: EventBacktestResult,
        stress: EventBacktestResult,
        promotion_metrics: Mapping[str, Any],
    ) -> str:
        payload = {
            "runner_version": RUNNER_VERSION,
            "registration_sha256": self.study.registration_sha256,
            "config_sha256": self.config.sha256,
            "research_data_sha256": data_identity,
            "target_hashes": [plan.target_sha256 for plan in plans],
            "base_nav_sha256": snapshot_sha256(base.nav),
            "stress_nav_sha256": snapshot_sha256(stress.nav),
            "promotion_metrics": dict(promotion_metrics),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_study_attestation(
    result: TrustedStudyResult, config: StrategyConfig
) -> bool:
    """Recompute the cryptographic evidence identity before DB promotion."""

    if (
        result.study.config_sha256 != config.sha256
        or result.promotion_metrics.get("trusted_study_runner") is not True
        or result.promotion_metrics.get("runner_version") != RUNNER_VERSION
        or result.promotion_metrics.get("study_attestation_sha256")
        != result.study_attestation_sha256
    ):
        return False
    attested_metrics = {
        key: value
        for key, value in result.promotion_metrics.items()
        if key
        not in {
            "trusted_study_runner",
            "study_attestation_sha256",
            "runner_version",
        }
    }
    expected = TrustedStudyRunner(config)._attestation(
        result.research_data_sha256,
        result.plans,
        result.champion_base,
        result.champion_stress,
        attested_metrics,
    )
    return expected == result.study_attestation_sha256


__all__ = [
    "FrozenStudyMarketData",
    "RUNNER_VERSION",
    "StudyRunError",
    "TrustedStudyResult",
    "TrustedStudyRunner",
    "verify_study_attestation",
]
