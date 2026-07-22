"""Pure benchmark-aware core/satellite portfolio construction."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.execution.v3.ids import build_target_hash

from .config import StrategyConfig, load_strategy_config
from .data import DataManifest, DataReadiness, SIGNAL_TIME_DATASETS


class DataFailure(RuntimeError):
    """A fail-closed market-data error, never a signal to buy SPY."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class SecurityMetadata:
    symbol: str
    issuer_id: str
    sector: str
    adv_30d: float

    def __post_init__(self) -> None:
        if not self.symbol or not self.issuer_id or not self.sector:
            raise ValueError("security metadata requires symbol, issuer_id and sector")
        if not math.isfinite(self.adv_30d) or self.adv_30d <= 0:
            raise ValueError("adv_30d must be a positive finite number")


@dataclass(frozen=True, slots=True)
class PortfolioInputs:
    signal_date: pd.Timestamp
    total_return_prices: pd.DataFrame
    securities: tuple[SecurityMetadata, ...]
    benchmark_sector_weights: Mapping[str, float]
    current_holdings: frozenset[str] = frozenset()
    manifests: tuple[DataManifest, ...] = ()
    current_drawdown: float = 0.0
    prior_de_risked: bool = False
    satellite_reentry_approved: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "signal_date", pd.Timestamp(self.signal_date))
        object.__setattr__(self, "securities", tuple(self.securities))
        object.__setattr__(self, "current_holdings", frozenset(self.current_holdings))
        sectors = {str(k): float(v) for k, v in self.benchmark_sector_weights.items()}
        object.__setattr__(self, "benchmark_sector_weights", MappingProxyType(dict(sorted(sectors.items()))))
        object.__setattr__(self, "manifests", tuple(self.manifests))
        if not 0 <= self.current_drawdown <= 1:
            raise ValueError("drawdown must be a positive magnitude in [0, 1]")


@dataclass(frozen=True, slots=True)
class SignalRecord:
    symbol: str
    momentum_return: float
    path_smoothness: float
    score: float
    rank: int
    percentile: float
    retained: bool


@dataclass(frozen=True, slots=True)
class PortfolioPlan:
    strategy_id: str
    strategy_version: str
    config_sha256: str
    benchmark_symbol: str
    signal_date: pd.Timestamp
    target_weights: tuple[tuple[str, float], ...]
    cash_weight: float
    selected_symbols: tuple[str, ...]
    signals: tuple[SignalRecord, ...]
    portfolio_beta: float
    tracking_error: float
    max_active_sector_deviation: float
    promotable: bool
    promotion_blockers: tuple[str, ...]
    diagnostics: tuple[tuple[str, str], ...]
    weight_sha256: str
    target_sha256: str

    @property
    def weights(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.target_weights))

    @property
    def invested_weight(self) -> float:
        return float(sum(weight for _, weight in self.target_weights))

    @property
    def satellite_weight(self) -> float:
        benchmark = next(weight for symbol, weight in self.target_weights if symbol == self.benchmark_symbol)
        return max(0.0, self.invested_weight - benchmark)


class PortfolioConstructor:
    """Construct deterministic target weights from frozen point-in-time inputs."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or load_strategy_config()

    def construct(self, inputs: PortfolioInputs) -> PortfolioPlan:
        readiness = DataReadiness.evaluate(inputs.manifests) if inputs.manifests else DataReadiness(
            promotable=False,
            manifests=(),
            blockers=("missing:data_manifests",),
        )
        risk = self.config.risk
        de_risked = inputs.current_drawdown >= risk.de_risk_drawdown or (
            inputs.prior_de_risked
            and not (
                inputs.current_drawdown < risk.warning_drawdown
                and inputs.satellite_reentry_approved
            )
        )
        if de_risked:
            # Operational loss containment must remain available when signal or
            # constituent data is unavailable.  This narrow plan can only cut
            # the satellite to zero and retain the configured 69% SPY core.
            return self._drawdown_plan(inputs, readiness)

        prices, securities = self._validate_and_prepare(inputs)
        self._reject_lookahead_manifests(inputs)

        signals = self._rank_signals(prices, securities, inputs.current_holdings)
        selected = self._select_by_sector(signals, securities, inputs.benchmark_sector_weights)

        selected_meta = {security.symbol: security for security in securities if security.symbol in selected}
        trial = self.config.allocation.satellite_weight
        last_failure = ""
        while trial >= -1e-12:
            satellite_budget = max(0.0, round(trial, 12))
            weights = self._allocate_weights(selected, selected_meta, inputs.benchmark_sector_weights, satellite_budget)
            beta, tracking_error, sector_deviation = self._risk_metrics(
                weights, prices, selected_meta, inputs.benchmark_sector_weights
            )
            failures: list[str] = []
            if not risk.min_beta <= beta <= risk.max_beta:
                failures.append(f"beta={beta:.6f}")
            if tracking_error > risk.tracking_error_hard_max + 1e-12:
                failures.append(f"tracking_error={tracking_error:.6f}")
            if sector_deviation > risk.active_sector_hard_max + 1e-12:
                failures.append(f"sector_deviation={sector_deviation:.6f}")
            if not failures:
                actual_satellite = sum(
                    weight for symbol, weight in weights.items() if symbol != self.config.benchmark
                )
                construction = "normal"
                if satellite_budget == 0:
                    construction = "spy_risk_fallback"
                elif satellite_budget < self.config.allocation.satellite_weight:
                    construction = "satellite_scaled"
                elif actual_satellite < satellite_budget - 1e-12:
                    construction = "capacity_released_to_core"
                return self._finalize(
                    inputs,
                    weights,
                    cash_weight=self.config.allocation.cash_weight,
                    signals=signals,
                    selected=tuple(s for s in selected if weights.get(s, 0) > 0),
                    prices=prices,
                    readiness=readiness,
                    diagnostics={
                        "construction": construction,
                        "requested_satellite_weight": f"{self.config.allocation.satellite_weight:.12f}",
                        "risk_budget_satellite_weight": f"{satellite_budget:.12f}",
                        "actual_satellite_weight": f"{actual_satellite:.12f}",
                        "risk_step_down_reason": last_failure,
                        "drawdown_warning": str(inputs.current_drawdown >= risk.warning_drawdown).lower(),
                    },
                    precomputed_risk=(beta, tracking_error, sector_deviation),
                )
            last_failure = ",".join(failures)
            trial = round(trial - risk.satellite_step_down, 12)

        # A 99% SPY / 1% cash portfolio has beta ~= 0.99 and should always pass
        # the configured bounds.  Reaching this line signals an internal bug.
        raise RuntimeError(f"risk fallback could not produce a valid portfolio: {last_failure}")

    def _drawdown_plan(
        self, inputs: PortfolioInputs, readiness: DataReadiness
    ) -> PortfolioPlan:
        weights = ((self.config.benchmark, round(self.config.allocation.core_weight, 12)),)
        cash_weight = round(1.0 - self.config.allocation.core_weight, 12)
        return PortfolioPlan(
            strategy_id=self.config.strategy_id,
            strategy_version=self.config.version,
            config_sha256=self.config.sha256,
            benchmark_symbol=self.config.benchmark,
            signal_date=inputs.signal_date,
            target_weights=weights,
            cash_weight=cash_weight,
            selected_symbols=(),
            signals=(),
            # These describe the explicit core/cash mix, not a normal-risk
            # estimate.  Normal beta/TE/sector gates do not apply to kill mode.
            portfolio_beta=self.config.allocation.core_weight,
            tracking_error=0.0,
            max_active_sector_deviation=0.0,
            promotable=readiness.promotable,
            promotion_blockers=readiness.blockers,
            diagnostics=(
                ("construction", "drawdown_de_risk"),
                ("current_drawdown", f"{inputs.current_drawdown:.12f}"),
                (
                    "satellite_reentry_approved",
                    str(inputs.satellite_reentry_approved).lower(),
                ),
            ),
            weight_sha256=build_target_hash(dict(weights)),
            target_sha256=self._target_provenance_hash(
                inputs.signal_date, dict(weights), cash_weight
            ),
        )

    def _validate_and_prepare(
        self, inputs: PortfolioInputs
    ) -> tuple[pd.DataFrame, tuple[SecurityMetadata, ...]]:
        if not inputs.securities:
            raise DataFailure("empty_universe", "no point-in-time constituents were supplied")
        symbols = [security.symbol for security in inputs.securities]
        if len(symbols) != len(set(symbols)):
            raise DataFailure("duplicate_symbol", "constituent metadata contains duplicate symbols")

        sectors = inputs.benchmark_sector_weights
        if not sectors or any(not math.isfinite(v) or v < 0 for v in sectors.values()):
            raise DataFailure("invalid_sector_weights", "benchmark sector weights are missing or invalid")
        if not math.isclose(sum(sectors.values()), 1.0, abs_tol=1e-8):
            raise DataFailure("invalid_sector_weights", "benchmark sector weights must sum to 1")
        unknown = sorted({s.sector for s in inputs.securities}.difference(sectors))
        if unknown:
            raise DataFailure("unknown_sector", f"constituents reference sectors absent from benchmark: {unknown}")

        prices = inputs.total_return_prices.copy(deep=True)
        if prices.empty or not isinstance(prices.index, pd.DatetimeIndex):
            raise DataFailure("invalid_prices", "total-return prices require a non-empty DatetimeIndex")
        if prices.index.has_duplicates:
            raise DataFailure("duplicate_price_date", "price index contains duplicate sessions")
        # Research prices are keyed by New York session labels, while frozen
        # signal timestamps are timezone-aware instants.  Compare normalized
        # session labels so an honest 16:00 ET month-end signal works with a
        # tz-naive daily price index without pandas aware/naive errors.
        if prices.index.tz is not None:
            prices.index = prices.index.tz_convert("America/New_York").normalize().tz_localize(None)
        else:
            prices.index = prices.index.normalize()
        if prices.index.has_duplicates:
            raise DataFailure("duplicate_price_date", "price timestamps collapse to duplicate NY sessions")
        signal_session = pd.Timestamp(inputs.signal_date)
        if signal_session.tzinfo is not None:
            signal_session = signal_session.tz_convert("America/New_York").normalize().tz_localize(None)
        else:
            signal_session = signal_session.normalize()
        prices = prices.sort_index().loc[lambda frame: frame.index <= signal_session]
        needed = self.config.momentum.lookback_sessions + self.config.momentum.skip_sessions + 1
        if len(prices) < needed:
            raise DataFailure("insufficient_history", f"need at least {needed} sessions, found {len(prices)}")
        if self.config.benchmark not in prices.columns:
            raise DataFailure("missing_benchmark", f"{self.config.benchmark} total-return prices are absent")

        deduped = self._dedupe_issuers(inputs.securities)
        usable = 0
        for security in deduped:
            if security.symbol not in prices.columns:
                continue
            sample = pd.to_numeric(prices[security.symbol], errors="coerce").iloc[-needed:]
            usable += int(sample.notna().all() and np.isfinite(sample).all() and (sample > 0).all())
        coverage = usable / len(deduped)
        if coverage + 1e-12 < self.config.risk.minimum_price_coverage:
            raise DataFailure(
                "insufficient_price_coverage",
                f"usable signal coverage {coverage:.4f} is below {self.config.risk.minimum_price_coverage:.4f}",
            )
        return prices, deduped

    @staticmethod
    def _dedupe_issuers(securities: Sequence[SecurityMetadata]) -> tuple[SecurityMetadata, ...]:
        chosen: dict[str, SecurityMetadata] = {}
        for security in securities:
            prior = chosen.get(security.issuer_id)
            if prior is None or (-security.adv_30d, security.symbol) < (-prior.adv_30d, prior.symbol):
                chosen[security.issuer_id] = security
        return tuple(sorted(chosen.values(), key=lambda security: security.symbol))

    def _reject_lookahead_manifests(self, inputs: PortfolioInputs) -> None:
        signal = inputs.signal_date
        if signal.tzinfo is None:
            signal = signal.tz_localize("UTC")
        else:
            signal = signal.tz_convert("UTC")
        future = sorted(
            m.dataset
            for m in inputs.manifests
            if m.dataset in SIGNAL_TIME_DATASETS
            and pd.Timestamp(m.availability_at) > signal
        )
        if future:
            raise DataFailure("lookahead_manifest", f"datasets were unavailable at signal time: {future}")
        low_coverage = sorted(
            m.dataset
            for m in inputs.manifests
            if m.dataset in SIGNAL_TIME_DATASETS
            and m.coverage + 1e-12 < self.config.risk.minimum_price_coverage
        )
        if low_coverage:
            raise DataFailure("manifest_coverage", f"dataset manifests fail coverage: {low_coverage}")

    def _rank_signals(
        self,
        prices: pd.DataFrame,
        securities: Sequence[SecurityMetadata],
        current_holdings: frozenset[str],
    ) -> tuple[SignalRecord, ...]:
        momentum = self.config.momentum
        end_index = len(prices) - 1 - momentum.skip_sessions
        start_index = end_index - momentum.lookback_sessions
        raw: list[tuple[str, float, float]] = []
        for security in securities:
            if security.symbol not in prices.columns:
                continue
            path = pd.to_numeric(prices[security.symbol], errors="coerce").iloc[start_index : end_index + 1]
            if len(path) != momentum.lookback_sessions + 1 or path.isna().any() or (path <= 0).any():
                continue
            total_return = float(path.iloc[-1] / path.iloc[0] - 1.0)
            log_path = np.log(path.to_numpy(dtype=float))
            x = np.arange(len(log_path), dtype=float)
            fitted = np.polyval(np.polyfit(x, log_path, 1), x)
            ss_total = float(np.square(log_path - log_path.mean()).sum())
            r_squared = 0.0 if ss_total <= 1e-20 else 1.0 - float(np.square(log_path - fitted).sum()) / ss_total
            raw.append((security.symbol, total_return, max(0.0, min(1.0, r_squared))))
        if not raw:
            raise DataFailure("no_signals", "no constituent had a valid 252/21 signal window")

        frame = pd.DataFrame(raw, columns=["symbol", "momentum", "smoothness"]).set_index("symbol")
        momentum_rank = frame["momentum"].rank(method="average", pct=True)
        smoothness_rank = frame["smoothness"].rank(method="average", pct=True)
        frame["score"] = (
            self.config.momentum.momentum_weight * momentum_rank
            + self.config.momentum.smoothness_weight * smoothness_rank
        )
        frame = frame.reset_index().sort_values(["score", "momentum", "symbol"], ascending=[False, False, True])
        count = len(frame)
        records: list[SignalRecord] = []
        for rank, row in enumerate(frame.itertuples(index=False), start=1):
            percentile = rank / count
            records.append(
                SignalRecord(
                    symbol=row.symbol,
                    momentum_return=float(row.momentum),
                    path_smoothness=float(row.smoothness),
                    score=float(row.score),
                    rank=rank,
                    percentile=float(percentile),
                    retained=row.symbol in current_holdings and percentile <= self.config.momentum.retention_percentile,
                )
            )
        return tuple(records)

    def _select_by_sector(
        self,
        signals: Sequence[SignalRecord],
        securities: Sequence[SecurityMetadata],
        sector_weights: Mapping[str, float],
    ) -> tuple[str, ...]:
        metadata = {security.symbol: security for security in securities}
        candidates = [
            signal
            for signal in signals
            if signal.percentile <= self.config.momentum.entry_percentile or signal.retained
        ]
        quotas = self._hamilton_quotas(sector_weights, self.config.momentum.target_holdings)
        selected: list[str] = []
        for sector in sorted(quotas):
            names = [signal for signal in candidates if metadata[signal.symbol].sector == sector]
            # Hysteresis is binding: an incumbent inside the retention band is
            # kept ahead of a new entrant, while score remains the tie-breaker
            # within incumbent/new groups.
            names.sort(key=lambda signal: (not signal.retained, signal.rank, signal.symbol))
            selected.extend(signal.symbol for signal in names[: quotas[sector]])
        selected_set = set(selected)
        return tuple(signal.symbol for signal in signals if signal.symbol in selected_set)

    @staticmethod
    def _hamilton_quotas(sector_weights: Mapping[str, float], total: int) -> dict[str, int]:
        exact = {sector: weight * total for sector, weight in sector_weights.items()}
        quotas = {sector: int(math.floor(value)) for sector, value in exact.items()}
        remaining = total - sum(quotas.values())
        order = sorted(exact, key=lambda sector: (-(exact[sector] - quotas[sector]), sector))
        for sector in order[:remaining]:
            quotas[sector] += 1
        return quotas

    def _allocate_weights(
        self,
        selected: Sequence[str],
        metadata: Mapping[str, SecurityMetadata],
        sector_weights: Mapping[str, float],
        satellite_budget: float,
    ) -> dict[str, float]:
        by_sector: dict[str, list[str]] = {sector: [] for sector in sector_weights}
        for symbol in selected:
            by_sector[metadata[symbol].sector].append(symbol)
        direct: dict[str, float] = {}
        cap = self.config.risk.max_direct_issuer_weight
        for sector, names in by_sector.items():
            if not names:
                continue
            per_name = min(satellite_budget * sector_weights[sector] / len(names), cap)
            for symbol in names:
                direct[symbol] = per_name
        actual_satellite = sum(direct.values())
        weights = {self.config.benchmark: 1.0 - self.config.allocation.cash_weight - actual_satellite}
        weights.update(direct)
        return dict(sorted(weights.items()))

    def _risk_metrics(
        self,
        weights: Mapping[str, float],
        prices: pd.DataFrame,
        metadata: Mapping[str, SecurityMetadata],
        sector_weights: Mapping[str, float],
    ) -> tuple[float, float, float]:
        symbols = list(weights)
        risk_prices = prices.reindex(columns=symbols).tail(self.config.risk.risk_estimation_sessions + 1)
        if risk_prices.isna().any().any() or len(risk_prices) < self.config.risk.risk_estimation_sessions + 1:
            raise DataFailure("risk_data_incomplete", "beta/tracking-error window is incomplete")
        returns = risk_prices.pct_change(fill_method=None).dropna()
        benchmark_returns = returns[self.config.benchmark]
        variance = float(benchmark_returns.var(ddof=1))
        if not math.isfinite(variance) or variance <= 1e-20:
            raise DataFailure("benchmark_zero_variance", "cannot estimate beta against a constant benchmark")
        portfolio_returns = sum(float(weights[symbol]) * returns[symbol] for symbol in symbols)
        beta = float(portfolio_returns.cov(benchmark_returns) / variance)
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = float(active_returns.std(ddof=1) * math.sqrt(252))

        direct_by_sector = {sector: 0.0 for sector in sector_weights}
        for symbol, weight in weights.items():
            if symbol != self.config.benchmark:
                direct_by_sector[metadata[symbol].sector] += weight
        core = weights[self.config.benchmark]
        deviations = [
            abs(core * benchmark_weight + direct_by_sector[sector] - benchmark_weight)
            for sector, benchmark_weight in sector_weights.items()
        ]
        return beta, tracking_error, max(deviations, default=0.0)

    def _finalize(
        self,
        inputs: PortfolioInputs,
        weights: Mapping[str, float],
        *,
        cash_weight: float,
        signals: Sequence[SignalRecord],
        selected: Sequence[str],
        prices: pd.DataFrame,
        readiness: DataReadiness,
        diagnostics: Mapping[str, str],
        precomputed_risk: tuple[float, float, float] | None = None,
    ) -> PortfolioPlan:
        selected_meta = {s.symbol: s for s in inputs.securities if s.symbol in selected}
        risk_values = precomputed_risk or self._risk_metrics(
            weights, prices, selected_meta, inputs.benchmark_sector_weights
        )
        rounded = tuple(sorted((symbol, round(float(weight), 12)) for symbol, weight in weights.items() if weight > 1e-14))
        if not math.isclose(sum(weight for _, weight in rounded) + cash_weight, 1.0, abs_tol=1e-9):
            raise RuntimeError("constructed weights and cash do not sum to 1")
        # The weight hash is the exact cross-runtime parity contract.  The
        # target hash additionally binds strategy/config/signal provenance.
        weight_hash = build_target_hash(dict(rounded))
        target_hash = self._target_provenance_hash(inputs.signal_date, dict(rounded), cash_weight)
        return PortfolioPlan(
            strategy_id=self.config.strategy_id,
            strategy_version=self.config.version,
            config_sha256=self.config.sha256,
            benchmark_symbol=self.config.benchmark,
            signal_date=inputs.signal_date,
            target_weights=rounded,
            cash_weight=round(cash_weight, 12),
            selected_symbols=tuple(selected),
            signals=tuple(signals),
            portfolio_beta=risk_values[0],
            tracking_error=risk_values[1],
            max_active_sector_deviation=risk_values[2],
            promotable=readiness.promotable,
            promotion_blockers=readiness.blockers,
            diagnostics=tuple(sorted((str(k), str(v)) for k, v in diagnostics.items())),
            weight_sha256=weight_hash,
            target_sha256=target_hash,
        )

    def _target_provenance_hash(
        self,
        signal_date: pd.Timestamp,
        weights: Mapping[str, float],
        cash_weight: float,
    ) -> str:
        timestamp = pd.Timestamp(signal_date)
        payload = {
            "strategy_id": self.config.strategy_id,
            "strategy_version": self.config.version,
            "config_sha256": self.config.sha256,
            "benchmark": self.config.benchmark,
            "signal_date": timestamp.isoformat(),
            "weights": [[symbol, f"{float(weight):.12f}"] for symbol, weight in sorted(weights.items())],
            "cash_weight": f"{float(cash_weight):.12f}",
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
