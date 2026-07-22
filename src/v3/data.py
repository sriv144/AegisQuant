"""Point-in-time market-data contracts and research-only adapter.

No implementation in this module performs network I/O.  Open/free datasets are
injected as frozen snapshots and are explicitly unable to claim promotability.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd


class DataContractError(RuntimeError):
    """Raised when a requested point-in-time dataset is absent or malformed."""


class DataTier(str, Enum):
    RESEARCH_ONLY = "RESEARCH_ONLY"
    PROMOTABLE = "PROMOTABLE"


# Stable, frozen datasets used to establish research truth and promotion
# identity.  Current arrival quotes and EOD marks are intentionally separate:
# they change every run and belong to the decision bundle hash, not the
# research-data hash.
REQUIRED_RESEARCH_DATASETS = (
    "constituents",
    "total_return_prices",
    "execution_prices",
    "corporate_actions",
    "delistings",
    "symbol_changes",
    "sector_history",
    "security_master",
    "benchmark_sector_weights",
    "benchmark_total_return",
    "adv",
    "historical_spreads",
)
REQUIRED_PORTFOLIO_DATASETS = REQUIRED_RESEARCH_DATASETS
EPHEMERAL_RUNTIME_DATASETS = frozenset(
    {"quotes", "arrival_quotes", "runtime_quotes", "benchmark_mark", "eod_benchmark_mark"}
)

# Dataset release times are checked against the decision point at which each
# value can first influence the system.  Executable prices/quotes observed in
# the next-month trading window are not look-ahead merely because they arrive
# after the prior-month signal close.
SIGNAL_TIME_DATASETS = frozenset(
    {
        "constituents",
        "total_return_prices",
        "corporate_actions",
        "delistings",
        "symbol_changes",
        "sector_history",
        "security_master",
        "benchmark_sector_weights",
        "benchmark_total_return",
    }
)
EXECUTION_TIME_DATASETS = frozenset(
    {"execution_prices", "adv", "historical_spreads", *EPHEMERAL_RUNTIME_DATASETS}
)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _canonical_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
        return stamp.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if not np.isfinite(value) else value
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_scalar(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_scalar(item) for item in value]
    return value


def snapshot_sha256(value: Any) -> str:
    """Content hash a tabular snapshot with deterministic row/column ordering."""

    if isinstance(value, pd.Series):
        value = value.to_frame(name=value.name or "value")
    if isinstance(value, pd.DataFrame):
        frame = value.copy(deep=True)
        frame.columns = [str(c) for c in frame.columns]
        frame = frame.reindex(sorted(frame.columns), axis=1)
        frame = frame.sort_index(kind="stable")
        payload = {
            "index": [_canonical_scalar(v) for v in frame.index.tolist()],
            "columns": frame.columns.tolist(),
            "data": [[_canonical_scalar(v) for v in row] for row in frame.to_numpy(dtype=object)],
        }
    elif isinstance(value, Mapping):
        payload = {str(k): _canonical_scalar(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    else:
        payload = _canonical_scalar(value)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _row_count(value: Any) -> int:
    if isinstance(value, (pd.DataFrame, pd.Series, Mapping, Sequence)) and not isinstance(value, (str, bytes)):
        return len(value)
    return 1


def research_data_sha256(manifests: Sequence["DataManifest"]) -> str:
    """Stable identity for validated research data, excluding runtime marks."""

    payload = [
        {
            "dataset": manifest.dataset,
            "source": manifest.source,
            "availability_at": manifest.availability_at.isoformat(),
            "frozen_at": manifest.frozen_at.isoformat(),
            "row_count": manifest.row_count,
            "coverage": f"{manifest.coverage:.12f}",
            "sha256": manifest.sha256,
            "tier": manifest.tier.value,
        }
        for manifest in sorted(manifests, key=lambda item: (item.dataset, item.sha256))
        if manifest.dataset in REQUIRED_RESEARCH_DATASETS
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DataManifest:
    dataset: str
    source: str
    availability_at: datetime
    frozen_at: datetime
    row_count: int
    coverage: float
    sha256: str
    tier: DataTier
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.dataset.strip() or not self.source.strip():
            raise ValueError("manifest dataset and source are required")
        object.__setattr__(self, "availability_at", _utc(self.availability_at))
        object.__setattr__(self, "frozen_at", _utc(self.frozen_at))
        if self.row_count < 0:
            raise ValueError("manifest row_count cannot be negative")
        if not 0 <= self.coverage <= 1:
            raise ValueError("manifest coverage must be in [0, 1]")
        if len(self.sha256) != 64:
            raise ValueError("manifest sha256 must be a 64-character digest")

    @classmethod
    def from_snapshot(
        cls,
        dataset: str,
        source: str,
        snapshot: Any,
        *,
        availability_at: datetime,
        frozen_at: datetime,
        coverage: float,
        tier: DataTier,
        warnings: Sequence[str] = (),
    ) -> "DataManifest":
        return cls(
            dataset=dataset,
            source=source,
            availability_at=availability_at,
            frozen_at=frozen_at,
            row_count=_row_count(snapshot),
            coverage=float(coverage),
            sha256=snapshot_sha256(snapshot),
            tier=tier,
            warnings=tuple(str(w) for w in warnings),
        )

    @property
    def promotable(self) -> bool:
        return (
            self.tier is DataTier.PROMOTABLE
            and self.coverage >= 0.98
            and not self.warnings
            and self.availability_at <= self.frozen_at
            and self.row_count > 0
        )


@dataclass(frozen=True, slots=True)
class DataReadiness:
    promotable: bool
    manifests: tuple[DataManifest, ...]
    blockers: tuple[str, ...]

    @property
    def research_only(self) -> bool:
        return not self.promotable

    @classmethod
    def evaluate(
        cls,
        manifests: Sequence[DataManifest],
        required: Sequence[str] = REQUIRED_PORTFOLIO_DATASETS,
    ) -> "DataReadiness":
        ordered = tuple(sorted(manifests, key=lambda m: m.dataset))
        by_name: dict[str, DataManifest] = {}
        blockers: list[str] = []
        for manifest in ordered:
            if manifest.dataset in by_name:
                blockers.append(f"duplicate:{manifest.dataset}")
            else:
                by_name[manifest.dataset] = manifest
        for name in required:
            manifest = by_name.get(name)
            if manifest is None:
                blockers.append(f"missing:{name}")
            elif not manifest.promotable:
                blockers.append(f"not_promotable:{name}")
        return cls(promotable=not blockers, manifests=ordered, blockers=tuple(blockers))


@runtime_checkable
class PointInTimeDataProvider(Protocol):
    """Minimum provider surface required by v3 research and execution."""

    def get_constituents(self, as_of: pd.Timestamp) -> pd.DataFrame: ...

    def get_total_return_prices(
        self, symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp, as_of: pd.Timestamp
    ) -> pd.DataFrame: ...

    def get_execution_prices(
        self, symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp, as_of: pd.Timestamp
    ) -> pd.DataFrame: ...

    def get_corporate_actions(self, start: pd.Timestamp, end: pd.Timestamp, as_of: pd.Timestamp) -> pd.DataFrame: ...

    def get_delistings(self, start: pd.Timestamp, end: pd.Timestamp, as_of: pd.Timestamp) -> pd.DataFrame: ...

    def get_symbol_changes(self, start: pd.Timestamp, end: pd.Timestamp, as_of: pd.Timestamp) -> pd.DataFrame: ...

    def get_sector_history(self, symbols: Sequence[str], as_of: pd.Timestamp) -> pd.DataFrame: ...

    def get_security_master(self, symbols: Sequence[str], as_of: pd.Timestamp) -> pd.DataFrame: ...

    def get_benchmark_sector_weights(self, as_of: pd.Timestamp) -> pd.Series: ...

    def get_benchmark_total_return(self, start: pd.Timestamp, end: pd.Timestamp, as_of: pd.Timestamp) -> pd.Series: ...

    def get_adv(self, symbols: Sequence[str], as_of: pd.Timestamp) -> pd.Series: ...

    def get_historical_spreads(
        self, symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp, as_of: pd.Timestamp
    ) -> pd.DataFrame: ...

    def get_quotes(self, symbols: Sequence[str], as_of: pd.Timestamp) -> pd.DataFrame: ...

    def manifest(self, dataset: str) -> DataManifest: ...

    def readiness(self, required: Sequence[str] = REQUIRED_PORTFOLIO_DATASETS) -> DataReadiness: ...


class OpenResearchProvider:
    """Injected free/public snapshots that are permanently research-only.

    The adapter makes the provenance limitation executable: callers receive
    valid data, but readiness can never pass a promotion gate merely because a
    free current-membership dataset happened to have high numeric coverage.
    """

    def __init__(
        self,
        snapshots: Mapping[str, Any],
        *,
        source: str = "open-public-snapshot",
        availability_at: datetime | Mapping[str, datetime] | None = None,
        frozen_at: datetime | Mapping[str, datetime] | None = None,
        coverage: Mapping[str, float] | None = None,
        warnings: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        self._enforce_manifest_availability = availability_at is not None
        self._snapshots = MappingProxyType({k: self._copy(v) for k, v in snapshots.items()})
        self._manifests = MappingProxyType(
            {
                name: DataManifest.from_snapshot(
                    name,
                    source,
                    value,
                    availability_at=self._dataset_time(availability_at, name, now),
                    frozen_at=self._dataset_time(frozen_at, name, now),
                    coverage=float((coverage or {}).get(name, 1.0)),
                    tier=DataTier.RESEARCH_ONLY,
                    warnings=tuple((warnings or {}).get(name, ())) + ("open_source_not_independently_validated",),
                )
                for name, value in self._snapshots.items()
            }
        )

    @staticmethod
    def _dataset_time(
        value: datetime | Mapping[str, datetime] | None,
        dataset: str,
        default: datetime,
    ) -> datetime:
        if isinstance(value, Mapping):
            if dataset not in value:
                raise DataContractError(f"missing timestamp provenance for {dataset!r}")
            return _utc(value[dataset])
        return _utc(value or default)

    @staticmethod
    def _copy(value: Any) -> Any:
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return value.copy(deep=True)
        if isinstance(value, Mapping):
            return MappingProxyType(dict(value))
        return value

    def _dataset(self, name: str) -> Any:
        if name not in self._snapshots:
            raise DataContractError(f"open research snapshot is missing {name!r}")
        return self._copy(self._snapshots[name])

    def _assert_available(self, name: str, as_of: pd.Timestamp) -> None:
        # Open snapshots without explicit release-time provenance remain useful
        # for non-promotable historical research.  Their mandatory warning
        # prevents promotion; explicit availability timestamps are always
        # enforced and validated below.
        if not self._enforce_manifest_availability:
            return
        requested = pd.Timestamp(as_of)
        if requested.tzinfo is None:
            requested = requested.tz_localize("UTC")
        else:
            requested = requested.tz_convert("UTC")
        available = pd.Timestamp(self.manifest(name).availability_at)
        if requested < available:
            raise DataContractError(
                f"{name!r} was not available at {requested.isoformat()} "
                f"(first available {available.isoformat()})"
            )

    @staticmethod
    def _filter_available_rows(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        date = pd.Timestamp(as_of)
        out = frame
        if "available_at" in out.columns:
            available = pd.to_datetime(out["available_at"], utc=True)
            comparison = date.tz_localize("UTC") if date.tzinfo is None else date.tz_convert("UTC")
            out = out.loc[available <= comparison]
        if {"effective_start", "effective_end"}.issubset(out.columns):
            start = pd.to_datetime(out["effective_start"], utc=True)
            end = pd.to_datetime(out["effective_end"], utc=True)
            comparison = date.tz_localize("UTC") if date.tzinfo is None else date.tz_convert("UTC")
            out = out.loc[(start <= comparison) & (end.isna() | (end >= comparison))]
        return out

    @staticmethod
    def _window(frame: pd.DataFrame | pd.Series, start: pd.Timestamp, end: pd.Timestamp):
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise DataContractError("time-series snapshot requires a DatetimeIndex")
        lower = pd.Timestamp(start)
        upper = pd.Timestamp(end)
        if frame.index.tz is None:
            lower = lower.tz_localize(None) if lower.tzinfo is not None else lower
            upper = upper.tz_localize(None) if upper.tzinfo is not None else upper
        else:
            lower = lower.tz_localize(frame.index.tz) if lower.tzinfo is None else lower.tz_convert(frame.index.tz)
            upper = upper.tz_localize(frame.index.tz) if upper.tzinfo is None else upper.tz_convert(frame.index.tz)
        out = frame.loc[(frame.index >= lower) & (frame.index <= upper)]
        return out.copy(deep=True)

    @classmethod
    def _event_window(
        cls,
        frame: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise DataContractError("event snapshot must be a DataFrame")
        if "effective_date" not in frame.columns:
            return cls._window(frame, start, min(pd.Timestamp(end), pd.Timestamp(as_of)))
        dates = pd.to_datetime(frame["effective_date"], utc=True)
        lower = pd.Timestamp(start)
        upper = min(pd.Timestamp(end), pd.Timestamp(as_of))
        lower = lower.tz_localize("UTC") if lower.tzinfo is None else lower.tz_convert("UTC")
        upper = upper.tz_localize("UTC") if upper.tzinfo is None else upper.tz_convert("UTC")
        out = frame.loc[(dates >= lower) & (dates <= upper)]
        if "available_at" in out.columns:
            out = cls._filter_available_rows(out, as_of)
        return out.copy(deep=True)

    def get_constituents(self, as_of: pd.Timestamp) -> pd.DataFrame:
        self._assert_available("constituents", as_of)
        frame = self._dataset("constituents")
        if not isinstance(frame, pd.DataFrame):
            raise DataContractError("constituents snapshot must be a DataFrame")
        frame = self._filter_available_rows(frame, as_of)
        return frame.copy(deep=True)

    def get_total_return_prices(self, symbols, start, end, as_of) -> pd.DataFrame:
        self._assert_available("total_return_prices", as_of)
        frame = self._dataset("total_return_prices")
        return self._window(frame, start, min(pd.Timestamp(end), pd.Timestamp(as_of))).reindex(columns=list(symbols))

    def get_execution_prices(self, symbols, start, end, as_of) -> pd.DataFrame:
        self._assert_available("execution_prices", as_of)
        frame = self._dataset("execution_prices")
        return self._window(frame, start, min(pd.Timestamp(end), pd.Timestamp(as_of))).reindex(columns=list(symbols))

    def get_corporate_actions(self, start, end, as_of) -> pd.DataFrame:
        self._assert_available("corporate_actions", as_of)
        frame = self._dataset("corporate_actions")
        return self._event_window(frame, start, end, as_of)

    def get_delistings(self, start, end, as_of) -> pd.DataFrame:
        self._assert_available("delistings", as_of)
        frame = self._dataset("delistings")
        return self._event_window(frame, start, end, as_of)

    def get_symbol_changes(self, start, end, as_of) -> pd.DataFrame:
        self._assert_available("symbol_changes", as_of)
        frame = self._dataset("symbol_changes")
        return self._event_window(frame, start, end, as_of)

    def get_sector_history(self, symbols, as_of) -> pd.DataFrame:
        self._assert_available("sector_history", as_of)
        frame = self._dataset("sector_history")
        if "symbol" in frame.columns:
            frame = frame.loc[frame["symbol"].isin(symbols)]
        frame = self._filter_available_rows(frame, as_of)
        return frame.copy(deep=True)

    def get_security_master(self, symbols, as_of) -> pd.DataFrame:
        self._assert_available("security_master", as_of)
        frame = self._dataset("security_master")
        if not isinstance(frame, pd.DataFrame):
            raise DataContractError("security_master snapshot must be a DataFrame")
        if "symbol" not in frame.columns:
            raise DataContractError("security_master requires a symbol column")
        frame = frame.loc[frame["symbol"].isin(symbols)]
        frame = self._filter_available_rows(frame, as_of)
        return frame.copy(deep=True)

    def get_benchmark_sector_weights(self, as_of) -> pd.Series:
        self._assert_available("benchmark_sector_weights", as_of)
        snapshot = self._dataset("benchmark_sector_weights")
        if isinstance(snapshot, Mapping):
            series = pd.Series(dict(snapshot), dtype=float)
        elif isinstance(snapshot, pd.Series):
            series = snapshot.astype(float)
        elif isinstance(snapshot, pd.DataFrame) and {"sector", "weight"}.issubset(snapshot.columns):
            frame = self._filter_available_rows(snapshot, as_of)
            order_column = next(
                (name for name in ("effective_start", "as_of", "available_at") if name in frame.columns),
                None,
            )
            if order_column is not None:
                frame = frame.assign(__order=pd.to_datetime(frame[order_column])).sort_values("__order")
                frame = frame.groupby("sector", as_index=False).tail(1)
            series = frame.set_index("sector")["weight"].astype(float)
        else:
            raise DataContractError("benchmark_sector_weights has no recognized schema")
        if series.empty or series.isna().any() or (series < 0).any():
            raise DataContractError("benchmark sector weights are missing or invalid")
        if not math.isclose(float(series.sum()), 1.0, abs_tol=1e-8):
            raise DataContractError("benchmark sector weights must sum to one")
        return series.sort_index().copy(deep=True)

    def get_benchmark_total_return(self, start, end, as_of) -> pd.Series:
        self._assert_available("benchmark_total_return", as_of)
        series = self._dataset("benchmark_total_return")
        if isinstance(series, pd.DataFrame):
            if series.shape[1] != 1:
                raise DataContractError("benchmark_total_return must have one column")
            series = series.iloc[:, 0]
        return self._window(series, start, min(pd.Timestamp(end), pd.Timestamp(as_of)))

    def get_adv(self, symbols, as_of) -> pd.Series:
        self._assert_available("adv", as_of)
        series = self._dataset("adv")
        if isinstance(series, pd.DataFrame):
            if {"symbol", "adv_30d"}.issubset(series.columns):
                frame = self._filter_available_rows(series.loc[series["symbol"].isin(symbols)], as_of)
                order_column = next(
                    (name for name in ("effective_start", "as_of", "available_at") if name in frame.columns),
                    None,
                )
                if order_column is not None:
                    frame = frame.assign(__order=pd.to_datetime(frame[order_column])).sort_values("__order")
                    frame = frame.groupby("symbol", as_index=False).tail(1)
                series = frame.set_index("symbol")["adv_30d"]
            elif series.shape[1] == 1:
                series = series.iloc[:, 0]
            else:
                raise DataContractError("adv snapshot has no recognized schema")
        return series.reindex(list(symbols)).copy(deep=True)

    def get_historical_spreads(self, symbols, start, end, as_of) -> pd.DataFrame:
        self._assert_available("historical_spreads", as_of)
        frame = self._dataset("historical_spreads")
        if not isinstance(frame, pd.DataFrame):
            raise DataContractError("historical_spreads snapshot must be a DataFrame")
        if {"timestamp", "symbol", "spread_bps"}.issubset(frame.columns):
            timestamps = pd.to_datetime(frame["timestamp"], utc=True)
            lower = pd.Timestamp(start)
            upper = min(pd.Timestamp(end), pd.Timestamp(as_of))
            lower = lower.tz_localize("UTC") if lower.tzinfo is None else lower.tz_convert("UTC")
            upper = upper.tz_localize("UTC") if upper.tzinfo is None else upper.tz_convert("UTC")
            return frame.loc[
                frame["symbol"].isin(symbols) & (timestamps >= lower) & (timestamps <= upper)
            ].copy(deep=True)
        return self._window(frame, start, min(pd.Timestamp(end), pd.Timestamp(as_of))).reindex(
            columns=list(symbols)
        )

    def get_quotes(self, symbols, as_of) -> pd.DataFrame:
        self._assert_available("quotes", as_of)
        frame = self._dataset("quotes")
        if not isinstance(frame, pd.DataFrame):
            raise DataContractError("quotes snapshot must be a DataFrame")
        requested = list(symbols)
        if "symbol" in frame.columns:
            frame = frame.loc[frame["symbol"].isin(requested)].copy()
            time_column = next(
                (name for name in ("quote_time", "timestamp", "available_at") if name in frame.columns),
                None,
            )
            if time_column is None:
                raise DataContractError("long-form quotes require quote_time/timestamp/available_at")
            frame[time_column] = pd.to_datetime(frame[time_column])
            frame = frame.loc[frame[time_column] <= pd.Timestamp(as_of)].sort_values(time_column)
            frame = frame.groupby("symbol", as_index=False).tail(1).set_index("symbol")
        elif frame.index.name == "symbol" or set(requested).issubset(frame.index.astype(str)):
            frame = frame.reindex(requested)
        else:
            raise DataContractError("quotes snapshot has no recognized symbol schema")
        return frame.reindex(requested).copy(deep=True)

    def manifest(self, dataset: str) -> DataManifest:
        try:
            return self._manifests[dataset]
        except KeyError as exc:
            raise DataContractError(f"missing manifest for {dataset!r}") from exc

    def readiness(self, required: Sequence[str] = REQUIRED_PORTFOLIO_DATASETS) -> DataReadiness:
        return DataReadiness.evaluate(tuple(self._manifests.values()), required)


class PromotableProvider(OpenResearchProvider):
    """Validated PIT snapshots whose provenance can pass promotion readiness.

    Construction is intentionally strict: a caller must supply an independent
    validation identifier and every required dataset must already satisfy the
    manifest contract.  The provider still performs no network I/O.
    """

    def __init__(
        self,
        snapshots: Mapping[str, Any],
        *,
        source: str,
        independent_validation_id: str,
        availability_at: datetime | Mapping[str, datetime],
        frozen_at: datetime | Mapping[str, datetime],
        coverage: Mapping[str, float],
        warnings: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        if not independent_validation_id.strip():
            raise DataContractError("independent_validation_id is required")
        missing = sorted(set(REQUIRED_PORTFOLIO_DATASETS).difference(snapshots))
        if missing:
            raise DataContractError(f"promotable provider is missing required datasets: {missing}")
        self._validate_promotable_snapshots(snapshots)
        super().__init__(
            snapshots,
            source=f"{source}#validation={independent_validation_id.strip()}",
            availability_at=availability_at,
            frozen_at=frozen_at,
            coverage=coverage,
            warnings=warnings,
        )
        # Rebuild manifests without the OpenResearchProvider's mandatory
        # research-only warning.  Snapshot hashes and provenance timestamps are
        # retained exactly.
        self._manifests = MappingProxyType(
            {
                name: DataManifest.from_snapshot(
                    name,
                    f"{source}#validation={independent_validation_id.strip()}",
                    value,
                    availability_at=self._dataset_time(availability_at, name, datetime.now(timezone.utc)),
                    frozen_at=self._dataset_time(frozen_at, name, datetime.now(timezone.utc)),
                    coverage=float(coverage.get(name, 0.0)),
                    tier=DataTier.PROMOTABLE,
                    warnings=tuple((warnings or {}).get(name, ())),
                )
                for name, value in self._snapshots.items()
            }
        )
        readiness = self.readiness()
        if not readiness.promotable:
            raise DataContractError(f"validated provider failed readiness: {list(readiness.blockers)}")

    @staticmethod
    def _require_columns(snapshot: Any, dataset: str, required: set[str]) -> pd.DataFrame:
        if not isinstance(snapshot, pd.DataFrame):
            raise DataContractError(f"{dataset} must be a DataFrame")
        missing = required.difference(str(column) for column in snapshot.columns)
        if missing:
            raise DataContractError(f"{dataset} is missing columns: {sorted(missing)}")
        return snapshot

    @classmethod
    def _validate_promotable_snapshots(cls, snapshots: Mapping[str, Any]) -> None:
        """Validate the canonical PIT schemas before a provider may promote.

        This is deliberately stricter than the research-only adapter.  It
        prevents a present-day membership list or a close-only execution
        series from being mislabeled as survivorship-safe executable history.
        """

        cls._require_columns(
            snapshots["constituents"],
            "constituents",
            {"symbol", "effective_start", "effective_end"},
        )
        cls._require_columns(
            snapshots["sector_history"],
            "sector_history",
            {"symbol", "sector", "effective_start", "effective_end"},
        )
        cls._require_columns(
            snapshots["security_master"],
            "security_master",
            {"symbol", "issuer_id", "share_class", "effective_start", "effective_end"},
        )
        sector_weights = snapshots["benchmark_sector_weights"]
        if isinstance(sector_weights, Mapping):
            sector_values = pd.Series(dict(sector_weights), dtype=float)
        elif isinstance(sector_weights, pd.Series):
            sector_values = sector_weights.astype(float)
        elif isinstance(sector_weights, pd.DataFrame) and {"sector", "weight"}.issubset(
            sector_weights.columns
        ):
            sector_values = sector_weights.groupby("sector")["weight"].last().astype(float)
        else:
            raise DataContractError("benchmark_sector_weights has no recognized schema")
        if sector_values.empty or sector_values.isna().any() or (sector_values < 0).any() or not math.isclose(
            float(sector_values.sum()), 1.0, abs_tol=1e-8
        ):
            raise DataContractError("benchmark_sector_weights must be finite, non-negative, and sum to one")
        cls._require_columns(
            snapshots["corporate_actions"],
            "corporate_actions",
            {"symbol", "effective_date", "action_type"},
        )
        cls._require_columns(
            snapshots["delistings"],
            "delistings",
            {"symbol", "effective_date", "delisting_return"},
        )
        cls._require_columns(
            snapshots["symbol_changes"],
            "symbol_changes",
            {"old_symbol", "new_symbol", "effective_date"},
        )

        total_return = snapshots["total_return_prices"]
        if not isinstance(total_return, pd.DataFrame) or total_return.empty:
            raise DataContractError("total_return_prices must be a non-empty DataFrame")
        if not isinstance(total_return.index, pd.DatetimeIndex):
            raise DataContractError("total_return_prices requires a DatetimeIndex")

        execution = snapshots["execution_prices"]
        if not isinstance(execution, pd.DataFrame) or execution.empty:
            raise DataContractError("execution_prices must be a non-empty DataFrame")
        if isinstance(execution.columns, pd.MultiIndex):
            fields = {str(value).lower() for level in execution.columns.levels for value in level}
            required_fields = {"open", "high", "low", "close", "volume"}
            if not required_fields.issubset(fields):
                raise DataContractError("execution_prices MultiIndex must contain unadjusted OHLCV fields")
        else:
            required_fields = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
            if not required_fields.issubset({str(column).lower() for column in execution.columns}):
                raise DataContractError("execution_prices must contain timestamped unadjusted OHLCV")

        benchmark = snapshots["benchmark_total_return"]
        if not isinstance(benchmark, (pd.Series, pd.DataFrame)) or len(benchmark) == 0:
            raise DataContractError("benchmark_total_return must be a non-empty time series")
        cls._require_columns(
            snapshots["adv"],
            "adv",
            {"symbol", "adv_30d", "available_at"},
        )
        spread_frame = snapshots["historical_spreads"]
        if isinstance(spread_frame, pd.DataFrame) and {
            "timestamp",
            "symbol",
            "spread_bps",
        }.issubset(spread_frame.columns):
            pass
        elif not isinstance(spread_frame, pd.DataFrame) or not isinstance(
            spread_frame.index, pd.DatetimeIndex
        ):
            raise DataContractError(
                "historical_spreads must be a timestamped long-form or wide DataFrame"
            )
