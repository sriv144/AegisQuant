"""Load frozen, explicit inputs for v3 shadow and paper decisions.

The runtime never downloads a universe or price history on demand.  A caller
must provide a content-addressable JSON bundle (and optionally a relative CSV)
whose point-in-time fields are visible to the portfolio constructor.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import pandas as pd

from src.execution.v3 import QuoteSnapshot

from .data import (
    REQUIRED_PORTFOLIO_DATASETS,
    DataManifest,
    DataTier,
    research_data_sha256,
    snapshot_sha256,
)
from .portfolio import PortfolioInputs, SecurityMetadata


class RuntimeInputError(ValueError):
    pass


def _aware(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise RuntimeInputError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise RuntimeInputError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class BenchmarkRuntimeMark:
    session_date: date
    symbol: str
    total_return_level: Decimal
    daily_total_return: Decimal | None
    source: str
    source_sha256: str
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class RuntimeInputBundle:
    signal_date: pd.Timestamp
    total_return_prices: pd.DataFrame
    securities: tuple[SecurityMetadata, ...]
    benchmark_sector_weights: Mapping[str, float]
    manifests: tuple[DataManifest, ...]
    quotes: Mapping[str, QuoteSnapshot]
    benchmark_mark: BenchmarkRuntimeMark | None
    scheduled_eligible_sessions: tuple[date, ...]
    starting_nav: Decimal
    prior_de_risked: bool
    satellite_reentry_approved: bool
    content_validated_datasets: frozenset[str]
    content_validation_blockers: tuple[str, ...]
    research_data_sha256: str
    bundle_sha256: str
    source_path: Path

    def portfolio_inputs(
        self,
        *,
        current_holdings: frozenset[str] = frozenset(),
        current_drawdown: float = 0.0,
        prior_de_risked: bool | None = None,
        satellite_reentry_approved: bool | None = None,
    ) -> PortfolioInputs:
        return PortfolioInputs(
            signal_date=self.signal_date,
            total_return_prices=self.total_return_prices.copy(deep=True),
            securities=self.securities,
            benchmark_sector_weights=self.benchmark_sector_weights,
            current_holdings=current_holdings,
            manifests=self.manifests,
            current_drawdown=current_drawdown,
            prior_de_risked=(
                self.prior_de_risked if prior_de_risked is None else prior_de_risked
            ),
            satellite_reentry_approved=(
                self.satellite_reentry_approved
                if satellite_reentry_approved is None
                else satellite_reentry_approved
            ),
        )


def load_runtime_input(path: str | Path) -> RuntimeInputBundle:
    source = Path(path).resolve()
    try:
        raw = source.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeInputError(f"cannot load frozen runtime input {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeInputError("runtime input root must be a JSON object")

    signal_date = pd.Timestamp(payload.get("signal_date"))
    if pd.isna(signal_date):
        raise RuntimeInputError("signal_date is required")
    if signal_date.tzinfo is None:
        raise RuntimeInputError("signal_date must include a timezone")

    prices = _load_prices(payload, source.parent)
    try:
        securities = tuple(
            SecurityMetadata(
                symbol=str(row["symbol"]).upper(),
                issuer_id=str(row["issuer_id"]),
                sector=str(row["sector"]),
                adv_30d=float(row["adv_30d"]),
            )
            for row in payload["securities"]
        )
        sector_weights = {
            str(sector): float(weight)
            for sector, weight in payload["benchmark_sector_weights"].items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeInputError(f"invalid security or sector metadata: {exc}") from exc

    manifests = tuple(_manifest(row) for row in payload.get("manifests", ()))
    quotes = MappingProxyType(
        {
            str(symbol).upper(): _quote(str(symbol).upper(), row)
            for symbol, row in payload.get("quotes", {}).items()
        }
    )
    benchmark_mark = (
        None
        if payload.get("benchmark_mark") is None
        else _benchmark_mark(payload["benchmark_mark"])
    )
    try:
        eligible = tuple(sorted(date.fromisoformat(value) for value in payload.get("scheduled_eligible_sessions", ())))
        starting_nav = Decimal(str(payload.get("starting_nav", "100000")))
    except (TypeError, ValueError) as exc:
        raise RuntimeInputError(f"invalid session or starting NAV input: {exc}") from exc
    if starting_nav <= 0:
        raise RuntimeInputError("starting_nav must be positive")

    manifests, validated_datasets, validation_blockers = _validate_manifest_content(
        manifests=manifests,
        encoded_snapshots=payload.get("manifest_snapshots", {}),
        runtime_content_manifest=payload.get("runtime_content_manifest", {}),
        prices=prices,
        signal_date=signal_date,
        securities=securities,
        sector_weights=sector_weights,
        quotes=quotes,
        benchmark_mark=benchmark_mark,
    )

    return RuntimeInputBundle(
        signal_date=signal_date,
        total_return_prices=prices,
        securities=securities,
        benchmark_sector_weights=MappingProxyType(dict(sorted(sector_weights.items()))),
        manifests=manifests,
        quotes=quotes,
        benchmark_mark=benchmark_mark,
        scheduled_eligible_sessions=eligible,
        starting_nav=starting_nav,
        prior_de_risked=_json_bool(payload, "prior_de_risked", False),
        satellite_reentry_approved=_json_bool(
            payload, "satellite_reentry_approved", False
        ),
        content_validated_datasets=validated_datasets,
        content_validation_blockers=validation_blockers,
        research_data_sha256=research_data_sha256(manifests),
        bundle_sha256=hashlib.sha256(
            raw + b"\x00" + snapshot_sha256(prices).encode("ascii")
        ).hexdigest(),
        source_path=source,
    )


def _decode_manifest_snapshot(dataset: str, encoded: Any) -> Any:
    if not isinstance(encoded, Mapping):
        raise RuntimeInputError(f"manifest snapshot {dataset} must be an object")
    kind = encoded.get("kind")
    if kind == "table":
        records = encoded.get("records")
        if not isinstance(records, list):
            raise RuntimeInputError(f"manifest snapshot {dataset} table requires records")
        frame = pd.DataFrame(records)
        for column in encoded.get("datetime_columns", ()):
            if column not in frame.columns:
                raise RuntimeInputError(f"manifest snapshot {dataset} has no datetime column {column}")
            frame[column] = pd.to_datetime(frame[column], utc=True)
        index = encoded.get("index")
        if index is not None:
            if not isinstance(index, list) or len(index) != len(frame):
                raise RuntimeInputError(f"manifest snapshot {dataset} index length mismatch")
            parsed_index = pd.to_datetime(index) if encoded.get("datetime_index", False) else index
            frame.index = parsed_index
        return frame
    if kind == "series":
        values = encoded.get("values")
        index = encoded.get("index")
        if not isinstance(values, list) or not isinstance(index, list) or len(values) != len(index):
            raise RuntimeInputError(f"manifest snapshot {dataset} series is malformed")
        parsed_index = pd.to_datetime(index) if encoded.get("datetime_index", False) else index
        return pd.Series(values, index=parsed_index, name=encoded.get("name"))
    if kind == "mapping":
        values = encoded.get("values")
        if not isinstance(values, Mapping):
            raise RuntimeInputError(f"manifest snapshot {dataset} mapping is malformed")
        return dict(values)
    if kind == "raw":
        return encoded.get("value")
    raise RuntimeInputError(f"manifest snapshot {dataset} has unsupported kind {kind!r}")


def _snapshot_row_count(snapshot: Any) -> int:
    if isinstance(snapshot, (pd.DataFrame, pd.Series, Mapping, Sequence)) and not isinstance(
        snapshot, (str, bytes, bytearray)
    ):
        return len(snapshot)
    return 1


def _with_validation_warning(manifest: DataManifest, warning: str) -> DataManifest:
    return DataManifest(
        dataset=manifest.dataset,
        source=manifest.source,
        availability_at=manifest.availability_at,
        frozen_at=manifest.frozen_at,
        row_count=manifest.row_count,
        coverage=manifest.coverage,
        sha256=manifest.sha256,
        tier=manifest.tier,
        warnings=tuple(dict.fromkeys((*manifest.warnings, warning))),
    )


def _validate_manifest_content(
    *,
    manifests: tuple[DataManifest, ...],
    encoded_snapshots: Any,
    runtime_content_manifest: Any,
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    securities: tuple[SecurityMetadata, ...],
    sector_weights: Mapping[str, float],
    quotes: Mapping[str, QuoteSnapshot],
    benchmark_mark: BenchmarkRuntimeMark | None,
) -> tuple[tuple[DataManifest, ...], frozenset[str], tuple[str, ...]]:
    if not isinstance(encoded_snapshots, Mapping):
        raise RuntimeInputError("manifest_snapshots must be an object")
    snapshots: dict[str, Any] = {"total_return_prices": prices}
    for dataset, encoded in encoded_snapshots.items():
        name = str(dataset)
        if name == "total_return_prices":
            raise RuntimeInputError("total_return_prices snapshot is loaded from the frozen price input")
        snapshots[name] = _decode_manifest_snapshot(name, encoded)

    validated: set[str] = set()
    blockers: list[str] = []
    checked: list[DataManifest] = []
    for manifest in manifests:
        snapshot = snapshots.get(manifest.dataset)
        if snapshot is None:
            blocker = f"unvalidated_runtime_content:{manifest.dataset}"
            blockers.append(blocker)
            checked.append(_with_validation_warning(manifest, blocker))
            continue
        actual_hash = snapshot_sha256(snapshot)
        actual_rows = _snapshot_row_count(snapshot)
        if manifest.sha256 != actual_hash:
            raise RuntimeInputError(
                f"{manifest.dataset} manifest SHA-256 does not match the frozen snapshot"
            )
        if manifest.row_count != actual_rows:
            raise RuntimeInputError(
                f"{manifest.dataset} manifest row_count {manifest.row_count} does not match {actual_rows}"
            )
        validated.add(manifest.dataset)
        checked.append(manifest)

    by_name = {manifest.dataset: manifest for manifest in checked}
    for dataset in REQUIRED_PORTFOLIO_DATASETS:
        if dataset not in by_name:
            blockers.append(f"missing_manifest:{dataset}")
    _validate_runtime_snapshot_consistency(
        snapshots=snapshots,
        validated=validated,
        signal_date=signal_date,
        securities=securities,
        sector_weights=sector_weights,
        quotes=quotes,
        benchmark_mark=benchmark_mark,
    )
    runtime_blockers, runtime_validated = _validate_runtime_content_manifest(
        runtime_content_manifest,
        securities=securities,
        sector_weights=sector_weights,
        quotes=quotes,
        benchmark_mark=benchmark_mark,
    )
    validated.update(f"runtime:{name}" for name in runtime_validated)
    blockers.extend(runtime_blockers)
    if runtime_blockers and checked:
        first, *rest = checked
        for blocker in runtime_blockers:
            first = _with_validation_warning(first, blocker)
        checked = [first, *rest]
    return tuple(checked), frozenset(validated), tuple(sorted(set(blockers)))


def _active_rows(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    out = frame.copy(deep=True)
    comparison = pd.Timestamp(as_of)
    comparison = (
        comparison.tz_localize("UTC")
        if comparison.tzinfo is None
        else comparison.tz_convert("UTC")
    )
    if "available_at" in out.columns:
        available = pd.to_datetime(out["available_at"], utc=True)
        out = out.loc[available <= comparison]
    if "effective_start" in out.columns:
        starts = pd.to_datetime(out["effective_start"], utc=True)
        out = out.loc[starts <= comparison]
    if "effective_end" in out.columns:
        ends = pd.to_datetime(out["effective_end"], utc=True)
        out = out.loc[ends.isna() | (ends >= comparison)]
    return out


def _latest_by_symbol(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if "symbol" not in frame.columns:
        raise RuntimeInputError("runtime snapshot requires a symbol column")
    out = _active_rows(frame, as_of)
    order_column = next(
        (
            name
            for name in ("quote_time", "timestamp", "available_at", "effective_start")
            if name in out.columns
        ),
        None,
    )
    if order_column is not None:
        out = out.assign(__order=pd.to_datetime(out[order_column], utc=True)).sort_values(
            "__order"
        )
        out = out.groupby("symbol", as_index=False).tail(1)
    if out["symbol"].astype(str).str.upper().duplicated().any():
        raise RuntimeInputError("runtime snapshot has duplicate active symbols")
    out = out.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    return out.set_index("symbol")


def _validate_runtime_snapshot_consistency(
    *,
    snapshots: Mapping[str, Any],
    validated: set[str],
    signal_date: pd.Timestamp,
    securities: tuple[SecurityMetadata, ...],
    sector_weights: Mapping[str, float],
    quotes: Mapping[str, QuoteSnapshot],
    benchmark_mark: BenchmarkRuntimeMark | None,
) -> None:
    security_by_symbol = {security.symbol.upper(): security for security in securities}
    security_symbols = set(security_by_symbol)

    if "constituents" in validated:
        snapshot = snapshots["constituents"]
        if not isinstance(snapshot, pd.DataFrame):
            raise RuntimeInputError("constituents runtime snapshot must be a table")
        active = _latest_by_symbol(snapshot, signal_date)
        if set(active.index) != security_symbols:
            raise RuntimeInputError("constituent snapshot does not match runtime securities")

    if "security_master" in validated:
        snapshot = snapshots["security_master"]
        if not isinstance(snapshot, pd.DataFrame) or "issuer_id" not in snapshot.columns:
            raise RuntimeInputError("security_master runtime snapshot requires issuer_id")
        active = _latest_by_symbol(snapshot, signal_date)
        if not security_symbols.issubset(active.index):
            raise RuntimeInputError("security master is missing runtime securities")
        for symbol, security in security_by_symbol.items():
            if str(active.at[symbol, "issuer_id"]) != security.issuer_id:
                raise RuntimeInputError(f"security master issuer mismatch for {symbol}")

    if "sector_history" in validated:
        snapshot = snapshots["sector_history"]
        if not isinstance(snapshot, pd.DataFrame) or "sector" not in snapshot.columns:
            raise RuntimeInputError("sector_history runtime snapshot requires sector")
        active = _latest_by_symbol(snapshot, signal_date)
        if not security_symbols.issubset(active.index):
            raise RuntimeInputError("sector history is missing runtime securities")
        for symbol, security in security_by_symbol.items():
            if str(active.at[symbol, "sector"]) != security.sector:
                raise RuntimeInputError(f"sector history mismatch for {symbol}")

    if "benchmark_sector_weights" in validated:
        snapshot = snapshots["benchmark_sector_weights"]
        if isinstance(snapshot, pd.Series):
            observed = {str(key): float(value) for key, value in snapshot.items()}
        elif isinstance(snapshot, Mapping):
            observed = {str(key): float(value) for key, value in snapshot.items()}
        elif isinstance(snapshot, pd.DataFrame) and {"sector", "weight"}.issubset(snapshot.columns):
            observed = {
                str(row.sector): float(row.weight)
                for row in snapshot.itertuples(index=False)
            }
        else:
            raise RuntimeInputError("benchmark sector-weight snapshot has no recognized schema")
        expected = {str(key): float(value) for key, value in sector_weights.items()}
        if observed.keys() != expected.keys() or any(
            not math.isclose(observed[key], expected[key], abs_tol=1e-12)
            for key in expected
        ):
            raise RuntimeInputError("benchmark sector weights do not match the runtime portfolio input")

    if "adv" in validated:
        snapshot = snapshots["adv"]
        if isinstance(snapshot, pd.Series):
            observed_adv = {str(key).upper(): float(value) for key, value in snapshot.items()}
        elif isinstance(snapshot, pd.DataFrame) and {"symbol", "adv_30d"}.issubset(snapshot.columns):
            active = _latest_by_symbol(snapshot, signal_date)
            observed_adv = {
                str(symbol).upper(): float(value)
                for symbol, value in active["adv_30d"].items()
            }
        else:
            raise RuntimeInputError("ADV runtime snapshot has no recognized schema")
        for symbol, security in security_by_symbol.items():
            if symbol not in observed_adv or not math.isclose(
                observed_adv[symbol], security.adv_30d, rel_tol=1e-12, abs_tol=1e-6
            ):
                raise RuntimeInputError(f"ADV snapshot mismatch for {symbol}")

    # Execution quotes and the current EOD benchmark mark are ephemeral
    # decision inputs.  They are bound by runtime_content_manifest below and
    # deliberately excluded from the stable research-data identity.


def _runtime_content_snapshots(
    *,
    securities: tuple[SecurityMetadata, ...],
    sector_weights: Mapping[str, float],
    quotes: Mapping[str, QuoteSnapshot],
    benchmark_mark: BenchmarkRuntimeMark | None,
) -> Mapping[str, Any]:
    ordered_securities = sorted(securities, key=lambda security: security.symbol)
    security_frame = pd.DataFrame(
        [
            {
                "symbol": security.symbol,
                "issuer_id": security.issuer_id,
                "sector": security.sector,
                "adv_30d": security.adv_30d,
            }
            for security in ordered_securities
        ]
    )
    quote_frame = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "bid_price": float(quote.bid_price),
                "ask_price": float(quote.ask_price),
                "observed_at": quote.observed_at,
                "adv_dollars_30d": float(quote.adv_dollars_30d),
            }
            for symbol, quote in sorted(quotes.items())
        ]
    )
    benchmark = (
        None
        if benchmark_mark is None
        else {
            "session_date": benchmark_mark.session_date.isoformat(),
            "symbol": benchmark_mark.symbol,
            "total_return_level": str(benchmark_mark.total_return_level),
            "daily_total_return": (
                None
                if benchmark_mark.daily_total_return is None
                else str(benchmark_mark.daily_total_return)
            ),
            "source": benchmark_mark.source,
            "source_sha256": benchmark_mark.source_sha256,
            "observed_at": benchmark_mark.observed_at,
        }
    )
    snapshots: dict[str, Any] = {
            "constituents": tuple(security.symbol for security in ordered_securities),
            "securities": security_frame,
            "benchmark_sector_weights": dict(sorted(sector_weights.items())),
            "adv": pd.Series(
                {security.symbol: security.adv_30d for security in ordered_securities},
                name="adv_30d",
                dtype=float,
            ),
            "quotes": quote_frame,
        }
    if benchmark is not None:
        snapshots["benchmark_mark"] = benchmark
    return MappingProxyType(snapshots)


def _validate_runtime_content_manifest(
    encoded: Any,
    *,
    securities: tuple[SecurityMetadata, ...],
    sector_weights: Mapping[str, float],
    quotes: Mapping[str, QuoteSnapshot],
    benchmark_mark: BenchmarkRuntimeMark | None,
) -> tuple[tuple[str, ...], frozenset[str]]:
    if not isinstance(encoded, Mapping):
        raise RuntimeInputError("runtime_content_manifest must be an object")
    snapshots = _runtime_content_snapshots(
        securities=securities,
        sector_weights=sector_weights,
        quotes=quotes,
        benchmark_mark=benchmark_mark,
    )
    blockers: list[str] = []
    validated: set[str] = set()
    for name, snapshot in snapshots.items():
        identity = encoded.get(name)
        if not isinstance(identity, Mapping):
            blockers.append(f"unvalidated_runtime_content:{name}")
            continue
        expected_hash = str(identity.get("sha256", ""))
        try:
            expected_rows = int(identity["row_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeInputError(f"runtime content manifest {name} has invalid row_count") from exc
        actual_hash = snapshot_sha256(snapshot)
        actual_rows = _snapshot_row_count(snapshot)
        if expected_hash != actual_hash or expected_rows != actual_rows:
            raise RuntimeInputError(f"runtime content manifest mismatch for {name}")
        validated.add(name)
    return tuple(blockers), frozenset(validated)


def build_runtime_content_manifest(
    *,
    securities: tuple[SecurityMetadata, ...],
    sector_weights: Mapping[str, float],
    quotes: Mapping[str, QuoteSnapshot],
    benchmark_mark: BenchmarkRuntimeMark | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    """Build the exact identities expected in a frozen runtime bundle."""

    snapshots = _runtime_content_snapshots(
        securities=securities,
        sector_weights=sector_weights,
        quotes=quotes,
        benchmark_mark=benchmark_mark,
    )
    return MappingProxyType(
        {
            name: MappingProxyType(
                {
                    "sha256": snapshot_sha256(snapshot),
                    "row_count": _snapshot_row_count(snapshot),
                }
            )
            for name, snapshot in snapshots.items()
        }
    )


def _load_prices(payload: Mapping[str, Any], base: Path) -> pd.DataFrame:
    if "total_return_prices_csv" in payload:
        csv_path = (base / str(payload["total_return_prices_csv"])).resolve()
        if base.resolve() not in (csv_path, *csv_path.parents):
            raise RuntimeInputError("price CSV must remain inside the bundle directory")
        try:
            frame = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        except (OSError, ValueError) as exc:
            raise RuntimeInputError(f"cannot load total-return price CSV: {exc}") from exc
    elif "total_return_prices" in payload:
        raw_prices = payload["total_return_prices"]
        if not isinstance(raw_prices, dict):
            raise RuntimeInputError("total_return_prices must be an object")
        dates = raw_prices.get("dates")
        values = raw_prices.get("values")
        if dates is not None and isinstance(values, dict):
            frame = pd.DataFrame(values, index=pd.to_datetime(dates))
        else:
            frame = pd.DataFrame(raw_prices)
            frame.index = pd.to_datetime(frame.index)
    else:
        raise RuntimeInputError("total_return_prices or total_return_prices_csv is required")
    if frame.empty or frame.index.has_duplicates:
        raise RuntimeInputError("total-return prices must be non-empty with unique sessions")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise RuntimeInputError("total-return prices require a DatetimeIndex")
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert("America/New_York").normalize().tz_localize(None)
    else:
        frame.index = frame.index.normalize()
    if frame.index.has_duplicates:
        raise RuntimeInputError("price timestamps collapse to duplicate New York sessions")
    frame = frame.sort_index()
    frame.columns = [str(column).upper() for column in frame.columns]
    return frame.apply(pd.to_numeric, errors="coerce")


def _json_bool(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise RuntimeInputError(f"{key} must be a JSON boolean")
    return value


def _manifest(row: Mapping[str, Any]) -> DataManifest:
    try:
        return DataManifest(
            dataset=str(row["dataset"]),
            source=str(row["source"]),
            availability_at=_aware(str(row["availability_at"]), "manifest availability_at"),
            frozen_at=_aware(str(row["frozen_at"]), "manifest frozen_at"),
            row_count=int(row["row_count"]),
            coverage=float(row["coverage"]),
            sha256=str(row["sha256"]),
            tier=DataTier(str(row["tier"])),
            warnings=tuple(str(value) for value in row.get("warnings", ())),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeInputError(f"invalid data manifest: {exc}") from exc


def _quote(symbol: str, row: Mapping[str, Any]) -> QuoteSnapshot:
    try:
        return QuoteSnapshot(
            symbol=symbol,
            bid_price=Decimal(str(row["bid_price"])),
            ask_price=Decimal(str(row["ask_price"])),
            observed_at=_aware(str(row["observed_at"]), f"quote {symbol} observed_at"),
            adv_dollars_30d=Decimal(str(row["adv_dollars_30d"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeInputError(f"invalid quote for {symbol}: {exc}") from exc


def _benchmark_mark(row: Mapping[str, Any]) -> BenchmarkRuntimeMark:
    try:
        symbol = str(row.get("symbol", "SPY")).upper()
        digest = str(row["source_sha256"])
        level = Decimal(str(row["total_return_level"]))
        daily = row.get("daily_total_return")
        mark = BenchmarkRuntimeMark(
            session_date=date.fromisoformat(str(row["session_date"])),
            symbol=symbol,
            total_return_level=level,
            daily_total_return=None if daily is None else Decimal(str(daily)),
            source=str(row["source"]),
            source_sha256=digest,
            observed_at=_aware(str(row["observed_at"]), "benchmark observed_at"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeInputError(f"invalid benchmark mark: {exc}") from exc
    if mark.symbol != "SPY" or mark.total_return_level <= 0 or len(mark.source_sha256) != 64:
        raise RuntimeInputError("benchmark mark must be positive, SHA-addressed SPY total return")
    return mark


__all__ = [
    "BenchmarkRuntimeMark",
    "RuntimeInputBundle",
    "RuntimeInputError",
    "build_runtime_content_manifest",
    "load_runtime_input",
]
