"""Durable EOD NAV and SPY total-return marks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from src.db.v3_models import (
    BenchmarkMark,
    PortfolioSnapshotRecord,
    PositionSnapshotRecord,
    ShadowAccountRecord,
    StrategyEpoch,
)
from src.execution.v3 import PositionSnapshot, QuoteSnapshot, ShadowAccount
from src.v3.runtime_input import BenchmarkRuntimeMark


@dataclass(frozen=True, slots=True)
class EODPerformanceResult:
    snapshot_id: str
    nav: Decimal
    drawdown: Decimal
    cumulative_return: Decimal
    cumulative_benchmark_return: Decimal
    cumulative_excess_return: Decimal


class EODPerformanceStore:
    def __init__(
        self,
        engine: Engine,
        *,
        strategy_id: str,
        strategy_version: str,
        config_sha256: str,
    ) -> None:
        self.engine = engine
        self.strategy_id = strategy_id
        self.strategy_version = strategy_version
        self.config_sha256 = config_sha256

    def record_shadow(
        self,
        *,
        run_id: str,
        account: ShadowAccount,
        quotes: Mapping[str, QuoteSnapshot],
        benchmark: BenchmarkRuntimeMark,
        observed_at: datetime,
    ) -> EODPerformanceResult:
        nav = account.net_asset_value(quotes)
        positions = tuple(
            (
                position.symbol,
                position.quantity,
                quotes[position.symbol].midpoint,
                "core" if position.symbol == "SPY" else "momentum_satellite",
                "V3_ATTRIBUTED",
            )
            for position in account.positions.values()
        )
        return self._record(
            run_id=run_id,
            account_key=account.account_key,
            mode="shadow",
            nav=nav,
            cash=account.cash,
            positions=positions,
            benchmark=benchmark,
            observed_at=observed_at,
        )

    def record_paper(
        self,
        *,
        run_id: str,
        account_key: str,
        nav: Decimal,
        cash: Decimal,
        positions: Sequence[PositionSnapshot],
        benchmark: BenchmarkRuntimeMark,
        observed_at: datetime,
    ) -> EODPerformanceResult:
        prior_lots = self._latest_position_lots(account_key)
        rows: list[tuple[str, Decimal, Decimal, str, str]] = []
        for position in positions:
            final_quantity = Decimal(position.quantity)
            if final_quantity <= 0:
                continue
            lots = prior_lots.get(position.symbol, {})
            legacy_quantity = min(
                final_quantity,
                lots.get("LEGACY_UNATTRIBUTED", Decimal("0")),
            )
            remaining = final_quantity - legacy_quantity
            v3_quantity = min(
                remaining,
                lots.get("V3_ATTRIBUTED", Decimal("0")),
            )
            # Any quantity not explained by the durable prior lots is manual or
            # otherwise unattributed and must not be credited to v3.
            legacy_quantity += remaining - v3_quantity
            if legacy_quantity > 0:
                rows.append(
                    (
                        position.symbol,
                        legacy_quantity,
                        position.market_price,
                        "legacy_unattributed",
                        "LEGACY_UNATTRIBUTED",
                    )
                )
            if v3_quantity > 0:
                rows.append(
                    (
                        position.symbol,
                        v3_quantity,
                        position.market_price,
                        "core" if position.symbol == "SPY" else "momentum_satellite",
                        "V3_ATTRIBUTED",
                    )
                )
        return self._record(
            run_id=run_id,
            account_key=account_key,
            mode="paper",
            nav=nav,
            cash=cash,
            positions=tuple(rows),
            benchmark=benchmark,
            observed_at=observed_at,
        )

    def _record(
        self,
        *,
        run_id: str,
        account_key: str,
        mode: str,
        nav: Decimal,
        cash: Decimal,
        positions: Sequence[tuple[str, Decimal, Decimal, str, str]],
        benchmark: BenchmarkRuntimeMark,
        observed_at: datetime,
    ) -> EODPerformanceResult:
        with Session(self.engine) as session:
            epoch = session.scalar(
                select(StrategyEpoch)
                .where(
                    StrategyEpoch.account_key == account_key,
                    StrategyEpoch.mode == mode,
                    StrategyEpoch.strategy_id == self.strategy_id,
                    StrategyEpoch.strategy_version == self.strategy_version,
                    StrategyEpoch.config_sha256 == self.config_sha256,
                )
                .order_by(StrategyEpoch.activated_at.desc())
                .limit(1)
            )
            if epoch is None:
                raise RuntimeError(f"{mode} strategy epoch is not bootstrapped")
            snapshot_id = hashlib.sha256(
                (
                    f"eod|{account_key}|{mode}|{epoch.epoch_id}|"
                    f"{benchmark.session_date.isoformat()}"
                ).encode("utf-8")
            ).hexdigest()

            existing_mark = session.scalar(
                select(BenchmarkMark).where(
                    BenchmarkMark.account_key == account_key,
                    BenchmarkMark.mode == mode,
                    BenchmarkMark.session_date == benchmark.session_date,
                    BenchmarkMark.symbol == "SPY",
                )
            )
            if existing_mark is None:
                session.add(
                    BenchmarkMark(
                        account_key=account_key,
                        mode=mode,
                        session_date=benchmark.session_date,
                        symbol="SPY",
                        total_return_level=benchmark.total_return_level,
                        daily_total_return=benchmark.daily_total_return,
                        source=benchmark.source,
                        source_sha256=benchmark.source_sha256,
                        observed_at=benchmark.observed_at,
                    )
                )
                session.flush()
            elif (
                Decimal(existing_mark.total_return_level) != benchmark.total_return_level
                or existing_mark.source_sha256 != benchmark.source_sha256
            ):
                raise RuntimeError("conflicting SPY total-return mark for the same NY session")

            first_mark = session.scalar(
                select(BenchmarkMark)
                .where(
                    BenchmarkMark.account_key == account_key,
                    BenchmarkMark.mode == mode,
                    BenchmarkMark.symbol == "SPY",
                    BenchmarkMark.session_date
                    >= (
                        epoch.activated_at.replace(tzinfo=UTC)
                        if epoch.activated_at.tzinfo is None
                        else epoch.activated_at
                    ).astimezone(ZoneInfo("America/New_York")).date(),
                )
                .order_by(BenchmarkMark.session_date)
                .limit(1)
            )
            assert first_mark is not None
            cumulative_return = nav / Decimal(epoch.starting_nav) - Decimal("1")
            cumulative_benchmark = (
                benchmark.total_return_level / Decimal(first_mark.total_return_level)
                - Decimal("1")
            )
            cumulative_excess = cumulative_return - cumulative_benchmark
            prior = session.scalar(
                select(PortfolioSnapshotRecord)
                .where(
                    PortfolioSnapshotRecord.account_key == account_key,
                    PortfolioSnapshotRecord.mode == mode,
                )
                .order_by(PortfolioSnapshotRecord.observed_at.desc())
                .limit(1)
            )
            peak = max(
                nav,
                Decimal(epoch.starting_nav),
                Decimal(prior.peak_nav) if prior is not None else Decimal("0"),
            )
            drawdown = Decimal("0") if peak <= 0 else max(Decimal("0"), (peak - nav) / peak)
            existing_snapshot = session.get(PortfolioSnapshotRecord, snapshot_id)
            if existing_snapshot is None:
                session.add(
                    PortfolioSnapshotRecord(
                        snapshot_id=snapshot_id,
                        run_id=run_id or None,
                        epoch_id=epoch.epoch_id,
                        account_key=account_key,
                        mode=mode,
                        observed_at=observed_at,
                        session_date=benchmark.session_date,
                        nav=nav,
                        cash=cash,
                        invested_weight=Decimal("0") if nav <= 0 else Decimal("1") - cash / nav,
                        peak_nav=peak,
                        drawdown=drawdown,
                        beta=None if prior is None else prior.beta,
                        tracking_error=None if prior is None else prior.tracking_error,
                        cumulative_return=cumulative_return,
                        cumulative_benchmark_return=cumulative_benchmark,
                        cumulative_excess_return=cumulative_excess,
                    )
                )
                session.flush()
                for symbol, quantity, price, sleeve, attribution in positions:
                    value = quantity * price
                    session.add(
                        PositionSnapshotRecord(
                            snapshot_id=snapshot_id,
                            symbol=symbol,
                            sleeve=sleeve,
                            attribution=attribution,
                            quantity=quantity,
                            market_price=price,
                            market_value=value,
                            weight=Decimal("0") if nav <= 0 else value / nav,
                        )
                    )
            elif (
                Decimal(existing_snapshot.nav) != nav
                or Decimal(existing_snapshot.cash) != cash
            ):
                raise RuntimeError("conflicting EOD portfolio snapshot for the same NY session")
            else:
                persisted_rows = session.scalars(
                    select(PositionSnapshotRecord).where(
                        PositionSnapshotRecord.snapshot_id == snapshot_id
                    )
                ).all()
                persisted = {
                    (row.symbol, row.attribution): (
                        Decimal(row.quantity),
                        Decimal(row.market_price),
                    )
                    for row in persisted_rows
                }
                expected = {
                    (symbol, attribution): (Decimal(quantity), Decimal(price))
                    for symbol, quantity, price, _sleeve, attribution in positions
                }
                if persisted != expected:
                    raise RuntimeError(
                        "conflicting EOD positions for the same NY session"
                    )

            if mode == "shadow":
                shadow = session.scalar(
                    select(ShadowAccountRecord).where(
                        ShadowAccountRecord.account_key == account_key,
                        ShadowAccountRecord.epoch_id == epoch.epoch_id,
                    )
                )
                if shadow is not None and Decimal(shadow.peak_nav) < peak:
                    shadow.peak_nav = peak
                    shadow.updated_at = observed_at
            session.commit()

        return EODPerformanceResult(
            snapshot_id=snapshot_id,
            nav=nav,
            drawdown=drawdown,
            cumulative_return=cumulative_return,
            cumulative_benchmark_return=cumulative_benchmark,
            cumulative_excess_return=cumulative_excess,
        )

    def _latest_position_lots(
        self, account_key: str
    ) -> dict[str, dict[str, Decimal]]:
        with Session(self.engine) as session:
            latest_id = session.scalar(
                select(PortfolioSnapshotRecord.snapshot_id)
                .where(
                    PortfolioSnapshotRecord.account_key == account_key,
                    PortfolioSnapshotRecord.mode == "paper",
                )
                .order_by(PortfolioSnapshotRecord.observed_at.desc())
                .limit(1)
            )
            if latest_id is None:
                return {}
            rows = session.scalars(
                select(PositionSnapshotRecord).where(
                    PositionSnapshotRecord.snapshot_id == latest_id
                )
            ).all()
            result: dict[str, dict[str, Decimal]] = {}
            for row in rows:
                result.setdefault(row.symbol, {})[row.attribution] = (
                    result.setdefault(row.symbol, {}).get(
                        row.attribution, Decimal("0")
                    )
                    + Decimal(row.quantity)
                )
            return result


__all__ = ["EODPerformanceResult", "EODPerformanceStore"]
