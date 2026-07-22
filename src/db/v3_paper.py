"""Transactional paper-completion persistence hook."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from src.db.v3_models import (
    ExecutionRun,
    PortfolioSnapshotRecord,
    PositionSnapshotRecord,
    StrategyEpoch,
)
from src.execution.v3 import OrderSide, OrderState, PaperCompletionSnapshot


def _id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


class SQLPaperCompletionRecorder:
    """Persist final paper NAV/positions before the account lease is released."""

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

    def current_peak(self, account_key: str) -> Decimal | None:
        """Return the durable all-time paper high-water mark for this account."""

        with Session(self.engine) as session:
            value = session.scalar(
                select(func.max(PortfolioSnapshotRecord.peak_nav)).where(
                    PortfolioSnapshotRecord.account_key == account_key,
                    PortfolioSnapshotRecord.mode == "paper",
                )
            )
        return None if value is None else Decimal(value)

    def is_de_risked(self, account_key: str) -> bool:
        """Return the latest completed, config-bound paper risk posture."""

        history = self._risk_posture_history(account_key)
        return bool(history and history[-1][1])

    def de_risked_since(self, account_key: str) -> datetime | None:
        """Return the start of the current uninterrupted de-risk episode."""

        history = self._risk_posture_history(account_key)
        if not history or not history[-1][1]:
            return None
        activated_at: datetime | None = None
        for observed_at, de_risked in history:
            if de_risked and activated_at is None:
                activated_at = observed_at
            elif not de_risked:
                activated_at = None
        return activated_at

    def _risk_posture_history(
        self, account_key: str
    ) -> list[tuple[datetime, bool]]:
        """Read config-bound completed postures in durable chronological order."""

        with Session(self.engine) as session:
            rows = session.scalars(
                select(ExecutionRun)
                .where(
                    ExecutionRun.account_key == account_key,
                    ExecutionRun.mode == "paper",
                    ExecutionRun.purpose == "rebalance",
                    ExecutionRun.status == "completed",
                    ExecutionRun.strategy_id == self.strategy_id,
                    ExecutionRun.strategy_version == self.strategy_version,
                )
                .order_by(ExecutionRun.completed_at.asc(), ExecutionRun.started_at.asc())
            ).all()
        history: list[tuple[datetime, bool]] = []
        for row in rows:
            metadata = dict(row.metadata_json or {})
            plan_metadata = metadata.get("plan_metadata")
            if not isinstance(plan_metadata, dict):
                continue
            if str(plan_metadata.get("config_sha256", "")) != self.config_sha256:
                continue
            observed_at = row.completed_at or row.started_at
            if observed_at.tzinfo is None:
                observed_at = observed_at.replace(tzinfo=UTC)
            history.append(
                (
                    observed_at,
                    bool(
                        plan_metadata.get("drawdown_kill") is True
                        or plan_metadata.get("de_risk_active") is True
                    ),
                )
            )
        return history

    def record_paper_completion(self, snapshot: PaperCompletionSnapshot) -> None:
        account = snapshot.account
        snapshot_id = _id("paper-completion", snapshot.run_id, snapshot.target_hash)
        epoch_id = _id(
            "strategy-epoch",
            account.account_key,
            "paper",
            self.strategy_id,
            self.strategy_version,
            self.config_sha256,
        )
        migration_cost = self._estimated_slippage_cost(snapshot)
        with Session(self.engine) as session:
            run = session.get(ExecutionRun, snapshot.run_id, with_for_update=True)
            if run is None:
                raise RuntimeError("paper completion references an unknown execution run")
            if run.account_key != account.account_key:
                raise RuntimeError("paper completion account does not match execution run")
            if run.target_hash != snapshot.target_hash:
                raise RuntimeError("paper completion target does not match execution run")
            existing = session.get(PortfolioSnapshotRecord, snapshot_id)
            if existing is not None:
                run_metadata = dict(run.metadata_json or {})
                if (
                    existing.run_id != snapshot.run_id
                    or existing.account_key != account.account_key
                    or existing.mode != "paper"
                    or run_metadata.get("paper_snapshot_id") != snapshot_id
                    or run_metadata.get("target_hash") != snapshot.target_hash
                ):
                    raise RuntimeError("conflicting paper completion snapshot")
                # This transaction committed final broker truth previously. A
                # retry may observe new marks, cash or manual account activity;
                # the immutable completion remains authoritative for this run.
                return

            attributed_positions = self._attributed_positions(session, snapshot, snapshot_id)

            epoch = session.get(StrategyEpoch, epoch_id)
            if epoch is None:
                starting_nav = account.equity + migration_cost
                epoch = StrategyEpoch(
                    epoch_id=epoch_id,
                    account_key=account.account_key,
                    account_fingerprint=_id("fingerprint", account.account_key),
                    mode="paper",
                    starting_nav=starting_nav,
                    strategy_id=self.strategy_id,
                    strategy_version=self.strategy_version,
                    config_sha256=self.config_sha256,
                    activated_at=snapshot.observed_at,
                    migration_cost=migration_cost,
                )
                session.add(epoch)
                session.flush()

            prior = session.scalar(
                select(PortfolioSnapshotRecord)
                .where(
                    PortfolioSnapshotRecord.account_key == account.account_key,
                    PortfolioSnapshotRecord.mode == "paper",
                    PortfolioSnapshotRecord.epoch_id == epoch_id,
                )
                .order_by(PortfolioSnapshotRecord.observed_at.desc())
                .limit(1)
            )
            peak = max(
                account.equity,
                Decimal(epoch.starting_nav),
                Decimal(prior.peak_nav) if prior is not None else Decimal("0"),
            )
            drawdown = (
                Decimal("0")
                if peak <= 0
                else max(Decimal("0"), (peak - account.equity) / peak)
            )
            run_metadata = dict(run.metadata_json or {})
            plan_metadata = dict(getattr(snapshot, "plan_metadata", {}) or {})
            beta = plan_metadata.get("portfolio_beta")
            tracking_error = plan_metadata.get("tracking_error")
            session.add(
                PortfolioSnapshotRecord(
                    snapshot_id=snapshot_id,
                    run_id=snapshot.run_id,
                    epoch_id=epoch_id,
                    account_key=account.account_key,
                    mode="paper",
                    observed_at=snapshot.observed_at,
                    session_date=snapshot.observed_at.astimezone(
                        ZoneInfo("America/New_York")
                    ).date(),
                    nav=account.equity,
                    cash=account.cash,
                    invested_weight=(
                        Decimal("0")
                        if account.equity <= 0
                        else Decimal("1") - account.cash / account.equity
                    ),
                    peak_nav=peak,
                    drawdown=drawdown,
                    beta=None if beta is None else Decimal(str(beta)),
                    tracking_error=(
                        None if tracking_error is None else Decimal(str(tracking_error))
                    ),
                    cumulative_return=account.equity / Decimal(epoch.starting_nav)
                    - Decimal("1"),
                    cumulative_benchmark_return=None,
                    cumulative_excess_return=None,
                )
            )
            session.flush()
            actual_weights: dict[str, Decimal] = {}
            for symbol, quantity, price, sleeve, attribution in attributed_positions:
                value = quantity * price
                actual_weight = (
                    Decimal("0") if account.equity <= 0 else value / account.equity
                )
                actual_weights[symbol] = actual_weights.get(symbol, Decimal("0")) + actual_weight
                session.add(
                    PositionSnapshotRecord(
                        snapshot_id=snapshot_id,
                        symbol=symbol,
                        sleeve=sleeve,
                        attribution=attribution,
                        quantity=quantity,
                        market_price=price,
                        market_value=value,
                        weight=actual_weight,
                    )
                )
            expected_weights = {
                symbol: Decimal(weight)
                for symbol, weight in snapshot.target_weights.items()
            }
            max_position_drift = max(
                (
                    abs(actual_weights.get(symbol, Decimal("0")) - target)
                    for symbol, target in expected_weights.items()
                ),
                default=Decimal("0"),
            )
            off_target_weight = sum(
                (
                    weight
                    for symbol, weight in actual_weights.items()
                    if symbol not in expected_weights
                ),
                Decimal("0"),
            )
            max_position_drift = max(max_position_drift, off_target_weight)
            invested_drift = abs(
                sum(actual_weights.values(), Decimal("0"))
                - sum(expected_weights.values(), Decimal("0"))
            )
            run_metadata.update(
                {
                    "paper_snapshot_id": snapshot_id,
                    "migration_cost": str(migration_cost),
                    "ending_nav": str(account.equity),
                    "ending_cash": str(account.cash),
                    "target_hash": snapshot.target_hash,
                    "max_position_drift_bps": str(
                        max_position_drift * Decimal("10000")
                    ),
                    "invested_weight_drift_bps": str(
                        invested_drift * Decimal("10000")
                    ),
                    "post_fill_drift_within_50bps": (
                        max_position_drift <= Decimal("0.005")
                        and invested_drift <= Decimal("0.005")
                    ),
                }
            )
            run.metadata_json = run_metadata
            session.commit()

    @staticmethod
    def _attributed_positions(
        session: Session,
        snapshot: PaperCompletionSnapshot,
        snapshot_id: str,
    ) -> tuple[tuple[str, Decimal, Decimal, str, str], ...]:
        """Split broker quantities into durable legacy and v3 lots conservatively.

        Filled sells consume legacy lots first. Filled buys create v3 lots. Any
        unexplained increase is deliberately left unattributed so manual or
        external activity can never be counted as strategy alpha.
        """

        prior_snapshot_id = session.scalar(
            select(PortfolioSnapshotRecord.snapshot_id)
            .where(
                PortfolioSnapshotRecord.account_key == snapshot.account.account_key,
                PortfolioSnapshotRecord.mode == "paper",
                PortfolioSnapshotRecord.snapshot_id != snapshot_id,
                PortfolioSnapshotRecord.observed_at <= snapshot.observed_at,
            )
            .order_by(
                PortfolioSnapshotRecord.observed_at.desc(),
                PortfolioSnapshotRecord.created_at.desc(),
            )
            .limit(1)
        )
        prior_legacy: defaultdict[str, Decimal] = defaultdict(Decimal)
        prior_v3: defaultdict[str, Decimal] = defaultdict(Decimal)
        if prior_snapshot_id is not None:
            prior_rows = session.scalars(
                select(PositionSnapshotRecord).where(
                    PositionSnapshotRecord.snapshot_id == prior_snapshot_id
                )
            ).all()
            for row in prior_rows:
                quantity = Decimal(row.quantity)
                if row.attribution == "V3_ATTRIBUTED":
                    prior_v3[row.symbol] += quantity
                else:
                    prior_legacy[row.symbol] += quantity

        intents = {intent.client_order_id: intent for intent in snapshot.intents}
        cumulative_fills: defaultdict[str, Decimal] = defaultdict(Decimal)
        for event in snapshot.events:
            if event.client_order_id in intents:
                cumulative_fills[event.client_order_id] = max(
                    cumulative_fills[event.client_order_id],
                    Decimal(event.filled_quantity),
                )
        buys: defaultdict[str, Decimal] = defaultdict(Decimal)
        sells: defaultdict[str, Decimal] = defaultdict(Decimal)
        for client_id, quantity in cumulative_fills.items():
            if quantity <= 0:
                continue
            intent = intents[client_id]
            if intent.side is OrderSide.BUY:
                buys[intent.symbol] += quantity
            else:
                sells[intent.symbol] += quantity

        rows: list[tuple[str, Decimal, Decimal, str, str]] = []
        for position in snapshot.positions:
            symbol = position.symbol
            final_quantity = Decimal(position.quantity)
            if final_quantity <= 0:
                continue
            sold = sells[symbol]
            legacy_after_sells = max(Decimal("0"), prior_legacy[symbol] - sold)
            residual_sells = max(Decimal("0"), sold - prior_legacy[symbol])
            v3_after_sells = max(Decimal("0"), prior_v3[symbol] - residual_sells)
            expected_v3 = v3_after_sells + buys[symbol]

            legacy_quantity = min(final_quantity, legacy_after_sells)
            remaining = final_quantity - legacy_quantity
            v3_quantity = min(remaining, expected_v3)
            # Broker quantity not explained by frozen v3 fills remains legacy.
            legacy_quantity += remaining - v3_quantity

            if legacy_quantity > 0:
                rows.append(
                    (
                        symbol,
                        legacy_quantity,
                        Decimal(position.market_price),
                        "legacy_unattributed",
                        "LEGACY_UNATTRIBUTED",
                    )
                )
            if v3_quantity > 0:
                rows.append(
                    (
                        symbol,
                        v3_quantity,
                        Decimal(position.market_price),
                        "core" if symbol == "SPY" else "momentum_satellite",
                        "V3_ATTRIBUTED",
                    )
                )
        return tuple(rows)

    @staticmethod
    def _estimated_slippage_cost(snapshot: PaperCompletionSnapshot) -> Decimal:
        intents = {intent.client_order_id: intent for intent in snapshot.intents}
        latest_fills = {}
        for event in snapshot.events:
            if event.state is OrderState.FILLED:
                latest_fills[event.client_order_id] = event
        cost = Decimal("0")
        for client_id, event in latest_fills.items():
            intent = intents.get(client_id)
            if (
                intent is None
                or event.filled_average_price is None
                or event.filled_quantity <= 0
            ):
                continue
            price_delta = (
                event.filled_average_price - intent.arrival_price
                if intent.side is OrderSide.BUY
                else intent.arrival_price - event.filled_average_price
            )
            cost += max(Decimal("0"), price_delta) * event.filled_quantity
        return cost


__all__ = ["SQLPaperCompletionRecorder"]
