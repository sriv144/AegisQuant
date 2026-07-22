"""Durable shadow account state and snapshots."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Mapping

from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import sessionmaker

from src.db.v3_models import (
    ExecutionRun,
    OrderEventRecord,
    OrderIntentRecord,
    PortfolioSnapshotRecord,
    PositionSnapshotRecord,
    ShadowAccountRecord,
    ShadowPositionRecord,
    StrategyEpoch,
)
from src.execution.v3 import (
    PortfolioPlan,
    QuoteSnapshot,
    ShadowAccount,
    ShadowExecutionResult,
    ShadowExecutor,
    ShadowFill,
    ShadowPosition,
)


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


class ShadowAccountStore:
    """Load and atomically checkpoint an isolated hypothetical portfolio."""

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
        self.SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    def load_or_create(self, account_key: str, starting_nav: Decimal) -> ShadowAccount:
        shadow_id = self.shadow_account_id(account_key)
        epoch_id = self.epoch_id(account_key)
        with self.SessionLocal.begin() as session:
            row = session.get(ShadowAccountRecord, shadow_id, with_for_update=True)
            if row is None:
                epoch = session.get(StrategyEpoch, epoch_id)
                if epoch is None:
                    session.add(
                        StrategyEpoch(
                            epoch_id=epoch_id,
                            account_key=account_key,
                            account_fingerprint=_stable_id("fingerprint", account_key),
                            mode="shadow",
                            starting_nav=starting_nav,
                            strategy_id=self.strategy_id,
                            strategy_version=self.strategy_version,
                            config_sha256=self.config_sha256,
                            activated_at=datetime.now(UTC),
                            migration_cost=Decimal("0"),
                        )
                    )
                elif Decimal(epoch.starting_nav) != starting_nav:
                    raise RuntimeError(
                        "shadow epoch starting NAV conflicts with audited bootstrap"
                    )
                row = ShadowAccountRecord(
                    shadow_account_id=shadow_id,
                    epoch_id=epoch_id,
                    account_key=account_key,
                    cash=starting_nav,
                    peak_nav=starting_nav,
                    version=0,
                    updated_at=datetime.now(UTC),
                )
                session.add(row)
                positions: list[ShadowPositionRecord] = []
            else:
                epoch = session.get(StrategyEpoch, row.epoch_id)
                if epoch is None:
                    raise RuntimeError("shadow account references a missing strategy epoch")
                if (
                    epoch.config_sha256 != self.config_sha256
                    or epoch.strategy_id != self.strategy_id
                    or epoch.strategy_version != self.strategy_version
                    or Decimal(epoch.starting_nav) != starting_nav
                ):
                    raise RuntimeError(
                        "shadow account identity or starting NAV conflicts with its epoch"
                    )
                positions = session.scalars(
                    select(ShadowPositionRecord).where(
                        ShadowPositionRecord.shadow_account_id == shadow_id
                    )
                ).all()
        return ShadowAccount(
            account_key=account_key,
            cash=Decimal(row.cash),
            positions={
                position.symbol: ShadowPosition(
                    symbol=position.symbol,
                    quantity=Decimal(position.quantity),
                    average_cost=Decimal(position.cost_basis),
                )
                for position in positions
            },
        )

    def checkpoint(
        self,
        *,
        account: ShadowAccount,
        plan: PortfolioPlan,
        quotes: Mapping[str, QuoteSnapshot],
        decision_key: str,
        observed_at: datetime,
        fills: tuple[ShadowFill, ...] = (),
    ) -> None:
        shadow_id = self.shadow_account_id(account.account_key)
        epoch_id = self.epoch_id(account.account_key)
        nav = account.net_asset_value(quotes)
        with self.SessionLocal.begin() as session:
            row = session.get(ShadowAccountRecord, shadow_id, with_for_update=True)
            if row is None:
                raise RuntimeError("shadow account was not bootstrapped")
            peak = max(Decimal(row.peak_nav), nav)
            row.cash = account.cash
            row.peak_nav = peak
            row.version += 1
            row.updated_at = observed_at
            session.execute(
                delete(ShadowPositionRecord).where(
                    ShadowPositionRecord.shadow_account_id == shadow_id
                )
            )
            for position in account.positions.values():
                session.add(
                    ShadowPositionRecord(
                        shadow_account_id=shadow_id,
                        symbol=position.symbol,
                        sleeve=("core" if position.symbol == "SPY" else "momentum_satellite"),
                        quantity=position.quantity,
                        cost_basis=position.average_cost,
                        updated_at=observed_at,
                    )
                )

            run = session.scalar(
                select(ExecutionRun).where(ExecutionRun.decision_key == decision_key)
            )
            if run is None:
                raise RuntimeError("shadow execution run was not persisted before fills")
            for fill in fills:
                if session.get(OrderIntentRecord, fill.client_order_id) is None:
                    is_buy = fill.side.value == "buy"
                    session.add(
                        OrderIntentRecord(
                            client_order_id=fill.client_order_id,
                            run_id=run.run_id,
                            decision_key=decision_key,
                            sleeve=("core" if fill.symbol == "SPY" else "momentum_satellite"),
                            symbol=fill.symbol,
                            side=fill.side.value,
                            requested_quantity=None if is_buy else fill.frozen_order_amount,
                            requested_notional=fill.frozen_order_amount if is_buy else None,
                            frozen_order_amount=format(
                                fill.frozen_order_amount.normalize(), "f"
                            ),
                            target_weight=plan.target_weights.get(fill.symbol, Decimal("0")),
                            arrival_bid=fill.arrival_price,
                            arrival_ask=fill.arrival_price,
                            arrival_quote_at=fill.filled_at,
                            created_at=fill.filled_at,
                        )
                    )
                    session.flush()
                    accepted_id = _stable_id(fill.client_order_id, "accepted")
                    filled_id = _stable_id(fill.client_order_id, "filled")
                    session.add_all(
                        (
                            OrderEventRecord(
                                event_id=accepted_id,
                                client_order_id=fill.client_order_id,
                                broker_order_id=f"shadow-{fill.client_order_id}",
                                state="accepted",
                                observed_at=fill.filled_at,
                                filled_quantity=Decimal("0"),
                                filled_average_price=None,
                                slippage_bps=None,
                                reason="shadow simulation",
                                raw_status="accepted",
                            ),
                            OrderEventRecord(
                                event_id=filled_id,
                                client_order_id=fill.client_order_id,
                                broker_order_id=f"shadow-{fill.client_order_id}",
                                state="filled",
                                observed_at=fill.filled_at,
                                filled_quantity=fill.quantity,
                                filled_average_price=fill.fill_price,
                                slippage_bps=(
                                    abs(fill.fill_price - fill.arrival_price)
                                    / fill.arrival_price
                                    * Decimal("10000")
                                ),
                                reason="shadow simulation",
                                raw_status="filled",
                            ),
                        )
                    )
            snapshot_id = str(uuid.uuid4())
            drawdown = Decimal("0") if peak <= 0 else max(Decimal("0"), (peak - nav) / peak)
            session.add(
                PortfolioSnapshotRecord(
                    snapshot_id=snapshot_id,
                    run_id=None if run is None else run.run_id,
                    epoch_id=epoch_id,
                    account_key=account.account_key,
                    mode="shadow",
                    observed_at=observed_at,
                    session_date=observed_at.date(),
                    nav=nav,
                    cash=account.cash,
                    invested_weight=(Decimal("0") if nav <= 0 else Decimal("1") - account.cash / nav),
                    peak_nav=peak,
                    drawdown=drawdown,
                    beta=plan.metadata.get("portfolio_beta"),
                    tracking_error=plan.metadata.get("tracking_error"),
                    cumulative_return=None,
                    cumulative_benchmark_return=None,
                    cumulative_excess_return=None,
                )
            )
            for position in account.positions.values():
                price = quotes[position.symbol].midpoint
                value = position.quantity * price
                session.add(
                    PositionSnapshotRecord(
                        snapshot_id=snapshot_id,
                        symbol=position.symbol,
                        sleeve=("core" if position.symbol == "SPY" else "momentum_satellite"),
                        attribution="V3_ATTRIBUTED",
                        quantity=position.quantity,
                        market_price=price,
                        market_value=value,
                        weight=Decimal("0") if nav <= 0 else value / nav,
                    )
                )

    def current_drawdown(
        self, account: ShadowAccount, quotes: Mapping[str, QuoteSnapshot]
    ) -> Decimal:
        with self.SessionLocal() as session:
            row = session.get(ShadowAccountRecord, self.shadow_account_id(account.account_key))
            if row is None:
                return Decimal("0")
            nav = account.net_asset_value(quotes)
            peak = Decimal(row.peak_nav)
            return Decimal("0") if peak <= 0 else max(Decimal("0"), (peak - nav) / peak)

    def shadow_account_id(self, account_key: str) -> str:
        return _stable_id(
            "shadow-account",
            account_key,
            self.strategy_id,
            self.strategy_version,
            self.config_sha256,
        )

    def epoch_id(self, account_key: str) -> str:
        return _stable_id(
            "strategy-epoch",
            account_key,
            "shadow",
            self.strategy_id,
            self.strategy_version,
            self.config_sha256,
        )


class DurableShadowExecutor(ShadowExecutor):
    """Checkpoint shadow state before the execution run is marked complete."""

    def __init__(self, store: ShadowAccountStore, **kwargs) -> None:
        super().__init__(**kwargs)
        self.store = store

    def execute(
        self,
        *,
        account: ShadowAccount,
        plan: PortfolioPlan,
        quotes: Mapping[str, QuoteSnapshot],
        decision_key: str,
        now: datetime,
    ) -> ShadowExecutionResult:
        result = super().execute(
            account=account,
            plan=plan,
            quotes=quotes,
            decision_key=decision_key,
            now=now,
        )
        self.store.checkpoint(
            account=account,
            plan=plan,
            quotes=quotes,
            decision_key=decision_key,
            observed_at=now,
            fills=result.fills,
        )
        return result


__all__ = ["DurableShadowExecutor", "ShadowAccountStore"]
