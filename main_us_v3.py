"""AegisQuant v3 one-shot runtime.

This is the only supported US strategy entrypoint.  It defaults to a read-only
shadow health probe, accepts only ``shadow`` and Alpaca ``paper`` modes, and
always emits a complete audit artifact bundle.  There is intentionally no
live mode or automatic market-data download path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, select

from src.db.v3_ledger import SQLAlchemyLedger, V3SchemaMissing
from src.db.v3_performance import EODPerformanceStore
from src.db.v3_paper import SQLPaperCompletionRecorder
from src.db.v3_research_store import DataManifestStore
from src.db.v3_models import (
    PortfolioSnapshotRecord,
    PositionSnapshotRecord,
)
from src.db.v3_shadow import DurableShadowExecutor, ShadowAccountStore
from src.execution.v3 import (
    AlpacaPyGateway,
    BrokerReadError,
    ExecutionCoordinator,
    InMemoryLedger,
    PaperPlanFactory,
    PaperPlanningContext,
    PortfolioPlan as ExecutionPortfolioPlan,
    OrderSide,
    RunPurpose,
    RunRecord,
    RunResult,
    RunStatus,
    RuntimeSettings,
    SettingsValidationError,
    SafetyBlock,
    TradingMode,
    build_decision_key,
    build_operational_key,
    build_target_hash,
)
from src.v3 import DataFailure, PortfolioConstructor, load_strategy_config
from src.v3.artifacts import ArtifactWriter
from src.v3.config import StrategyConfigError
from src.v3.runtime_input import RuntimeInputBundle, RuntimeInputError, load_runtime_input
from src.v3.promotion import ExperimentRegistry


NEW_YORK = ZoneInfo("America/New_York")
PAPER_URL = "https://paper-api.alpaca.markets"


class RuntimeBlock(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AegisQuant v3 safe one-shot runtime")
    parser.add_argument("--mode", default=None, help="shadow or paper; live is invalid")
    parser.add_argument(
        "--purpose",
        default=None,
        help="health, eod, rebalance, reconcile, or bootstrap",
    )
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="deprecated alias for --mode shadow",
    )
    parser.add_argument("--input-bundle", default=None, help="frozen point-in-time JSON input")
    parser.add_argument("--strategy-config", default=None, help="tracked immutable strategy JSON")
    return parser


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeBlock(f"{name} must be a boolean")


def _decimal_env(name: str, default: str) -> Decimal:
    try:
        return Decimal(os.getenv(name, default))
    except Exception as exc:
        raise RuntimeBlock(f"{name} must be numeric") from exc


def _safe_artifact_run_id() -> str:
    supplied = os.getenv("AEGISQUANT_RUN_ID", "").strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if supplied and supplied[0].isalnum() and len(supplied) <= 128 and set(supplied) <= allowed:
        return supplied
    return f"local-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def _load_dotenv_if_enabled() -> None:
    if _bool_env("AEGISQUANT_SKIP_DOTENV", False):
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        return


def _settings(args: argparse.Namespace, config) -> RuntimeSettings:
    raw_mode = args.mode or os.getenv("TRADING_MODE", "shadow")
    if args.dry_run:
        if args.mode and args.mode != TradingMode.SHADOW.value:
            raise SettingsValidationError("--dry-run cannot be combined with a non-shadow mode")
        raw_mode = TradingMode.SHADOW.value
    raw_purpose = args.purpose or os.getenv("RUN_PURPOSE", "health")
    mode = TradingMode(raw_mode)
    account_key = os.getenv("AEGISQUANT_ACCOUNT_KEY", "").strip()
    if mode is TradingMode.SHADOW and not account_key:
        account_key = f"shadow-{config.sha256[:16]}"

    environment_strategy = os.getenv("STRATEGY_ID", config.strategy_id)
    environment_version = os.getenv("STRATEGY_VERSION", config.version)
    if environment_strategy != config.strategy_id or environment_version != config.version:
        raise RuntimeBlock("runtime strategy identity does not match the tracked configuration")
    if os.getenv("BENCHMARK_SYMBOL", config.benchmark).upper() != config.benchmark:
        raise RuntimeBlock("runtime benchmark does not match the tracked configuration")
    if _bool_env("RL_ENABLED", False):
        raise RuntimeBlock("RL is quarantined and cannot load or alter v3 targets")

    database_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL", "")
    return RuntimeSettings(
        mode=mode,
        purpose=RunPurpose(raw_purpose),
        strategy_id=config.strategy_id,
        strategy_version=config.version,
        strategy_config_sha256=config.sha256,
        commit_sha=os.getenv("GITHUB_SHA", os.getenv("COMMIT_SHA", "unknown")),
        benchmark=config.benchmark,
        database_url=database_url,
        execution_enabled=_bool_env("PAPER_EXECUTION_ENABLED", False),
        kill_switch=_bool_env("KILL_SWITCH", True),
        alpaca_base_url=os.getenv("ALPACA_BASE_URL", PAPER_URL),
        alpaca_api_key=os.getenv("ALPACA_API_KEY", ""),
        alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        account_key=account_key,
        quote_max_age_seconds=int(os.getenv("V3_QUOTE_MAX_AGE_SECONDS", "60")),
        unresolved_order_minutes=int(os.getenv("V3_ORDER_TIMEOUT_MINUTES", "15")),
        min_trade_notional=_decimal_env("V3_MIN_ORDER_NOTIONAL_USD", "100"),
        min_drift_fraction=_decimal_env("V3_MIN_DRIFT_BPS", "20") / Decimal("10000"),
        adv_limit_fraction=Decimal("0.05"),
        buying_power_buffer_fraction=Decimal("0.005"),
    )


def _bootstrap_schema(database_url: str) -> None:
    if not database_url:
        raise RuntimeBlock("bootstrap requires DATABASE_URL or POSTGRES_URL")
    root = Path(__file__).resolve().parent
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    command.upgrade(config, "head")


def _make_ledger(settings: RuntimeSettings):
    if settings.database_url:
        return SQLAlchemyLedger(settings.database_url)
    if settings.mode is TradingMode.PAPER or settings.purpose in {
        RunPurpose.REBALANCE,
        RunPurpose.RECONCILE,
        RunPurpose.EOD,
        RunPurpose.BOOTSTRAP,
    }:
        raise RuntimeBlock(f"{settings.mode.value}/{settings.purpose.value} requires durable v3 state")
    return InMemoryLedger()


def _load_bundle(args: argparse.Namespace) -> RuntimeInputBundle:
    path = args.input_bundle or os.getenv("V3_RUNTIME_INPUT", "")
    if not path:
        raise RuntimeBlock("this purpose requires --input-bundle or V3_RUNTIME_INPUT")
    return load_runtime_input(path)


def _scheduled_probe_reason(
    settings: RuntimeSettings,
    bundle: RuntimeInputBundle | None,
    now: datetime,
) -> str | None:
    if os.getenv("GITHUB_EVENT_NAME", "") != "schedule":
        return None
    local = now.astimezone(NEW_YORK)
    if settings.purpose is RunPurpose.REBALANCE:
        if local.hour != 10 or not 5 <= local.minute <= 35:
            return "paired rebalance probe is not the 10:05-10:35 ET instance"
        if bundle is None or local.date() not in bundle.scheduled_eligible_sessions[:3]:
            return "session is outside the frozen first-three-NYSE-session window"
    elif settings.purpose is RunPurpose.EOD:
        if local.hour != 16 or not 35 <= local.minute <= 59:
            return "paired EOD probe is not the 16:35-16:59 ET instance"
    return None


def _validate_monthly_signal(bundle: RuntimeInputBundle, now: datetime) -> None:
    local_now = now.astimezone(NEW_YORK)
    signal = bundle.signal_date.tz_convert(NEW_YORK)
    previous_month = 12 if local_now.month == 1 else local_now.month - 1
    previous_year = local_now.year - 1 if local_now.month == 1 else local_now.year
    if (signal.year, signal.month) != (previous_year, previous_month):
        raise RuntimeBlock("signal timestamp must be a completed session in the prior month")
    if signal > local_now:
        raise RuntimeBlock("signal timestamp cannot be in the future")
    last_price_session = bundle.total_return_prices.index.max()
    if getattr(last_price_session, "tzinfo", None) is None:
        last_price_session = last_price_session.tz_localize(NEW_YORK)
    else:
        last_price_session = last_price_session.tz_convert(NEW_YORK)
    if last_price_session.date() != signal.date():
        raise RuntimeBlock(
            "signal timestamp must equal the final session in the frozen total-return history"
        )


def _validate_shadow_quotes(
    bundle: RuntimeInputBundle,
    required_symbols: set[str],
    now: datetime,
    settings: RuntimeSettings,
) -> None:
    missing = required_symbols - set(bundle.quotes)
    if missing:
        raise RuntimeBlock(f"frozen shadow quotes are missing: {', '.join(sorted(missing))}")
    for symbol in sorted(required_symbols):
        quote = bundle.quotes[symbol]
        age = (now - quote.observed_at.astimezone(UTC)).total_seconds()
        if age < -5 or age > settings.quote_max_age_seconds:
            raise RuntimeBlock(f"quote for {symbol} is stale or future-dated")
        if quote.adv_dollars_30d <= 0:
            raise RuntimeBlock(f"ADV for {symbol} is unavailable")
        _ = quote.midpoint


def _validate_shadow_liquidity(
    account,
    plan: ExecutionPortfolioPlan,
    quotes: Mapping[str, Any],
    settings: RuntimeSettings,
) -> None:
    nav = account.net_asset_value(quotes)
    for symbol in set(plan.target_weights) | set(account.positions):
        quote = quotes[symbol]
        current_value = (
            account.positions[symbol].quantity * quote.midpoint
            if symbol in account.positions
            else Decimal("0")
        )
        target_value = nav * plan.target_weights.get(symbol, Decimal("0"))
        if abs(target_value - current_value) > quote.adv_dollars_30d * settings.adv_limit_fraction:
            raise RuntimeBlock(f"shadow order for {symbol} would exceed 5% of 30-day ADV")


def _migration_delta_preview(
    *,
    account: Any,
    positions: Sequence[Any],
    open_orders: Sequence[Any],
    research_plan: Any,
    bundle: RuntimeInputBundle,
    settings: RuntimeSettings,
) -> dict[str, Any]:
    """Create an attributable paper-migration preview without broker writes."""

    target_weights = {
        symbol: Decimal(str(weight)) for symbol, weight in research_plan.target_weights
    }
    quantities = {position.symbol: Decimal(position.quantity) for position in positions}
    effective = dict(quantities)
    blocked_symbols: set[str] = set()
    for order in open_orders:
        if order.state.is_terminal:
            continue
        remaining = Decimal(order.remaining_quantity)
        effective[order.symbol] = effective.get(order.symbol, Decimal("0")) + (
            remaining if order.side is OrderSide.BUY else -remaining
        )
        if not order.client_order_id.startswith("aq3-p-"):
            blocked_symbols.add(order.symbol)

    symbols = set(target_weights) | set(effective)
    _validate_shadow_quotes(bundle, symbols, account.observed_at, settings)
    rows: list[dict[str, Any]] = []
    gross_notional = Decimal("0")
    for symbol in sorted(symbols):
        quote = bundle.quotes[symbol]
        current_quantity = effective.get(symbol, Decimal("0"))
        current_value = current_quantity * quote.midpoint
        target_weight = target_weights.get(symbol, Decimal("0"))
        target_value = Decimal(account.equity) * target_weight
        delta = target_value - current_value
        notional = abs(delta)
        drift = (
            Decimal("0")
            if Decimal(account.equity) <= 0
            else notional / Decimal(account.equity)
        )
        adv_fraction = notional / quote.adv_dollars_30d
        meets_minimums = (
            drift >= settings.min_drift_fraction
            and notional >= settings.min_trade_notional
        )
        within_adv = adv_fraction <= settings.adv_limit_fraction
        if meets_minimums:
            gross_notional += notional
        rows.append(
            {
                "symbol": symbol,
                "side": "buy" if delta > 0 else "sell" if delta < 0 else "none",
                "current_quantity_including_open_orders": str(current_quantity),
                "current_weight": str(
                    Decimal("0")
                    if Decimal(account.equity) <= 0
                    else current_value / Decimal(account.equity)
                ),
                "target_weight": str(target_weight),
                "estimated_notional": str(notional),
                "drift_bps": str(drift * Decimal("10000")),
                "adv_participation": str(adv_fraction),
                "meets_minimum_trade_rules": meets_minimums,
                "within_five_percent_adv": within_adv,
                "blocked_by_unattributed_open_order": symbol in blocked_symbols,
            }
        )
    estimated_cost = gross_notional * Decimal("0.0005")
    return {
        "preview_only": True,
        "broker_post_count": 0,
        "target_weight_sha256": research_plan.weight_sha256,
        "research_data_sha256": bundle.research_data_sha256,
        "gross_trade_notional_after_minimums": str(gross_notional),
        "estimated_base_cost_at_5bps": str(estimated_cost),
        "unattributed_open_order_symbols": sorted(blocked_symbols),
        "preview_constraints_pass": bool(
            research_plan.promotable
            and not blocked_symbols
            and all(
                row["within_five_percent_adv"]
                for row in rows
                if row["meets_minimum_trade_rules"]
            )
        ),
        "implementation_ready": False,
        "approval_required": "protected manual paper rebalance workflow",
        "deltas": rows,
    }


def _execution_plan(
    research_plan,
    drawdown: Decimal,
    *,
    bundle_sha256: str,
    research_data_sha256: str,
) -> ExecutionPortfolioPlan:
    diagnostics = dict(research_plan.diagnostics)
    de_risk_active = diagnostics.get("construction") == "drawdown_de_risk"
    return ExecutionPortfolioPlan(
        strategy_id=research_plan.strategy_id,
        strategy_version=research_plan.strategy_version,
        as_of=research_plan.signal_date.to_pydatetime(),
        target_weights={
            symbol: Decimal(str(weight)) for symbol, weight in research_plan.target_weights
        },
        drawdown=drawdown,
        metadata={
            "plan_origin": "src.v3.portfolio.PortfolioConstructor",
            "config_sha256": research_plan.config_sha256,
            "source_target_sha256": research_plan.target_sha256,
            "source_weight_hash": research_plan.weight_sha256,
            "benchmark_symbol": research_plan.benchmark_symbol,
            "data_bundle_sha256": bundle_sha256,
            "research_data_sha256": research_data_sha256,
            "cash_weight": str(research_plan.cash_weight),
            "portfolio_beta": str(research_plan.portfolio_beta),
            "tracking_error": str(research_plan.tracking_error),
            "max_active_sector_deviation": str(research_plan.max_active_sector_deviation),
            "promotable": research_plan.promotable,
            "promotion_blockers": list(research_plan.promotion_blockers),
            "drawdown_kill": de_risk_active and drawdown >= Decimal("0.15"),
            "de_risk_active": de_risk_active,
            "satellite_reentry_approved": (
                not de_risk_active
                and bool(diagnostics.get("satellite_reentry_approved") == "true")
            ),
        },
    )


class _LeaseSafePaperPlanFactory(PaperPlanFactory):
    """Bridge fresh broker truth to the single deterministic constructor."""

    def __init__(
        self,
        *,
        config: Any,
        bundle: RuntimeInputBundle,
        engine: Engine,
        commit_sha: str,
    ) -> None:
        self.config = config
        self.bundle = bundle
        self.engine = engine
        self.commit_sha = commit_sha
        self.last_research_plan: Any = None

    def construct(self, context: PaperPlanningContext) -> ExecutionPortfolioPlan:
        reentry_allowed = (
            context.prior_de_risked
            and context.fresh_drawdown < Decimal("0.10")
            and self.bundle.satellite_reentry_approved
        )
        containment_active = (
            context.fresh_drawdown >= Decimal("0.15")
            or (context.prior_de_risked and not reentry_allowed)
        )
        if not containment_active:
            try:
                _validate_monthly_signal(self.bundle, context.now)
            except RuntimeBlock as exc:
                raise SafetyBlock(str(exc)) from exc
        try:
            research_plan = PortfolioConstructor(self.config).construct(
                self.bundle.portfolio_inputs(
                    current_holdings=context.current_holdings,
                    current_drawdown=float(context.fresh_drawdown),
                    prior_de_risked=context.prior_de_risked,
                    satellite_reentry_approved=(
                        self.bundle.satellite_reentry_approved
                        and reentry_allowed
                    ),
                )
            )
        except DataFailure as exc:
            raise SafetyBlock(str(exc)) from exc

        de_risk_plan = (
            dict(research_plan.diagnostics).get("construction")
            == "drawdown_de_risk"
        )
        if not de_risk_plan:
            if not research_plan.promotable:
                raise SafetyBlock(
                    "paper target is not promotable: "
                    + ", ".join(research_plan.promotion_blockers)
                )
            if not ExperimentRegistry(self.engine).has_passing_evidence(
                config=self.config,
                data_manifest_sha256=self.bundle.research_data_sha256,
                commit_sha=self.commit_sha,
                require_trusted_runner=True,
            ):
                raise SafetyBlock(
                    "paper execution lacks passing config/data/commit-bound promotion evidence"
                )

        self.last_research_plan = research_plan
        execution_plan = _execution_plan(
            research_plan,
            context.fresh_drawdown,
            bundle_sha256=self.bundle.bundle_sha256,
            research_data_sha256=self.bundle.research_data_sha256,
        )
        metadata = dict(execution_plan.metadata)
        metadata["drawdown_kill"] = context.fresh_drawdown >= Decimal("0.15")
        metadata["de_risk_active"] = de_risk_plan
        metadata["satellite_reentry_approved"] = bool(reentry_allowed)
        return ExecutionPortfolioPlan(
            strategy_id=execution_plan.strategy_id,
            strategy_version=execution_plan.strategy_version,
            as_of=execution_plan.as_of,
            target_weights=execution_plan.target_weights,
            sleeve=execution_plan.sleeve,
            drawdown=execution_plan.drawdown,
            metadata=metadata,
        )


def _synthetic_result(status: RunStatus, exit_code: int, message: str) -> RunResult:
    return RunResult(
        run_id="",
        status=status,
        exit_code=exit_code,
        message=message,
        decision_key="",
        target_hash="",
    )


def _record_runtime_failure(
    *,
    ledger: Any,
    settings: RuntimeSettings | None,
    previous: RunResult,
    status: RunStatus,
    exit_code: int,
    message: str,
    now: datetime,
    invocation_id: str,
    preflight: dict[str, Any],
) -> RunResult:
    """Persist main-level failures that occur outside coordinator boundaries."""

    run_id = previous.run_id
    decision_key = previous.decision_key
    target_hash = previous.target_hash or build_target_hash({})
    durable = False
    try:
        if run_id and hasattr(ledger, "update_run"):
            ledger.update_run(
                run_id,
                status,
                now,
                failure_reason=message,
                metadata={"runtime_failure": True, "artifact_run_id": invocation_id},
            )
            durable = True
        elif (
            settings is not None
            and settings.mode is TradingMode.PAPER
            and getattr(ledger, "paper_durable_truth", False) is True
            and settings.account_key.strip()
        ):
            local_now = now.astimezone(NEW_YORK)
            intended_key = (
                build_decision_key(
                    settings.strategy_id,
                    settings.strategy_version,
                    settings.account_key,
                    settings.mode,
                    local_now,
                )
                if settings.purpose is RunPurpose.REBALANCE
                else build_operational_key(
                    settings.strategy_id,
                    settings.strategy_version,
                    settings.account_key,
                    settings.mode,
                    settings.purpose,
                    local_now,
                )
            )
            attempt_key = f"{intended_key}|runtime-attempt|{invocation_id}"
            record = ledger.create_run(
                RunRecord(
                    run_id=uuid.uuid4().hex,
                    decision_key=attempt_key,
                    strategy_id=settings.strategy_id,
                    strategy_version=settings.strategy_version,
                    account_key=settings.account_key,
                    mode=settings.mode,
                    purpose=settings.purpose,
                    target_hash=target_hash,
                    created_at=now,
                    metadata={
                        "trigger": os.getenv("GITHUB_EVENT_NAME", "cli"),
                        "commit_sha": settings.commit_sha,
                        "intended_decision_key": intended_key,
                        "runtime_failure": True,
                        "artifact_run_id": invocation_id,
                    },
                )
            )
            run_id = record.run_id
            decision_key = intended_key
            ledger.update_run(
                run_id,
                status,
                now,
                failure_reason=message,
                metadata={"runtime_failure_reason": message},
            )
            durable = True
    except Exception:
        durable = False
    preflight["durable_failure_recorded"] = durable
    return RunResult(
        run_id=run_id if durable else previous.run_id,
        status=status,
        exit_code=exit_code,
        message=message,
        decision_key=decision_key,
        target_hash=target_hash,
        order_client_ids=previous.order_client_ids,
        metadata=dict(previous.metadata),
    )


def _validate_eod_bundle(bundle: RuntimeInputBundle, now: datetime) -> None:
    mark = bundle.benchmark_mark
    if mark is None:
        raise RuntimeBlock("EOD requires a SHA-addressed SPY total-return benchmark mark")
    local_date = now.astimezone(NEW_YORK).date()
    if mark.session_date != local_date:
        raise RuntimeBlock("benchmark mark is not for the current New York session")
    if mark.observed_at > now:
        raise RuntimeBlock("benchmark mark cannot be future-dated")


def _record_legacy_paper_snapshot(
    *,
    engine: Engine,
    run_id: str,
    account: Any,
    positions: Sequence[Any],
    observed_at: datetime,
) -> None:
    snapshot_id = "bootstrap-" + uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"aegisquant|paper|{account.account_key}|{observed_at.date().isoformat()}",
    ).hex
    with SQLAlchemyLedger(engine=engine).SessionLocal.begin() as session:
        existing = session.get(PortfolioSnapshotRecord, snapshot_id)
        if existing is not None:
            persisted_positions = session.scalars(
                select(PositionSnapshotRecord).where(
                    PositionSnapshotRecord.snapshot_id == snapshot_id
                )
            ).all()
            persisted = {
                row.symbol: (Decimal(row.quantity), Decimal(row.market_price))
                for row in persisted_positions
            }
            expected = {
                position.symbol: (
                    Decimal(position.quantity),
                    Decimal(position.market_price),
                )
                for position in positions
            }
            if (
                Decimal(existing.nav) != Decimal(account.equity)
                or Decimal(existing.cash) != Decimal(account.cash)
                or persisted != expected
            ):
                raise RuntimeBlock(
                    "same-session paper bootstrap conflicts with the prior audited snapshot"
                )
            return
        nav = Decimal(account.equity)
        cash = Decimal(account.cash)
        session.add(
            PortfolioSnapshotRecord(
                snapshot_id=snapshot_id,
                run_id=run_id or None,
                epoch_id=None,
                account_key=account.account_key,
                mode="paper",
                observed_at=observed_at,
                session_date=observed_at.astimezone(NEW_YORK).date(),
                nav=nav,
                cash=cash,
                invested_weight=Decimal("0") if nav <= 0 else Decimal("1") - cash / nav,
                peak_nav=nav,
                drawdown=Decimal("0"),
                beta=None,
                tracking_error=None,
                cumulative_return=None,
                cumulative_benchmark_return=None,
                cumulative_excess_return=None,
            )
        )
        session.flush()
        for position in positions:
            value = Decimal(position.quantity) * Decimal(position.market_price)
            session.add(
                PositionSnapshotRecord(
                    snapshot_id=snapshot_id,
                    symbol=position.symbol,
                    sleeve="legacy_unattributed",
                    attribution="LEGACY_UNATTRIBUTED",
                    quantity=position.quantity,
                    market_price=position.market_price,
                    market_value=value,
                    weight=Decimal("0") if nav <= 0 else value / nav,
                )
            )


def _orders_payload(ledger: Any, run_id: str) -> dict[str, Any]:
    if not run_id or not hasattr(ledger, "intents_for_run"):
        return {"intents": [], "events": []}
    try:
        intents = tuple(ledger.intents_for_run(run_id))
        events = [event for intent in intents for event in ledger.events_for(intent.client_order_id)]
        return {"intents": intents, "events": events}
    except (KeyError, RuntimeError):
        return {"intents": [], "events": []}


def _write_artifacts(
    invocation_id: str,
    *,
    started_at: datetime,
    finished_at: datetime,
    args: argparse.Namespace | None,
    settings: RuntimeSettings | None,
    config: Any,
    bundle: RuntimeInputBundle | None,
    research_plan: Any,
    result: RunResult,
    ledger: Any,
    extra_preflight: Mapping[str, Any],
) -> None:
    mode = settings.mode.value if settings else (getattr(args, "mode", None) or "invalid")
    purpose = settings.purpose.value if settings else (getattr(args, "purpose", None) or "invalid")
    orders = _orders_payload(ledger, result.run_id)
    writer = ArtifactWriter.from_environment(Path("artifacts"))
    writer.write_outcome(
        invocation_id,
        manifest={
            "execution_run_id": result.run_id,
            "mode": mode,
            "purpose": purpose,
            "status": result.status.value,
            "exit_code": result.exit_code,
            "strategy_id": getattr(config, "strategy_id", None),
            "strategy_version": getattr(config, "version", None),
            "config_sha256": getattr(config, "sha256", None),
            "commit_sha": os.getenv("COMMIT_SHA", "unknown"),
            "started_at": started_at,
            "finished_at": finished_at,
            "force_recompute": bool(getattr(args, "force_recompute", False)),
        },
        preflight={
            **dict(extra_preflight),
            "mode": mode,
            "purpose": purpose,
            "paper_gate_errors": (
                list(settings.paper_gate_errors()) if settings is not None else []
            ),
            "database_configured": bool(settings and settings.database_url),
            "paper_endpoint_exact": bool(
                settings and settings.alpaca_base_url == PAPER_URL
            ),
            "rl_enabled": os.getenv("RL_ENABLED", "false").strip().lower()
            in {"1", "true", "yes", "on"},
            "message": result.message,
        },
        targets=(
            {
                "available": True,
                "weights": dict(research_plan.target_weights),
                "cash_weight": research_plan.cash_weight,
                "target_sha256": research_plan.target_sha256,
                "weight_sha256": research_plan.weight_sha256,
                "promotable": research_plan.promotable,
                "promotion_blockers": research_plan.promotion_blockers,
                "selected_symbols": research_plan.selected_symbols,
                "portfolio_beta": research_plan.portfolio_beta,
                "tracking_error": research_plan.tracking_error,
                "max_active_sector_deviation": research_plan.max_active_sector_deviation,
                "bundle_sha256": bundle.bundle_sha256 if bundle else None,
            }
            if research_plan is not None
            else {"available": False}
        ),
        orders=orders,
        reconciliation={
            "required": result.status is RunStatus.RECONCILIATION_REQUIRED,
            "order_client_ids": result.order_client_ids,
            "message": result.message,
        },
        performance={
            "available": "ending_nav" in result.metadata,
            "benchmark": "SPY",
            **dict(result.metadata),
        },
        log_lines=(
            f"{finished_at.isoformat()} status={result.status.value} exit={result.exit_code}",
            result.message,
            "--dry-run is deprecated; shadow mode was used" if args and args.dry_run else "",
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    invocation_id = _safe_artifact_run_id()
    started_at = datetime.now(UTC)
    args: argparse.Namespace | None = None
    settings: RuntimeSettings | None = None
    config = None
    bundle: RuntimeInputBundle | None = None
    research_plan = None
    ledger: Any = InMemoryLedger()
    preflight: dict[str, Any] = {}
    result = _synthetic_result(RunStatus.FAILED, 1, "runtime did not start")

    try:
        _load_dotenv_if_enabled()
        args = _parser().parse_args(argv)
        config_path = args.strategy_config or os.getenv("STRATEGY_CONFIG_PATH")
        config = load_strategy_config(config_path)
        settings = _settings(args, config)
        preflight["config_identity"] = config.identity

        if settings.purpose is RunPurpose.BOOTSTRAP:
            if settings.mode is TradingMode.PAPER and not settings.database_url.lower().startswith(
                ("postgresql://", "postgresql+")
            ):
                raise RuntimeBlock("paper bootstrap requires durable PostgreSQL")
            _bootstrap_schema(settings.database_url)
            preflight["migration"] = "alembic head"

        ledger = _make_ledger(settings)
        gateway = None
        shadow_account = None
        shadow_quotes = None
        shadow_executor = None
        execution_plan = None
        paper_plan_factory: _LeaseSafePaperPlanFactory | None = None

        if settings.purpose is RunPurpose.REBALANCE:
            bundle = _load_bundle(args)
            if isinstance(ledger, SQLAlchemyLedger):
                preflight["data_manifest_ids"] = DataManifestStore(ledger.engine).persist(
                    bundle.manifests
                )
            if settings.mode is TradingMode.SHADOW:
                _validate_monthly_signal(bundle, started_at)
                if not isinstance(ledger, SQLAlchemyLedger):
                    raise RuntimeBlock("shadow rebalance requires a durable v3 ledger")
                store = ShadowAccountStore(
                    ledger.engine,
                    strategy_id=config.strategy_id,
                    strategy_version=config.version,
                    config_sha256=config.sha256,
                )
                shadow_account = store.load_or_create(settings.account_key, bundle.starting_nav)
                required_before = set(shadow_account.positions) | {"SPY"}
                _validate_shadow_quotes(bundle, required_before, started_at, settings)
                drawdown = store.current_drawdown(shadow_account, bundle.quotes)
                inputs = bundle.portfolio_inputs(
                    current_holdings=frozenset(shadow_account.positions),
                    current_drawdown=float(drawdown),
                )
                research_plan = PortfolioConstructor(config).construct(inputs)
                execution_plan = _execution_plan(
                    research_plan,
                    drawdown,
                    bundle_sha256=bundle.bundle_sha256,
                    research_data_sha256=bundle.research_data_sha256,
                )
                required = set(execution_plan.target_weights) | set(shadow_account.positions)
                _validate_shadow_quotes(bundle, required, started_at, settings)
                _validate_shadow_liquidity(
                    shadow_account, execution_plan, bundle.quotes, settings
                )
                shadow_quotes = bundle.quotes
                shadow_executor = DurableShadowExecutor(
                    store,
                    one_way_cost_bps=config.research.base_cost_bps_one_way,
                    min_trade_notional=settings.min_trade_notional,
                    min_drift_fraction=settings.min_drift_fraction,
                )
            else:
                if not isinstance(ledger, SQLAlchemyLedger):
                    raise RuntimeBlock("paper rebalance requires durable PostgreSQL")
                commit_sha = os.getenv("GITHUB_SHA", os.getenv("COMMIT_SHA", "unknown"))
                paper_plan_factory = _LeaseSafePaperPlanFactory(
                    config=config,
                    bundle=bundle,
                    engine=ledger.engine,
                    commit_sha=commit_sha,
                )

            schedule_reason = _scheduled_probe_reason(settings, bundle, started_at)
            if schedule_reason:
                result = _synthetic_result(RunStatus.SKIPPED_NOT_DUE, 0, schedule_reason)
            else:
                if settings.mode is TradingMode.PAPER and not settings.paper_gate_errors():
                    gateway = AlpacaPyGateway(settings)
                paper_state = (
                    SQLPaperCompletionRecorder(
                        ledger.engine,
                        strategy_id=config.strategy_id,
                        strategy_version=config.version,
                        config_sha256=config.sha256,
                    )
                    if settings.mode is TradingMode.PAPER
                    and isinstance(ledger, SQLAlchemyLedger)
                    else None
                )
                coordinator = ExecutionCoordinator(
                    settings,
                    ledger,
                    gateway=gateway,
                    shadow_executor=shadow_executor,
                    paper_completion_recorder=paper_state,
                    paper_risk_state_provider=paper_state,
                )
                if settings.mode is TradingMode.PAPER:
                    assert paper_plan_factory is not None
                    result = coordinator.run(
                        paper_plan_factory=paper_plan_factory,
                        now=started_at,
                        trigger=os.getenv("GITHUB_EVENT_NAME", "cli"),
                    )
                    research_plan = paper_plan_factory.last_research_plan
                else:
                    result = coordinator.run(
                        plan=execution_plan,
                        now=started_at,
                        trigger=os.getenv("GITHUB_EVENT_NAME", "cli"),
                        shadow_account=shadow_account,
                        shadow_quotes=shadow_quotes,
                    )
        else:
            schedule_reason = _scheduled_probe_reason(settings, None, started_at)
            if schedule_reason:
                result = _synthetic_result(RunStatus.SKIPPED_NOT_DUE, 0, schedule_reason)
            else:
                bootstrap_account = None
                bootstrap_positions: tuple[Any, ...] = ()
                if settings.purpose is RunPurpose.BOOTSTRAP and settings.mode is TradingMode.PAPER:
                    if not isinstance(ledger, SQLAlchemyLedger) or not ledger.paper_durable_truth:
                        raise RuntimeBlock("paper bootstrap requires verified PostgreSQL v3 truth")
                    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
                        raise RuntimeBlock("paper bootstrap requires Alpaca paper credentials")
                    bundle = _load_bundle(args)
                    _validate_monthly_signal(bundle, started_at)
                    preflight["data_manifest_ids"] = DataManifestStore(
                        ledger.engine
                    ).persist(bundle.manifests)
                    gateway = AlpacaPyGateway(settings)
                    bootstrap_account = gateway.get_account()
                    bootstrap_positions = gateway.get_positions()
                    bootstrap_open_orders = gateway.get_open_orders()
                    bootstrap_order_history = gateway.get_order_history()
                    _ = gateway.get_clock()
                    if settings.account_key and settings.account_key != bootstrap_account.account_key:
                        raise RuntimeBlock("configured account fingerprint does not match Alpaca")
                    settings = replace(settings, account_key=bootstrap_account.account_key)
                    history_state_counts: dict[str, int] = {}
                    history_identity: list[dict[str, str]] = []
                    for order in bootstrap_order_history:
                        history_state_counts[order.state.value] = (
                            history_state_counts.get(order.state.value, 0) + 1
                        )
                        history_identity.append(
                            {
                                "broker_order_id": order.broker_order_id,
                                "client_order_id": order.client_order_id,
                                "symbol": order.symbol,
                                "side": order.side.value,
                                "state": order.state.value,
                                "filled_quantity": str(order.filled_quantity),
                            }
                        )
                    history_sha256 = hashlib.sha256(
                        json.dumps(
                            history_identity,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    research_plan = PortfolioConstructor(config).construct(
                        bundle.portfolio_inputs(
                            current_holdings=frozenset(
                                position.symbol for position in bootstrap_positions
                            ),
                            current_drawdown=0.0,
                        )
                    )
                    migration_preview = _migration_delta_preview(
                        account=bootstrap_account,
                        positions=bootstrap_positions,
                        open_orders=bootstrap_open_orders,
                        research_plan=research_plan,
                        bundle=bundle,
                        settings=settings,
                    )
                    preflight.update(
                        {
                            "discovered_account_key": bootstrap_account.account_key,
                            "audited_starting_nav": str(bootstrap_account.equity),
                            "legacy_position_count": len(bootstrap_positions),
                            "legacy_positions": [
                                {
                                    "symbol": position.symbol,
                                    "quantity": str(position.quantity),
                                    "attribution": "LEGACY_UNATTRIBUTED",
                                }
                                for position in bootstrap_positions
                            ],
                            "open_order_count": len(bootstrap_open_orders),
                            "historical_order_audit": {
                                "read_limit": 500,
                                "row_count": len(bootstrap_order_history),
                                "state_counts": history_state_counts,
                                "sha256": history_sha256,
                            },
                            "migration_delta": migration_preview,
                        }
                    )
                    ShadowAccountStore(
                        ledger.engine,
                        strategy_id=config.strategy_id,
                        strategy_version=config.version,
                        config_sha256=config.sha256,
                    ).load_or_create(
                        bootstrap_account.account_key,
                        Decimal(bootstrap_account.equity),
                    )

                if settings.purpose is RunPurpose.EOD:
                    bundle = _load_bundle(args)
                    _validate_eod_bundle(bundle, started_at)
                    if isinstance(ledger, SQLAlchemyLedger):
                        preflight["data_manifest_ids"] = DataManifestStore(
                            ledger.engine
                        ).persist(bundle.manifests)

                if settings.mode is TradingMode.PAPER and settings.purpose in {
                    RunPurpose.RECONCILE,
                    RunPurpose.EOD,
                }:
                    if settings.alpaca_api_key and settings.alpaca_secret_key:
                        gateway = AlpacaPyGateway(settings)
                paper_state = (
                    SQLPaperCompletionRecorder(
                        ledger.engine,
                        strategy_id=config.strategy_id,
                        strategy_version=config.version,
                        config_sha256=config.sha256,
                    )
                    if settings.mode is TradingMode.PAPER
                    and isinstance(ledger, SQLAlchemyLedger)
                    else None
                )
                coordinator = ExecutionCoordinator(
                    settings,
                    ledger,
                    gateway=gateway,
                    paper_completion_recorder=paper_state,
                    paper_risk_state_provider=paper_state,
                )
                result = coordinator.run(
                    plan=None,
                    now=started_at,
                    trigger=os.getenv("GITHUB_EVENT_NAME", "cli"),
                )
                if settings.purpose is RunPurpose.BOOTSTRAP and result.exit_code == 0:
                    if bootstrap_account is not None:
                        assert isinstance(ledger, SQLAlchemyLedger)
                        _record_legacy_paper_snapshot(
                            engine=ledger.engine,
                            run_id=result.run_id,
                            account=bootstrap_account,
                            positions=bootstrap_positions,
                            observed_at=started_at,
                        )
                    if result.run_id and hasattr(ledger, "update_run"):
                        ledger.update_run(result.run_id, RunStatus.COMPLETED, started_at)
                    result = replace(
                        result,
                        status=RunStatus.COMPLETED,
                        message="v3 schema bootstrap completed",
                    )
                elif settings.purpose is RunPurpose.EOD and result.exit_code == 0:
                    assert bundle is not None and bundle.benchmark_mark is not None
                    if not isinstance(ledger, SQLAlchemyLedger):
                        raise RuntimeBlock("EOD persistence requires the durable v3 ledger")
                    performance_store = EODPerformanceStore(
                        ledger.engine,
                        strategy_id=config.strategy_id,
                        strategy_version=config.version,
                        config_sha256=config.sha256,
                    )
                    if settings.mode is TradingMode.SHADOW:
                        store = ShadowAccountStore(
                            ledger.engine,
                            strategy_id=config.strategy_id,
                            strategy_version=config.version,
                            config_sha256=config.sha256,
                        )
                        shadow_account = store.load_or_create(
                            settings.account_key, bundle.starting_nav
                        )
                        _validate_shadow_quotes(
                            bundle,
                            set(shadow_account.positions),
                            started_at,
                            settings,
                        )
                        performance = performance_store.record_shadow(
                            run_id=result.run_id,
                            account=shadow_account,
                            quotes=bundle.quotes,
                            benchmark=bundle.benchmark_mark,
                            observed_at=started_at,
                        )
                    else:
                        if gateway is None or not ledger.paper_durable_truth:
                            raise RuntimeBlock("paper EOD requires verified PostgreSQL and gateway")
                        account = gateway.get_account()
                        if account.account_key != settings.account_key:
                            raise RuntimeBlock(
                                "configured account fingerprint does not match Alpaca"
                            )
                        positions = gateway.get_positions()
                        open_orders = gateway.get_open_orders()
                        if any(not order.state.is_terminal for order in open_orders):
                            raise RuntimeBlock(
                                "paper EOD found unresolved broker orders; reconcile first"
                            )
                        clock = gateway.get_clock()
                        _ = gateway.get_calendar(
                            started_at.astimezone(NEW_YORK).date(),
                            started_at.astimezone(NEW_YORK).date(),
                        )
                        if abs(
                            (started_at - account.observed_at.astimezone(UTC)).total_seconds()
                        ) > 60:
                            raise RuntimeBlock("Alpaca EOD account snapshot is stale")
                        if abs(
                            (started_at - clock.timestamp.astimezone(UTC)).total_seconds()
                        ) > 60:
                            raise RuntimeBlock("Alpaca EOD clock snapshot is stale")
                        performance = performance_store.record_paper(
                            run_id=result.run_id,
                            account_key=account.account_key,
                            nav=account.equity,
                            cash=account.cash,
                            positions=positions,
                            benchmark=bundle.benchmark_mark,
                            observed_at=started_at,
                        )
                    ledger.update_run(
                        result.run_id,
                        RunStatus.COMPLETED,
                        started_at,
                        metadata={
                            "ending_nav": str(performance.nav),
                            "drawdown": str(performance.drawdown),
                            "cumulative_return": str(performance.cumulative_return),
                            "cumulative_benchmark_return": str(
                                performance.cumulative_benchmark_return
                            ),
                            "cumulative_excess_return": str(
                                performance.cumulative_excess_return
                            ),
                        },
                    )
                    result = replace(
                        result,
                        status=RunStatus.COMPLETED,
                        message="EOD NAV and SPY total-return performance persisted",
                        metadata={
                            "ending_nav": str(performance.nav),
                            "drawdown": str(performance.drawdown),
                            "cumulative_return": str(performance.cumulative_return),
                            "cumulative_benchmark_return": str(
                                performance.cumulative_benchmark_return
                            ),
                            "cumulative_excess_return": str(
                                performance.cumulative_excess_return
                            ),
                        },
                    )
    except (
        DataFailure,
        RuntimeBlock,
        RuntimeInputError,
        SettingsValidationError,
        StrategyConfigError,
        V3SchemaMissing,
        BrokerReadError,
        ValueError,
    ) as exc:
        result = _record_runtime_failure(
            ledger=ledger,
            settings=settings,
            previous=result,
            status=RunStatus.BLOCKED,
            exit_code=2,
            message=str(exc),
            now=datetime.now(UTC),
            invocation_id=invocation_id,
            preflight=preflight,
        )
    except Exception as exc:  # fail closed without leaking a secret-bearing repr
        result = _record_runtime_failure(
            ledger=ledger,
            settings=settings,
            previous=result,
            status=RunStatus.FAILED,
            exit_code=1,
            message=f"unexpected v3 runtime failure: {type(exc).__name__}",
            now=datetime.now(UTC),
            invocation_id=invocation_id,
            preflight=preflight,
        )

    finished_at = datetime.now(UTC)
    try:
        _write_artifacts(
            invocation_id,
            started_at=started_at,
            finished_at=finished_at,
            args=args,
            settings=settings,
            config=config,
            bundle=bundle,
            research_plan=research_plan,
            result=result,
            ledger=ledger,
            extra_preflight=preflight,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "exit_code": 1,
                    "message": f"artifact write failed: {type(exc).__name__}",
                }
            ),
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "status": result.status.value,
                "exit_code": result.exit_code,
                "message": result.message,
                "artifact_run_id": invocation_id,
                "execution_run_id": result.run_id,
            },
            sort_keys=True,
        )
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
