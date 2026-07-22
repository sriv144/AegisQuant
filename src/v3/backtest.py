"""Event-driven, self-financing accounting for v3 research.

Signals are decisions, not fills: every target executes on the first available
session strictly after its signal timestamp.  Shares then drift naturally until
another event changes them; no target-weight return smoothing is performed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from src.execution.v3.ids import build_target_hash

from .portfolio import PortfolioPlan


class BacktestDataError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Trade:
    execution_date: pd.Timestamp
    signal_date: pd.Timestamp
    symbol: str
    side: str
    quantity: float
    price: float
    notional: float
    cost: float


@dataclass(frozen=True, slots=True)
class DailyAccount:
    date: pd.Timestamp
    cash: float
    market_value: float
    nav: float
    positions: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class EventBacktestResult:
    nav: pd.Series
    daily_returns: pd.Series
    accounts: tuple[DailyAccount, ...]
    trades: tuple[Trade, ...]
    ending_positions: Mapping[str, float]
    total_transaction_cost: float
    total_dividends: float
    total_delisting_proceeds: float
    one_way_turnover: float
    annualized_one_way_turnover: float
    executed_signal_dates: tuple[pd.Timestamp, ...]
    executed_target_hashes: tuple[tuple[pd.Timestamp, str], ...]


class EventDrivenBacktester:
    def __init__(
        self,
        *,
        initial_cash: float = 100_000.0,
        transaction_cost_bps: float = 5.0,
        fractional_shares: bool = True,
    ) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
        self.initial_cash = float(initial_cash)
        self.transaction_cost_rate = float(transaction_cost_bps) / 10_000.0
        self.fractional_shares = bool(fractional_shares)

    def run(
        self,
        execution_prices: pd.DataFrame,
        targets_by_signal_date: Mapping[pd.Timestamp, PortfolioPlan | Mapping[str, float]],
        *,
        dividends: pd.DataFrame | None = None,
        splits: pd.DataFrame | None = None,
        delistings: pd.DataFrame | None = None,
        delisting_returns: pd.DataFrame | None = None,
        symbol_changes: pd.DataFrame | None = None,
        fractionable: Mapping[str, bool] | None = None,
    ) -> EventBacktestResult:
        prices = self._validate_prices(execution_prices)
        dividend_frame = self._action_frame(dividends, prices, default=0.0)
        split_frame = self._action_frame(splits, prices, default=1.0)
        delisting_frame = self._action_frame(delistings, prices, default=np.nan)
        delisting_return_frame = self._action_frame(delisting_returns, prices, default=np.nan)
        symbol_change_schedule = self._symbol_change_schedule(symbol_changes, prices.index)
        execution_schedule = self._execution_schedule(prices.index, targets_by_signal_date)

        cash = self.initial_cash
        positions: dict[str, float] = {}
        trades: list[Trade] = []
        accounts: list[DailyAccount] = []
        executed_signals: list[pd.Timestamp] = []
        executed_target_hashes: list[tuple[pd.Timestamp, str]] = []
        total_cost = 0.0
        total_dividends = 0.0
        total_delisting_proceeds = 0.0
        turnover_sum = 0.0
        prior_marks: dict[str, float] = {}
        symbol_aliases: dict[str, str] = {}
        inactive_symbols: set[str] = set()
        fractionable_by_symbol = {str(k): bool(v) for k, v in (fractionable or {}).items()}

        for date in prices.index:
            row = prices.loc[date]
            for old_symbol, new_symbol, ratio in symbol_change_schedule.get(date, ()):
                resolved_new = self._resolve_symbol(new_symbol, symbol_aliases)
                if resolved_new in inactive_symbols:
                    raise BacktestDataError(f"symbol change maps to inactive symbol {resolved_new}")
                symbol_aliases[old_symbol] = resolved_new
                if old_symbol not in positions:
                    continue
                positions[resolved_new] = positions.get(resolved_new, 0.0) + positions.pop(old_symbol) * ratio
                if old_symbol in prior_marks:
                    prior_marks[resolved_new] = prior_marks.pop(old_symbol) / ratio
            for symbol in list(positions):
                ratio = float(split_frame.at[date, symbol]) if symbol in split_frame.columns else 1.0
                if not math.isfinite(ratio) or ratio <= 0:
                    raise BacktestDataError(f"invalid split ratio for {symbol} on {date.date()}")
                if not math.isclose(ratio, 1.0):
                    positions[symbol] *= ratio
                    if symbol in prior_marks:
                        prior_marks[symbol] /= ratio

                dividend = float(dividend_frame.at[date, symbol]) if symbol in dividend_frame.columns else 0.0
                if not math.isfinite(dividend) or dividend < 0:
                    raise BacktestDataError(f"invalid dividend for {symbol} on {date.date()}")
                dividend_cash = positions[symbol] * dividend
                cash += dividend_cash
                total_dividends += dividend_cash

                recovery = (
                    float(delisting_frame.at[date, symbol])
                    if symbol in delisting_frame.columns and pd.notna(delisting_frame.at[date, symbol])
                    else math.nan
                )
                delisting_return = (
                    float(delisting_return_frame.at[date, symbol])
                    if symbol in delisting_return_frame.columns
                    and pd.notna(delisting_return_frame.at[date, symbol])
                    else math.nan
                )
                if math.isfinite(recovery) and math.isfinite(delisting_return):
                    raise BacktestDataError(
                        f"both recovery price and delisting return supplied for {symbol} on {date.date()}"
                    )
                if math.isfinite(recovery):
                    if recovery < 0:
                        raise BacktestDataError(f"negative delisting recovery for {symbol} on {date.date()}")
                    proceeds = positions.pop(symbol) * recovery
                    cash += proceeds
                    total_delisting_proceeds += proceeds
                    inactive_symbols.add(symbol)
                    prior_marks.pop(symbol, None)
                elif math.isfinite(delisting_return):
                    if delisting_return < -1.0:
                        raise BacktestDataError(f"delisting return below -100% for {symbol} on {date.date()}")
                    if symbol not in prior_marks:
                        raise BacktestDataError(f"no prior mark for delisting return on {symbol} at {date.date()}")
                    proceeds = positions.pop(symbol) * prior_marks[symbol] * (1.0 + delisting_return)
                    cash += proceeds
                    total_delisting_proceeds += proceeds
                    inactive_symbols.add(symbol)
                    prior_marks.pop(symbol, None)

            for symbol in delisting_frame.columns:
                has_recovery = pd.notna(delisting_frame.at[date, symbol])
                has_return = (
                    symbol in delisting_return_frame.columns
                    and pd.notna(delisting_return_frame.at[date, symbol])
                )
                if has_recovery or has_return:
                    inactive_symbols.add(symbol)

            if date in execution_schedule:
                signal_date, target = execution_schedule[date]
                source_target_weights = self._target_weights(target)
                source_target_hash = build_target_hash(source_target_weights)
                if isinstance(target, PortfolioPlan) and target.weight_sha256 != source_target_hash:
                    raise BacktestDataError("PortfolioPlan weight hash does not match its target weights")
                target_weights = self._normalize_target_symbols(
                    source_target_weights,
                    aliases=symbol_aliases,
                    inactive=inactive_symbols,
                )
                self._require_prices(row, set(positions).union(target_weights))
                pre_trade_nav = cash + sum(positions[symbol] * float(row[symbol]) for symbol in positions)
                if pre_trade_nav <= 0:
                    raise BacktestDataError("portfolio NAV became non-positive")
                desired = {
                    symbol: self._round_quantity(
                        symbol,
                        pre_trade_nav * weight / float(row[symbol]),
                        fractionable_by_symbol,
                    )
                    for symbol, weight in target_weights.items()
                }
                for symbol in positions:
                    desired.setdefault(symbol, 0.0)

                # Sells are terminal before buys, matching the paper execution
                # coordinator's required cash and failure ordering.
                sell_notional = 0.0
                for symbol in sorted(desired):
                    delta = desired[symbol] - positions.get(symbol, 0.0)
                    if delta >= -1e-12:
                        continue
                    quantity = -delta
                    price = float(row[symbol])
                    notional = quantity * price
                    cost = notional * self.transaction_cost_rate
                    cash += notional - cost
                    positions[symbol] = desired[symbol]
                    if positions[symbol] <= 1e-12:
                        positions.pop(symbol, None)
                    trades.append(Trade(date, signal_date, symbol, "sell", quantity, price, notional, cost))
                    total_cost += cost
                    sell_notional += notional

                buys: list[tuple[str, float, float]] = []
                for symbol in sorted(desired):
                    delta = desired[symbol] - positions.get(symbol, 0.0)
                    if delta > 1e-12:
                        buys.append((symbol, delta, float(row[symbol])))
                required_cash = sum(quantity * price * (1.0 + self.transaction_cost_rate) for _, quantity, price in buys)
                scale = 1.0 if required_cash <= cash + 1e-9 else max(0.0, cash / required_cash)
                buy_notional = 0.0
                for symbol, desired_quantity, price in buys:
                    quantity = self._round_quantity(
                        symbol,
                        desired_quantity * scale,
                        fractionable_by_symbol,
                    )
                    if quantity <= 1e-12:
                        continue
                    notional = quantity * price
                    cost = notional * self.transaction_cost_rate
                    cash -= notional + cost
                    if cash < -1e-7:
                        raise RuntimeError("buy scaling produced negative cash")
                    positions[symbol] = positions.get(symbol, 0.0) + quantity
                    trades.append(Trade(date, signal_date, symbol, "buy", quantity, price, notional, cost))
                    total_cost += cost
                    buy_notional += notional
                cash = max(0.0, cash)
                # One-way turnover includes the cash leg: entry/liquidation is
                # fully counted, while a same-size rotation is not double-counted.
                turnover_sum += max(buy_notional, sell_notional) / pre_trade_nav
                executed_signals.append(signal_date)
                executed_target_hashes.append((signal_date, source_target_hash))

            self._require_prices(row, set(positions))
            market_value = sum(quantity * float(row[symbol]) for symbol, quantity in positions.items())
            nav = cash + market_value
            for symbol in positions:
                prior_marks[symbol] = float(row[symbol])
            accounts.append(
                DailyAccount(
                    date=date,
                    cash=float(cash),
                    market_value=float(market_value),
                    nav=float(nav),
                    positions=tuple(sorted((symbol, float(quantity)) for symbol, quantity in positions.items())),
                )
            )

        nav = pd.Series({account.date: account.nav for account in accounts}, name="nav", dtype=float)
        returns = nav.pct_change(fill_method=None).fillna(0.0).rename("portfolio_return")
        elapsed_sessions = max(1, len(nav) - 1)
        return EventBacktestResult(
            nav=nav,
            daily_returns=returns,
            accounts=tuple(accounts),
            trades=tuple(trades),
            ending_positions=MappingProxyType(dict(sorted(positions.items()))),
            total_transaction_cost=float(total_cost),
            total_dividends=float(total_dividends),
            total_delisting_proceeds=float(total_delisting_proceeds),
            one_way_turnover=float(turnover_sum),
            annualized_one_way_turnover=float(turnover_sum * 252 / elapsed_sessions),
            executed_signal_dates=tuple(executed_signals),
            executed_target_hashes=tuple(executed_target_hashes),
        )

    @staticmethod
    def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty or not isinstance(prices.index, pd.DatetimeIndex):
            raise BacktestDataError("execution prices require a non-empty DatetimeIndex")
        if prices.index.has_duplicates:
            raise BacktestDataError("execution prices contain duplicate sessions")
        frame = prices.sort_index().copy(deep=True)
        frame.columns = [str(column) for column in frame.columns]
        if frame.columns.duplicated().any():
            raise BacktestDataError("execution prices contain duplicate symbols")
        return frame

    @staticmethod
    def _action_frame(
        action: pd.DataFrame | None,
        prices: pd.DataFrame,
        *,
        default: float,
    ) -> pd.DataFrame:
        if action is None:
            return pd.DataFrame(default, index=prices.index, columns=prices.columns, dtype=float)
        frame = action.copy(deep=True).reindex(index=prices.index, columns=prices.columns)
        return frame.fillna(default) if math.isfinite(default) else frame

    @staticmethod
    def _execution_schedule(
        sessions: pd.DatetimeIndex,
        targets: Mapping[pd.Timestamp, PortfolioPlan | Mapping[str, float]],
    ) -> dict[pd.Timestamp, tuple[pd.Timestamp, PortfolioPlan | Mapping[str, float]]]:
        schedule: dict[pd.Timestamp, tuple[pd.Timestamp, PortfolioPlan | Mapping[str, float]]] = {}
        for raw_signal, target in sorted(targets.items(), key=lambda item: pd.Timestamp(item[0])):
            signal = pd.Timestamp(raw_signal)
            comparable = signal
            if sessions.tz is None and comparable.tzinfo is not None:
                comparable = comparable.tz_convert("America/New_York").normalize().tz_localize(None)
            elif sessions.tz is not None and comparable.tzinfo is None:
                comparable = comparable.tz_localize("America/New_York").tz_convert(sessions.tz)
            elif sessions.tz is not None:
                comparable = comparable.tz_convert(sessions.tz)
            position = sessions.searchsorted(comparable, side="right")
            if position >= len(sessions):
                continue
            execution_date = sessions[position]
            if execution_date in schedule:
                raise BacktestDataError(f"multiple signals map to execution session {execution_date.date()}")
            schedule[execution_date] = (signal, target)
        return schedule

    @staticmethod
    def _symbol_change_schedule(
        symbol_changes: pd.DataFrame | None,
        sessions: pd.DatetimeIndex,
    ) -> dict[pd.Timestamp, tuple[tuple[str, str, float], ...]]:
        if symbol_changes is None or symbol_changes.empty:
            return {}
        required = {"effective_date", "old_symbol", "new_symbol"}
        missing = required.difference(symbol_changes.columns)
        if missing:
            raise BacktestDataError(f"symbol changes are missing columns: {sorted(missing)}")
        schedule: dict[pd.Timestamp, list[tuple[str, str, float]]] = {}
        seen_old: set[tuple[pd.Timestamp, str]] = set()
        for row in symbol_changes.itertuples(index=False):
            effective = pd.Timestamp(getattr(row, "effective_date"))
            if effective not in sessions:
                raise BacktestDataError(f"symbol change date is not an execution session: {effective}")
            old_symbol = str(getattr(row, "old_symbol"))
            new_symbol = str(getattr(row, "new_symbol"))
            ratio_value = getattr(row, "ratio", 1.0)
            ratio = 1.0 if pd.isna(ratio_value) else float(ratio_value)
            if not old_symbol or not new_symbol or old_symbol == new_symbol:
                raise BacktestDataError("symbol changes require distinct non-empty symbols")
            if not math.isfinite(ratio) or ratio <= 0:
                raise BacktestDataError("symbol-change ratio must be positive and finite")
            key = (effective, old_symbol)
            if key in seen_old:
                raise BacktestDataError(f"duplicate symbol change for {old_symbol} on {effective.date()}")
            seen_old.add(key)
            schedule.setdefault(effective, []).append((old_symbol, new_symbol, ratio))
        return {date: tuple(sorted(events)) for date, events in schedule.items()}

    @staticmethod
    def _target_weights(target: PortfolioPlan | Mapping[str, float]) -> dict[str, float]:
        source = target.weights if isinstance(target, PortfolioPlan) else target
        converted = {str(symbol): float(weight) for symbol, weight in source.items()}
        if any(not math.isfinite(weight) or weight < 0 for weight in converted.values()):
            raise BacktestDataError("target weights must be finite and non-negative")
        weights = {symbol: weight for symbol, weight in converted.items() if weight > 0}
        if sum(weights.values()) > 1.0 + 1e-9:
            raise BacktestDataError("target invested weight cannot exceed 100%")
        return dict(sorted(weights.items()))

    @staticmethod
    def _resolve_symbol(symbol: str, aliases: Mapping[str, str]) -> str:
        current = str(symbol)
        seen: set[str] = set()
        while current in aliases:
            if current in seen:
                raise BacktestDataError(f"cyclic symbol-change mapping involving {current}")
            seen.add(current)
            current = aliases[current]
        return current

    @classmethod
    def _normalize_target_symbols(
        cls,
        weights: Mapping[str, float],
        *,
        aliases: Mapping[str, str],
        inactive: set[str],
    ) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for symbol, weight in weights.items():
            active_symbol = cls._resolve_symbol(symbol, aliases)
            if active_symbol in inactive:
                raise BacktestDataError(f"target contains delisted symbol {active_symbol}")
            normalized[active_symbol] = normalized.get(active_symbol, 0.0) + float(weight)
        if sum(normalized.values()) > 1.0 + 1e-9:
            raise BacktestDataError("normalized target invested weight cannot exceed 100%")
        return dict(sorted(normalized.items()))

    @staticmethod
    def _require_prices(row: pd.Series, symbols: set[str]) -> None:
        missing = sorted(
            symbol
            for symbol in symbols
            if symbol not in row.index or not math.isfinite(float(row[symbol])) or float(row[symbol]) <= 0
        )
        if missing:
            raise BacktestDataError(f"missing non-positive execution/mark prices: {missing}")

    def _round_quantity(
        self,
        symbol: str,
        quantity: float,
        fractionable_by_symbol: Mapping[str, bool],
    ) -> float:
        supports_fractional = fractionable_by_symbol.get(symbol, self.fractional_shares)
        if supports_fractional:
            return math.floor(max(0.0, quantity) * 1_000_000) / 1_000_000
        return float(math.floor(max(0.0, quantity)))
