"""
Paper Portfolio - Mark-to-Market Simulator
==========================================
In paper mode the real broker (Groww) never places orders, so `open_positions`
and `daily_pnl` stay empty and the dashboard shows no movement.

This module closes that gap:
  1. `simulate_fills` - turns each run's target weights into simulated OPEN
     positions in `open_positions` (idempotent - re-runs don't double-fill).
  2. `mark_to_market` - recomputes `pnl_pct` on every OPEN row against the
     current yfinance quote.
  3. `write_daily_pnl` - upserts a `daily_pnl` row keyed on today's IST date.

Reused so the wheel isn't reinvented:
  - `src.engine.position_manager.Position` dataclass + factory methods
  - `src.engine.position_manager.position_manager.open_position` (ORM upsert)
  - `src.db.models` ORM schema
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, DailyPnL, OpenPosition, PaperFill, PaperOrder
from src.engine.position_manager import (
    Position,
    PositionManager,
    position_manager as shared_position_manager,
)


IST = ZoneInfo("Asia/Kolkata")


@dataclass
class PortfolioSnapshot:
    portfolio_value: float
    base_capital: float
    unrealized_pnl: float
    today_pnl: float
    open_count: int
    winners: List[dict]   # [{ticker, pnl_pct, pnl_inr}]
    losers: List[dict]
    drawdown: float


class PaperPortfolio:
    """Paper-mode MTM tracker. Writes to open_positions + daily_pnl."""

    def __init__(
        self,
        base_capital: float = 250_000.0,
        db_url: Optional[str] = None,
        position_manager: Optional[PositionManager] = None,
    ):
        self.base_capital = float(base_capital)
        if position_manager is not None:
            self.position_manager = position_manager
            self.db_url = position_manager.db_url
        elif db_url is None:
            self.position_manager = shared_position_manager
            self.db_url = shared_position_manager.db_url
        else:
            self.position_manager = PositionManager(db_url)
            self.db_url = db_url

        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    # Core operations

    def mark_to_market(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Walk every OPEN row, update pnl_pct vs. current price.
        Returns {ticker: unrealized_pnl_inr}.
        """
        out: Dict[str, float] = {}
        session = self.Session()
        try:
            rows = session.query(OpenPosition).filter(OpenPosition.status == "OPEN").all()
            now_iso = datetime.utcnow().isoformat()
            for row in rows:
                cur = prices.get(row.ticker)
                if not cur or cur <= 0 or not row.entry_price:
                    continue
                sign = self._sign(row.quantity)
                pnl_pct = sign * (cur - row.entry_price) / row.entry_price
                pnl_inr = self._pnl_inr(row.entry_price, row.quantity, pnl_pct)
                row.pnl_pct = float(pnl_pct)
                row.updated_at = now_iso
                out[row.ticker] = float(pnl_inr)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"[PaperPortfolio] mark_to_market failed: {e}")
        finally:
            session.close()
        return out

    def simulate_fills(
        self,
        tickers: List[str],
        weights: np.ndarray,
        prices: Dict[str, float],
        trade_types: Dict[str, str],
        equity: Optional[float] = None,
    ) -> int:
        """
        Open simulated positions for every ticker with |w| >= 1e-3 and
        trade_type != SKIP. Idempotent: if a row already exists OPEN with the
        same signed direction, leave it alone. If the sign flipped, close-and-
        reopen (close via position_manager, open new row).

        Returns count of new rows opened.
        """
        equity = float(equity) if equity is not None else self.base_capital
        opened = 0

        # Build map of currently open positions (signed by quantity)
        session = self.Session()
        try:
            existing = {
                r.ticker: r for r in session.query(OpenPosition)
                .filter(OpenPosition.status == "OPEN").all()
            }
        finally:
            session.close()

        for i, ticker in enumerate(tickers):
            w = float(weights[i])
            if abs(w) < 1e-3:
                continue
            if trade_types.get(ticker, "SKIP") == "SKIP":
                continue
            price = prices.get(ticker)
            if not price or price <= 0:
                continue

            target_notional = abs(w) * equity
            qty = int(target_notional / price)
            if qty <= 0:
                continue
            signed_qty = qty if w > 0 else -qty

            prior = existing.get(ticker)
            if prior is not None:
                prior_sign = 1 if prior.quantity >= 0 else -1
                new_sign = 1 if signed_qty >= 0 else -1
                if prior_sign == new_sign:
                    # Same direction: keep the original entry price so MTM math
                    # reflects true unrealized P&L across runs.
                    continue
                # Direction flip: close old, open new.
                self.position_manager.close_position(ticker, price, reason="FLIP")

            trade_type = trade_types.get(ticker, "CNC")
            factory = Position.default_mis if trade_type == "MIS" else Position.default_cnc
            pos = factory(
                ticker=ticker,
                entry_price=float(price),
                quantity=signed_qty,
                strategy="paper_sim",
            )
            self.position_manager.open_position(pos)
            opened += 1

        if opened:
            print(f"[PaperPortfolio] Simulated {opened} new fills")
        return opened

    def execute_target_weights(
        self,
        run_id: str,
        tickers: List[str],
        weights: np.ndarray,
        prices: Dict[str, float],
        trade_types: Dict[str, str],
        equity: Optional[float] = None,
        strategies: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """
        Broker-like paper execution. Creates `paper_orders` and `paper_fills`,
        updates `open_positions`, and returns ticker -> simulated fill price.

        Rejections are explicit rows, not silent skips, so the terminal can show
        why a candidate did not become a trade.
        """
        equity = float(equity) if equity is not None else self.base_capital
        strategies = strategies or {}
        fills: Dict[str, float] = {}
        opened_weights = np.zeros(len(tickers), dtype=float)
        accepted_trade_types: Dict[str, str] = {}

        for i, ticker in enumerate(tickers):
            w = float(weights[i])
            trade_type = trade_types.get(ticker, "SKIP")
            price = float(prices.get(ticker) or 0.0)

            if abs(w) < 1e-3 or trade_type == "SKIP":
                continue

            target_notional = abs(w) * equity
            qty = int(target_notional / price) if price > 0 else 0
            side = "BUY" if w > 0 else "SELL"
            status = "PLACED"
            rejection_reason = ""

            if price <= 0:
                status = "REJECTED"
                rejection_reason = "Missing or invalid reference price."
            elif qty <= 0:
                status = "REJECTED"
                rejection_reason = "Target notional is too small for one share."

            order_id = self._order_id(run_id, ticker, i)
            self._record_order(
                run_id=run_id,
                order_id=order_id,
                ticker=ticker,
                side=side,
                product_type=trade_type,
                quantity=max(0, qty),
                target_weight=w,
                notional=target_notional,
                status=status,
                rejection_reason=rejection_reason,
                strategy=strategies.get(ticker, ""),
            )

            if status == "REJECTED":
                continue

            fill_price, slippage_bps = self._simulated_fill_price(price, side, trade_type)
            fees, fee_breakdown = self._india_fee_breakdown(fill_price, qty, side, trade_type)
            self._record_fill(
                run_id=run_id,
                order_id=order_id,
                ticker=ticker,
                side=side,
                product_type=trade_type,
                quantity=qty,
                price=fill_price,
                slippage_bps=slippage_bps,
                fees=fees,
                fee_breakdown=fee_breakdown,
            )
            fills[ticker] = fill_price
            opened_weights[i] = w
            accepted_trade_types[ticker] = trade_type

        if np.any(np.abs(opened_weights) >= 1e-3):
            self.simulate_fills(
                tickers=tickers,
                weights=opened_weights,
                prices={**prices, **fills},
                trade_types=accepted_trade_types,
                equity=equity,
            )
        return fills

    def write_daily_pnl(self, cb_reason: str, intraday_ratio: float) -> None:
        """
        Upsert today's daily_pnl row. Totals split by trade_type.
        Portfolio value = base_capital + sum of unrealized P&L across OPEN rows.
        Drawdown computed vs. running peak.
        """
        today = datetime.now(IST).strftime("%Y-%m-%d")
        session = self.Session()
        try:
            opens = session.query(OpenPosition).filter(OpenPosition.status == "OPEN").all()
            intraday = sum(
                self._pnl_inr(o.entry_price, o.quantity, o.pnl_pct or 0.0)
                for o in opens if o.trade_type == "MIS"
            )
            delivery = sum(
                self._pnl_inr(o.entry_price, o.quantity, o.pnl_pct or 0.0)
                for o in opens if o.trade_type == "CNC"
            )
            total_unrealized = intraday + delivery
            portfolio_value = self.base_capital + total_unrealized

            # Peak equity across history (for drawdown)
            prior_peak = session.query(DailyPnL).order_by(
                DailyPnL.total_portfolio_value.desc()
            ).first()
            peak = max(
                portfolio_value,
                (prior_peak.total_portfolio_value if prior_peak else portfolio_value),
            )
            drawdown = max(0.0, (peak - portfolio_value) / peak) if peak > 0 else 0.0

            existing = session.query(DailyPnL).filter(DailyPnL.date == today).first()
            if existing:
                existing.total_portfolio_value = float(portfolio_value)
                existing.intraday_pnl = float(intraday)
                existing.delivery_pnl = float(delivery)
                existing.total_pnl = float(total_unrealized)
                existing.drawdown = float(drawdown)
                existing.intraday_ratio_used = float(intraday_ratio)
            else:
                session.add(DailyPnL(
                    date=today,
                    total_portfolio_value=float(portfolio_value),
                    intraday_pnl=float(intraday),
                    delivery_pnl=float(delivery),
                    total_pnl=float(total_unrealized),
                    drawdown=float(drawdown),
                    intraday_ratio_used=float(intraday_ratio),
                ))
            session.commit()
            print(
                f"[PaperPortfolio] daily_pnl {today}: value=Rs {portfolio_value:,.0f} "
                f"pnl=Rs {total_unrealized:,.0f} dd={drawdown*100:.2f}% cb={cb_reason}"
            )
        except Exception as e:
            session.rollback()
            print(f"[PaperPortfolio] write_daily_pnl failed: {e}")
        finally:
            session.close()

    def auto_close_intraday(
        self,
        run_id: str,
        prices: Dict[str, float],
        reason: str = "MIS_AUTO_CLOSE",
    ) -> int:
        """Close all open MIS paper positions at supplied prices."""
        session = self.Session()
        closed = 0
        try:
            rows = session.query(OpenPosition).filter(
                OpenPosition.status == "OPEN",
                OpenPosition.trade_type == "MIS",
            ).all()
        finally:
            session.close()

        for row in rows:
            price = float(prices.get(row.ticker) or 0.0)
            if price <= 0:
                continue
            side = "SELL" if row.quantity >= 0 else "BUY"
            qty = abs(int(row.quantity or 0))
            order_id = self._order_id(run_id, row.ticker, closed, suffix="AUTO")
            self._record_order(
                run_id=run_id,
                order_id=order_id,
                ticker=row.ticker,
                side=side,
                product_type="MIS",
                quantity=qty,
                target_weight=0.0,
                notional=qty * price,
                status="PLACED",
                rejection_reason="",
                strategy=reason,
            )
            fill_price, slippage_bps = self._simulated_fill_price(price, side, "MIS")
            fees, fee_breakdown = self._india_fee_breakdown(fill_price, qty, side, "MIS")
            self._record_fill(
                run_id=run_id,
                order_id=order_id,
                ticker=row.ticker,
                side=side,
                product_type="MIS",
                quantity=qty,
                price=fill_price,
                slippage_bps=slippage_bps,
                fees=fees,
                fee_breakdown=fee_breakdown,
            )
            self.position_manager.close_position(row.ticker, fill_price, reason=reason)
            closed += 1
        return closed

    # Reporting

    def snapshot(self, prices: Optional[Dict[str, float]] = None) -> PortfolioSnapshot:
        """Quick summary for the Slack digest / weekly review."""
        session = self.Session()
        try:
            opens = session.query(OpenPosition).filter(OpenPosition.status == "OPEN").all()
            movers = []
            for o in opens:
                pnl_pct = float(o.pnl_pct or 0.0)
                pnl_inr = self._pnl_inr(o.entry_price or 0.0, o.quantity or 0, pnl_pct)
                movers.append({
                    "ticker": o.ticker,
                    "pnl_pct": pnl_pct,
                    "pnl_inr": pnl_inr,
                    "trade_type": o.trade_type,
                })
            unrealized = sum(m["pnl_inr"] for m in movers)
            portfolio_value = self.base_capital + unrealized

            today = datetime.now(IST).strftime("%Y-%m-%d")
            today_row = session.query(DailyPnL).filter(DailyPnL.date == today).first()
            today_pnl = float(today_row.total_pnl) if today_row else unrealized
            drawdown = float(today_row.drawdown) if today_row else 0.0
        finally:
            session.close()

        movers.sort(key=lambda m: m["pnl_inr"], reverse=True)
        winners = [m for m in movers if m["pnl_inr"] > 0][:3]
        losers = [m for m in reversed(movers) if m["pnl_inr"] < 0][:3]

        return PortfolioSnapshot(
            portfolio_value=portfolio_value,
            base_capital=self.base_capital,
            unrealized_pnl=unrealized,
            today_pnl=today_pnl,
            open_count=len(movers),
            winners=winners,
            losers=losers,
            drawdown=drawdown,
        )

    def current_weights(
        self,
        tickers: List[str],
        prices: Optional[Dict[str, float]] = None,
        portfolio_value: Optional[float] = None,
    ) -> np.ndarray:
        """Return signed current weights for the requested ticker order."""
        prices = prices or {}
        session = self.Session()
        weights = np.zeros(len(tickers), dtype=float)
        try:
            opens = session.query(OpenPosition).filter(OpenPosition.status == "OPEN").all()
            price_by_ticker = {
                o.ticker: float(prices.get(o.ticker) or o.entry_price or 0.0)
                for o in opens
            }
            if portfolio_value is None:
                unrealized = sum(
                    self._pnl_inr(o.entry_price or 0.0, o.quantity or 0, o.pnl_pct or 0.0)
                    for o in opens
                )
                portfolio_value = self.base_capital + unrealized

            denom = float(portfolio_value or self.base_capital or 1.0)
            if denom <= 0:
                denom = 1.0

            index = {ticker: i for i, ticker in enumerate(tickers)}
            for o in opens:
                if o.ticker not in index:
                    continue
                weights[index[o.ticker]] = (o.quantity or 0) * price_by_ticker[o.ticker] / denom
        finally:
            session.close()
        return weights

    @staticmethod
    def _sign(quantity: int) -> int:
        return 1 if quantity >= 0 else -1

    @staticmethod
    def _pnl_inr(entry_price: float, quantity: int, pnl_pct: float) -> float:
        return float(pnl_pct) * float(entry_price) * abs(int(quantity))

    @staticmethod
    def _order_id(run_id: str, ticker: str, idx: int, suffix: str = "ORD") -> str:
        clean = ticker.replace(".", "_").replace("-", "_")
        return f"{run_id}_{suffix}_{idx}_{clean}"[:120]

    @staticmethod
    def _simulated_fill_price(price: float, side: str, trade_type: str) -> tuple[float, float]:
        slippage_bps = 3.0 if trade_type == "MIS" else 5.0
        sign = 1 if side == "BUY" else -1
        fill = float(price) * (1 + sign * slippage_bps / 10_000)
        return round(fill, 4), slippage_bps

    @staticmethod
    def _india_fee_breakdown(price: float, quantity: int, side: str, trade_type: str) -> tuple[float, dict]:
        """
        Conservative paper-trading Indian equity cost approximation.
        Rates are placeholders for simulation, not broker/legal advice.
        """
        turnover = abs(float(price) * int(quantity))
        brokerage = min(20.0, turnover * 0.0003) if turnover else 0.0
        stt_rate = 0.00025 if trade_type == "MIS" else 0.001
        stt = turnover * stt_rate if side == "SELL" or trade_type == "CNC" else 0.0
        exchange_txn = turnover * 0.0000345
        sebi = turnover * 0.000001
        stamp = turnover * (0.00003 if trade_type == "MIS" else 0.00015) if side == "BUY" else 0.0
        gst = 0.18 * (brokerage + exchange_txn)
        total = brokerage + stt + exchange_txn + sebi + stamp + gst
        breakdown = {
            "brokerage": round(brokerage, 4),
            "stt": round(stt, 4),
            "exchange_txn": round(exchange_txn, 4),
            "sebi": round(sebi, 4),
            "stamp": round(stamp, 4),
            "gst": round(gst, 4),
            "total": round(total, 4),
            "turnover": round(turnover, 4),
        }
        return round(total, 4), breakdown

    def _record_order(
        self,
        run_id: str,
        order_id: str,
        ticker: str,
        side: str,
        product_type: str,
        quantity: int,
        target_weight: float,
        notional: float,
        status: str,
        rejection_reason: str,
        strategy: str,
    ) -> None:
        session = self.Session()
        try:
            row = session.query(PaperOrder).filter(PaperOrder.order_id == order_id).first()
            if row is None:
                row = PaperOrder(order_id=order_id, run_id=run_id)
                session.add(row)
            row.timestamp = datetime.utcnow().isoformat()
            row.ticker = ticker
            row.side = side
            row.product_type = product_type
            row.order_type = "MARKET"
            row.quantity = int(quantity)
            row.target_weight = float(target_weight)
            row.notional = float(notional)
            row.status = status
            row.rejection_reason = rejection_reason
            row.strategy = strategy
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"[PaperPortfolio] record_order failed: {e}")
        finally:
            session.close()

    def _record_fill(
        self,
        run_id: str,
        order_id: str,
        ticker: str,
        side: str,
        product_type: str,
        quantity: int,
        price: float,
        slippage_bps: float,
        fees: float,
        fee_breakdown: dict,
    ) -> None:
        session = self.Session()
        try:
            fill_id = f"{order_id}_FILL"
            row = session.query(PaperFill).filter(PaperFill.fill_id == fill_id).first()
            if row is None:
                row = PaperFill(fill_id=fill_id, order_id=order_id, run_id=run_id)
                session.add(row)
            row.timestamp = datetime.utcnow().isoformat()
            row.ticker = ticker
            row.side = side
            row.product_type = product_type
            row.quantity = int(quantity)
            row.price = float(price)
            row.slippage_bps = float(slippage_bps)
            row.fees = float(fees)
            row.fee_breakdown = json.dumps(fee_breakdown)
            row.status = "FILLED"
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"[PaperPortfolio] record_fill failed: {e}")
        finally:
            session.close()
