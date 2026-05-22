"""
Capital Allocator with RL Meta-Learner
=======================================
Allocates portfolio capital between intraday (MIS) and delivery (CNC) trading.
RL meta-model learns optimal split ratio based on observed performance.
Initial split: 20% intraday / 80% delivery.
"""

import logging
import json
import os
from typing import Tuple, Dict, Optional
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, CapitalAllocatorState

logger = logging.getLogger(__name__)


class CapitalAllocator:
    """
    RL-based meta-learner for intraday:delivery capital split.
    Learns which allocation ratio works best over time.
    """

    def __init__(self, initial_intraday_ratio: float = 0.20):
        """
        Args:
            initial_intraday_ratio: Initial split (0.20 = 20% intraday, 80% delivery)
        """
        self.initial_intraday_ratio = initial_intraday_ratio

        # Current allocation (before RL adjusts)
        self.current_intraday_ratio = initial_intraday_ratio

        # RL model state (minimal)
        self.rl_enabled = False
        self.rl_model = None
        self.performance_history = []  # List of (week, intraday_pnl, delivery_pnl, ratio_used)
        self.db_url = os.getenv("POSTGRES_URL") or "sqlite:///aegisquant_live.db"
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self._load_state()

        logger.info(
            f"[CapitalAllocator] Initialized with "
            f"{initial_intraday_ratio*100:.0f}% intraday / "
            f"{(1-initial_intraday_ratio)*100:.0f}% delivery"
        )

    def _load_state(self) -> None:
        """Restore allocator state from the most recent persisted row."""
        session = self.Session()
        try:
            row = session.query(CapitalAllocatorState).order_by(CapitalAllocatorState.updated_at.desc()).first()
            if not row:
                return
            self.current_intraday_ratio = float(row.current_intraday_ratio or self.initial_intraday_ratio)
            self.rl_enabled = bool(row.rl_enabled)
            try:
                self.performance_history = json.loads(row.performance_history or "[]")
            except Exception:
                self.performance_history = []
        except Exception as exc:
            logger.warning(f"[CapitalAllocator] State restore failed: {exc}")
        finally:
            session.close()

    def persist_state(self, as_of: Optional[str] = None) -> None:
        """Persist current intraday/delivery split and history."""
        as_of = as_of or datetime.utcnow().strftime("%Y-%m-%d")
        session = self.Session()
        try:
            row = session.query(CapitalAllocatorState).filter(CapitalAllocatorState.as_of == as_of).first()
            if row is None:
                row = CapitalAllocatorState(as_of=as_of)
                session.add(row)
            row.current_intraday_ratio = float(self.current_intraday_ratio)
            row.rl_enabled = 1 if self.rl_enabled else 0
            row.weeks_of_data = int(len(self.performance_history))
            row.performance_history = json.dumps(self.performance_history, default=str)
            row.updated_at = datetime.utcnow().isoformat()
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.warning(f"[CapitalAllocator] State persist failed: {exc}")
        finally:
            session.close()

    def get_budgets(self, portfolio_state: Optional[Dict] = None) -> Tuple[float, float]:
        """
        Compute intraday_budget and delivery_budget.

        Args:
            portfolio_state: Current portfolio state (drawdown, portfolio_value, etc.)
                            If provided, adjusts budgets based on risk state

        Returns:
            (intraday_budget, delivery_budget) in rupees
        """
        # Start with current ratio
        ratio = self.current_intraday_ratio

        # Adjust for drawdown if provided
        if portfolio_state and "current_drawdown" in portfolio_state:
            drawdown = portfolio_state["current_drawdown"]
            if drawdown > 0.15:  # > 15% drawdown
                # Cut intraday allocation by 50%
                ratio = ratio * 0.5
                logger.warning(f"[CapitalAllocator] Drawdown {drawdown*100:.1f}% — reducing intraday ratio to {ratio*100:.0f}%")
            elif drawdown > 0.08:  # > 8% drawdown
                # Cut by 25%
                ratio = ratio * 0.75

        # Cap intraday at 50% max
        ratio = min(ratio, 0.50)

        capital = portfolio_state.get("portfolio_value", 250000.0) if portfolio_state else 250000.0

        intraday_budget = capital * ratio
        delivery_budget = capital * (1 - ratio)

        logger.debug(f"[CapitalAllocator] Budgets: {intraday_budget:.0f} intraday, {delivery_budget:.0f} delivery")
        return (intraday_budget, delivery_budget)

    def update_rl_model(self, weekly_intraday_pnl: float, weekly_delivery_pnl: float) -> None:
        """
        Weekly training step: update RL model based on performance.

        Args:
            weekly_intraday_pnl: P&L from intraday trades this week
            weekly_delivery_pnl: P&L from delivery trades this week
        """
        # Log performance
        self.performance_history.append({
            "week": datetime.now().isoformat(),
            "intraday_pnl": weekly_intraday_pnl,
            "delivery_pnl": weekly_delivery_pnl,
            "ratio_used": self.current_intraday_ratio,
        })

        # Need at least 4 weeks of data to train RL
        if len(self.performance_history) < 4:
            logger.info(f"[CapitalAllocator] Week {len(self.performance_history)}/4 — accumulating data before RL training")
            return

        # Simple RL update: if intraday has been consistently negative, reduce ratio
        # If delivery has been consistently positive and intraday negative, skew more to delivery
        recent_weeks = self.performance_history[-4:]
        intraday_sharpe = np.mean([w["intraday_pnl"] for w in recent_weeks])
        delivery_sharpe = np.mean([w["delivery_pnl"] for w in recent_weeks])

        if intraday_sharpe < 0 and delivery_sharpe > 0:
            # Intraday underperforming, shift toward delivery
            new_ratio = self.current_intraday_ratio * 0.9  # Reduce by 10%
            logger.info(
                f"[CapitalAllocator] Intraday Sharpe {intraday_sharpe:.2f}, "
                f"Delivery Sharpe {delivery_sharpe:.2f} — adjusting ratio from "
                f"{self.current_intraday_ratio*100:.0f}% to {new_ratio*100:.0f}%"
            )
            self.current_intraday_ratio = max(new_ratio, 0.05)  # Floor at 5%
        elif intraday_sharpe > 0 and delivery_sharpe < 0:
            # Intraday outperforming, shift toward intraday
            new_ratio = min(self.current_intraday_ratio * 1.1, 0.40)  # Increase 10%, cap at 40%
            logger.info(
                f"[CapitalAllocator] Intraday Sharpe {intraday_sharpe:.2f}, "
                f"Delivery Sharpe {delivery_sharpe:.2f} — adjusting ratio from "
                f"{self.current_intraday_ratio*100:.0f}% to {new_ratio*100:.0f}%"
            )
            self.current_intraday_ratio = new_ratio
        else:
            logger.info(
                f"[CapitalAllocator] Both performing well (Intraday: {intraday_sharpe:.2f}, "
                f"Delivery: {delivery_sharpe:.2f}) — holding ratio at {self.current_intraday_ratio*100:.0f}%"
            )

        self.rl_enabled = True
        self.persist_state()

    def to_dict(self) -> dict:
        """Serialize state for logging."""
        return {
            "current_intraday_ratio": self.current_intraday_ratio,
            "rl_enabled": self.rl_enabled,
            "weeks_of_data": len(self.performance_history),
        }


# Module-level singleton
capital_allocator = CapitalAllocator()
