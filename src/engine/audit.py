"""
Paper-trading audit persistence.

This module is the terminal's memory spine: it records what the agents saw,
why they acted, what the paper broker did, and how the RL layer is performing.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import (
    AgentReasoning,
    Base,
    MarketObservation,
    RLModelEvaluation,
)


def _json(value: Any) -> str:
    try:
        return json.dumps(value, default=str)
    except Exception:
        return json.dumps(str(value))


class AuditLogger:
    """Writes the explainability/audit tables used by the Quant Terminal."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("POSTGRES_URL") or "sqlite:///aegisquant_live.db"
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def record_market_observation(
        self,
        run_id: str,
        vix: float,
        universe: Iterable[str],
        data_quality: Dict[str, Any],
        prices: Optional[Dict[str, float]] = None,
        alt_data: Optional[Dict[str, dict]] = None,
    ) -> dict:
        prices = prices or {}
        alt_data = alt_data or {}
        universe = list(universe)
        index_move = 0.0
        notable_news = []

        for ticker, payload in alt_data.items():
            if int(payload.get("news_volume", 0) or 0) > 0 or abs(float(payload.get("sentiment_score", 0.0) or 0.0)) >= 0.35:
                notable_news.append({
                    "ticker": ticker,
                    "sentiment": payload.get("sentiment_score", 0.0),
                    "news_volume": payload.get("news_volume", 0),
                })

        valid_prices = [float(v) for v in prices.values() if v and float(v) > 0]
        sector_breadth = float(len(valid_prices) / max(1, len(universe)))

        row = MarketObservation(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat(),
            vix=float(vix or 0.0),
            index_move=float(index_move),
            sector_breadth=sector_breadth,
            universe_size=len(universe),
            data_quality_status=str(data_quality.get("status", "UNKNOWN")),
            notable_news=_json(notable_news[:25]),
            notes=str(data_quality.get("notes") or "Market observation recorded."),
        )
        session = self.Session()
        try:
            session.add(row)
            session.commit()
            return self._observation_to_dict(row)
        finally:
            session.close()

    def record_agent_reasoning(self, run_id: str, ticker: str, state: Dict[str, Any]) -> int:
        """Persist one reasoning row per research/committee/PM/risk/execution agent."""
        rows: List[AgentReasoning] = []
        timestamp = datetime.utcnow().isoformat()
        committee_direction = (state.get("committee_decision") or {}).get("direction", "NONE")

        for signal in state.get("research_signals", []) or []:
            rows.append(self._reasoning_row(
                run_id,
                timestamp,
                ticker,
                signal.get("agent_name", "Research_Agent"),
                signal.get("action", "UNKNOWN"),
                signal.get("confidence", 0.0),
                signal.get("rationale", ""),
                state,
                committee_direction,
            ))

        for key, agent_name in [
            ("committee_decision", "Strategy_Committee"),
            ("allocation_request", "Portfolio_Manager"),
            ("risk_approval", "Risk_Officer"),
            ("execution_result", "Trade_Execution_Agent"),
        ]:
            decision = state.get(key) or {}
            if not decision:
                continue
            rows.append(self._reasoning_row(
                run_id,
                timestamp,
                ticker,
                decision.get("agent_name", agent_name),
                decision.get("action", "UNKNOWN"),
                decision.get("confidence", 0.0),
                decision.get("rationale", ""),
                state,
                committee_direction,
            ))

        if not rows:
            return 0

        session = self.Session()
        try:
            session.add_all(rows)
            session.commit()
            return len(rows)
        finally:
            session.close()

    def record_rl_evaluation(
        self,
        run_id: str,
        model_version: str,
        target_weights: Iterable[float],
        safe_weights: Iterable[float],
        performance_summary: Dict[str, Any],
        turnover_from: Optional[Iterable[float]] = None,
    ) -> dict:
        target = np.asarray(list(target_weights), dtype=float)
        safe = np.asarray(list(safe_weights), dtype=float)
        prior = np.asarray(list(turnover_from), dtype=float) if turnover_from is not None else np.zeros_like(safe)
        if prior.shape != safe.shape:
            prior = np.zeros_like(safe)

        turnover = float(np.sum(np.abs(safe - prior)))
        excess = float(performance_summary.get("excess_return", 0.0) or 0.0)
        drawdown = float(performance_summary.get("max_drawdown", 0.0) or 0.0)
        sharpe = float(performance_summary.get("rolling_sharpe_30", 0.0) or 0.0)
        reward = float(excess - (0.02 * turnover) - max(0.0, drawdown - 0.05))
        readiness = str(performance_summary.get("readiness_status", "BLOCKED"))
        promotion = 1 if readiness == "LIVE_READY" and reward > 0 else 0

        improvement_note = "RL sizing improved excess return this run." if excess > 0 else ""
        degradation_note = "" if excess >= 0 else "RL sizing lagged benchmark on latest marked performance."

        row = RLModelEvaluation(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat(),
            model_version=model_version,
            benchmark_symbol=str(performance_summary.get("benchmark_symbol", "NIFTYBEES.NS")),
            reward=reward,
            turnover=turnover,
            drawdown=drawdown,
            rolling_sharpe=sharpe,
            benchmark_return=float(performance_summary.get("benchmark_return", 0.0) or 0.0),
            excess_return=excess,
            readiness_status=readiness,
            improvement_note=improvement_note,
            degradation_note=degradation_note,
            promotion_eligible=promotion,
            metadata_json=_json({
                "gross_target": float(np.sum(np.abs(target))) if target.size else 0.0,
                "gross_safe": float(np.sum(np.abs(safe))) if safe.size else 0.0,
                "days_observed": performance_summary.get("days_observed", 0),
                "verdict": performance_summary.get("verdict", "INSUFFICIENT_DATA"),
                "reasons": performance_summary.get("reasons", []),
            }),
        )
        session = self.Session()
        try:
            session.add(row)
            session.commit()
            return self._rl_to_dict(row)
        finally:
            session.close()

    def _reasoning_row(
        self,
        run_id: str,
        timestamp: str,
        ticker: str,
        agent_name: str,
        action: str,
        confidence: Any,
        rationale: str,
        state: Dict[str, Any],
        committee_direction: str,
    ) -> AgentReasoning:
        monitored = self._monitored_signals(state)
        disagreement = self._disagreement(action, committee_direction)
        return AgentReasoning(
            run_id=run_id,
            timestamp=timestamp,
            ticker=ticker,
            agent_name=str(agent_name),
            action=str(action),
            confidence=float(confidence or 0.0),
            rationale=str(rationale or ""),
            source_inputs=_json({
                "market_data": state.get("market_data", {}),
                "technical_indicators": state.get("technical_indicators", {}),
                "alternative_data": state.get("alternative_data", {}),
                "strategy": state.get("current_strategy", ""),
            }),
            monitored_signals=_json(monitored),
            disagreement=disagreement,
        )

    @staticmethod
    def _monitored_signals(state: Dict[str, Any]) -> List[str]:
        technicals = state.get("technical_indicators", {}) or {}
        alt = state.get("alternative_data", {}) or {}
        portfolio = state.get("portfolio_state", {}) or {}
        signals = []
        for key in ["RSI_14", "MACD", "MACD_Signal", "SMA_50", "Daily_Return", "Volatility_20"]:
            if key in technicals:
                signals.append(f"{key}={technicals[key]}")
        if "sentiment_score" in alt:
            signals.append(f"sentiment={alt.get('sentiment_score')}")
        if "news_volume" in alt:
            signals.append(f"news_volume={alt.get('news_volume')}")
        if "vix_raw" in portfolio:
            signals.append(f"india_vix={portfolio.get('vix_raw')}")
        return signals[:12]

    @staticmethod
    def _disagreement(action: str, committee_direction: str) -> str:
        action = str(action or "").upper()
        direction = str(committee_direction or "NONE").upper()
        if direction == "NONE":
            return "NONE"
        if "LONG" in action and direction == "SHORT":
            return "AGAINST_COMMITTEE"
        if "SHORT" in action and direction == "LONG":
            return "AGAINST_COMMITTEE"
        return "NONE"

    @staticmethod
    def _observation_to_dict(row: MarketObservation) -> dict:
        return {
            "run_id": row.run_id,
            "timestamp": row.timestamp,
            "vix": float(row.vix or 0.0),
            "universe_size": int(row.universe_size or 0),
            "sector_breadth": float(row.sector_breadth or 0.0),
            "data_quality_status": row.data_quality_status,
            "notes": row.notes,
        }

    @staticmethod
    def _rl_to_dict(row: RLModelEvaluation) -> dict:
        return {
            "run_id": row.run_id,
            "model_version": row.model_version,
            "reward": float(row.reward or 0.0),
            "turnover": float(row.turnover or 0.0),
            "excess_return": float(row.excess_return or 0.0),
            "readiness_status": row.readiness_status,
            "promotion_eligible": bool(row.promotion_eligible),
        }


audit_logger = AuditLogger()
