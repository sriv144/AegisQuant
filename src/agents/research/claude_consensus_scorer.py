"""Claude-backed consensus scorer for AegisQuant research agents.

This module provides an opt-in scoring layer that uses Anthropic's Claude
models to cross-check the JSON decisions produced by the existing
OpenAI/OpenRouter-backed research agents (fundamental_agent, macro_agent,
quant_agent, sentiment_agent).

Usage::

    from src.agents.research.claude_consensus_scorer import score_consensus

    result = score_consensus(
        ticker="AAPL",
        proposals=[fundamental_out, macro_out, quant_out, sentiment_out],
    )
    # -> {"consensus": "BUY", "confidence": 0.72, "rationale": "..."}

The scorer never raises and never blocks the main pipeline. It returns the
`ABSTAIN` dict whenever:

* `ANTHROPIC_API_KEY` is unset,
* the optional `anthropic` package is not installed,
* the proposal slate is empty, or
* the LLM call / JSON parse fails for any reason.

The existing OpenAI / OpenRouter flow in `base_agent.py` is untouched.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aegisquant.claude_consensus")

DEFAULT_MODEL = "claude-sonnet-4-6"
_VALID_DIRECTIONS = {"BUY", "SELL", "HOLD", "ABSTAIN"}

ABSTAIN: Dict[str, Any] = {
    "consensus": "ABSTAIN",
    "confidence": 0.0,
    "rationale": "Claude consensus scorer disabled or unavailable.",
    "model": None,
}


def _resolve_model(override: Optional[str] = None) -> str:
    if override and override.strip():
        return override.strip()
    env_choice = os.getenv("ANTHROPIC_MODEL", "").strip()
    return env_choice or DEFAULT_MODEL


def _build_prompt(ticker: str, proposals: List[Dict[str, Any]]) -> str:
    proposal_block = json.dumps(proposals, indent=2, default=str, sort_keys=True)
    return (
        f"You are a hedge-fund risk committee chair scoring a slate of "
        f"agent proposals for ticker {ticker}.\n\n"
        f"Proposals (one per research agent):\n{proposal_block}\n\n"
        "Return ONLY a JSON object with keys:\n"
        '  consensus: one of "BUY", "SELL", "HOLD", "ABSTAIN"\n'
        "  confidence: float in [0.0, 1.0]\n"
        "  rationale: <= 60 words explaining the call\n"
        "Do not include any text outside the JSON object."
    )


def _extract_text(message: Any) -> str:
    out: List[str] = []
    for block in getattr(message, "content", []) or []:
        block_text = getattr(block, "text", None)
        if block_text:
            out.append(block_text)
    return "".join(out).strip()


def _parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace == -1 or last_brace <= first_brace:
        return None
    try:
        candidate = json.loads(text[first_brace : last_brace + 1])
    except json.JSONDecodeError:
        return None
    return candidate if isinstance(candidate, dict) else None


def score_consensus(
    ticker: str,
    proposals: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """Score a slate of research proposals via Claude.

    Returns a fresh copy of the ABSTAIN dict (with `model` populated when
    available) on every disabled or failure path. Never raises.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return dict(ABSTAIN)

    if not proposals:
        return dict(ABSTAIN)

    try:
        from anthropic import Anthropic  # type: ignore import-not-found
    except ImportError:
        logger.warning(
            "ANTHROPIC_API_KEY is set but the `anthropic` package is not "
            "installed. Run `pip install anthropic` to enable."
        )
        return dict(ABSTAIN)

    chosen_model = _resolve_model(model)
    prompt = _build_prompt(ticker, proposals)

    try:
        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model=chosen_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # noqa: BLE001 - never crash the trading pipeline
        logger.exception("Claude consensus call failed: %s", exc)
        return dict(ABSTAIN)

    parsed = _parse_json_block(_extract_text(message))
    if parsed is None:
        result = dict(ABSTAIN)
        result["model"] = chosen_model
        return result

    consensus = str(parsed.get("consensus", "ABSTAIN")).upper()
    if consensus not in _VALID_DIRECTIONS:
        consensus = "ABSTAIN"

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    rationale = str(parsed.get("rationale", "")).strip()[:600]

    return {
        "consensus": consensus,
        "confidence": confidence,
        "rationale": rationale,
        "model": chosen_model,
    }


__all__ = ["ABSTAIN", "DEFAULT_MODEL", "score_consensus"]
