"""
TextFeature contract.

Every text-feature agent returns a TextFeature dataclass with:
  - score: float in roughly [-3, +3]  (z-score-like; -3 = very bearish, +3 = very bullish)
  - confidence: float in [0, 1]      (0 = no signal / parse failed, 1 = strong)
  - metadata: dict                    (raw extracted bits — useful for explainability)

These features are NEVER directly traded on. They modulate sleeve weights
(Phase 4) or feature the Risk Officer's regime gating.

Confidence semantics:
  - 0.0 → ignore this feature (LLM down, no recent text, parse error)
  - 0.5 → moderate confidence (some text but ambiguous, or stale)
  - 1.0 → high confidence (clear signal in fresh text)
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TextFeature:
    """Output of a TextFeatureAgent.compute() call."""
    feature_name: str
    score: float = 0.0
    confidence: float = 0.0
    as_of: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def is_actionable(self, min_confidence: float = 0.3) -> bool:
        """True if confidence is high enough to use this feature."""
        return self.confidence >= min_confidence

    @classmethod
    def empty(cls, feature_name: str, reason: str = "no data") -> "TextFeature":
        return cls(feature_name=feature_name, score=0.0, confidence=0.0,
                   as_of=datetime.utcnow(), rationale=reason)


class TextFeatureAgent(ABC):
    """Base class for agents that turn text into numeric features."""

    name: str = "unnamed_text_agent"
    cache_ttl_seconds: int = 3600   # 1 hour default — most text is sticky

    def __init__(self, openai_client=None):
        self._client = openai_client   # injected for testing; auto-resolved on first use
        self._cache: Dict[str, tuple] = {}    # key → (expiry_ts, TextFeature)

    @abstractmethod
    def compute(self, *args, **kwargs) -> TextFeature:
        """Compute the feature. Must NOT raise — return TextFeature.empty() on failure."""
        ...

    # ── shared helpers ──────────────────────────────────────────────────────

    def _get_openai(self):
        """Lazy-init the OpenAI client. Returns None if no API key."""
        if self._client is not None:
            return self._client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            return self._client
        except Exception as e:
            logger.warning(f"{self.name}: failed to init OpenAI client: {e}")
            return None

    def _llm_score(self, prompt: str, system: str = "", model: str = "gpt-4o-mini",
                   max_tokens: int = 200) -> Optional[Dict[str, Any]]:
        """
        Run a small LLM completion and parse a JSON response.
        Returns dict on success, None on any failure.
        Convention: prompts ask the LLM to respond with JSON containing
          {"score": float, "confidence": float, "rationale": "..."}.
        """
        client = self._get_openai()
        if not client:
            return None
        try:
            sys_prompt = system or (
                "You are a financial text analyst. Respond ONLY with valid JSON. "
                "score: number from -3 (very bearish) to +3 (very bullish). "
                "confidence: number from 0 (uncertain) to 1 (clear signal). "
                "rationale: brief one-sentence explanation."
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            import json
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.warning(f"{self.name}: LLM call failed: {e}")
            return None

    def _cache_get(self, key: str) -> Optional[TextFeature]:
        entry = self._cache.get(key)
        if entry and time.time() < entry[0]:
            return entry[1]
        return None

    def _cache_set(self, key: str, feat: TextFeature):
        self._cache[key] = (time.time() + self.cache_ttl_seconds, feat)
