"""
Macro Regime Agent
==================

Classifies the current macro environment into a small set of regimes that the
PM combiner uses to modulate sleeve weights. Uses ONLY free data:

  - VIX (^VIX) — equity vol regime
  - 10Y minus 2Y Treasury yield spread (^TNX vs ^IRX proxy) — yield curve
  - Trailing 60d SPY return — risk-on / risk-off

Output is a single numeric "macro regime score" in [-3, +3]:
  +3 : very risk-on (low VIX, normal curve, SPY up)
   0 : neutral
  -3 : very risk-off (high VIX, inverted curve, SPY down)

This score is consumed by the PMAgent (Phase 4) to scale sleeve weights:
  - sleeves with high equity beta (VQM, XS momentum) get UNDERWEIGHTED in -2 or worse
  - defensive / trend sleeves can get OVERWEIGHTED in risk-off

No LLM needed for the numeric path. Optionally, if OPENAI_API_KEY is set, we
ALSO fetch the most recent FOMC press release and ask the LLM for a "tone delta"
that nudges the regime score. The numeric path is the source of truth — LLM is
a small modulator.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from src.agents.text_features.base import TextFeature, TextFeatureAgent

logger = logging.getLogger(__name__)


class MacroRegimeAgent(TextFeatureAgent):
    name = "macro_regime"
    cache_ttl_seconds = 3600   # 1 hour — macro changes slowly

    # Calibration thresholds (rough — backtest in Phase 5 to refine)
    VIX_HIGH = 28.0
    VIX_LOW = 15.0
    CURVE_INVERT_BPS = -25     # 10Y - 2Y < -25bp = recession watch
    SPY_60D_BAD = -0.05        # -5% in 60d = clearly risk-off
    SPY_60D_GOOD = +0.05

    def compute(self, _ignored=None) -> TextFeature:
        """Compute the current macro regime score."""
        cache_key = "macro::regime"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        vix = self._fetch_close("^VIX")
        ten_y = self._fetch_close("^TNX")    # 10-year, % * 10
        two_y = self._fetch_close("^IRX")    # 13-wk T-bill as 2Y proxy (yfinance has no ^FVX-free)
        spy_60d_return = self._fetch_n_day_return("SPY", 60)

        components = []
        rationale_parts = []

        # VIX component: lower = better, in [-1, +1]
        if vix is not None:
            if vix <= self.VIX_LOW:
                vix_score = 1.0
            elif vix >= self.VIX_HIGH:
                vix_score = -1.0
            else:
                vix_score = 1.0 - 2.0 * (vix - self.VIX_LOW) / (self.VIX_HIGH - self.VIX_LOW)
            components.append(vix_score)
            rationale_parts.append(f"VIX={vix:.1f}({vix_score:+.2f})")

        # Curve component: positive (normal) = good; inversion = bad
        if ten_y is not None and two_y is not None:
            # both are reported in basis points/10. Approximate spread.
            spread_bps = (ten_y - two_y) * 10.0
            if spread_bps >= 50:
                curve_score = 1.0
            elif spread_bps <= self.CURVE_INVERT_BPS:
                curve_score = -1.0
            else:
                curve_score = (spread_bps - self.CURVE_INVERT_BPS) / (50 - self.CURVE_INVERT_BPS) * 2.0 - 1.0
            components.append(curve_score)
            rationale_parts.append(f"curve={spread_bps:+.0f}bp({curve_score:+.2f})")

        # 60d SPY return component
        if spy_60d_return is not None:
            if spy_60d_return >= self.SPY_60D_GOOD:
                spy_score = 1.0
            elif spy_60d_return <= self.SPY_60D_BAD:
                spy_score = -1.0
            else:
                spy_score = (spy_60d_return - self.SPY_60D_BAD) / (self.SPY_60D_GOOD - self.SPY_60D_BAD) * 2.0 - 1.0
            components.append(spy_score)
            rationale_parts.append(f"SPY_60d={spy_60d_return*100:+.1f}%({spy_score:+.2f})")

        if not components:
            feat = TextFeature.empty(self.name, "no macro data available")
            self._cache_set(cache_key, feat)
            return feat

        # Scale combined score into [-3, +3]
        combined = float(np.mean(components) * 3.0)
        confidence = min(1.0, len(components) / 3.0)   # full confidence if all 3 ingredients present

        feat = TextFeature(
            feature_name=self.name,
            score=combined,
            confidence=confidence,
            as_of=datetime.utcnow(),
            metadata={
                "vix": vix,
                "ten_y": ten_y,
                "two_y_proxy": two_y,
                "spy_60d_return": spy_60d_return,
                "n_components": len(components),
            },
            rationale=" | ".join(rationale_parts) + f" -> regime={combined:+.2f}",
        )
        self._cache_set(cache_key, feat)
        return feat

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fetch_close(symbol: str) -> Optional[float]:
        try:
            import yfinance as yf
            df = yf.download(symbol, period="5d", interval="1d",
                             progress=False, auto_adjust=True)
            if df is None or df.empty:
                return None
            # df['Close'] may be Series or DataFrame depending on yfinance version
            close = df["Close"]
            val = close.iloc[-1]
            if hasattr(val, "iloc"):
                val = val.iloc[0]
            return float(val)
        except Exception as e:
            logger.warning(f"MacroRegimeAgent: failed to fetch {symbol}: {e}")
            return None

    @staticmethod
    def _fetch_n_day_return(symbol: str, n_days: int) -> Optional[float]:
        try:
            import yfinance as yf
            df = yf.download(symbol, period=f"{n_days + 10}d", interval="1d",
                             progress=False, auto_adjust=True)
            if df is None or df.empty or len(df) < n_days:
                return None
            close = df["Close"]
            first = close.iloc[-n_days]
            last = close.iloc[-1]
            if hasattr(first, "iloc"):
                first = first.iloc[0]
            if hasattr(last, "iloc"):
                last = last.iloc[0]
            return float(last / first - 1.0)
        except Exception as e:
            logger.warning(f"MacroRegimeAgent: failed return calc for {symbol}: {e}")
            return None
