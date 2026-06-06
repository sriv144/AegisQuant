"""
Trend Factor — Robert Carver's EWMAC (Exponentially Weighted Moving Average Crossover)
================================================================================

Three speeds combined: (8, 32), (16, 64), (32, 128) — fast / medium / slow trend.

For each speed:
  raw = EMA_fast(price) - EMA_slow(price)
  vol = ewmstd(daily_return, span=36) * price            # vol in price units
  forecast = raw / vol                                    # vol-normalised
  capped = clip(forecast, -20, +20) * forecast_scalar     # see Carver Ch.6
  scaled = capped to target an average abs forecast of 10

Final factor score = mean of the 3 capped forecasts.

Reference: Robert Carver, "Systematic Trading" (2015) chapters 4–8.

This is a TIME-SERIES (not cross-sectional) signal — each ticker has a
forecast independent of the others. We still z-score across the universe at
the end so the sleeve combiner can compare it to the other factors on the
same scale.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.factors.base import Factor, FactorResult

logger = logging.getLogger(__name__)


class TrendFactor(Factor):
    name = "trend"
    rebalance_freq = "weekly"
    requires = ["prices"]

    SPEEDS: List[Tuple[int, int]] = [(8, 32), (16, 64), (32, 128)]
    FORECAST_CAP = 20.0
    TARGET_ABS_FORECAST = 10.0
    VOL_SPAN = 36

    # Empirical forecast scalars from Carver Ch.7 Table 8 (calibrated so the
    # avg |forecast| is ≈ 10 across many instruments). Per (fast,slow) pair.
    FORECAST_SCALARS = {(8, 32): 5.3, (16, 64): 3.75, (32, 128): 2.65}

    def compute(self, universe: List[str], as_of: Optional[datetime] = None) -> FactorResult:
        as_of = pd.Timestamp(as_of or datetime.utcnow())
        start = (as_of - timedelta(days=500)).strftime("%Y-%m-%d")
        end = as_of.strftime("%Y-%m-%d")
        prices = self.dp.get_prices(universe, start=start, end=end)
        if prices is None or prices.empty:
            return FactorResult(self.name, as_of, {}, {}, {}, notes="no price data")

        raw_metrics: Dict[str, Dict[str, float]] = {}
        composite_forecast: Dict[str, float] = {}

        for t in universe:
            if t not in prices.columns:
                continue
            series = prices[t].dropna()
            if len(series) < max(s for _, s in self.SPEEDS) + 20:
                continue

            ret = series.pct_change()
            vol_pct = ret.ewm(span=self.VOL_SPAN, min_periods=10).std()
            vol_price = (vol_pct * series).iloc[-1]
            if not np.isfinite(vol_price) or vol_price == 0:
                continue

            per_speed = {}
            forecasts = []
            for (fast, slow) in self.SPEEDS:
                ema_f = series.ewm(span=fast, min_periods=fast).mean().iloc[-1]
                ema_s = series.ewm(span=slow, min_periods=slow).mean().iloc[-1]
                raw = ema_f - ema_s
                f_raw = raw / vol_price
                f_scaled = f_raw * self.FORECAST_SCALARS[(fast, slow)]
                f_capped = float(np.clip(f_scaled, -self.FORECAST_CAP, self.FORECAST_CAP))
                forecasts.append(f_capped)
                per_speed[f"f_{fast}_{slow}"] = f_capped

            avg_forecast = float(np.mean(forecasts))
            composite_forecast[t] = avg_forecast
            per_speed["avg_forecast"] = avg_forecast
            raw_metrics[t] = per_speed

        # Z-score the composite forecast across the universe so it's commensurate
        # with the other factors when the sleeve combiner sums them.
        scores = self.zscore(composite_forecast, winsorize=3.0)
        confidence = {t: 1.0 for t in scores}

        return FactorResult(
            factor_name=self.name,
            as_of=as_of,
            scores=scores,
            confidence=confidence,
            raw=raw_metrics,
            notes=f"Trend (Carver EWMAC {self.SPEEDS}) on {len(scores)} tickers",
        )
