"""
Multi-Asset Benchmarks
======================
Extends baseline evaluation formulas to explicitly test equal-weight and momentum 
rebalancing over an N-dimensional portfolio space.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

from src.backtest.metrics import sharpe_ratio, max_drawdown, profit_factor as compute_profit_factor

class MultiAssetBenchmarkSuite:
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        
    def evaluate_equal_weight(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """
        If the portfolio equally weighted every asset daily.
        returns_df: A DataFrame where columns are ticker strings and values are daily returns.
        """
        if returns_df.empty:
            return {}
            
        # Cross-sectional mean calculates daily EW portfolio return
        ew_returns = returns_df.mean(axis=1)
        
        # Calculate standard KPIs
        compounded = (1 + ew_returns).prod() - 1
        sharpe = sharpe_ratio(ew_returns)
        mdd = max_drawdown(ew_returns)
        win_rate = (ew_returns > 0).mean()
        pf = compute_profit_factor(ew_returns)
        
        return {
            "label": "Equal-Weight Benchmark",
            "annualised_return": (1 + compounded) ** (252 / len(ew_returns)) - 1 if len(ew_returns) > 0 else 0,
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(mdd),
            "win_rate": float(win_rate),
            "profit_factor": float(pf)
        }

    def evaluate_cross_sectional_momentum(self, returns_df: pd.DataFrame, momentum_lookback: int = 252) -> Dict[str, float]:
        """
        Selects top 50% of assets mathematically determined by N-day lookback returns, and equally weights them.
        """
        if len(returns_df) <= momentum_lookback:
            logger.warning("Not enough data to calculate cross-sectional momentum benchmark.")
            return {}
            
        # Calculate rolling momentum
        rolling_mom = returns_df.rolling(window=momentum_lookback).apply(lambda x: (1 + x).prod() - 1)
        
        mom_returns = []
        # Step through history starting after the lookback initializes
        for i in range(momentum_lookback, len(returns_df)):
            # T-1 momentum determines target weights for T
            mom_scores = rolling_mom.iloc[i - 1]
            if mom_scores.isna().all():
                mom_returns.append(0.0)
                continue
                
            # Filter top 50%
            threshold = mom_scores.median()
            long_targets = mom_scores[mom_scores >= threshold].index
            
            # Sub-slice the row for today's returns against those assets
            today_returns = returns_df.iloc[i][long_targets]
            # Equally weight the selected
            if len(today_returns) > 0:
                mom_returns.append(today_returns.mean())
            else:
                mom_returns.append(0.0)
                
        mom_returns = pd.Series(mom_returns)
        
        compounded = (1 + mom_returns).prod() - 1
        sharpe = sharpe_ratio(mom_returns)
        
        return {
            "label": f"Momentum Benchmark ({momentum_lookback}d)",
            "annualised_return": (1 + compounded) ** (252 / len(mom_returns)) - 1 if len(mom_returns) > 0 else 0,
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown(mom_returns)),
            "win_rate": float((mom_returns > 0).mean()),
            "profit_factor": float(compute_profit_factor(mom_returns))
        }
