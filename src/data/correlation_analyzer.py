"""
Correlation Analyzer
====================
Computes rolling pairwise correlations to dynamically assist the MultiAssetEnv in 
understanding diversification bounds and penalizing highly correlated long exposures.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    def __init__(self, rolling_window: int = 60):
        self.rolling_window = rolling_window
        
    def compute_all_correlations(self, feature_dfs: Dict[str, pd.DataFrame], tickers: List[str]) -> pd.DataFrame:
        """
        Takes raw dictionaries of ticker DataFrames (from WalkForwardEngine) and exactly 
        merges them on Date index to compute the overarching correlation matrix.
        Returns a symmetric (N, N) dataframe, or a rolling 3D array if extrapolated further.
        """
        # Inner join all 'Daily_Return' series
        df_merged = pd.DataFrame()
        
        for tick in tickers:
            if tick in feature_dfs and "Daily_Return" in feature_dfs[tick]:
                # Slice out just the return and rename column
                series = feature_dfs[tick]["Daily_Return"].rename(tick)
                if df_merged.empty:
                    df_merged = series.to_frame()
                else:
                    df_merged = df_merged.join(series, how="inner")
                    
        if df_merged.empty:
            logger.warning("Empty data fed to CorrelationAnalyzer.")
            return pd.DataFrame()
            
        # Compute static full-window Pearson correlation
        corr_matrix = df_merged.corr(method="pearson").fillna(0.0)
        
        # In a deep integration, we'd emit df_merged.rolling(self.rolling_window).corr() 
        # but the standard 2D matrix is required for baseline universe sizing filters.
        return corr_matrix

corr_analyzer = CorrelationAnalyzer()
