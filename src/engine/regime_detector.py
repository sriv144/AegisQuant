"""
Regime Detector
===============
HMM-based market regime classifier with 4 hidden states:

    State 0 — Bull Quiet    : trending up, low volatility
    State 1 — Bull Volatile : trending up, high vol (earnings, momentum)
    State 2 — Bear Quiet    : drifting down, low vol (slow distribution)
    State 3 — Bear Volatile : crisis — sharp drawdowns, VIX-spike regime

Trained on 4 microstructure features derived from daily OHLCV:
    [daily_return, Volatility_20, RSI_14_Z, MACD_Z]

Usage:
    from src.engine.regime_detector import RegimeDetector
    detector = RegimeDetector()
    detector.fit(features_df)            # features_df from feature_engineer
    regimes = detector.predict(features_df)  # np.ndarray of int in {0,1,2,3}
    onehot  = detector.get_onehot(regimes[t])  # [1,0,0,0] style encoding
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns expected in features_df
_FEATURE_COLS = ["Daily_Return", "Volatility_20", "RSI_14_Z", "MACD_Z"]

# Default save path
_DEFAULT_MODEL_PATH = Path("models/regime_detector.pkl")


class RegimeDetector:
    """
    HMM-based 4-state market regime classifier.

    Args:
        n_states:      Number of hidden Markov states (default 4).
        random_state:  Seed for reproducibility.
    """

    # Regime labels for human readability
    REGIME_NAMES = {
        0: "Bull Quiet",
        1: "Bull Volatile",
        2: "Bear Quiet",
        3: "Bear Volatile",
    }

    def __init__(self, n_states: int = 4, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self._model = None
        self._is_fitted = False
        self._regime_map: dict = {}  # raw HMM state -> canonical regime id

    # ------------------------------------------------------------------ fit
    def fit(self, features_df: pd.DataFrame) -> "RegimeDetector":
        """
        Fit the HMM on historical feature data.

        Args:
            features_df: DataFrame with columns including Daily_Return,
                         Volatility_20, RSI_14_Z, MACD_Z. May contain NaNs
                         (rows are dropped before fitting).

        Returns:
            self (fluent interface)
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError(
                "hmmlearn is required for RegimeDetector. "
                "Install it with: pip install hmmlearn>=0.3.2"
            )

        X = self._extract_features(features_df)
        if len(X) < 50:
            logger.warning(
                "RegimeDetector.fit: only %d observations after NaN drop — "
                "skipping fit, will default all regimes to 0", len(X)
            )
            self._is_fitted = False
            return self

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=200,
            random_state=self.random_state,
        )
        model.fit(X)
        self._model = model
        self._is_fitted = True

        # Build a stable mapping from HMM state id -> canonical regime id.
        # HMM state ordering is arbitrary; we sort by mean daily return
        # so that low-return states = Bear, high-return = Bull,
        # and within each group low-vol = Quiet, high-vol = Volatile.
        self._regime_map = self._build_regime_map(model)
        logger.info(
            "RegimeDetector fitted on %d observations. "
            "Regime map (HMM state -> canonical): %s", len(X), self._regime_map
        )
        return self

    # --------------------------------------------------------------- predict
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict regime label for every row in features_df.

        Returns:
            Integer array of length len(features_df) with values in {0,1,2,3}.
            Rows that are NaN are assigned regime 0 (Bull Quiet fallback).
        """
        if not self._is_fitted:
            return np.zeros(len(features_df), dtype=int)

        raw_labels = np.zeros(len(features_df), dtype=int)
        valid_mask = features_df[_FEATURE_COLS].notna().all(axis=1)

        if valid_mask.any():
            X_valid = features_df.loc[valid_mask, _FEATURE_COLS].values
            raw_valid = self._model.predict(X_valid)
            mapped_valid = np.array(
                [self._regime_map.get(int(s), 0) for s in raw_valid], dtype=int
            )
            raw_labels[valid_mask.values] = mapped_valid

        return raw_labels

    # ---------------------------------------------------------- predict_proba
    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Return (N, n_states) state probability matrix.
        Rows with NaN features get uniform 0.25 probability.
        """
        N = len(features_df)
        proba = np.full((N, self.n_states), 1.0 / self.n_states, dtype=float)

        if not self._is_fitted:
            return proba

        valid_mask = features_df[_FEATURE_COLS].notna().all(axis=1)
        if valid_mask.any():
            X_valid = features_df.loc[valid_mask, _FEATURE_COLS].values
            # posteriors: shape (len(X_valid), n_states)
            _, posteriors = self._model.decode(X_valid, algorithm="viterbi")
            # posteriors from decode is not probabilities — use score_samples
            # for actual posteriors
            log_proba = self._model.predict_proba(X_valid)  # (T, n_states)
            # re-order columns to canonical regime order
            reordered = np.zeros_like(log_proba)
            for hmm_state, canonical in self._regime_map.items():
                if hmm_state < log_proba.shape[1]:
                    reordered[:, canonical] = log_proba[:, hmm_state]
            proba[valid_mask.values] = reordered

        return proba

    # -------------------------------------------------------------- get_onehot
    @staticmethod
    def get_onehot(regime_id: int, n_states: int = 4) -> np.ndarray:
        """Return one-hot encoding for a regime id."""
        v = np.zeros(n_states, dtype=np.float32)
        v[int(regime_id) % n_states] = 1.0
        return v

    # ------------------------------------------------------------------ save
    def save(self, path: str | Path = _DEFAULT_MODEL_PATH) -> None:
        """Persist the fitted model to disk using joblib."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "regime_map": self._regime_map,
                     "n_states": self.n_states, "is_fitted": self._is_fitted}, path)
        logger.info("RegimeDetector saved to %s", path)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str | Path = _DEFAULT_MODEL_PATH) -> "RegimeDetector":
        """Restore a previously saved RegimeDetector from disk."""
        import joblib
        data = joblib.load(Path(path))
        det = cls(n_states=data["n_states"])
        det._model = data["model"]
        det._regime_map = data["regime_map"]
        det._is_fitted = data["is_fitted"]
        logger.info("RegimeDetector loaded from %s", path)
        return det

    # ---------------------------------------------------------------- summary
    def print_regime_distribution(self, regimes: np.ndarray) -> None:
        """Print count and percentage of each regime in a predicted array."""
        total = len(regimes)
        print("\nRegime Distribution:")
        for rid in range(self.n_states):
            count = int((regimes == rid).sum())
            pct = count / total * 100 if total > 0 else 0
            name = self.REGIME_NAMES.get(rid, f"State {rid}")
            print(f"  [{rid}] {name:<18}: {count:>5} days  ({pct:5.1f}%)")

    # ----------------------------------------------------------- private utils
    @staticmethod
    def _extract_features(features_df: pd.DataFrame) -> np.ndarray:
        """Extract and clean the 4 HMM input features."""
        available = [c for c in _FEATURE_COLS if c in features_df.columns]
        if not available:
            raise ValueError(
                f"features_df must contain columns: {_FEATURE_COLS}. "
                f"Found: {list(features_df.columns)}"
            )
        missing = set(_FEATURE_COLS) - set(available)
        if missing:
            logger.warning("Missing HMM feature columns: %s — filling with 0", missing)
            for col in missing:
                features_df = features_df.copy()
                features_df[col] = 0.0

        X = features_df[_FEATURE_COLS].dropna().values.astype(float)
        return X

    def _build_regime_map(self, model) -> dict:
        """
        Map each HMM state index to one of {0,1,2,3} canonical regime ids.

        Canonical assignment:
          - Rank states by mean daily return (highest = most bullish)
          - Within top-2 (bull states): lower vol = state 0, higher vol = state 1
          - Within bottom-2 (bear states): lower vol = state 2, higher vol = state 3
        """
        n = model.n_components
        # Mean return is feature index 0; volatility is feature index 1
        means = model.means_  # (n_components, n_features)
        mean_return = means[:, 0]
        mean_vol = means[:, 1]

        ranked = np.argsort(mean_return)[::-1]  # highest return first
        half = n // 2
        bull_states = ranked[:half]
        bear_states = ranked[half:]

        # Within bull states: sort by volatility ascending (Quiet=0, Volatile=1)
        bull_sorted = sorted(bull_states, key=lambda s: mean_vol[s])
        # Within bear states: sort by volatility ascending (Quiet=2, Volatile=3)
        bear_sorted = sorted(bear_states, key=lambda s: mean_vol[s])

        regime_map = {}
        for i, s in enumerate(bull_sorted):
            regime_map[int(s)] = i          # 0 = Bull Quiet, 1 = Bull Volatile
        for i, s in enumerate(bear_sorted):
            regime_map[int(s)] = half + i   # 2 = Bear Quiet, 3 = Bear Volatile

        return regime_map
