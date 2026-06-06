"""
Purged K-Fold Cross-Validation with Embargo
============================================

López de Prado (Advances in Financial Machine Learning, Ch. 7).

Standard k-fold leaks future into past for time-series with overlapping labels.
The fix:

  - **Purge**: drop training samples whose LABELS overlap the test set's
    observation window. (Example: if test sample's label is "60-day forward
    return", any training sample whose 60-day window overlaps with the test
    sample must be removed from train.)

  - **Embargo**: additionally drop training samples that occur within `h`
    bars AFTER the test set ends — to prevent serial correlation in returns
    or features from leaking the test outcome into adjacent training labels.

Usage:
    cv = PurgedKFold(n_splits=5, t1=label_end_times, embargo_pct=0.01)
    for train_idx, test_idx in cv.split(X):
        ...

`t1` is a pd.Series mapping each observation index to the end-time of its
label window (when the next-bar/forward-return computation finishes). For
"label = 1-day return", t1 = index + 1 day. For "60-day triple-barrier",
t1 = index + 60 days (or earlier if barrier hit).
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd


class PurgedKFold:
    """Time-series-aware k-fold CV with purging and embargo.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    t1 : pd.Series
        For each observation (index), the timestamp at which its label is
        determined / its information window ends. Required.
    embargo_pct : float
        Fraction of total samples to embargo AFTER each test fold.
        Typical: 0.01 (1%). For 1000 samples, 10-bar embargo.
    """

    def __init__(self, n_splits: int = 5, t1: Optional[pd.Series] = None,
                 embargo_pct: float = 0.01):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if t1 is None:
            raise ValueError("t1 (label end times) is required for purging")
        if not t1.index.is_monotonic_increasing:
            raise ValueError("t1.index must be monotonically increasing")
        self.n_splits = n_splits
        self.t1 = t1
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) for each fold."""
        if X.shape[0] != self.t1.shape[0]:
            raise ValueError(f"X and t1 length mismatch: {X.shape[0]} vs {self.t1.shape[0]}")

        indices = np.arange(X.shape[0])
        embargo_size = int(X.shape[0] * self.embargo_pct)
        fold_size = X.shape[0] // self.n_splits

        for k in range(self.n_splits):
            # Test fold: contiguous block
            test_start = k * fold_size
            test_end = (k + 1) * fold_size if k < self.n_splits - 1 else X.shape[0]
            test_idx = indices[test_start:test_end]

            # Compute purge window: any train sample whose label end-time falls
            # within the test fold's observation window must be purged.
            test_t0 = self.t1.index[test_start]
            test_t1 = self.t1.iloc[test_end - 1]  # the end of the latest label

            train_mask = np.ones(X.shape[0], dtype=bool)
            train_mask[test_start:test_end] = False

            # Purge: drop train samples whose t1 falls inside [test_t0, test_t1]
            # OR whose observation start falls inside [test_t0, test_t1]
            t1_arr = self.t1.values
            obs_start = self.t1.index.values
            # Convert to comparable numpy datetimes
            ts0 = np.datetime64(test_t0)
            ts1 = np.datetime64(test_t1)
            label_in_test = (t1_arr >= ts0) & (t1_arr <= ts1)
            obs_in_test = (obs_start >= ts0) & (obs_start <= ts1)
            purge_mask = label_in_test | obs_in_test
            train_mask &= ~purge_mask

            # Embargo: drop the first `embargo_size` samples AFTER the test fold
            if embargo_size > 0 and test_end < X.shape[0]:
                emb_end = min(test_end + embargo_size, X.shape[0])
                train_mask[test_end:emb_end] = False

            train_idx = indices[train_mask]
            yield train_idx, test_idx
