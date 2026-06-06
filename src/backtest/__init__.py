# AegisQuant — Backtesting Framework
#
# Two complementary toolkits live here:
#
# 1. Legacy RL-environment backtests
#    historical_env.py, multi_asset_env.py, walk_forward.py — gym environments
#    used by the (currently paused) PPO RL training pipeline.
#
# 2. Factor / Sleeve backtests (Phase 5 of strategy redesign)
#    purged_cv.py            López de Prado purged k-fold CV with embargo
#    triple_barrier.py       Profit/stop/timeout labels (no next-bar bias)
#    deflated_sharpe.py      Bailey & López de Prado deflated Sharpe ratio
#    sleeve_backtester.py    Walks a Sleeve through history, computes metrics
#
# The factor toolkit is the gating layer for live deployment: each sleeve must
# pass deflated Sharpe > 0.4 OOS before being added to the live combiner.

from src.backtest.purged_cv import PurgedKFold
from src.backtest.triple_barrier import triple_barrier_labels
from src.backtest.deflated_sharpe import (
    deflated_sharpe_ratio,
    probability_backtest_overfit,
)
from src.backtest.sleeve_backtester import SleeveBacktester, BacktestResult

__all__ = [
    "PurgedKFold",
    "triple_barrier_labels",
    "deflated_sharpe_ratio",
    "probability_backtest_overfit",
    "SleeveBacktester",
    "BacktestResult",
]
