"""AegisQuant v3 benchmark-aware research primitives.

The package is deliberately independent from the legacy trading runtime.  It
contains only deterministic, broker-free portfolio and research components.
"""

from .config import StrategyConfig, load_strategy_config
from .data import (
    DataManifest,
    DataReadiness,
    DataTier,
    OpenResearchProvider,
    PointInTimeDataProvider,
    PromotableProvider,
)
from .portfolio import (
    DataFailure,
    PortfolioConstructor,
    PortfolioInputs,
    PortfolioPlan,
    SecurityMetadata,
)
from .backtest import EventBacktestResult, EventDrivenBacktester
from .metrics import SpyRelativeMetrics, compute_spy_relative_metrics
from .research import (
    EXTERNAL_DATA_REQUIREMENTS,
    ExperimentEvidence,
    PreRegisteredStudy,
    PromotionDecision,
    PromotionThresholds,
    evaluate_promotion,
)
from .rl import (
    RLAllocationAction,
    RLEligibilityEvidence,
    RLQuarantineError,
    RLRegistryState,
    RegisteredRLCheckpoint,
    RegisteredRLLoader,
    rl_eligibility_failures,
)

__all__ = [
    "DataFailure",
    "DataManifest",
    "DataReadiness",
    "DataTier",
    "EventBacktestResult",
    "EventDrivenBacktester",
    "OpenResearchProvider",
    "PointInTimeDataProvider",
    "PromotableProvider",
    "PortfolioConstructor",
    "PortfolioInputs",
    "PortfolioPlan",
    "SecurityMetadata",
    "SpyRelativeMetrics",
    "StrategyConfig",
    "EXTERNAL_DATA_REQUIREMENTS",
    "ExperimentEvidence",
    "PreRegisteredStudy",
    "PromotionDecision",
    "PromotionThresholds",
    "RLAllocationAction",
    "RLEligibilityEvidence",
    "RLQuarantineError",
    "RLRegistryState",
    "RegisteredRLCheckpoint",
    "RegisteredRLLoader",
    "compute_spy_relative_metrics",
    "evaluate_promotion",
    "load_strategy_config",
    "rl_eligibility_failures",
]
