"""
AegisQuant Portfolio Construction
==================================

Sleeves are independent strategies that each output {ticker -> weight}
summing to 1.0 within the sleeve. The PM agent (Phase 4) combines sleeves
via risk parity at the portfolio level.

Currently implemented sleeves:
  - ValueQualityMomentumSleeve  (long-only equity, monthly)
  - CrossSectionalMomentumSleeve  (long-only momentum top decile, monthly)
  - PEADSleeve  (event-driven, weekly)
  - InsiderBuyingSleeve  (event-driven, weekly)
"""

from src.portfolio.sleeves import (
    Sleeve,
    SleeveResult,
    ValueQualityMomentumSleeve,
    CrossSectionalMomentumSleeve,
    PEADSleeve,
    InsiderBuyingSleeve,
    all_sleeves,
)
from src.portfolio.combiner import Combiner, PortfolioTarget
from src.portfolio.risk_officer import RiskOfficer, RiskReview

__all__ = [
    "Sleeve",
    "SleeveResult",
    "ValueQualityMomentumSleeve",
    "CrossSectionalMomentumSleeve",
    "PEADSleeve",
    "InsiderBuyingSleeve",
    "all_sleeves",
    "Combiner",
    "PortfolioTarget",
    "RiskOfficer",
    "RiskReview",
]
