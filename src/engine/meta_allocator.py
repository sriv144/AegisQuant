from __future__ import annotations

from typing import Dict


class MetaAllocator:
    """Tiny RL-ready allocator over sleeves and cash, not single names."""

    def __init__(self, max_total_invested: float = 0.65, max_sleeve_nav: float = 0.325):
        self.max_total_invested = float(max_total_invested)
        self.max_sleeve_nav = float(max_sleeve_nav)

    def allocate(self, sleeve_scores: Dict[str, float]) -> Dict[str, float]:
        active = {k: max(0.0, float(v)) for k, v in sleeve_scores.items() if float(v) > 0.0}
        if not active:
            return {"cash": 1.0}

        equal_weight = self.max_total_invested / len(active)
        out = {k: min(self.max_sleeve_nav, equal_weight) for k in active}
        invested = sum(out.values())
        if invested > self.max_total_invested:
            scale = self.max_total_invested / invested
            out = {k: v * scale for k, v in out.items()}
            invested = sum(out.values())

        out["cash"] = round(max(0.0, 1.0 - invested), 10)
        return out
