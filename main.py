"""Disabled legacy Alpaca/RL scheduler compatibility entrypoint.

The old default command could construct an Alpaca executor and route PPO
weights to broker orders.  AegisQuant v3 deliberately has no live mode and
paper execution is available only through the guarded one-shot
``main_us_v3.py`` coordinator.
"""

from __future__ import annotations


def main() -> int:
    print(
        "Legacy Alpaca/RL execution is disabled. "
        "Use: python main_us_v3.py --mode shadow --purpose health"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
