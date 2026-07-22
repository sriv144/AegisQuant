"""Disabled compatibility entry point for the retired US trading runtime.

Use :mod:`main_us_v3` for shadow or explicitly approved Alpaca paper runs.
This module deliberately imports no broker, scheduler, strategy, or order code.
"""

from __future__ import annotations


_MESSAGE = "Legacy main_us.py execution is disabled; use main_us_v3.py in shadow mode."


def _legacy_main_us_live_loop_disabled(*_args: object, **_kwargs: object) -> None:
    """Fail closed for callers that retained a reference to the old loop."""

    raise RuntimeError("Legacy US execution is disabled")


def main() -> None:
    """Reject execution of the retired entry point."""

    raise SystemExit(_MESSAGE)


if __name__ == "__main__":
    main()
