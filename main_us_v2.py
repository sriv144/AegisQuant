"""Disabled compatibility entry point for the retired v2 trading runtime.

AegisQuant v3 is the sole US runtime.  Keeping this file as a tiny shim makes
old commands fail clearly without importing any legacy Alpaca writer.
"""

from __future__ import annotations


_MESSAGE = "Legacy main_us_v2.py execution is disabled; use main_us_v3.py in shadow mode."


def _legacy_main_us_v2_execution_disabled(*_args: object, **_kwargs: object) -> None:
    """Fail closed for code that retained a reference to the v2 cycle."""

    raise RuntimeError("Legacy US v2 execution is disabled")


def main() -> None:
    """Reject execution of the retired entry point."""

    raise SystemExit(_MESSAGE)


if __name__ == "__main__":
    main()
