"""Regression tests for retirement of the unsafe v2 runtime."""

from __future__ import annotations

import runpy

import pytest

import main_us_v2


def test_v2_runtime_has_no_execution_or_schedule_surface():
    assert not hasattr(main_us_v2, "run_cycle")
    assert not hasattr(main_us_v2, "_build_delta_orders")
    assert not hasattr(main_us_v2, "_loop_cron_minute")


def test_v2_command_fails_closed():
    with pytest.raises(SystemExit, match="Legacy main_us_v2.py execution is disabled"):
        runpy.run_module("main_us_v2", run_name="__main__")
