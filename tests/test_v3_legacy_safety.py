import runpy

import pytest


def test_default_legacy_rl_scheduler_is_disabled():
    import main

    assert main.main() == 2
    assert not hasattr(main, "main_live_loop")


def test_legacy_us_entrypoint_exits_without_starting_scheduler():
    with pytest.raises(SystemExit, match="Legacy main_us.py execution is disabled"):
        runpy.run_module("main_us", run_name="__main__")


def test_legacy_us_loop_is_not_publicly_callable():
    import main_us

    assert not hasattr(main_us, "main_us_live_loop")
    with pytest.raises(RuntimeError, match="Legacy US execution is disabled"):
        main_us._legacy_main_us_live_loop_disabled()


def test_legacy_v2_entrypoint_exits_without_importing_order_code():
    with pytest.raises(SystemExit, match="Legacy main_us_v2.py execution is disabled"):
        runpy.run_module("main_us_v2", run_name="__main__")


def test_legacy_v2_cycle_is_not_publicly_callable():
    import main_us_v2

    assert not hasattr(main_us_v2, "run_cycle")
    with pytest.raises(RuntimeError, match="Legacy US v2 execution is disabled"):
        main_us_v2._legacy_main_us_v2_execution_disabled()
