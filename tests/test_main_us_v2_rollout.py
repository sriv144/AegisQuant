from datetime import datetime, timezone

import main_us_v2


def test_live_start_gate_uses_new_york_date(monkeypatch):
    monkeypatch.setattr(main_us_v2, "V2_LIVE_START_DATE", "2026-06-08")

    before = datetime(2026, 6, 8, 1, 0, tzinfo=timezone.utc)   # Jun 7 NY evening
    after = datetime(2026, 6, 8, 14, 0, tzinfo=timezone.utc)   # Jun 8 NY morning

    assert main_us_v2._live_start_reached(before) is False
    assert main_us_v2._live_start_reached(after) is True


def test_resolve_dry_run_before_live_start(monkeypatch):
    monkeypatch.setattr(main_us_v2, "KILL_SWITCH", False)
    monkeypatch.setattr(main_us_v2, "V2_REQUIRE_PRETRADE_REVIEW", False)
    monkeypatch.setattr(main_us_v2, "V2_LIVE_START_DATE", "2026-06-08")

    dry, reason = main_us_v2._resolve_dry_run(
        False,
        datetime(2026, 6, 8, 1, 0, tzinfo=timezone.utc),
    )

    assert dry is True
    assert "V2_LIVE_START_DATE" in reason


def test_build_delta_orders_respects_threshold(monkeypatch):
    monkeypatch.setattr(main_us_v2, "V2_MIN_TRADE_NAV_PCT", 0.005)
    monkeypatch.setattr(main_us_v2, "_current_positions_from_alpaca", lambda: {"A": 10, "B": 0})
    monkeypatch.setattr(main_us_v2, "_latest_prices", lambda symbols: {"A": 100.0, "B": 100.0})

    orders = main_us_v2._build_delta_orders({"A": 0.011, "B": 0.20}, nav=100_000)

    assert [o["symbol"] for o in orders] == ["B"]
    assert orders[0]["side"] == "buy"
    assert orders[0]["qty"] == 200
