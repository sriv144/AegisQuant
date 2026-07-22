from __future__ import annotations

from pathlib import Path

import pytest
import requests
import yaml

from .no_network_order_post import (
    NetworkOrderPostBlocked,
    is_alpaca_order_post,
    pytest_configure,
)


ROOT = Path(__file__).resolve().parents[2]


def test_compose_has_only_opt_in_one_shot_v3_shadow_trader():
    compose_path = ROOT / "docker-compose.yml"
    compose = yaml.load(compose_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    services = compose["services"]
    shadow = services["manual-shadow"]

    assert "trader" not in services
    assert shadow["profiles"] == ["manual-shadow"]
    assert shadow["restart"] == "no"
    assert shadow["command"] == [
        "python",
        "main_us_v3.py",
        "--mode",
        "shadow",
        "--purpose",
        "rebalance",
    ]
    assert shadow["environment"]["EXECUTION_ENABLED"] == "false"
    assert shadow["environment"]["PAPER_EXECUTION_ENABLED"] == "false"
    assert shadow["environment"]["ENABLE_BROKER_EXECUTION"] == "false"
    assert shadow["environment"]["RL_ENABLED"] == "false"
    assert shadow["environment"]["V3_RUNTIME_INPUT"] == "/app/data/v3_runtime_input.json"
    assert "env_file" not in shadow
    assert "depends_on" not in services["dashboard"]
    assert services["dashboard"]["environment"]["POSTGRES_URL"].endswith(
        "/app/data/aegisquant_v3.db}"
    )

    text = compose_path.read_text(encoding="utf-8")
    assert "main_us.py" not in text
    assert "main_us_v2.py" not in text
    assert "ALPACA_API_KEY" not in text
    assert "ALPACA_SECRET_KEY" not in text

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'CMD ["python", "-m", "src.webapp.server"]' in dockerfile
    assert "COPY requirements.lock ." in dockerfile
    assert "requirements.txt" not in dockerfile
    assert "alpaca-trade-api>=3.0.0" not in dockerfile
    assert 'CMD ["python", "main_us.py"' not in dockerfile
    assert 'CMD ["python", "main_us_v2.py"' not in dockerfile


def test_network_guard_recognizes_only_real_alpaca_order_posts():
    assert is_alpaca_order_post(
        "POST", "https://paper-api.alpaca.markets/v2/orders"
    )
    assert is_alpaca_order_post("post", "https://api.alpaca.markets/v2/orders/123")
    assert not is_alpaca_order_post(
        "GET", "https://paper-api.alpaca.markets/v2/orders"
    )
    assert not is_alpaca_order_post("POST", "https://example.test/v2/orders")
    assert not is_alpaca_order_post(
        "POST", "https://paper-api.alpaca.markets/v2/assets"
    )


def test_network_guard_blocks_before_requests_transport():
    pytest_configure(None)
    request = requests.Request(
        "POST",
        "https://paper-api.alpaca.markets/v2/orders",
        json={"symbol": "SPY"},
    ).prepare()

    with pytest.raises(NetworkOrderPostBlocked, match="network order POST blocked"):
        requests.Session().send(request)


def test_codeowners_covers_safety_critical_surfaces():
    text = (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    for owned_path in (
        "/.github/CODEOWNERS",
        "/.github/workflows/",
        "/Dockerfile",
        "/.dockerignore",
        "/docker-compose.yml",
        "/main_us.py",
        "/main_us_v2.py",
        "/main_us_v3.py",
        "/scripts/fetch_v3_runtime_input.py",
        "/src/v3/",
        "/src/execution/",
        "/src/execution/v3/",
        "/config/strategies/",
        "/alembic/",
    ):
        assert f"{owned_path} @sriv144" in text
