"""Pytest plugin that makes a real Alpaca order POST a hard test failure.

Load this plugin for the entire suite with::

    PYTHONPATH=tests pytest -p v3_workflow.no_network_order_post

The guard is intentionally host- and path-specific: tests may use in-memory
fakes and local HTTP fixtures, but neither the paper nor production Alpaca
order endpoint can receive a POST from a test process.
"""

from __future__ import annotations

from typing import Any, Callable
from urllib.parse import urlsplit


class NetworkOrderPostBlocked(RuntimeError):
    """Raised before a test can send an Alpaca order over the network."""


def is_alpaca_order_post(method: object, url: object) -> bool:
    """Return whether *method* and *url* identify an Alpaca order POST."""

    if str(method).upper() != "POST":
        return False
    parsed = urlsplit(str(url))
    hostname = (parsed.hostname or "").lower()
    is_alpaca = hostname == "alpaca.markets" or hostname.endswith(".alpaca.markets")
    path_parts = {part for part in parsed.path.lower().split("/") if part}
    return is_alpaca and "orders" in path_parts


def _blocked(method: object, url: object) -> None:
    if is_alpaca_order_post(method, url):
        raise NetworkOrderPostBlocked(
            f"network order POST blocked during tests: {urlsplit(str(url)).hostname}"
        )


def pytest_configure(config: Any) -> None:
    """Install process-wide guards before test modules are imported."""

    del config

    import requests

    if not getattr(requests.sessions.Session.send, "_aegisquant_order_guard", False):
        original_requests_send = requests.sessions.Session.send

        def guarded_requests_send(
            session: requests.sessions.Session,
            request: requests.PreparedRequest,
            **kwargs: Any,
        ) -> requests.Response:
            _blocked(request.method, request.url)
            return original_requests_send(session, request, **kwargs)

        guarded_requests_send._aegisquant_order_guard = True  # type: ignore[attr-defined]
        requests.sessions.Session.send = guarded_requests_send

    try:
        import httpx
    except ImportError:
        return

    def install_httpx_guard(owner: type[Any]) -> None:
        original: Callable[..., Any] = owner.send
        if getattr(original, "_aegisquant_order_guard", False):
            return

        if owner is httpx.AsyncClient:
            async def guarded_async_send(
                client: Any,
                request: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                _blocked(request.method, request.url)
                return await original(client, request, *args, **kwargs)

            guarded_async_send._aegisquant_order_guard = True  # type: ignore[attr-defined]
            owner.send = guarded_async_send
        else:
            def guarded_send(
                client: Any,
                request: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                _blocked(request.method, request.url)
                return original(client, request, *args, **kwargs)

            guarded_send._aegisquant_order_guard = True  # type: ignore[attr-defined]
            owner.send = guarded_send

    install_httpx_guard(httpx.Client)
    install_httpx_guard(httpx.AsyncClient)
