from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.fetch_v3_runtime_input import fetch


class _Response:
    status = 200
    headers: dict[str, str] = {}

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit: int) -> bytes:
        return self.payload[:limit]


class _Opener:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def open(self, *_args, **_kwargs):
        return _Response(self.payload)


def test_fetch_requires_https_and_exact_digest(tmp_path: Path):
    with pytest.raises(ValueError, match="credential-free HTTPS"):
        fetch(url="http://example.test/input.json", expected_sha256="0" * 64, output=tmp_path / "x")


def test_fetch_writes_only_verified_self_contained_bundle(monkeypatch, tmp_path: Path):
    payload = json.dumps({"total_return_prices": {"dates": [], "values": {}}}).encode()
    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *_args: _Opener(payload)
    )
    output = tmp_path / "bundle.json"
    digest = hashlib.sha256(payload).hexdigest()

    assert fetch(url="https://example.test/input.json", expected_sha256=digest, output=output) == digest
    assert output.read_bytes() == payload

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch(url="https://example.test/input.json", expected_sha256="0" * 64, output=output)
    assert output.read_bytes() == payload


def test_fetch_rejects_sidecar_csv_bundle(monkeypatch, tmp_path: Path):
    payload = json.dumps({"total_return_prices_csv": "prices.csv"}).encode()
    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *_args: _Opener(payload)
    )
    with pytest.raises(ValueError, match="must inline"):
        fetch(
            url="https://example.test/input.json",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            output=tmp_path / "bundle.json",
        )
