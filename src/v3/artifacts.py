"""Complete, redacted audit bundles for every v3 runtime outcome."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


REQUIRED_ARTIFACTS = (
    "manifest.json",
    "preflight.json",
    "targets.json",
    "orders.json",
    "reconciliation.json",
    "performance.json",
    "run.log",
)

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SENSITIVE_KEY_PARTS = (
    "password",
    "secret",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "account_id",
    "database_url",
    "postgres_url",
)
_SECRET_URL = re.compile(
    r"(?i)\b(?:postgres(?:ql)?(?:\+\w+)?|redis|mysql|https?)://[^\s\"']+"
)


class ArtifactError(RuntimeError):
    pass


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


class ArtifactWriter:
    """Write one atomic and complete artifact directory.

    The workflow is expected to reject an incomplete directory.  The writer
    applies key-based and value-based redaction before data reaches disk.
    """

    def __init__(
        self,
        root: str | Path = "artifacts",
        *,
        secret_values: Iterable[str] = (),
    ) -> None:
        self.root = Path(root)
        self._secret_values = tuple(
            sorted(
                {value for value in secret_values if value and len(value) >= 4},
                key=len,
                reverse=True,
            )
        )

    @classmethod
    def from_environment(cls, root: str | Path = "artifacts") -> "ArtifactWriter":
        secret_names = (
            "ALPACA_API_KEY",
            "ALPACA_SECRET_KEY",
            "DATABASE_URL",
            "POSTGRES_URL",
            "OPENAI_API_KEY",
        )
        return cls(root, secret_values=(os.getenv(name, "") for name in secret_names))

    def write_outcome(
        self,
        run_id: str,
        *,
        manifest: Mapping[str, Any],
        preflight: Mapping[str, Any] | None = None,
        targets: Mapping[str, Any] | None = None,
        orders: Mapping[str, Any] | None = None,
        reconciliation: Mapping[str, Any] | None = None,
        performance: Mapping[str, Any] | None = None,
        log_lines: Iterable[str] = (),
    ) -> Path:
        if not _SAFE_RUN_ID.fullmatch(run_id):
            raise ArtifactError("run id contains unsafe path characters")
        run_dir = (self.root / run_id).resolve()
        root = self.root.resolve()
        if root != run_dir.parent:
            raise ArtifactError("artifact path escaped its configured root")
        run_dir.mkdir(parents=True, exist_ok=True)

        payloads = {
            "preflight.json": preflight or {},
            "targets.json": targets or {},
            "orders.json": orders or {"intents": [], "events": []},
            "reconciliation.json": reconciliation or {"required": False, "unresolved": []},
            "performance.json": performance or {"available": False},
        }
        for filename, payload in payloads.items():
            self._write_json(run_dir / filename, payload)

        safe_lines = [self._redact_text(str(line).rstrip("\n")) for line in log_lines]
        if not safe_lines:
            safe_lines = ["AegisQuant v3 run completed without additional log messages."]
        self._write_text(run_dir / "run.log", "\n".join(safe_lines) + "\n")

        hashes = {
            filename: self._sha256(run_dir / filename)
            for filename in REQUIRED_ARTIFACTS
            if filename != "manifest.json"
        }
        full_manifest = {
            **dict(manifest),
            "run_id": run_id,
            "artifact_schema": "aegisquant.v3.run-artifacts.v1",
            "required_files": list(REQUIRED_ARTIFACTS),
            "sha256": hashes,
            "complete": True,
        }
        self._write_json(run_dir / "manifest.json", full_manifest)
        self.verify(run_dir)
        return run_dir

    @staticmethod
    def verify(run_dir: str | Path) -> None:
        directory = Path(run_dir)
        missing = [name for name in REQUIRED_ARTIFACTS if not (directory / name).is_file()]
        empty = [
            name
            for name in REQUIRED_ARTIFACTS
            if (directory / name).is_file() and (directory / name).stat().st_size == 0
        ]
        if missing or empty:
            raise ArtifactError(f"incomplete artifact bundle; missing={missing}, empty={empty}")

    def _redact(self, value: Any, *, key: str = "") -> Any:
        if any(part in key.lower() for part in _SENSITIVE_KEY_PARTS):
            return "[REDACTED]"
        value = _json_value(value)
        if isinstance(value, Mapping):
            return {str(k): self._redact(v, key=str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact(item, key=key) for item in value]
        if isinstance(value, str):
            return self._redact_text(value)
        return value

    def _redact_text(self, value: str) -> str:
        redacted = value
        for secret in self._secret_values:
            redacted = redacted.replace(secret, "[REDACTED]")
        return _SECRET_URL.sub("[REDACTED_URL]", redacted)

    def _write_json(self, path: Path, payload: Any) -> None:
        safe = self._redact(payload)
        self._write_text(
            path,
            json.dumps(safe, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["ArtifactError", "ArtifactWriter", "REQUIRED_ARTIFACTS"]
