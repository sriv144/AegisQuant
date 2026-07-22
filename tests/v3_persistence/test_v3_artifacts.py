from __future__ import annotations

import json

import pytest

from src.v3.artifacts import ArtifactError, ArtifactWriter, REQUIRED_ARTIFACTS


def test_artifact_writer_always_emits_complete_redacted_bundle(tmp_path) -> None:
    writer = ArtifactWriter(tmp_path, secret_values=("alpaca-secret",))
    run_dir = writer.write_outcome(
        "gha-123-1",
        manifest={"mode": "shadow", "database_url": "postgresql://user:pass@db/aegis"},
        preflight={"account_id": "raw-account", "account_key": "hashed-account"},
        targets={"weights": {"SPY": "0.69"}},
        log_lines=("failed with alpaca-secret at https://secret.example/path",),
    )

    assert {path.name for path in run_dir.iterdir()} == set(REQUIRED_ARTIFACTS)
    ArtifactWriter.verify(run_dir)
    all_text = "\n".join(path.read_text(encoding="utf-8") for path in run_dir.iterdir())
    assert "alpaca-secret" not in all_text
    assert "raw-account" not in all_text
    assert "postgresql://" not in all_text
    assert "secret.example" not in all_text
    assert "hashed-account" in all_text
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert set(manifest["sha256"]) == set(REQUIRED_ARTIFACTS) - {"manifest.json"}


def test_artifact_writer_rejects_path_traversal(tmp_path) -> None:
    with pytest.raises(ArtifactError):
        ArtifactWriter(tmp_path).write_outcome("../escape", manifest={})
