"""Persistence helpers for immutable v3 data provenance."""

from __future__ import annotations

import hashlib
from datetime import UTC

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from src.db.v3_models import DataManifestRecord
from src.v3.data import DataManifest


def _same_time(left, right) -> bool:
    if left.tzinfo is None:
        left = left.replace(tzinfo=UTC)
    if right.tzinfo is None:
        right = right.replace(tzinfo=UTC)
    return left.astimezone(UTC) == right.astimezone(UTC)


class DataManifestStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def persist(self, manifests: tuple[DataManifest, ...]) -> tuple[str, ...]:
        ids: list[str] = []
        with Session(self.engine) as session:
            for manifest in manifests:
                manifest_id = hashlib.sha256(
                    "|".join(
                        (
                            manifest.dataset,
                            manifest.sha256,
                            manifest.frozen_at.isoformat(),
                        )
                    ).encode("utf-8")
                ).hexdigest()
                existing = session.scalar(
                    select(DataManifestRecord).where(
                        DataManifestRecord.dataset_name == manifest.dataset,
                        DataManifestRecord.sha256 == manifest.sha256,
                    )
                )
                if existing is None:
                    session.add(
                        DataManifestRecord(
                            manifest_id=manifest_id,
                            dataset_name=manifest.dataset,
                            data_tier=manifest.tier.value,
                            source=manifest.source,
                            availability_at=manifest.availability_at,
                            freeze_at=manifest.frozen_at,
                            row_count=manifest.row_count,
                            coverage=manifest.coverage,
                            sha256=manifest.sha256,
                            validation_status=(
                                "promotable" if manifest.promotable else "research_only"
                            ),
                            warnings_json=list(manifest.warnings),
                        )
                    )
                elif (
                    existing.source != manifest.source
                    or existing.data_tier != manifest.tier.value
                    or not _same_time(existing.availability_at, manifest.availability_at)
                    or not _same_time(existing.freeze_at, manifest.frozen_at)
                    or existing.row_count != manifest.row_count
                    or float(existing.coverage) != manifest.coverage
                    or existing.validation_status
                    != ("promotable" if manifest.promotable else "research_only")
                    or list(existing.warnings_json or []) != list(manifest.warnings)
                ):
                    raise ValueError(
                        f"manifest identity collision for {manifest.dataset}/{manifest.sha256}"
                    )
                ids.append(existing.manifest_id if existing is not None else manifest_id)
            session.commit()
        return tuple(ids)


__all__ = ["DataManifestStore"]
