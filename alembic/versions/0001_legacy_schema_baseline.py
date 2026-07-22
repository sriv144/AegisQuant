"""Baseline the pre-v3 legacy schema without altering or repurposing it.

Revision ID: 0001_legacy_baseline
Revises:
"""

from typing import Sequence


revision: str = "0001_legacy_baseline"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Existing deployments stamp this revision.  Fresh v3 databases deliberately
    # do not recreate the India-oriented legacy tables.
    pass


def downgrade() -> None:
    pass
