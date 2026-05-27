"""Project-root conftest.

Makes the repository root pytest's rootdir so the `src.*` import
layout resolves without requiring an editable install. Pytest's
default --import-mode=prepend adds this file's parent directory to
sys.path; we also do it explicitly for tooling that imports tests
outside of pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
