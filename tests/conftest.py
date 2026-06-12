"""Shared fixtures for the teu_amoc invariant test suite.

Two flavours of fixture:
- ``functions`` imports the project library once per session and shares it.
- ``cached_data_path`` resolves the on-disk cache directory and lets tests
  skip cleanly when a cached file is missing rather than erroring.

Baseline-update mode: set the env var ``TEU_AMOC_UPDATE_BASELINES=1`` and
run ``pytest`` to regenerate fingerprint baselines instead of asserting
against them. The gate exists so a baseline can never be silently
rewritten by an accidental ``pytest -k baseline`` invocation.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
CACHED_DATA_DIR = REPO_ROOT / "data"
BASELINES_DIR = Path(__file__).resolve().parent / "baselines"


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def scripts_dir():
    return SCRIPTS_DIR


@pytest.fixture(scope="session")
def cached_data_path():
    return CACHED_DATA_DIR


@pytest.fixture(scope="session")
def baselines_dir():
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    return BASELINES_DIR


@pytest.fixture(scope="session")
def functions():
    """Import ``scripts/functions.py`` once per session.

    We add ``scripts/`` to ``sys.path`` rather than installing the project
    as a package — matches the way figure scripts import it.
    """
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import functions as _functions  # noqa: E402

    return _functions


@pytest.fixture(scope="session")
def update_baselines() -> bool:
    return os.environ.get("TEU_AMOC_UPDATE_BASELINES", "") == "1"


def require_cached(path: Path) -> None:
    """Skip a test cleanly when a cached artefact is missing.

    Tier-2 tests run on the user's local cache. If a file is absent
    (e.g. on a fresh clone before the canonical pipeline has run), skip
    rather than fail — the test exists to verify the *contents* of a
    cache, not its presence.
    """
    if not path.exists():
        pytest.skip(f"cached artefact missing: {path}")
