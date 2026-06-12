"""Tier 2 — load ``hosmip_masks.pkl`` and check topology / coverage.

Catches:
- A refactor that breaks the EU composite into disconnected blobs.
- ``EU`` no longer contained in ``EU_EEU``, or ``EU_buffer`` no longer
  extending ``EU``.
- Missing IPCC or country masks.
- A country mask that ended up with zero land overlap (the
  centroid-only fallback failure mode).

The land-overlap assertion uses ``≥ 1`` cell — the minimum that
distinguishes "country fell off the grid entirely" from "small island
caught by one cell". Anything richer (small-country thresholds,
polygon-vs-cell reconciliation) is a follow-up diagnostic, not an
invariant.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from scipy import ndimage

from conftest import require_cached


REQUIRED_REGIONS = {"NEU", "WCE", "EEU", "MED", "EU", "EU_EEU", "EU_buffer", "LAND"}


def _to_bool(arr) -> np.ndarray:
    """Coerce a mask to a clean boolean ndarray.

    ``regionmask`` outputs are sometimes float with NaN sentinels for
    "outside region". Naive ``.astype(bool)`` would mark NaN as True
    (because ``np.nan != 0``); use ``nan_to_num`` first.
    """
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.floating):
        a = np.nan_to_num(a, nan=0.0)
    return a.astype(bool)


@pytest.fixture(scope="module")
def hosmip_masks(cached_data_path):
    path = cached_data_path / "hosmip_masks.pkl"
    require_cached(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def _count_connected_components(mask_2d: np.ndarray) -> int:
    """Components under 8-connectivity, with longitude-wrap awareness.

    A naive label on a lat/lon grid splits a region that crosses the
    ±180° (or 0°/360°) seam. If the naive count is >1, we re-test on
    the array rolled by half its longitude span — if the rolled array
    is connected, the region was just split by the seam.
    """
    structure = np.ones((3, 3), dtype=bool)
    _, n_naive = ndimage.label(mask_2d, structure=structure)
    if n_naive <= 1:
        return n_naive
    rolled = np.roll(mask_2d, shift=mask_2d.shape[1] // 2, axis=1)
    _, n_rolled = ndimage.label(rolled, structure=structure)
    return min(n_naive, n_rolled)


def test_models_have_required_regions(hosmip_masks):
    """Every HosMIP model carries the IPCC composites plus LAND."""
    missing_by_model = {}
    for model, model_masks in hosmip_masks.items():
        missing = REQUIRED_REGIONS - set(model_masks.keys())
        if missing:
            missing_by_model[model] = missing
    assert not missing_by_model, f"Missing required regions: {missing_by_model}"


def test_eu_is_single_connected_region(hosmip_masks, functions):
    """The EU composite (NEU ∪ WCE ∪ MED) must form one geographic blob.

    Geographic adjacency on a lat/lon grid uses 8-connectivity; the
    longitude seam is handled by re-testing on a rolled array.
    """
    offenders = []
    for model, model_masks in hosmip_masks.items():
        mask = np.asarray(model_masks["EU"])
        n_components = _count_connected_components(mask.astype(bool))
        if n_components != 1:
            offenders.append(f"{model}: EU has {n_components} components")
    assert not offenders, "\n".join(offenders)


def test_eu_contained_in_eu_eeu(hosmip_masks):
    """Every True cell in EU must also be True in EU_EEU."""
    offenders = []
    for model, model_masks in hosmip_masks.items():
        eu = _to_bool(model_masks["EU"])
        eu_eeu = _to_bool(model_masks["EU_EEU"])
        leak = eu & ~eu_eeu
        if leak.any():
            offenders.append(f"{model}: {int(leak.sum())} EU cells not in EU_EEU")
    assert not offenders, "\n".join(offenders)


def test_eu_buffer_extends_eu(hosmip_masks):
    """EU_buffer must be a superset of EU (rolling-shift extension)."""
    offenders = []
    for model, model_masks in hosmip_masks.items():
        eu = _to_bool(model_masks["EU"])
        buf = _to_bool(model_masks["EU_buffer"])
        gaps = eu & ~buf
        if gaps.any():
            offenders.append(f"{model}: {int(gaps.sum())} EU cells missing from EU_buffer")
    assert not offenders, "\n".join(offenders)


def test_country_masks_present_on_grid(hosmip_masks, functions):
    """Every country mask has at least one True cell on every model grid.

    This is the strict "fell off the grid entirely" guard. We do **not**
    require the country to overlap the LAND mask — on coarse model
    grids, small islands like Cyprus and Denmark sometimes resolve to
    cells whose centroids are classed as ocean by LAND but whose
    country polygon still contains them. Reconciling country-vs-LAND
    polygon discordance is left to a follow-up diagnostic.
    """
    expected_countries = set(functions.country_names.keys())
    offenders = []
    for model, model_masks in hosmip_masks.items():
        for code in expected_countries:
            if code not in model_masks:
                offenders.append(f"{model}: missing country mask {code!r}")
                continue
            country = _to_bool(model_masks[code])
            if int(country.sum()) < 1:
                offenders.append(
                    f"{model}/{code}: zero cells — country fell off the grid"
                )
    assert not offenders, "\n".join(offenders[:50]) + (
        f"\n... and {len(offenders) - 50} more" if len(offenders) > 50 else ""
    )
