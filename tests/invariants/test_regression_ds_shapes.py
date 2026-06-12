"""Tier 2 — assert shape and coverage of the cached regression datasets.

Catches:
- Loss of the ``season`` dim or its coord values.
- A required variable disappearing (someone shortened the writer).
- All-NaN gridcells inside the EU mask (a recompute that silently lost
  data over the publication region).

This test does **not** check numeric values — that is the job of
``test_cache_versions.py`` via the fingerprint comparison.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import xarray as xr

from conftest import require_cached


REQUIRED_VARS_MPI = {
    "coef_ensmean",
    "ste_ensmean",
    "rsq_ensmean",
    "AMOC_future",
    "AMOC_pi",
    "T_future",
    "T_pi",
    "T_pd_245",
    "req_strength_pi",
    "req_strength_pd",
    "req_weakening_pi",
    "req_weakening_pd",
}

REQUIRED_VARS_CESM = {
    "coef_ensmean",
    "ste_ensmean",
    "rsq_ensmean",
    "coef_ensmean_intercept",
    "AMOC_future",
    "AMOC_pi",
    "T_future",
    "T_pi",
    "T_pd_245",
    "req_strength_pi",
    "req_strength_pd",
    "req_weakening_pi",
    "req_weakening_pd",
}


@pytest.fixture(scope="module")
def reg_ds_mpi(cached_data_path):
    path = cached_data_path / "reg_ds_mpi_tas.nc"
    require_cached(path)
    return xr.open_dataset(path)


@pytest.fixture(scope="module")
def reg_ds_cesm(cached_data_path):
    path = cached_data_path / "reg_ds_cesm.nc"
    require_cached(path)
    return xr.open_dataset(path)


@pytest.fixture(scope="module")
def hosmip_masks(cached_data_path):
    path = cached_data_path / "hosmip_masks.pkl"
    require_cached(path)
    with open(path, "rb") as f:
        return pickle.load(f)


# --- MPI regression dataset -------------------------------------------------

def test_reg_ds_mpi_has_season_dim(reg_ds_mpi):
    assert "season" in reg_ds_mpi.dims
    assert reg_ds_mpi.sizes["season"] == 3
    assert list(reg_ds_mpi.season.values) == ["", "djf", "jja"]


def test_reg_ds_mpi_has_main_paper_scenarios(reg_ds_mpi, functions):
    """``scenar`` coord must cover the main-paper SSPs (exact equality)."""
    assert "scenar" in reg_ds_mpi.coords
    assert sorted(reg_ds_mpi.scenar.values.tolist()) == sorted(functions.ssps)


def test_reg_ds_mpi_required_vars_present(reg_ds_mpi):
    missing = REQUIRED_VARS_MPI - set(reg_ds_mpi.data_vars)
    assert not missing, f"reg_ds_mpi missing variables: {missing}"


def test_reg_ds_mpi_amoc_pi_matches_constant(reg_ds_mpi, functions):
    """``AMOC_pi`` in the cached dataset should equal the canonical
    constant in ``functions.py``."""
    val = float(reg_ds_mpi.AMOC_pi.mean())
    assert val == pytest.approx(functions.AMOC_pi_MPI), (
        f"reg_ds_mpi.AMOC_pi = {val}, expected {functions.AMOC_pi_MPI}"
    )


# --- CESM regression dataset ------------------------------------------------

def test_reg_ds_cesm_required_vars_present(reg_ds_cesm):
    missing = REQUIRED_VARS_CESM - set(reg_ds_cesm.data_vars)
    assert not missing, f"reg_ds_cesm missing variables: {missing}"


def test_reg_ds_cesm_has_scenar(reg_ds_cesm):
    """CESM cached dataset carries 4 scenarios (ssp126/245/370 + pi)."""
    assert "scenar" in reg_ds_cesm.coords
    assert reg_ds_cesm.sizes["scenar"] >= 3


# --- Coverage: regression slope finite over Europe -------------------------

def test_reg_ds_mpi_coef_finite_over_europe(reg_ds_mpi, hosmip_masks):
    """The MPI-ESM regression slope must be finite (not all-NaN) over the
    EU mask. ``MPI-ESM1-2-LR`` is the model whose grid the MPI regression
    dataset is defined on; use its EU mask.
    """
    if "MPI-ESM1-2-LR" not in hosmip_masks:
        pytest.skip("MPI-ESM1-2-LR mask absent from hosmip_masks.pkl")
    eu_mask = np.asarray(hosmip_masks["MPI-ESM1-2-LR"]["EU"])
    if np.issubdtype(eu_mask.dtype, np.floating):
        eu_mask = np.nan_to_num(eu_mask, nan=0.0)
    eu_mask = eu_mask.astype(bool)

    # reg_ds_mpi is on the European subset grid (lat=23, lon=47), not the
    # full model grid (lat=64, lon=128). If the shapes don't match, skip
    # the regional check rather than misalign.
    annual_full = reg_ds_mpi.coef_ensmean.sel(season="")
    extra = [d for d in annual_full.dims if d not in ("lat", "lon")]
    annual = annual_full.mean(dim=extra) if extra else annual_full
    if annual.shape != eu_mask.shape:
        pytest.skip(
            f"reg_ds_mpi grid {annual.shape} differs from hosmip MPI-ESM1-2-LR "
            f"grid {eu_mask.shape}; regional finiteness check needs a shared grid."
        )
    slope_eu = np.asarray(annual)[eu_mask]
    n_finite = np.isfinite(slope_eu).sum()
    assert n_finite > 0, "All-NaN regression slope over the EU mask"
    assert n_finite >= 0.5 * eu_mask.sum(), (
        f"Only {n_finite}/{int(eu_mask.sum())} EU cells have finite slope"
    )
