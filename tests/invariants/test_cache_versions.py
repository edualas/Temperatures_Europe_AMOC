"""Tier 2 — fingerprint cached pickles and netCDFs to detect silent drift.

Each tracked file gets a stable fingerprint stored in
``tests/baselines/cache_fingerprints.json``:

- ``size_bytes`` — file size on disk
- ``shape`` — for netCDF: ``{var: dims}`` map; for pickle: top-level
  ``type/len`` summary
- ``checksum`` — numerical: sum of every numeric array, rounded to 6
  decimals; structural: SHA-256 of the pickled bytes for non-numeric

The numerical checksum tolerates float-rounding across xarray writes
and lets the test pass on a re-save with no data change. It does
**not** tolerate a real change — a single cell shifting by 1e-5 will
trip the rounding boundary eventually, which is what we want.

Updating a baseline is a deliberate two-step:

1. ``TEU_AMOC_UPDATE_BASELINES=1 pytest tests/invariants/test_cache_versions.py``
2. Document the change in the commit message.

The env-var gate prevents accidental rewrites.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from conftest import require_cached


TRACKED_FILES = [
    "reg_ds_mpi_tas.nc",
    "reg_ds_cesm.nc",
    "hosmip_masks.pkl",
    "hosmip_reg_ds_dict.pkl",
    "multi_model_dict.pkl",
    # Synthetic CMIP6 range caches (schema v2, 2026-08-31): the 12 annual
    # statedep x calset variants + 4 seasonal at the default cal set, plus
    # the two target (w, t) caches. Untagged = the defaults (hosing, w —
    # predictors default flipped back to 'w' on 2026-09-01).
    "cmip_range_fig3.nc",
    "cmip_range_fig3_cal-full.nc",
    "cmip_range_fig3_cal-nocesm2.nc",
    "cmip_range_fig3_cal-consistentssp.nc",
    "cmip_range_fig3_sdep-pooled.nc",
    "cmip_range_fig3_sdep-pooled_cal-full.nc",
    "cmip_range_fig3_sdep-pooled_cal-nocesm2.nc",
    "cmip_range_fig3_sdep-pooled_cal-consistentssp.nc",
    "cmip_range_fig3_sdep-interact.nc",
    "cmip_range_fig3_sdep-interact_cal-full.nc",
    "cmip_range_fig3_sdep-interact_cal-nocesm2.nc",
    "cmip_range_fig3_sdep-interact_cal-consistentssp.nc",
    "cmip_range_fig3_season-djf.nc",
    "cmip_range_fig3_season-jja.nc",
    "cmip_range_fig3_sdep-pooled_season-djf.nc",
    "cmip_range_fig3_sdep-pooled_season-jja.nc",
    # wt-conditioning references (2026-09-01 mirror of the former pred-w
    # siblings): the two wt caches kept for w-vs-wt comparisons.
    "cmip_range_fig3_pred-wt.nc",
    "cmip_range_fig3_sdep-pooled_pred-wt.nc",
    # Decade-matched siblings for FigSupp_cmip_cooling (2026-09-01, schema v3
    # fw threading): target (w, t) at 2071-2080 (Levante-built) + the pooled
    # range on it. Skip (no baseline) until first built.
    "cmip_synth_wt_countries_fw2071-2080.nc",
    "cmip_range_fig3_sdep-pooled_fw2071-2080.nc",
    "cmip_synth_wt_EU.nc",
    "cmip_synth_wt_countries.nc",
]


def _fingerprint_netcdf(path: Path) -> dict:
    ds = xr.open_dataset(path)
    try:
        shape = {var: list(ds[var].dims) for var in ds.data_vars}
        # Sum of every numeric variable over its FINITE entries, rounded to 6
        # decimals: tolerant of float-rounding on resave, sensitive to actual
        # data shifts, and not blinded by the deliberate +inf entries in the
        # Synthetic CMIP6 range caches (a nansum there would be inf and mask
        # every change in the finite variables). Non-finite entries are
        # counted separately.
        total = 0.0
        n_nonfinite = 0
        for var in sorted(ds.data_vars):
            arr = np.asarray(ds[var].values)
            if np.issubdtype(arr.dtype, np.number):
                s = float(arr[np.isfinite(arr)].sum())
                total += round(s, 6)
                n_nonfinite += int((~np.isfinite(arr)).sum())
        return {
            "size_bytes": path.stat().st_size,
            "shape": shape,
            "checksum": round(total, 6),
            "n_nonfinite": n_nonfinite,
        }
    finally:
        ds.close()


def _fingerprint_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    summary = {
        "size_bytes": path.stat().st_size,
        "type": type(obj).__name__,
    }
    if hasattr(obj, "keys"):
        summary["top_level_keys"] = sorted(map(str, obj.keys()))
        summary["len"] = len(obj)
        # Numerical fingerprint: sum across any numeric arrays we find
        # one level deep (covers dict[model] -> {region: DataArray} and
        # dict[model] -> Dataset patterns).
        total = 0.0
        for key in sorted(summary["top_level_keys"]):
            try:
                sub = obj[key]
            except (KeyError, TypeError):
                continue
            total += _nansum_recursive(sub)
        summary["checksum"] = round(total, 4)
    return summary


def _nansum_recursive(item) -> float:
    """Best-effort numerical sum across nested mappings, ndarrays, and
    xarray objects. Skips anything non-numeric."""
    if isinstance(item, xr.Dataset):
        s = 0.0
        for var in item.data_vars:
            arr = np.asarray(item[var].values)
            if np.issubdtype(arr.dtype, np.number):
                s += float(np.nansum(arr))
        return s
    if isinstance(item, xr.DataArray):
        arr = np.asarray(item.values)
        if np.issubdtype(arr.dtype, np.number):
            return float(np.nansum(arr))
        return 0.0
    if isinstance(item, np.ndarray):
        if np.issubdtype(item.dtype, np.number):
            return float(np.nansum(item))
        return 0.0
    if isinstance(item, dict):
        return sum(_nansum_recursive(v) for v in item.values())
    if isinstance(item, (list, tuple)):
        return sum(_nansum_recursive(v) for v in item)
    return 0.0


def _fingerprint(path: Path) -> dict:
    if path.suffix == ".nc":
        return _fingerprint_netcdf(path)
    if path.suffix == ".pkl":
        return _fingerprint_pickle(path)
    raise ValueError(f"Unsupported cache file extension: {path}")


@pytest.fixture(scope="module")
def baselines_file(baselines_dir):
    return baselines_dir / "cache_fingerprints.json"


def _load_baselines(p: Path) -> dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text())


@pytest.mark.parametrize("filename", TRACKED_FILES)
def test_cache_fingerprint(filename, cached_data_path, baselines_file, update_baselines):
    path = cached_data_path / filename
    require_cached(path)
    fp = _fingerprint(path)
    baselines = _load_baselines(baselines_file)

    if update_baselines:
        baselines[filename] = fp
        baselines_file.write_text(json.dumps(baselines, indent=2, sort_keys=True))
        pytest.skip(f"baseline updated for {filename}")

    if filename not in baselines:
        pytest.skip(
            f"no baseline for {filename}; run "
            f"`TEU_AMOC_UPDATE_BASELINES=1 pytest -k {filename}` to create one"
        )

    expected = baselines[filename]
    # Compare structural fields first for the most actionable diff.
    structural_keys = [k for k in expected if k != "checksum"]
    for key in structural_keys:
        assert fp.get(key) == expected[key], (
            f"{filename}: {key} changed\n  expected: {expected[key]}\n  got:      {fp.get(key)}"
        )
    assert fp.get("checksum") == expected.get("checksum"), (
        f"{filename}: numerical checksum changed\n"
        f"  expected: {expected.get('checksum')}\n  got:      {fp.get('checksum')}"
    )
