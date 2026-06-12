"""Derive AMOC@26.5°N annual-mean for MRI-ESM2-0 ssp126 r2-r5.

Adds r2-r5 to the ensemble that previously contained only r1 (sourced
from /work/uo1075/m300817/.../ssp126/mri-esm2-0/). Raw msftmz chunks
come from the ESGF-West / ALCF replica:

  https://g-52ba3.fd635.8443.data.globus.org/css03_data/CMIP6/...

Layout of raw chunks (already downloaded into bu1431, one chunk per
member, 2015-01 to 2100-12):

  /work/bu1431/T_EU_AMOC/CMIP6/mri-esm2-0/ssp126/r{N}i1p1f1/Omon/msftmz/gr2z/{VERSION}/

Versions:
  r1  → v20200222 (matches /pool replica; downloaded so the gate
        exercises ESGF end-to-end and is byte-identical to /pool)
  r2-r5 → v20210910 (later MRI publication of additional members;
        not available on /pool)

Output files (mirror the legacy r1 naming/layout):

  /work/bu1431/T_EU_AMOC/CMIP6/mri-esm2-0/ssp126/processed/
      mri-esm2-0_r{N}i1p1f1_amoc26_yr.nc

Caveat: the downstream loader in scripts/cmip_cooling.py
currently reads `/work/uo1075/m300817/teu_amoc/data/CMIP6/...`. Wiring
this new bu1431 location in is a separate edit; this script only
produces the cache and runs all verification gates.

Run from scripts/ as
    python process_mri_ssp126_amoc.py
or interactively with VS Code cells.
"""

# %% imports + config ---------------------------------------------------------
import os
import hashlib

import numpy as np
import xarray as xr


SSP = "ssp126"
VER_R1   = "v20200222"
VER_NEW  = "v20210910"
MEMBERS_ALL = [f"r{i}i1p1f1" for i in (1, 2, 3, 4, 5)]
MEMBERS_NEW = [f"r{i}i1p1f1" for i in (2, 3, 4, 5)]

BU1431_BASE = "/work/bu1431/T_EU_AMOC/CMIP6/mri-esm2-0/ssp126"
POOL_R1 = (
    "/pool/data/CMIP6/data/ScenarioMIP/MRI/MRI-ESM2-0/ssp126/"
    "r1i1p1f1/Omon/msftmz/gr2z/v20200222/"
    "msftmz_Omon_MRI-ESM2-0_ssp126_r1i1p1f1_gr2z_201501-210012.nc"
)
LEGACY_R1 = (
    "/work/uo1075/m300817/teu_amoc/data/CMIP6/ssp126/mri-esm2-0/"
    "mri-esm2-0_r1i1p1f1_amoc26_yr.nc"
)
OUT_DIR = f"{BU1431_BASE}/processed"
os.makedirs(OUT_DIR, exist_ok=True)


def _esgf_chunk_path(member):
    v = VER_R1 if member == "r1i1p1f1" else VER_NEW
    return (f"{BU1431_BASE}/{member}/Omon/msftmz/gr2z/{v}/"
            f"msftmz_Omon_MRI-ESM2-0_ssp126_{member}_gr2z_201501-210012.nc")


# %% weighted month-to-year mean (same as process_giss_ssp126_amoc) ----------
def weighted_mon_to_year_mean(ds, var):
    """Annual mean weighted by days in each month."""
    month_length = ds.time.dt.days_in_month
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)
    obs = ds[var]
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)
    obs_sum = (obs * wgts).resample(time="YS").sum(dim="time")
    ones_out = (ones * wgts).resample(time="YS").sum(dim="time")
    return obs_sum / ones_out


def derive_amoc26_yr(msftmz_path):
    """msftmz at gr2z → basin=0 (atlantic_arctic_ocean), lat=26.5 nearest →
    days-in-month-weighted annual mean → idxmax over lev → /(1025*1e6).
    Returns DataArray on (time,) with a 'lev' time-varying coord."""
    ds = xr.open_dataset(msftmz_path, use_cftime=True)
    msftmz_sub = ds.sel(basin=0, drop=True).sel(lat=26.5, method="nearest", drop=True)
    amoc_yr = weighted_mon_to_year_mean(msftmz_sub, "msftmz")
    max_lev_idx = amoc_yr.idxmax(dim="lev")
    amoc_yr = amoc_yr.sel(lev=max_lev_idx, drop=True) / (1025.0 * 1e6)
    return amoc_yr.load()


def md5(path, bs=2**20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bs), b""):
            h.update(chunk)
    return h.hexdigest()


# %% Gate A — md5 r1 ESGF vs /pool --------------------------------------------
if __name__ == "__main__":
    print("=== Gate A: md5(r1 ESGF) vs md5(r1 /pool) ===")
    h_esgf = md5(_esgf_chunk_path("r1i1p1f1"))
    h_pool = md5(POOL_R1)
    print(f"  ESGF  : {h_esgf}")
    print(f"  /pool : {h_pool}")
    assert h_esgf == h_pool, (
        "Gate A FAILED: ESGF replica differs from /pool master. Stop here.")
    print("  Gate A PASSED — ESGF replica is bit-identical to /pool.")


# %% Gate B — recipe via /pool, compare to legacy uo1075 cache ----------------
if __name__ == "__main__":
    print()
    print("=== Gate B: derive r1 from /pool, compare to legacy uo1075 cache ===")
    new_r1_pool = derive_amoc26_yr(POOL_R1)
    legacy = xr.open_dataarray(LEGACY_R1, use_cftime=True)
    new_aligned = new_r1_pool.assign_coords(time=legacy.time)
    diff_B = float(np.abs(legacy.values - new_aligned.values).max())
    print(f"  n_time legacy={legacy.size}  new={new_r1_pool.size}")
    print(f"  legacy mean         = {float(legacy.mean()):.6f} Sv")
    print(f"  new(/pool) mean     = {float(new_r1_pool.mean()):.6f} Sv")
    print(f"  max|new-legacy|     = {diff_B:.3e} Sv")
    assert diff_B < 1e-6, (
        f"Gate B FAILED: recipe drift > 1e-6 Sv ({diff_B:.3e}). Stop here.")
    print("  Gate B PASSED — recipe matches m300817's.")


# %% Gate C — recipe via ESGF r1, compare to legacy --------------------------
if __name__ == "__main__":
    print()
    print("=== Gate C: derive r1 from ESGF chunk, compare to legacy ===")
    new_r1_esgf = derive_amoc26_yr(_esgf_chunk_path("r1i1p1f1"))
    legacy = xr.open_dataarray(LEGACY_R1, use_cftime=True)
    new_aligned = new_r1_esgf.assign_coords(time=legacy.time)
    diff_C = float(np.abs(legacy.values - new_aligned.values).max())
    print(f"  legacy mean         = {float(legacy.mean()):.6f} Sv")
    print(f"  new(ESGF) mean      = {float(new_r1_esgf.mean()):.6f} Sv")
    print(f"  max|new-legacy|     = {diff_C:.3e} Sv")
    assert diff_C < 1e-6, (
        f"Gate C FAILED: ESGF→recipe drift > 1e-6 Sv ({diff_C:.3e}). Stop here.")
    print("  Gate C PASSED — ESGF end-to-end matches legacy.")


# %% Derive r2-r5 and write output --------------------------------------------
if __name__ == "__main__":
    print()
    print("=== Derive r2-r5 amoc26_yr ===")
    for member in MEMBERS_NEW:
        out = f"{OUT_DIR}/mri-esm2-0_{member}_amoc26_yr.nc"
        if os.path.exists(out):
            print(f"  exists, skipping: {out}")
            continue
        amoc_yr = derive_amoc26_yr(_esgf_chunk_path(member))
        # Match legacy file layout: a single unnamed data variable + 'lev' coord on time.
        amoc_yr.to_netcdf(out)
        print(f"  wrote {out}")
        print(f"    n_time={amoc_yr.size}, mean={float(amoc_yr.mean()):.4f} Sv, "
              f"min={float(amoc_yr.min()):.4f}, max={float(amoc_yr.max()):.4f}")
        # NaN check
        n_nan = int(np.isnan(amoc_yr.values).sum())
        assert n_nan == 0, f"  {member}: {n_nan} NaN values"


# %% Sanity plot --------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    PLOT_PATH = "/home/m/m300940/teu_amoc/plots/mri_ssp126_amoc_ensemble.png"
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = cm.viridis(np.linspace(0.1, 0.9, len(MEMBERS_ALL)))
    for c, m in zip(colors, MEMBERS_ALL):
        if m == "r1i1p1f1":
            da = xr.open_dataarray(LEGACY_R1, use_cftime=True)
            tag = " (legacy uo1075)"
        else:
            da = xr.open_dataarray(f"{OUT_DIR}/mri-esm2-0_{m}_amoc26_yr.nc",
                                   use_cftime=True)
            tag = " (ESGF v20210910)"
        try:
            years = da.time.dt.year.values
        except Exception:
            years = np.array([t.year for t in da.time.values])
        ax.plot(years, da.values, color=c, alpha=0.35, lw=0.6)
        ax.plot(years, da.rolling(time=10, center=True).mean().values,
                color=c, lw=2.0, label=f"{m}{tag}")
    ax.set_xlabel("year")
    ax.set_ylabel("AMOC @ 26.5°N (Sv)")
    ax.set_title("MRI-ESM2-0 ssp126 — AMOC@26.5°N, 5 members")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"\nsanity plot → {PLOT_PATH}")
