"""
Stage and process GISS-E2-1-G ssp126 AMOC@26.5°N annual caches for
r2-r5 i1p1f2, lifting `reg_ds_giss.req_weakening_*.sel(scenar='ssp126')`
from a 1-member to a 5-member ensemble.

Why r2-r5: NCCS publishes post-erratum tas for r1-r5 (already staged
under cmip_proj_staging/ssp126/giss-e2-1-g/), but only r1 has an
existing msftmz-derived AMOC@26.5°N cache (legacy uo1075 archive,
identical tracking_id to the NCCS post-erratum file). r6-r10 don't
have an ssp126 simulation — only ssp245 was extended in the Romanou
et al. (2023) long-run ensemble.

Companion to scripts/_archive/process_giss_long_runs.py. Mirrors its
recipe (`weighted_mon_to_year_mean`, basin=0, lat=26.5 nearest, max
over lev, divide by rho*1e6) but parameterised for ssp126 and
parallelised over the 8 chunks (4 members × 2 time slices).

Run interactively cell-by-cell, or as a standalone script.
"""

# %% imports and configuration ------------------------------------------------
import os
import subprocess
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

PORTAL = (
    "https://portal.nccs.nasa.gov/datashare/giss_cmip6/"
    "ScenarioMIP/NASA-GISS/GISS-E2-1-G/ssp126"
)
VERSION = "v20200115"

RAW_DIR = "/work/bu1431/T_EU_AMOC/CMIP6/giss-e2-1-g/ssp126"
OUT_DIR = "/work/bu1431/T_EU_AMOC/CMIP6/giss-e2-1-g/cmip_proj_staging/ssp126/giss-e2-1-g"

# r1 is the existing single member (legacy uo1075 cache + DKRZ msftmz);
# r2-r5 are the new ones to download and process. r1 also gets
# re-derived from NCCS as a byte-level verification gate before we
# trust the recipe on r2-r5.
MEMBERS_NEW = [f"r{i}i1p1f2" for i in (2, 3, 4, 5)]
MEMBERS_ALL = [f"r{i}i1p1f2" for i in (1, 2, 3, 4, 5)]
CHUNKS = ["201501-205012", "205101-210012"]  # CMIP6 ssp126 covers 2015-2100 only

os.makedirs(OUT_DIR, exist_ok=True)
for r in MEMBERS_ALL:
    os.makedirs(f"{RAW_DIR}/{r}", exist_ok=True)


def filename(member, chunk):
    return f"msftmz_Omon_GISS-E2-1-G_ssp126_{member}_gn_{chunk}.nc"


def url(member, chunk):
    return (
        f"{PORTAL}/{member}/Omon/msftmz/gn/{VERSION}/"
        f"{filename(member, chunk)}"
    )


def local(member, chunk):
    return f"{RAW_DIR}/{member}/{filename(member, chunk)}"


# %% spider-check 5 members × 2 chunks = 10 URLs (parallel HEAD) -------------
def _spider(member, chunk):
    u = url(member, chunk)
    r = subprocess.run(["wget", "--spider", "-q", u], capture_output=True)
    return (member, chunk, u, r.returncode)


if __name__ == "__main__":
    tasks = [(m, c) for m in MEMBERS_ALL for c in CHUNKS]
    missing = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(_spider, m, c) for m, c in tasks]):
            m, c, u, rc = fut.result()
            if rc != 0:
                missing.append(u)
                print(f"MISSING: {u}")
    print(f"spider check: {len(missing)}/{len(tasks)} missing")
    assert not missing, "URL(s) missing from NCCS portal — investigate before proceeding"


# %% download all 10 chunks from NCCS portal in parallel ---------------------
def _wget(member, chunk):
    out = local(member, chunk)
    if os.path.exists(out):
        return (member, chunk, "skip")
    u = url(member, chunk)
    target_dir = f"{RAW_DIR}/{member}"
    r = subprocess.run(
        ["wget", "-nc", "-q", "--wait=0.5", "--random-wait", "-P", target_dir, u],
        capture_output=True,
    )
    return (member, chunk, "ok" if r.returncode == 0 else f"err rc={r.returncode}")


if __name__ == "__main__":
    print("downloading 10 msftmz chunks (5 members × 2) in parallel...")
    with ThreadPoolExecutor(max_workers=5) as ex:
        for fut in as_completed([ex.submit(_wget, m, c) for m, c in tasks]):
            m, c, status = fut.result()
            print(f"  {m} {c}: {status}")
    # sanity: every chunk should now be on disk
    for m, c in tasks:
        assert os.path.exists(local(m, c)), f"missing after download: {local(m, c)}"
    print("download complete")


# %% weighted month-to-year mean (copied from _archive/process_giss_long_runs) -
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


def derive_amoc26_yr(member):
    """msftmz → basin=0, lat=26.5 nearest → days-in-month annual mean →
    max over lev → divide by rho*1e6. Returns DataArray on (time,)."""
    paths = sorted(glob(f"{RAW_DIR}/{member}/msftmz_Omon_*.nc"))
    assert len(paths) == len(CHUNKS), (
        f"{member}: expected {len(CHUNKS)} msftmz chunks, got {len(paths)}")
    msftmz = xr.open_mfdataset(paths, parallel=False, use_cftime=True)
    amoc_yr = weighted_mon_to_year_mean(
        msftmz.sel(basin=0, drop=True).sel(lat=26.5, method="nearest", drop=True),
        "msftmz",
    )
    max_lev_idx = amoc_yr.idxmax(dim="lev")
    amoc_yr = amoc_yr.sel(lev=max_lev_idx, drop=True) / (1025.0 * 1e6)
    return amoc_yr.load()


# %% verification gate: re-derive r1 from NCCS, compare to legacy uo1075 -----
if __name__ == "__main__":
    LEGACY_R1 = (
        "/work/uo1075/m300817/teu_amoc/data/CMIP6/ssp126/giss-e2-1-g/"
        "giss-e2-1-g_r1i1p1f2_amoc26_yr.nc"
    )
    print(f"verification: re-derive r1 from NCCS msftmz...")
    new_r1 = derive_amoc26_yr("r1i1p1f2")
    legacy = xr.open_dataarray(LEGACY_R1, use_cftime=True)
    new_r1_aligned = new_r1.assign_coords(time=legacy.time)
    diff = np.abs(legacy.values - new_r1_aligned.values).max()
    print(f"  r1 max|new-legacy| = {diff:.3e} Sv  over n={legacy.size} years")
    print(f"  legacy mean      = {float(legacy.mean()):.4f} Sv")
    print(f"  new(NCCS) mean   = {float(new_r1.mean()):.4f} Sv")
    assert diff < 1e-6, (
        f"r1 recipe drift > 1e-6 Sv ({diff:.3e}) — STOP, do not proceed to r2-r5")
    print("  GATE PASSED — proceeding to r2-r5")


# %% process r2-r5 and write to cmip_proj_staging ----------------------------
if __name__ == "__main__":
    for member in MEMBERS_NEW:
        out = f"{OUT_DIR}/giss-e2-1-g_{member}_amoc26_yr.nc"
        if os.path.isfile(out):
            print(f"  exists, skipping: {out}")
            continue
        amoc_yr = derive_amoc26_yr(member)
        amoc_yr.to_netcdf(out)
        print(f"  wrote {out}  ({float(amoc_yr.mean()):.4f} Sv mean, "
              f"{int(amoc_yr.size)} yrs)")


# %% sanity plot: 5-member AMOC ensemble vs legacy r1 -------------------------
if __name__ == "__main__":
    PLOTS_DIR = "/home/m/m300940/teu_amoc/plots/giss_ssp126"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    members = MEMBERS_ALL
    series = {}
    for m in members:
        if m == "r1i1p1f2":
            f = LEGACY_R1
        else:
            f = f"{OUT_DIR}/giss-e2-1-g_{m}_amoc26_yr.nc"
        da = xr.open_dataarray(f, use_cftime=True)
        series[m] = da

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = cm.viridis(np.linspace(0.1, 0.9, len(members)))
    for c, m in zip(colors, members):
        da = series[m]
        ax.plot(da.time.dt.year, da.values, color=c, alpha=0.4, lw=0.6)
        ax.plot(da.time.dt.year, da.rolling(time=10, center=True).mean().values,
                color=c, label=m, lw=2)
    ax.axhline(np.nan)
    ax.set_xlabel("year")
    ax.set_ylabel("AMOC at 26.5°N [Sv]")
    ax.set_title("GISS-E2-1-G ssp126 r1-r5 i1p1f2 — 10yr rolling mean (thick), annual (thin)")
    ax.legend(fontsize=9, ncol=5, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/giss_ssp126_amoc_r1-r5.png", dpi=150)
    fig.savefig(f"{PLOTS_DIR}/giss_ssp126_amoc_r1-r5.pdf")
    print(f"wrote {PLOTS_DIR}/giss_ssp126_amoc_r1-r5.{{png,pdf}}")

    # Print 2091-2100 mean per member — this is what AMOC_future sees
    print("\n2091-2100 mean AMOC@26.5°N per member, Sv:")
    for m in members:
        v = float(series[m].sel(time=slice("2091", "2100")).mean())
        print(f"  {m}: {v:+7.4f} Sv")
    print(f"  5-member mean: "
          f"{np.mean([float(series[m].sel(time=slice('2091','2100')).mean()) for m in members]):+7.4f} Sv")
    print(f"  (previously, single-r1 mean = {float(series['r1i1p1f2'].sel(time=slice('2091','2100')).mean()):+7.4f} Sv)")

# %%
