"""
Replace bu1431 cmip_proj_staging ssp370 r1 tas_yr.nc with the NCCS
post-erratum version. Also (added 2026-05-23) replace the broken
ssp370 r1 amoc26_yr symlink with a regular file derived from DKRZ
msftmz; see Open issues / project_giss_ssp370_r1_amoc_broken_symlink.md
in memory.

Why: 2026-05-23 audit found that the only `cmip_proj_staging/ssp{126,245,370}/...`
file still tied to DKRZ pre-erratum is ssp370 r1 (DKRZ created 2020-02-08;
DKRZ r2-r10 were re-uploaded 2020-05-20 but r1 was never replaced).
Per-member Ireland 2091-2100 shift is small (−0.04 K vs +1 K for ssp126);
ensemble 10-member effect is −0.004 K. Worth fixing for correctness and
audit-trail cleanliness even though numerically tiny.

Mirrors the recipe in process_giss_ssp126_amoc.py: download NCCS monthly
chunks, days-in-month weighted annual mean, overwrite the staging file.
"""

# %% imports + config
import os, subprocess, glob
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xarray as xr

PORTAL = (
    "https://portal.nccs.nasa.gov/datashare/giss_cmip6/"
    "ScenarioMIP/NASA-GISS/GISS-E2-1-G/ssp370"
)
VERSION = "v20200115"
RAW_DIR = "/work/bu1431/T_EU_AMOC/CMIP6/giss-e2-1-g/ssp370"  # for raw NCCS chunks
OUT_PATH = ("/work/bu1431/T_EU_AMOC/CMIP6/giss-e2-1-g/cmip_proj_staging/"
            "ssp370/giss-e2-1-g/giss-e2-1-g_r1i1p1f2_tas_yr.nc")
BAK_PATH = OUT_PATH + ".bak_dkrz_preerratum_2026-05-23"

MEMBER = "r1i1p1f2"
CHUNKS = ["201501-205012", "205101-210012"]


def filename(chunk):
    return f"tas_Amon_GISS-E2-1-G_ssp370_{MEMBER}_gn_{chunk}.nc"


def nccs_url(chunk):
    return f"{PORTAL}/{MEMBER}/Amon/tas/gn/{VERSION}/{filename(chunk)}"


def local_raw(chunk):
    return f"{RAW_DIR}/{MEMBER}/{filename(chunk)}"


os.makedirs(f"{RAW_DIR}/{MEMBER}", exist_ok=True)


# %% spider check (2 URLs, parallel — overkill but consistent with ssp126 script)
def _spider(chunk):
    r = subprocess.run(["wget", "--spider", "-q", nccs_url(chunk)], capture_output=True)
    return chunk, r.returncode


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as ex:
        for fut in as_completed([ex.submit(_spider, c) for c in CHUNKS]):
            c, rc = fut.result()
            print(f"  spider {c}: {'OK' if rc == 0 else f'FAIL rc={rc}'}")
            assert rc == 0, f"NCCS missing chunk {c}"


# %% download (parallel) ------------------------------------------------------
def _wget(chunk):
    out = local_raw(chunk)
    if os.path.exists(out):
        return chunk, "skip"
    r = subprocess.run(
        ["wget", "-nc", "-q", "--wait=0.5", "--random-wait",
         "-P", f"{RAW_DIR}/{MEMBER}", nccs_url(chunk)],
        capture_output=True,
    )
    return chunk, "ok" if r.returncode == 0 else f"err rc={r.returncode}"


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as ex:
        for fut in as_completed([ex.submit(_wget, c) for c in CHUNKS]):
            c, status = fut.result()
            print(f"  {c}: {status}")
    for c in CHUNKS:
        assert os.path.exists(local_raw(c)), f"missing after download: {local_raw(c)}"


# %% days-in-month weighted annual mean ---------------------------------------
def weighted_mon_to_year_mean(ds, var):
    ml = ds.time.dt.days_in_month
    wgts = ml.groupby("time.year") / ml.groupby("time.year").sum()
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)
    obs = ds[var]
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)
    obs_sum = (obs * wgts).resample(time="YS").sum(dim="time")
    ones_out = (ones * wgts).resample(time="YS").sum(dim="time")
    return obs_sum / ones_out


# %% verification + replacement -----------------------------------------------
if __name__ == "__main__":
    # Old (DKRZ pre-erratum) — back up before overwriting
    if not os.path.exists(BAK_PATH):
        import shutil
        shutil.copy2(OUT_PATH, BAK_PATH)
        print(f"backed up: {BAK_PATH}")

    # Derive the new yearly mean
    paths = sorted(glob.glob(f"{RAW_DIR}/{MEMBER}/tas_Amon_*.nc"))
    assert len(paths) == 2, f"expected 2 chunks, got {len(paths)}"
    ds = xr.open_mfdataset(paths, parallel=False, use_cftime=True)
    new_yr = weighted_mon_to_year_mean(ds, "tas").load()

    # Quick sanity: compare to the about-to-be-overwritten DKRZ-pre file
    old_yr = xr.open_dataarray(OUT_PATH, use_cftime=True)
    # Align by year
    def by_yr(da):
        return da.assign_coords(year=da.time.dt.year).swap_dims(time="year").drop_vars("time")
    n_y, o_y = by_yr(new_yr), by_yr(old_yr)
    yrs = sorted(set(n_y.year.values) & set(o_y.year.values))
    diff = (n_y.sel(year=yrs) - o_y.sel(year=yrs)).load()
    print(f"\nNCCS post-erratum vs DKRZ pre-erratum (about-to-be-overwritten):")
    print(f"  max|diff| = {float(np.abs(diff).max()):.4e} K  "
          f"(zero ⇒ identical sources; nonzero ⇒ erratum genuine)")
    print(f"  mean diff = {float(diff.mean()):+.4e} K")

    # Need to write the DataArray, not a Dataset. Match the legacy file's
    # naming convention: a bare DataArray to_netcdf gives variable
    # `__xarray_dataarray_variable__`. The loader at functions.py:2495 uses
    # xr.open_dataarray, which is happy with either name. To preserve the
    # existing file's variable name (tas), we use to_dataset(name='tas').
    # Check what the existing file uses:
    src = xr.open_dataset(OUT_PATH)
    var_names = [v for v in src.data_vars]
    src.close()
    out_var = var_names[0] if var_names else "tas"
    print(f"  preserving variable name: {out_var!r}")

    new_yr.to_dataset(name=out_var).to_netcdf(OUT_PATH + ".tmp")
    os.replace(OUT_PATH + ".tmp", OUT_PATH)
    print(f"  wrote {OUT_PATH}")


# %% Cell B: replace broken ssp370 r1 amoc26_yr symlink ----------------------
# The bu1431 staging file was a symlink to
# /work/uo1075/m300817/teu_amoc/data/CMIP6/ssp370/giss-e2-1-g/
#   giss-e2-1-g_r1i1p1f2_amoc26_yr.nc
# but that target never existed (uo1075 only has the p3f1/p3f2/p5f1
# variants of r1 for ssp370, not the canonical p1f2). All other
# ssp370 r2-r10 symlinks under cmip_proj_staging resolve cleanly.
# DKRZ /pool/data/CMIP6 has the msftmz file (creation 2020-03-16,
# post-erratum). Recipe: days-in-month-weighted annual mean, basin=0,
# lat=26.5 nearest, idxmax over lev, divide by rho*1e6.

AMOC_OUT = ("/work/bu1431/T_EU_AMOC/CMIP6/giss-e2-1-g/cmip_proj_staging/"
            "ssp370/giss-e2-1-g/giss-e2-1-g_r1i1p1f2_amoc26_yr.nc")

if __name__ == "__main__":
    dkrz_paths = sorted(glob.glob(
        "/pool/data/CMIP6/data/ScenarioMIP/NASA-GISS/GISS-E2-1-G/ssp370/"
        "r1i1p1f2/Omon/msftmz/gn/v*/msftmz_*.nc"))
    assert len(dkrz_paths) == 2, (
        f"expected 2 DKRZ msftmz chunks, got {len(dkrz_paths)}")
    print(f"\ncell B: deriving ssp370 r1 amoc26_yr from DKRZ msftmz...")
    print(f"  inputs: {[p.rsplit('/',1)[1] for p in dkrz_paths]}")
    ds = xr.open_mfdataset(dkrz_paths, parallel=False, use_cftime=True)
    amoc_yr = weighted_mon_to_year_mean(
        ds.sel(basin=0, drop=True).sel(lat=26.5, method="nearest", drop=True),
        "msftmz",
    )
    max_lev_idx = amoc_yr.idxmax(dim="lev")
    amoc_yr = (amoc_yr.sel(lev=max_lev_idx, drop=True) / (1025.0 * 1e6)).load()

    # Sanity: matches what was reported on 2026-05-23 (+17.7037 Sv full
    # window, +11.2905 Sv at 2091-2100).
    full_mean = float(amoc_yr.mean())
    f2090 = float(amoc_yr.sel(time=slice("2091", "2100")).mean())
    print(f"  full-window mean = {full_mean:+.4f} Sv")
    print(f"  2091-2100 mean   = {f2090:+.4f} Sv")
    assert abs(full_mean - 17.7037) < 1e-3, (
        f"sanity check failed: full mean {full_mean} ≠ expected 17.7037")

    # Remove broken symlink; write regular file
    if os.path.islink(AMOC_OUT) or os.path.exists(AMOC_OUT):
        os.unlink(AMOC_OUT)
        print(f"  removed broken symlink at {AMOC_OUT}")
    amoc_yr.to_netcdf(AMOC_OUT)
    print(f"  wrote regular file: {AMOC_OUT}")

    # Compare to neighbouring members so the new file isn't an outlier
    sibs = [xr.open_dataarray(
        f"/work/bu1431/T_EU_AMOC/CMIP6/giss-e2-1-g/cmip_proj_staging/"
        f"ssp370/giss-e2-1-g/giss-e2-1-g_r{r}i1p1f2_amoc26_yr.nc",
        use_cftime=True).sel(time=slice("2091","2100")).mean()
        for r in range(2, 11)]
    sibs = [float(s) for s in sibs]
    print(f"  r2-r10 2091-2100 means: {[f'{s:+.2f}' for s in sibs]}")
    print(f"  r1 (new) vs r2-r10 mean: {f2090:+.2f} vs {np.mean(sibs):+.2f} Sv "
          f"(σ={np.std(sibs):.2f})")

# %%
