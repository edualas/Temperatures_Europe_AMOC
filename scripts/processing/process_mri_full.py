"""Fill the MRI-ESM2-0 r2-r5 i1p1f1 ensemble across all scenarios for tas + amoc26.

Reads from:
- /pool/data/CMIP6/...   (preferred when available, free)
- /work/bu1431/T_EU_AMOC/CMIP6/mri-esm2-0/raw/...  (ESGF downloads, mirror layout)

Writes annual-mean caches to the bu1431 mirror of the uo1075 tree:

  /work/bu1431/T_EU_AMOC/CMIP6/{scenario}/mri-esm2-0/
      mri-esm2-0_r{N}i1p1f1_{tas,amoc26}_yr.nc

These caches have the same layout as the legacy r1 caches at
/work/uo1075/m300817/teu_amoc/data/CMIP6/{scenario}/mri-esm2-0/ —
single unnamed DataArray, time dim, optional scalar 'lev' coord on
amoc files. So once cmip_cooling.py learns to also search
the bu1431 root, these files drop in transparently.

Run from scripts/ as
    python process_mri_full.py
or interactively with VS Code cells.
"""
# %% imports + config ---------------------------------------------------------
import os
import hashlib
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import xarray as xr


POOL_BASE = '/pool/data/CMIP6/data'
RAW_BASE = '/work/bu1431/T_EU_AMOC/CMIP6/mri-esm2-0/raw'
LEGACY_BASE = '/work/uo1075/m300817/teu_amoc/data/CMIP6'
OUT_BASE = '/work/bu1431/T_EU_AMOC/CMIP6'

# tas: (scenario, member) → list of chunk paths (in order)
# Source preference: /pool (where present), else bu1431/raw/ (ESGF download).

def _pool_tas(sce, mem, ver):
    mip = 'CMIP' if sce == 'historical' else 'ScenarioMIP'
    base = f'{POOL_BASE}/{mip}/MRI/MRI-ESM2-0/{sce}/{mem}/Amon/tas/gn/v{ver}'
    if not os.path.isdir(base):
        return []
    return sorted(f'{base}/{f}' for f in os.listdir(base) if f.endswith('.nc'))

def _pool_msftmz(sce, mem, ver):
    mip = 'CMIP' if sce == 'historical' else 'ScenarioMIP'
    base = f'{POOL_BASE}/{mip}/MRI/MRI-ESM2-0/{sce}/{mem}/Omon/msftmz/gr2z/v{ver}'
    if not os.path.isdir(base):
        return []
    return sorted(f'{base}/{f}' for f in os.listdir(base) if f.endswith('.nc'))

def _raw_chunks(sce, var, mem, table, grid):
    base = f'{RAW_BASE}/{sce}/{mem}/{table}/{var}/{grid}'
    if not os.path.isdir(base):
        return []
    versions = sorted(os.listdir(base))
    if not versions:
        return []
    pick = versions[-1]
    d = f'{base}/{pick}'
    return sorted(f'{d}/{f}' for f in os.listdir(d) if f.endswith('.nc'))


# What needs to be produced. For each: (sce, var, member, source_kind, version)
# source_kind ∈ {'pool', 'raw'}
PRODUCE = [
    # tas via /pool
    ('historical', 'tas', 'r3i1p1f1', 'pool', 'v20190222'),
    ('historical', 'tas', 'r5i1p1f1', 'pool', 'v20190222'),
    # tas via ESGF (in bu1431 raw)
    ('ssp126', 'tas', 'r2i1p1f1', 'raw', None),
    ('ssp126', 'tas', 'r3i1p1f1', 'raw', None),
    ('ssp126', 'tas', 'r4i1p1f1', 'raw', None),
    ('ssp126', 'tas', 'r5i1p1f1', 'raw', None),
    ('ssp245', 'tas', 'r2i1p1f1', 'raw', None),
    ('ssp245', 'tas', 'r3i1p1f1', 'raw', None),
    ('ssp245', 'tas', 'r4i1p1f1', 'raw', None),
    ('ssp245', 'tas', 'r5i1p1f1', 'raw', None),
    ('ssp585', 'tas', 'r2i1p1f1', 'raw', None),
    ('ssp585', 'tas', 'r3i1p1f1', 'raw', None),
    ('ssp585', 'tas', 'r4i1p1f1', 'raw', None),
    ('ssp585', 'tas', 'r5i1p1f1', 'raw', None),
    # amoc via /pool
    ('historical', 'amoc26', 'r5i1p1f1', 'pool', 'v20191210'),
    # amoc via ESGF (in bu1431 raw)
    ('historical', 'amoc26', 'r3i1p1f1', 'raw', None),
    ('ssp585', 'amoc26', 'r2i1p1f1', 'raw', None),
    ('ssp585', 'amoc26', 'r3i1p1f1', 'raw', None),
    ('ssp585', 'amoc26', 'r4i1p1f1', 'raw', None),
    ('ssp585', 'amoc26', 'r5i1p1f1', 'raw', None),
]


def _sources_for(sce, var, mem, kind, ver):
    if kind == 'pool':
        if var == 'tas':
            return _pool_tas(sce, mem, ver.lstrip('v'))
        else:
            return _pool_msftmz(sce, mem, ver.lstrip('v'))
    elif kind == 'raw':
        table = 'Amon' if var == 'tas' else 'Omon'
        gridvar = 'tas' if var == 'tas' else 'msftmz'
        grid = 'gn' if var == 'tas' else 'gr2z'
        return _raw_chunks(sce, gridvar, mem, table, grid)
    raise ValueError(kind)


def out_path(sce, var, mem):
    return f'{OUT_BASE}/{sce}/mri-esm2-0/mri-esm2-0_{mem}_{var}_yr.nc'


def legacy_path(sce, var, mem):
    return f'{LEGACY_BASE}/{sce}/mri-esm2-0/mri-esm2-0_{mem}_{var}_yr.nc'


# %% recipes ------------------------------------------------------------------
def weighted_mon_to_year_mean(ds, var):
    """Days-in-month weighted monthly→annual mean. Same recipe as
    scripts/process_mri_ssp126_amoc.py / process_giss_ssp126_amoc.py."""
    month_length = ds.time.dt.days_in_month
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)
    obs = ds[var]
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)
    obs_sum = (obs * wgts).resample(time="YS").sum(dim="time")
    ones_out = (ones * wgts).resample(time="YS").sum(dim="time")
    return obs_sum / ones_out


def derive_tas_yr(chunk_paths):
    """tas Amon → days-in-month-weighted annual mean. Preserves (lat, lon)."""
    ds = xr.open_mfdataset(chunk_paths, parallel=False, use_cftime=True,
                           combine='by_coords')
    out = weighted_mon_to_year_mean(ds, 'tas').load()
    return out


def derive_amoc26_yr(chunk_paths):
    """msftmz gr2z → basin=0 (Atlantic+Arctic) → lat=26.5 nearest →
    weighted_mon_to_year_mean → idxmax(lev) → /(1025·1e6)."""
    ds = xr.open_mfdataset(chunk_paths, parallel=False, use_cftime=True,
                           combine='by_coords')
    sub = ds.sel(basin=0, drop=True).sel(lat=26.5, method='nearest', drop=True)
    amoc_yr = weighted_mon_to_year_mean(sub, 'msftmz')
    max_lev_idx = amoc_yr.idxmax(dim='lev')
    amoc_yr = amoc_yr.sel(lev=max_lev_idx, drop=True) / (1025.0 * 1e6)
    return amoc_yr.load()


def md5(path, bs=2**20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(bs), b''):
            h.update(chunk)
    return h.hexdigest()


# %% Gate A — pool source coverage check -------------------------------------
def gate_a():
    print('=== Gate A: confirm every /pool source exists ===')
    failed = []
    for sce, var, mem, kind, ver in PRODUCE:
        if kind != 'pool':
            continue
        paths = _sources_for(sce, var, mem, kind, ver)
        if not paths:
            failed.append((sce, var, mem))
            print(f'  MISSING {sce} {var} {mem}')
        else:
            sz = sum(os.path.getsize(p) for p in paths) / 1e6
            print(f'  OK      {sce} {var} {mem}  ({len(paths)} chunk(s), {sz:.0f} MB)')
    assert not failed, f'Gate A FAILED: /pool sources missing for {failed}'
    print('  Gate A PASSED')


# %% Gate B — raw (ESGF) source coverage check -------------------------------
def gate_b(strict=True):
    """If strict=False, return list of (sce, var, mem) with missing sources
    instead of asserting. produce_all() skips items in that list."""
    print('\n=== Gate B: confirm every ESGF (raw) source exists ===')
    missing = []
    for sce, var, mem, kind, ver in PRODUCE:
        if kind != 'raw':
            continue
        paths = _sources_for(sce, var, mem, kind, ver)
        if not paths:
            missing.append((sce, var, mem))
            print(f'  MISSING {sce} {var} {mem}')
        else:
            sz = sum(os.path.getsize(p) for p in paths) / 1e6
            print(f'  OK      {sce} {var} {mem}  ({len(paths)} chunk(s), {sz:.0f} MB)')
    if strict:
        assert not missing, f'Gate B FAILED: ESGF sources missing for {missing}'
        print('  Gate B PASSED (all sources present)')
    else:
        if missing:
            print(f'  Gate B partial: {len(missing)} source(s) missing — '
                  f'these tuples will be skipped: {missing}')
        else:
            print('  Gate B PASSED (all sources present)')
    return missing


# %% Gate C — recipe sanity: tas r1 hist re-derived vs legacy -----------------
def gate_c():
    print('\n=== Gate C: tas recipe — re-derive hist r1 from /pool vs legacy ===')
    src = _pool_tas('historical', 'r1i1p1f1', '20190222')
    assert src, 'no /pool source for hist r1 tas'
    new = derive_tas_yr(src)
    legacy = xr.open_dataarray(legacy_path('historical', 'tas', 'r1i1p1f1'),
                                use_cftime=True)
    print(f'  legacy shape={legacy.shape}, new shape={new.shape}')
    print(f'  legacy time[0..2]: {[str(t)[:4] for t in legacy.time.values[:3]]}')
    print(f'  new    time[0..2]: {[str(t)[:4] for t in new.time.values[:3]]}')
    # Compare values, ignoring tiny coord differences
    new_aligned = new.assign_coords(time=legacy.time)
    diff = float(np.nanmax(np.abs(legacy.values - new_aligned.values)))
    print(f'  legacy mean (global, time-mean) = {float(legacy.mean()):.4f} K')
    print(f'  new    mean (global, time-mean) = {float(new.mean()):.4f} K')
    print(f'  max|diff|                       = {diff:.3e} K')
    assert diff < 1e-4, f'Gate C FAILED: recipe drift {diff:.3e} K'
    print('  Gate C PASSED')


# %% Gate D — amoc recipe sanity vs legacy ssp245 r2 (NOT r1!) ---------------
def gate_d():
    print('\n=== Gate D: amoc recipe — re-derive ssp245 r2 from /pool vs legacy ===')
    src = _pool_msftmz('ssp245', 'r2i1p1f1', '20210830')
    assert src, 'no /pool source for ssp245 r2 msftmz'
    new = derive_amoc26_yr(src)
    legacy = xr.open_dataarray(legacy_path('ssp245', 'amoc26', 'r2i1p1f1'),
                                use_cftime=True)
    new_aligned = new.assign_coords(time=legacy.time)
    diff = float(np.nanmax(np.abs(legacy.values - new_aligned.values)))
    print(f'  legacy mean = {float(legacy.mean()):.4f} Sv')
    print(f'  new    mean = {float(new.mean()):.4f} Sv')
    print(f'  max|diff|   = {diff:.3e} Sv')
    # Gate D is the "non-r1 check" — proves m300817 used the same recipe across members
    assert diff < 1e-6, f'Gate D FAILED: recipe drift on ssp245 r2 {diff:.3e}'
    print('  Gate D PASSED')


# %% Produce: all 20 outputs --------------------------------------------------
def produce_all(skip=None):
    skip = set(skip or [])
    print(f'\n=== Produce {len(PRODUCE)} outputs (skipping {len(skip)}) ===')
    for sce, var, mem, kind, ver in PRODUCE:
        if (sce, var, mem) in skip:
            print(f'  skip (no source): {sce}/{var}/{mem}')
            continue
        out = out_path(sce, var, mem)
        if os.path.exists(out):
            print(f'  exists, skipping: {out}')
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        src = _sources_for(sce, var, mem, kind, ver)
        if var == 'tas':
            da = derive_tas_yr(src)
        else:
            da = derive_amoc26_yr(src)
        n_nan_dom = int(np.isnan(da.values).sum())
        # Sanity: tas global-mean should be in [260, 300] K, amoc in [5, 30] Sv
        if var == 'tas':
            mean = float(da.mean())
            assert 260 <= mean <= 300, f'{sce} {mem} tas mean {mean:.1f} out of range'
        else:
            mean = float(da.mean())
            assert 5 <= mean <= 30, f'{sce} {mem} amoc mean {mean:.2f} out of range'
        # Write as the same naked DataArray layout as legacy r1
        da.to_netcdf(out)
        print(f'  wrote {sce}/{var}/{mem}  '
              f'n_time={da.time.size} mean={mean:.4f} nans={n_nan_dom} → {out}')


# %% Gate E — file count: 20 new files exist and open via legacy loader ------
def gate_e(skip=None):
    skip = set(skip or [])
    print(f'\n=== Gate E: produced files exist, openable, no NaN ===')
    bad = []
    for sce, var, mem, kind, ver in PRODUCE:
        if (sce, var, mem) in skip:
            print(f'  skip {sce}/{var}/{mem}')
            continue
        out = out_path(sce, var, mem)
        if not os.path.exists(out):
            bad.append((sce, var, mem, 'MISSING'))
            continue
        try:
            da = xr.open_dataarray(out, use_cftime=True)
            if var == 'amoc26' and 'lev' in da.coords:
                da = da.drop_vars('lev')
            vals = da.values if da.values.ndim == 1 else da.mean(['lat','lon']).values
            n_nan = int(np.isnan(vals).sum())
            yr0, yr1 = da.time.values[0].year, da.time.values[-1].year
            expected = 165 if sce == 'historical' else 86
            ok = (da.time.size == expected) and (n_nan == 0)
            tag = 'OK' if ok else 'BAD'
            print(f'  {tag:<3} {sce}/{var}/{mem}  n_time={da.time.size} '
                  f'(expected {expected})  yrs={yr0}-{yr1}  nan={n_nan}')
            if not ok:
                bad.append((sce, var, mem, f'n_time={da.time.size}, nan={n_nan}'))
        except Exception as e:
            bad.append((sce, var, mem, f'open failed: {e}'))
    assert not bad, f'Gate E FAILED: {bad}'
    print('  Gate E PASSED')


# %% Per-member 2091-2100 AMOC report (for METHODS Changelog) -----------------
def report_amoc_ensemble():
    print('\n=== Per-member 2091-2100 AMOC@26.5°N (Sv) by scenario ===')
    for sce in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
        vals = {}
        for mem in [f'r{i}i1p1f1' for i in range(1, 6)]:
            # Try legacy first, then bu1431
            for p in (legacy_path(sce, 'amoc26', mem), out_path(sce, 'amoc26', mem)):
                if os.path.exists(p):
                    da = xr.open_dataarray(p, use_cftime=True)
                    years = np.array([t.year for t in da.time.values])
                    sel = da.values[(years>=2091) & (years<=2100)]
                    vals[mem] = float(sel.mean())
                    break
        if not vals:
            print(f'  {sce}: no data')
            continue
        arr = np.array(list(vals.values()))
        line = ', '.join(f'{m}={v:.2f}' for m, v in vals.items())
        print(f'  {sce} N={len(vals)}  mean={arr.mean():.3f}  std={arr.std():.3f}  [{line}]')


# %% Main ---------------------------------------------------------------------
if __name__ == '__main__':
    gate_a()
    skip = gate_b(strict=False)
    gate_c()
    gate_d()
    produce_all(skip=skip)
    gate_e(skip=skip)
    report_amoc_ensemble()
    if skip:
        print(f'\nSKIPPED {len(skip)} tuple(s): {skip}')
        print(f'Re-run after downloading those raw sources.')
    else:
        print(f'\nALL GATES PASSED. {len(PRODUCE)} files ready.')
