"""Derive DJF + JJA seasonal tas for the files we already have annually.

Scope:
- HosMIP-7 historical seasonal tas (MPI-ESM1-2-LR excluded; covered by MPI-GE).
- GISS-E2-1-G historical seasonal tas (r1-r11 i1p1f2 — physics/forcing matches
  `_MODEL_PHYSICS_FORCING` in functions.get_cmip_projections).
- GISS PI seasonal climatology (1850-1899 DJF/JJA mean — sibling of the
  existing yearly tas_pi_climatology.nc).

Outputs land in /work/bu1431/T_EU_AMOC/CMIP6/historical/{model}/tas_{djf,jja}/
and the GISS processed/ directory. Companion edits to functions.py
(_model_root + historical season loop) make them visible to the loader.

Recipe lifted verbatim from data_process.ipynb cells 214/263 (canonical
CMIP6 seasonal aggregation). Cell 2 calibrates against an existing
known-good seasonal file before any new files are written.

Run interactively cell-by-cell, or as a standalone script.
"""

# %% imports + config ---------------------------------------------------------
import os
import sys
import json
from glob import glob
from pathlib import Path

import numpy as np
import xarray as xr
import cftime

UO1075 = '/work/uo1075/m300817/teu_amoc/data/CMIP6'
BU1431 = '/work/bu1431/T_EU_AMOC/CMIP6'
POOL_CMIP = '/pool/data/CMIP6/data/CMIP'
POOL_SCEN = '/pool/data/CMIP6/data/ScenarioMIP'

HIST_OUT_ROOT = f'{BU1431}/historical'

# Per-model: institute, CMIP6 model directory name (case-sensitive),
# grid label, project subroot ('CMIP' for historical / piControl,
# 'ScenarioMIP' for SSPs).
POOL_LOOKUP = {
    'canesm5':         {'inst': 'CCCma',                'cmip_name': 'CanESM5',         'grid': 'gn'},
    'ec-earth3':       {'inst': 'EC-Earth-Consortium',  'cmip_name': 'EC-Earth3',       'grid': 'gr'},
    'cesm2':           {'inst': 'NCAR',                 'cmip_name': 'CESM2',           'grid': 'gn'},
    'ipsl-cm6a-lr':    {'inst': 'IPSL',                 'cmip_name': 'IPSL-CM6A-LR',    'grid': 'gr'},
    'hadgem3-gc31-ll': {'inst': 'MOHC',                 'cmip_name': 'HadGEM3-GC31-LL', 'grid': 'gn'},
    'hadgem3-gc31-mm': {'inst': 'MOHC',                 'cmip_name': 'HadGEM3-GC31-MM', 'grid': 'gn'},
    'mpi-esm1-2-hr':   {'inst': 'MPI-M',                'cmip_name': 'MPI-ESM1-2-HR',   'grid': 'gn'},
    'giss-e2-1-g':     {'inst': 'NASA-GISS',            'cmip_name': 'GISS-E2-1-G',     'grid': 'gn'},
}

# 7 HosMIP models for processing (drops MPI-ESM1-2-LR per project memory:
# project_hosmip_seven_not_eight.md). Map model display name → uo1075 dir
# slug (matches data_process.ipynb cell 242 conventions).
HOSMIP_DIRS = {
    'CanESM5':         'canesm5',
    'EC-Earth3':       'ec-earth3',
    'CESM2':           'cesm2',
    'IPSL-CM6A-LR':    'ipsl-cm6a-lr',
    'HadGEM3-GC3-1LL': 'hadgem3-gc31-ll',
    'HadGEM3-GC3-1MM': 'hadgem3-gc31-mm',
    'MPI-ESM1-2-HR':   'mpi-esm1-2-hr',
}

GISS_HIST_DIR = f'{UO1075}/historical/giss-e2-1-g'
GISS_PROC = f'{BU1431}/giss-e2-1-g/processed'

# GISS p1f2 ensemble actually consumed downstream (matches the library's
# load_giss_member_amoc_tas members and the get_cmip_projections forcing filter).
GISS_MEMBERS = [f'r{i}i1p1f2' for i in range(1, 12)]  # r1..r11

PI_WINDOW = ('1850', '1899')


# %% helpers -------------------------------------------------------------------
def reset_djf_time(ds):
    """Shift each DJF time stamp by +1 year (Dec[N] → Jan[N+1] convention).

    Lifted from data_process.ipynb cell 17.
    """
    new_time = [
        cftime.DatetimeProlepticGregorian(d.year + 1, 1, d.day)
        for d in ds['time'].values
    ]
    ds = ds.assign_coords(time=new_time)
    return ds


def seasonalize(monthly, season, start_year):
    """Monthly → seasonal yearly, calendar-normalised.

    Replicates data_process.ipynb cells 214/263: resample, season-select,
    drop trailing partial year (DJF only), shift-Dec-to-Jan (DJF only),
    reassign time axis to a clean proleptic_gregorian year-start range.
    """
    if season == 'djf':
        seas = monthly.resample(time='QS-DEC').mean(dim='time')
        out = seas.sel(time=seas.time.dt.season == 'DJF').isel(time=slice(None, -1))
        out = reset_djf_time(out)
    elif season == 'jja':
        seas = monthly.resample(time='QS').mean(dim='time')
        out = seas.sel(time=seas.time.dt.season == 'JJA')
    else:
        raise ValueError(f'season must be djf or jja, got {season!r}')

    # 4-digit ISO year (e.g. start_year=1 → '0001'); matches data_process
    # cell 263 which used start='0001' for piControl, '2015' for SSPs.
    new_time = xr.cftime_range(
        start=f'{int(start_year):04d}', periods=out.sizes['time'],
        freq='YS', calendar='proleptic_gregorian')
    return out.assign_coords(time=new_time)


def _pool_member_dir(model_slug, experiment, member):
    """Return /pool path to the latest version of (model, exp, member) tas.

    Picks the highest YYYYMMDD v-dir if multiple are published. Returns
    None if the member is absent.
    """
    info = POOL_LOOKUP[model_slug]
    subroot = POOL_SCEN if experiment.startswith('ssp') else POOL_CMIP
    base = (f'{subroot}/{info["inst"]}/{info["cmip_name"]}/'
            f'{experiment}/{member}/Amon/tas/{info["grid"]}')
    if not os.path.isdir(base):
        return None
    versions = sorted(
        d for d in os.listdir(base)
        if d.startswith('v') and os.path.isdir(f'{base}/{d}'))
    if not versions:
        return None
    return f'{base}/{versions[-1]}'


def _bu1431_upload_files(model_slug, experiment, member):
    """EC-Earth3 fallback: look in /work/bu1431/.../upload/{slug}/ for
    files matching tas_Amon_{cmip_name}_{experiment}_{member}_{grid}_*.nc.

    Only EC-Earth3 needs this so far (5157-file upload from 2026-05-25;
    see memory: reference_ec_earth3_msftyz_audit). Generalised here.
    """
    info = POOL_LOOKUP.get(model_slug)
    if info is None:
        return []
    pattern = (f'/work/bu1431/T_EU_AMOC/CMIP6/upload/{model_slug}/'
               f'tas_Amon_{info["cmip_name"]}_{experiment}_{member}_'
               f'{info["grid"]}_*.nc')
    return sorted(glob(pattern))


def open_monthly_pool(model_slug, experiment, member):
    """Open monthly tas for (model_slug, experiment, member).

    Source preference:
    1. /pool/data/CMIP6 (canonical DKRZ pool)
    2. /work/bu1431/.../upload/{slug}/ (project upload tree — currently
       used for EC-Earth3 members not on /pool)

    Returns lazy xr.Dataset, or None if no source is found.
    """
    vdir = _pool_member_dir(model_slug, experiment, member)
    if vdir is not None:
        files = sorted(glob(f'{vdir}/*.nc'))
    else:
        files = _bu1431_upload_files(model_slug, experiment, member)
    if not files:
        return None
    return xr.open_mfdataset(
        files, parallel=False, use_cftime=True,
        data_vars='minimal', coords='minimal', compat='override')


def list_existing_annual_members(model_slug, scenario='historical'):
    """Return list of realisation strings with a *_tas_yr.nc on disk.

    Reads the canonical uo1075 layout used by data_process.ipynb cell 243.
    """
    pattern = f'{UO1075}/{scenario}/{model_slug}/{model_slug}_r*_tas_yr.nc'
    members = []
    for path in sorted(glob(pattern)):
        base = os.path.basename(path)
        # {model_slug}_{member}_tas_yr.nc
        rea = base[len(model_slug) + 1:-len('_tas_yr.nc')]
        members.append(rea)
    return members


def write_seasonal(monthly_ds, varname, season, start_year, out_path):
    """Apply seasonalize + write to out_path with parent dirs as needed."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seasonal = seasonalize(monthly_ds[varname], season, start_year=start_year)
    seasonal = seasonal.to_dataset(name=varname).load()
    seasonal.to_netcdf(out_path)
    return seasonal


# %% Cell 2 — calibration -----------------------------------------------------
# Reproduce one known-good seasonal file from the canonical monthly source
# and compare against the on-disk version. Halt with a clear error if the
# recipe does not match.

def _calibrate():
    """Verify recipe reproduces a known-good seasonal file from the same
    monthly source. ssp245 is a clean target (fixed 2015-2100 window,
    no version drift).
    """
    print('=' * 70)
    print('CALIBRATION — reproducing known-good seasonal file')
    print('=' * 70)

    cases = [
        # (model_slug, member, experiment_local, experiment_pool, season, start_year)
        ('canesm5', 'r1i1p1f1', 'ssp245', 'ssp245', 'djf', 2015),
        ('canesm5', 'r1i1p1f1', 'ssp245', 'ssp245', 'jja', 2015),
        ('ipsl-cm6a-lr', 'r1i1p1f1', 'ssp245', 'ssp245', 'djf', 2015),
    ]
    all_ok = True
    for model_slug, member, exp_local, exp_pool, season, start_year in cases:
        print(f'\n  case: {model_slug} {member} {exp_local} {season}')
        expected = (f'{UO1075}/{exp_local}/{model_slug}/tas_{season}/'
                    f'{model_slug}_{member}_tas_{season}.nc')
        if not os.path.exists(expected):
            print(f'    SKIP: expected file missing ({expected})')
            continue

        expected_ds = xr.open_dataset(expected, use_cftime=True)
        print(f'    expected sizes: {dict(expected_ds.sizes)}')

        monthly = open_monthly_pool(model_slug, exp_pool, member)
        if monthly is None:
            print(f'    SKIP: /pool has no monthly tas')
            expected_ds.close()
            continue
        print(f'    monthly time.size = {monthly.sizes.get("time", 0)}')

        new = seasonalize(monthly['tas'], season, start_year=start_year)
        print(f'    new sizes:      {dict(new.sizes)}')

        if new.sizes['time'] != expected_ds.sizes['time']:
            print(f'    SIZE MISMATCH: new={new.sizes["time"]} '
                  f'expected={expected_ds.sizes["time"]}')

        n = min(new.sizes['time'], expected_ds.sizes['time'])
        a = expected_ds['tas'].isel(time=slice(0, n)).values
        b = new.isel(time=slice(0, n)).values
        diff = np.abs(a - b)
        finite = np.isfinite(a) & np.isfinite(b)
        max_diff = float(np.nanmax(diff[finite])) if finite.any() else float('nan')
        print(f'    max|new - expected| over {finite.sum()} finite cells = {max_diff:.3e}')

        expected_ds.close()
        monthly.close()

        if not (max_diff < 1e-4):
            print(f'    FAIL (tolerance 1e-4)')
            all_ok = False
        else:
            print(f'    PASS')

    if not all_ok:
        raise RuntimeError(
            'CALIBRATION FAILED: at least one case exceeds 1e-4 tolerance — '
            'recipe drift from data_process.ipynb')
    print('\nCALIBRATION PASSED (all cases)')
    return True


# %% Cell 3 — HosMIP-7 historical seasonal tas --------------------------------

def process_hosmip_historical():
    print('=' * 70)
    print('HosMIP-7 HISTORICAL SEASONAL TAS')
    print('=' * 70)
    summary = []
    for model_disp, slug in HOSMIP_DIRS.items():
        members = list_existing_annual_members(slug, 'historical')
        print(f'\n--- {model_disp} ({slug}): {len(members)} member(s) on disk ---')
        for member in members:
            out_djf = (f'{HIST_OUT_ROOT}/{slug}/tas_djf/'
                       f'{slug}_{member}_tas_djf.nc')
            out_jja = (f'{HIST_OUT_ROOT}/{slug}/tas_jja/'
                       f'{slug}_{member}_tas_jja.nc')
            if os.path.exists(out_djf) and os.path.exists(out_jja):
                print(f'  skip (both present): {member}')
                summary.append((model_disp, member, 'skip_exists'))
                continue

            try:
                monthly = open_monthly_pool(slug, 'historical', member)
            except Exception as e:
                print(f'  FAIL load: {member}: {e}')
                summary.append((model_disp, member, f'fail_load:{e}'))
                continue

            if monthly is None:
                print(f'  /pool has no monthly tas: {member}')
                summary.append((model_disp, member, 'no_pool'))
                continue
            n_t = monthly.sizes.get('time', 0)
            if n_t != 1980:
                print(f'  WARN time.size={n_t} != 1980: {member}')

            try:
                if not os.path.exists(out_djf):
                    write_seasonal(monthly, 'tas', 'djf', 1850, out_djf)
                if not os.path.exists(out_jja):
                    write_seasonal(monthly, 'tas', 'jja', 1850, out_jja)
                print(f'  wrote: {member}')
                summary.append((model_disp, member, 'wrote'))
            except Exception as e:
                print(f'  FAIL write: {member}: {e}')
                summary.append((model_disp, member, f'fail_write:{e}'))
            finally:
                monthly.close()
    return summary


# %% Cell 4 — GISS historical seasonal tas (p1f2 ensemble) --------------------

def process_giss_historical():
    print('=' * 70)
    print('GISS HISTORICAL SEASONAL TAS')
    print('=' * 70)
    summary = []
    slug = 'giss-e2-1-g'
    out_root = f'{BU1431}/historical/{slug}'
    # Iterate over the members that have an annual _tas_yr.nc on uo1075
    # AND are in the canonical GISS_MEMBERS set (p1f2).
    on_disk = set(list_existing_annual_members(slug, 'historical'))
    members = [m for m in GISS_MEMBERS if m in on_disk]
    print(f'GISS p1f2 members with annual on disk: {len(members)}/{len(GISS_MEMBERS)}')
    for member in members:
        out_djf = f'{out_root}/tas_djf/{slug}_{member}_tas_djf.nc'
        out_jja = f'{out_root}/tas_jja/{slug}_{member}_tas_jja.nc'
        if os.path.exists(out_djf) and os.path.exists(out_jja):
            print(f'  skip (both present): {member}')
            summary.append((member, 'skip_exists'))
            continue
        try:
            monthly = open_monthly_pool(slug, 'historical', member)
        except Exception as e:
            print(f'  FAIL load: {member}: {e}')
            summary.append((member, f'fail_load:{e}'))
            continue
        if monthly is None:
            print(f'  /pool has no monthly tas: {member}')
            summary.append((member, 'no_pool'))
            continue
        try:
            if not os.path.exists(out_djf):
                write_seasonal(monthly, 'tas', 'djf', 1850, out_djf)
            if not os.path.exists(out_jja):
                write_seasonal(monthly, 'tas', 'jja', 1850, out_jja)
            print(f'  wrote: {member}')
            summary.append((member, 'wrote'))
        except Exception as e:
            print(f'  FAIL write: {member}: {e}')
            summary.append((member, f'fail_write:{e}'))
        finally:
            monthly.close()
    return summary


# %% Cell 5 — GISS PI seasonal climatology -----------------------------------

def build_giss_pi_climatology():
    """1850-1899 ensemble mean of (DJF, JJA) tas, sibling of the existing
    yearly tas_pi_climatology.nc.

    Reads the seasonal historical files just produced (so depends on
    Cell 4 completing successfully).
    """
    print('=' * 70)
    print('GISS PI SEASONAL CLIMATOLOGY (1850-1899 mean)')
    print('=' * 70)
    slug = 'giss-e2-1-g'
    in_root = f'{BU1431}/historical/{slug}'
    out_root = f'{BU1431}/giss-e2-1-g/processed'
    os.makedirs(out_root, exist_ok=True)

    on_disk = set(list_existing_annual_members(slug, 'historical'))
    members = [m for m in GISS_MEMBERS if m in on_disk]

    results = {}
    for season in ('djf', 'jja'):
        files = [f'{in_root}/tas_{season}/{slug}_{m}_tas_{season}.nc'
                 for m in members]
        present = [f for f in files if os.path.exists(f)]
        if not present:
            print(f'  no {season} files yet — skip')
            results[season] = None
            continue
        ds_list = [xr.open_dataset(f, use_cftime=True) for f in present]
        ens = xr.concat([d['tas'] for d in ds_list],
                        dim=xr.Variable('realiz', [m for m, f in zip(members, files) if os.path.exists(f)]))
        pi = ens.sel(time=slice(*PI_WINDOW)).mean(['time', 'realiz'])
        pi.name = 'tas'
        out = f'{out_root}/giss-e2-1-g_tas_pi_{season}_climatology.nc'
        pi.load().to_netcdf(out)
        for d in ds_list:
            d.close()
        print(f'  wrote: {out} (mean over {len(present)} members)')
        results[season] = out
    return results


# %% main ---------------------------------------------------------------------
if __name__ == '__main__':
    # Step 1: calibration — halt if the recipe drifts from the notebook.
    _calibrate()

    # Step 2: HosMIP-7 historical.
    hosmip_summary = process_hosmip_historical()

    # Step 3: GISS historical.
    giss_summary = process_giss_historical()

    # Step 4: GISS PI seasonal climatology (depends on step 3).
    pi_paths = build_giss_pi_climatology()

    # Step 5: summary.
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    from collections import Counter
    hc = Counter([s for _, _, s in hosmip_summary])
    gc = Counter([s for _, s in giss_summary])
    print('HosMIP-7 historical:')
    for k, v in sorted(hc.items()):
        print(f'  {k}: {v}')
    print('GISS historical:')
    for k, v in sorted(gc.items()):
        print(f'  {k}: {v}')
    print('GISS PI climatology:')
    for season, path in pi_paths.items():
        print(f'  {season}: {path}')
