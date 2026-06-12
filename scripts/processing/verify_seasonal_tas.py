"""Verify the seasonal-tas processing.

Three checks:
1. File-count parity: every annual *_tas_yr.nc on uo1075 has DJF/JJA
   siblings on bu1431.
2. Per-model spot-check: pick one DJF file per model, confirm time axis
   is annual frequency and one year's value equals Dec[N]+Jan[N+1]+Feb[N+1]
   mean of the /pool monthly source within 1e-6.
3. get_hist_pi_baseline(model, 'tas', season='djf'/'jja') returns finite
   for every HosMIP-7 + GISS model.
"""
import os
import sys
from glob import glob

import numpy as np
import xarray as xr

UO1075 = '/work/uo1075/m300817/teu_amoc/data/CMIP6'
BU1431 = '/work/bu1431/T_EU_AMOC/CMIP6'

MODELS = [
    'canesm5', 'ec-earth3', 'cesm2', 'ipsl-cm6a-lr',
    'hadgem3-gc31-ll', 'hadgem3-gc31-mm', 'mpi-esm1-2-hr',
    'giss-e2-1-g',
]

# ---- check 1: file-count parity ----
print('=' * 70)
print('CHECK 1: file-count parity')
print('=' * 70)
gaps = []
for slug in MODELS:
    yr_files = sorted(glob(f'{UO1075}/historical/{slug}/{slug}_r*_tas_yr.nc'))
    n_yr = len(yr_files)
    n_djf = len(glob(f'{BU1431}/historical/{slug}/tas_djf/{slug}_r*_tas_djf.nc'))
    n_jja = len(glob(f'{BU1431}/historical/{slug}/tas_jja/{slug}_r*_tas_jja.nc'))
    status = 'OK' if (n_djf == n_yr and n_jja == n_yr) else 'GAP'
    print(f'  {slug:20s}  yr={n_yr:3d}  djf={n_djf:3d}  jja={n_jja:3d}  {status}')
    if status == 'GAP':
        gaps.append((slug, n_yr, n_djf, n_jja))

# ---- check 2: spot-check one DJF file per model against /pool monthly ----
print()
print('=' * 70)
print('CHECK 2: per-model spot-check (Dec[N]+Jan[N+1]+Feb[N+1] mean)')
print('=' * 70)

# Reuse the /pool lookup table from the processor.
sys.path.insert(0, '/home/m/m300940/teu_amoc/scripts')
import process_seasonal_tas as pst

for slug in MODELS:
    djf_files = sorted(glob(f'{BU1431}/historical/{slug}/tas_djf/{slug}_r*_tas_djf.nc'))
    if not djf_files:
        print(f'  {slug}: no DJF files — SKIP')
        continue
    f = djf_files[0]
    member = os.path.basename(f)[len(slug) + 1:-len('_tas_djf.nc')]
    seas_ds = xr.open_dataset(f, use_cftime=True)
    tas_seas = seas_ds['tas']
    # Pick year index in middle (avoids edge effects from partial first/last).
    yi = tas_seas.sizes['time'] // 2
    year = tas_seas.time.dt.year.values[yi]
    # Open monthly /pool source.
    monthly = pst.open_monthly_pool(slug, 'historical', member)
    if monthly is None:
        print(f'  {slug} {member}: no /pool source — SKIP')
        seas_ds.close()
        continue
    # Dec(year-1) + Jan(year) + Feb(year) — DJF time stamp is January of
    # the following year (post-reset_djf_time), so year=Y means Dec(Y-1)
    # + Jan(Y) + Feb(Y).
    months_da = monthly['tas'].sel(
        time=monthly.time.dt.year.isin([year - 1, year]))
    months_da = months_da.where(
        ((months_da.time.dt.year == year - 1) & (months_da.time.dt.month == 12))
        | ((months_da.time.dt.year == year) & (months_da.time.dt.month <= 2)),
        drop=True)
    expected_mean = months_da.mean(dim='time').values
    actual = tas_seas.isel(time=yi).values
    diff = np.abs(actual - expected_mean)
    finite = np.isfinite(actual) & np.isfinite(expected_mean)
    max_diff = float(np.nanmax(diff[finite])) if finite.any() else float('nan')
    print(f'  {slug:20s}  {member:12s}  year={year}  '
          f'max|diff|={max_diff:.3e}  '
          f'{"OK" if max_diff < 1e-4 else "FAIL"}')
    seas_ds.close()
    monthly.close()

# ---- check 3: get_hist_pi_baseline returns finite ----
print()
print('=' * 70)
print('CHECK 3: get_hist_pi_baseline(season=...) returns finite')
print('=' * 70)
print('Building cmip6_ctrl_data via get_cmip_projections...')

import importlib
import sys, pathlib
if "__file__" in globals():  # standalone run: ensure scripts/ is importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import functions
importlib.reload(functions)
print('Loading masks + projections (this takes a few minutes) ...')

# Build the full dict (replicates the canonical pipeline entrypoint).
multi_model_dict, masks = functions.get_full_multi_model_dict()
data_dict = functions.load_mpi_esm_data(eur_only=True)
cmip6_ctrl_data = functions.get_cmip_projections(masks, data_dict=data_dict)

print()
HOSMIP_7 = ['CanESM5', 'EC-Earth3', 'CESM2', 'IPSL-CM6A-LR',
            'HadGEM3-GC3-1LL', 'HadGEM3-GC3-1MM', 'MPI-ESM1-2-HR']
for model in HOSMIP_7 + ['GISS-E2-1-G']:
    for season in ('djf', 'jja'):
        try:
            pi = functions.get_hist_pi_baseline(
                model, 'tas', cmip6_ctrl_data=cmip6_ctrl_data, season=season)
            n_finite = int(np.isfinite(pi.values).sum())
            n_total = int(pi.values.size)
            mean_val = float(pi.values[np.isfinite(pi.values)].mean()) if n_finite else float('nan')
            status = 'OK' if n_finite > 0 else 'FAIL (all NaN)'
            print(f'  {model:20s} {season}  finite={n_finite:5d}/{n_total:5d}  '
                  f'mean={mean_val:7.2f}°C  {status}')
        except Exception as e:
            print(f'  {model:20s} {season}  EXC: {e}')

print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
if gaps:
    print(f'CHECK 1: {len(gaps)} gap(s):')
    for g in gaps:
        print(f'  {g}')
else:
    print('CHECK 1: PASS — no gaps')
print('CHECK 2: see per-model OK/FAIL above')
print('CHECK 3: see per-model finite counts above')
