"""Derive DJF + JJA seasonal tas for the files we already have annually.

Scope:
- HosMIP-7 historical seasonal tas (MPI-ESM1-2-LR excluded; covered by MPI-GE).
- CNRM-CM6-1, CNRM-ESM2-1, UKESM1-0-LL and MRI-ESM2-0 historical, whose annual
  ensembles had no seasonal counterpart (HIST_EXTRA_TARGETS).
- GISS-E2-1-G historical seasonal tas (r1-r11 i1p1f2 — physics/forcing matches
  `_MODEL_PHYSICS_FORCING` in functions.get_cmip_projections).
- GISS PI seasonal climatology (1850-1899 DJF/JJA mean — sibling of the
  existing yearly tas_pi_climatology.nc).
- SSP members whose seasonal ensemble lags the annual one (SSP_TARGETS).

Outputs land in /work/bu1431/T_EU_AMOC/CMIP6/{scenario}/{model}/tas_{djf,jja}/
and the GISS processed/ directory. Companion edits to functions.py
(_model_root + the historical and SSP season loops, both of which resolve
per file across uo1075 and bu1431) make them visible to the loader.

The DJF recipe is data_process.ipynb cells 214/263. The JJA recipe followed
those cells until 2026-08-21, when `functions.seasonalize` moved to the
`QS-DEC` anchor: the notebook's January-anchored `QS` put months 7-8-9 in the
bin labelled JJA. Cell 2 calibrates against known-good seasonal files on
uo1075 before any new files are written, and is what caught the mismatch.

Run interactively cell-by-cell, or as a standalone script.
"""

# %% imports + config ---------------------------------------------------------
import os
import re
import sys
import json
from glob import glob
from pathlib import Path

import numpy as np
import xarray as xr
import cftime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import functions  # type: ignore

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
    'mri-esm2-0':      {'inst': 'MRI',                  'cmip_name': 'MRI-ESM2-0',      'grid': 'gn'},
    'cnrm-cm6-1':      {'inst': 'CNRM-CERFACS',         'cmip_name': 'CNRM-CM6-1',      'grid': 'gr'},
    'cnrm-esm2-1':     {'inst': 'CNRM-CERFACS',         'cmip_name': 'CNRM-ESM2-1',     'grid': 'gr'},
    'ukesm1-0-ll':     {'inst': 'MOHC',                 'cmip_name': 'UKESM1-0-LL',     'grid': 'gn'},
    'access-cm2':      {'inst': 'CSIRO-ARCCSS',         'cmip_name': 'ACCESS-CM2',      'grid': 'gn'},
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

# CMIP6 models outside the HosMIP set that have annual historical tas but no
# seasonal counterpart. Added 2026-08-21 so the seasonal member basis matches
# the annual one; MRI-ESM2-0 r3 is the member the old uo1075-only member glob
# could not see (it is staged on bu1431).
HIST_EXTRA_TARGETS = {
    'CNRM-CM6-1':  'cnrm-cm6-1',
    'CNRM-ESM2-1': 'cnrm-esm2-1',
    'UKESM1-0-LL': 'ukesm1-0-ll',
    'MRI-ESM2-0':  'mri-esm2-0',
}

GISS_HIST_DIR = f'{UO1075}/historical/giss-e2-1-g'
GISS_PROC = f'{BU1431}/giss-e2-1-g/processed'

# GISS p1f2 ensemble actually consumed downstream (matches the library's
# load_giss_member_amoc_tas members and the get_cmip_projections forcing filter).
GISS_MEMBERS = [f'r{i}i1p1f2' for i in range(1, 12)]  # r1..r11

PI_WINDOW = ('1850', '1899')


# %% helpers -------------------------------------------------------------------
# Both live in the functions library since 2026-08-17 (the Boot et al.
# seasonal loader is a second caller). Re-exported here so `pst.seasonalize`
# keeps working for process_giss_seasonal_to2500.py.
reset_djf_time = functions.reset_djf_time
seasonalize = functions.seasonalize


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


def _bu1431_mirror_dir(model_slug, experiment, member):
    """Return the bu1431 /pool-mirror version dir for (slug, exp, member).

    ``process_mri_full.py`` stages ESGF downloads under
    ``{BU1431}/{slug}/raw/`` in the /pool directory layout; MRI-ESM2-0 SSP
    members are absent from /pool and only available there.
    """
    info = POOL_LOOKUP.get(model_slug)
    if info is None:
        return None
    base = (f'{BU1431}/{model_slug}/raw/{experiment}/{member}/'
            f'Amon/tas/{info["grid"]}')
    if not os.path.isdir(base):
        return None
    versions = sorted(
        d for d in os.listdir(base)
        if d.startswith('v') and os.path.isdir(f'{base}/{d}'))
    return f'{base}/{versions[-1]}' if versions else None


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
    2. /work/bu1431/.../{slug}/raw/ (/pool-layout mirror of ESGF downloads
       — MRI-ESM2-0 SSP members)
    3. /work/bu1431/.../upload/{slug}/ (project upload tree — EC-Earth3
       members not on /pool)

    Returns lazy xr.Dataset, or None if no source is found.
    """
    vdir = (_pool_member_dir(model_slug, experiment, member)
            or _bu1431_mirror_dir(model_slug, experiment, member))
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

    Goes through the inventory rather than globbing uo1075 directly, so
    members staged on bu1431 are seen too — MRI-ESM2-0 historical r3 lives
    only there and a uo1075-only glob silently omitted it.
    """
    return functions.cmip6_inventory.available_realisations(
        scenario, 'tas', models=[model_slug], esgf_names=False)[model_slug]


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

def process_hosmip_historical(targets=None):
    targets = HOSMIP_DIRS if targets is None else targets
    print('=' * 70)
    print('HISTORICAL SEASONAL TAS')
    print('=' * 70)
    summary = []
    for model_disp, slug in targets.items():
        members = list_existing_annual_members(slug, 'historical')
        print(f'\n--- {model_disp} ({slug}): {len(members)} member(s) on disk ---')
        for member in members:
            out_djf = (f'{HIST_OUT_ROOT}/{slug}/tas_djf/'
                       f'{slug}_{member}_tas_djf.nc')
            out_jja = (f'{HIST_OUT_ROOT}/{slug}/tas_jja/'
                       f'{slug}_{member}_tas_jja.nc')
            # a member counts as done only if both seasons exist on *either*
            # root — the loader resolves per file across both, so a bu1431-only
            # check would restage members uo1075 already carries
            if all(any(os.path.exists(f'{root}/historical/{slug}/tas_{s}/'
                                      f'{slug}_{member}_tas_{s}.nc')
                       for root in (UO1075, BU1431))
                   for s in ('djf', 'jja')):
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


# %% Cell 6 — SSP seasonal tas for the members the annual set already has -----
# The consumed seasonal SSP ensembles are proper subsets of their annual
# counterparts (get_cmip_projections prints the shortfall and loads the subset).
# Targets are declared explicitly rather than looped over POOL_LOOKUP, because
# three documented integrity traps intersect exactly this path — see
# _ssp_gate below and the project memory note project_cmip6_bad_member_preflight.

SSP_OUT_ROOT = BU1431
SSP_TARGETS = {           # slug -> scenarios to fill
    'ec-earth3':  ['ssp245'],
    'cesm2':      ['ssp126', 'ssp370'],
    'mri-esm2-0': ['ssp126', 'ssp245'],   # monthly source on bu1431, not /pool
    'access-cm2': ['ssp126'],
}
# GISS ssp245 tas on the DKRZ replicas (/pool, /work/ik1017) is the February
# 2020 pre-erratum publication for r1-r5, ~0.5 K off at 2100; the canonical
# source is the NCCS mirror under bu1431. Refuse rather than omit, so a later
# widening of SSP_TARGETS cannot walk into it.
SSP_FORBIDDEN = {('giss-e2-1-g', 'ssp245'):
                 'pre-erratum on DKRZ replicas; use the NCCS mirror '
                 '(GISS-E2-1-G ssp245 r1-r5 pre-erratum)'}
# CESM2 ssp370 r5 and r6 carry byte-for-byte identical tas upstream (verified
# 2026-08-17 on the uo1075 annual caches and on an independent ESGF download:
# max|r5-r6| = 0 at every timestep, while every other pair differs by 6-8 K).
# Staging both would double-count one realisation, so neither is staged until
# the annual ensemble's own duplicate is resolved.
SSP_FORBIDDEN_MEMBERS = {
    ('cesm2', 'ssp370', 'r5i1p1f1'): 'duplicate of r6i1p1f1 upstream',
    ('cesm2', 'ssp370', 'r6i1p1f1'): 'duplicate of r5i1p1f1 upstream',
}
SSP_MONTHS = 1032                    # 2015-01..2100-12
SSP_SPAN = (201501, 210012)


def _file_coverage(files):
    """Months covered by ``files``, from their YYYYMM-YYYYMM suffixes.

    Cheap stand-in for opening them: chunking is model-specific (EC-Earth3
    publishes one file per year, CESM2 one or two per scenario), so file count
    is not a usable invariant but the covered span is.
    """
    months = set()
    for f in files:
        m = re.search(r'_(\d{6})-(\d{6})\.nc$', f)
        if not m:
            return None
        for y in range(int(m.group(1)) // 100, int(m.group(2)) // 100 + 1):
            lo = int(m.group(1)) % 100 if y == int(m.group(1)) // 100 else 1
            hi = int(m.group(2)) % 100 if y == int(m.group(2)) // 100 else 12
            months |= {y * 100 + mm for mm in range(lo, hi + 1)}
    return months


def _ssp_source(slug, scenario, member):
    """(label, files) for the monthly source open_monthly_pool will use."""
    vdir = _pool_member_dir(slug, scenario, member)
    if vdir is not None:
        return os.path.basename(vdir), sorted(glob(f'{vdir}/*.nc'))
    mdir = _bu1431_mirror_dir(slug, scenario, member)
    if mdir is not None:
        return f'bu1431-raw/{os.path.basename(mdir)}', sorted(glob(f'{mdir}/*.nc'))
    up = _bu1431_upload_files(slug, scenario, member)
    return ('bu1431-upload', up) if up else (None, [])


def _ssp_gate(slug, scenario, member):
    """Return None if (slug, scenario, member) is safe to stage, else why not."""
    if (slug, scenario) in SSP_FORBIDDEN:
        return f'forbidden: {SSP_FORBIDDEN[(slug, scenario)]}'
    if (slug, scenario, member) in SSP_FORBIDDEN_MEMBERS:
        return f'forbidden: {SSP_FORBIDDEN_MEMBERS[(slug, scenario, member)]}'
    cmip_name = POOL_LOOKUP[slug]['cmip_name']
    if (cmip_name, member) in functions.cmip6_inventory.RETRACTED:
        return 'retracted realisation (NCAR forcing-data bug)'
    label, files = _ssp_source(slug, scenario, member)
    if not files:
        return 'on neither /pool nor the bu1431 upload tree (needs an ESGF fetch)'
    # EC-Earth3 publishes 2-3 version dirs per member and _pool_member_dir
    # takes the newest; ssp245 r23 is the counterexample where the newest holds
    # a single year. Refuse rather than write a truncated seasonal file.
    cov = _file_coverage(files)
    if cov is None:
        return f'{label}: unparseable filename span'
    want = {y * 100 + m for y in range(SSP_SPAN[0] // 100, SSP_SPAN[1] // 100 + 1)
            for m in range(1, 13)}
    if not want <= cov:
        miss = sorted(want - cov)
        return (f'{label} covers {len(cov)} months, missing '
                f'{len(miss)} incl. {miss[0]}')
    return None


def process_ssp_seasonal(targets=None, dry_run=False):
    targets = targets or SSP_TARGETS
    print('=' * 70)
    print(f'SSP SEASONAL TAS{" (DRY RUN)" if dry_run else ""}')
    print('=' * 70)
    summary, provenance = [], []
    for slug, scenarios in targets.items():
        for scenario in scenarios:
            annual = list_existing_annual_members(slug, scenario)
            out_dir = f'{SSP_OUT_ROOT}/{scenario}/{slug}'
            # a member counts as done only if both seasons exist on *either*
            # root — the loader looks in both, so the skip must too, or a
            # re-run silently rewrites everything it already staged
            def _done(m):
                return all(any(os.path.exists(f'{root}/{scenario}/{slug}/tas_{s}/{slug}_{m}_tas_{s}.nc')
                               for root in (UO1075, SSP_OUT_ROOT))
                           for s in ('djf', 'jja'))
            todo = [m for m in annual if not _done(m)]
            print(f'\n--- {slug} {scenario}: {len(annual)} annual, '
                  f'{len(annual) - len(todo)} already seasonal, {len(todo)} to fill ---')
            for member in todo:
                why = _ssp_gate(slug, scenario, member)
                if why:
                    print(f'  SKIP {member}: {why}')
                    summary.append((slug, scenario, member, f'skip:{why}'))
                    continue
                label, src_files = _ssp_source(slug, scenario, member)
                if dry_run:
                    # the open is the expensive part (86 files/member); the gate
                    # above already covered everything a dry run can check
                    print(f'  OK   {member}: would write from {label}')
                    summary.append((slug, scenario, member, 'dry_ok'))
                    continue
                monthly = open_monthly_pool(slug, scenario, member)
                if monthly is None:
                    print(f'  SKIP {member}: open returned None')
                    summary.append((slug, scenario, member, 'skip:open_none'))
                    continue
                # Some members publish a 2101-2300 extension in the same
                # version dir (ACCESS-CM2 ssp126 r1), which would otherwise
                # produce a seasonal file on a different axis from its
                # ensemble siblings. Crop to the SSP window before the check.
                monthly = monthly.sel(time=slice(str(SSP_SPAN[0] // 100),
                                                 str(SSP_SPAN[1] // 100)))
                n_t = monthly.sizes.get('time', 0)
                if n_t != SSP_MONTHS:
                    print(f'  SKIP {member}: time.size={n_t} != {SSP_MONTHS}')
                    summary.append((slug, scenario, member, f'skip:time{n_t}'))
                    monthly.close()
                    continue
                provenance.append({'slug': slug, 'scenario': scenario, 'member': member,
                                   'version_dir': label,
                                   'n_files': len(src_files), 'time_size': n_t})
                try:
                    for season in ('djf', 'jja'):
                        out = f'{out_dir}/tas_{season}/{slug}_{member}_tas_{season}.nc'
                        if not os.path.exists(out):
                            write_seasonal(monthly, 'tas', season, 2015, out)
                    print(f'  wrote {member} (from {label})')
                    summary.append((slug, scenario, member, 'wrote'))
                except Exception as e:
                    print(f'  FAIL write {member}: {e}')
                    summary.append((slug, scenario, member, f'fail_write:{e}'))
                finally:
                    monthly.close()
    if provenance and not dry_run:
        path = f'{SSP_OUT_ROOT}/ssp_seasonal_provenance.json'
        old = json.load(open(path)) if os.path.exists(path) else []
        # keyed on (slug, scenario, member) so a re-run replaces a member's
        # record instead of appending a second one
        merged = {(r['slug'], r['scenario'], r['member']): r
                  for r in old + provenance}
        json.dump([merged[k] for k in sorted(merged)], open(path, 'w'), indent=1)
        print(f'\nprovenance -> {path} ({len(provenance)} new, '
              f'{len(merged)} total records)')
    return summary


# %% main ---------------------------------------------------------------------
if __name__ == '__main__':
    # Step 1: calibration — halt if the recipe drifts from the notebook.
    _calibrate()

    # Step 2: HosMIP-7 historical, then the non-HosMIP models whose annual
    # historical ensembles had no seasonal counterpart.
    hosmip_summary = process_hosmip_historical()
    hosmip_summary += process_hosmip_historical(HIST_EXTRA_TARGETS)

    # Step 3: GISS historical.
    giss_summary = process_giss_historical()

    # Step 4: GISS PI seasonal climatology (depends on step 3).
    pi_paths = build_giss_pi_climatology()

    # Step 5: SSP members whose seasonal ensemble lags the annual one.
    ssp_summary = process_ssp_seasonal()

    # Step 6: summary.
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    from collections import Counter
    hc = Counter([s for _, _, s in hosmip_summary])
    gc = Counter([s for _, s in giss_summary])
    sc = Counter([s for _, _, _, s in ssp_summary])
    print('Historical (HosMIP-7 + extras):')
    for k, v in sorted(hc.items()):
        print(f'  {k}: {v}')
    print('GISS historical:')
    for k, v in sorted(gc.items()):
        print(f'  {k}: {v}')
    print('GISS PI climatology:')
    for season, path in pi_paths.items():
        print(f'  {season}: {path}')
    print('SSP seasonal:')
    for k, v in sorted(sc.items()):
        print(f'  {k}: {v}')
