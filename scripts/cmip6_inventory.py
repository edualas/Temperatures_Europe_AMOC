"""CMIP6 filesystem inventory — single source of truth.

Globs the legacy uo1075 archive and the bu1431 staging mirror for files of
the form ``{model_lower}_{variant_id}_{tas|amoc26}_yr.nc``, dedupes across
the two roots, and returns the available realisations. Consumers:

- ``functions.compute_amoc_extent`` — CMIP6 weakening envelope per SSP.
- ``functions.get_cmip_projections`` — paper-set projection load.
- ``cmip_cooling`` — broader CMIP6 decadal-cooling scan.

Disk scans are memoised process-locally. A change-triggered snapshot of the
full inventory lives at ``data/cmip6_inventory_snapshots/latest.json``; a new
dated archive is created only when the on-disk set actually differs from
``latest.json`` (so calling ``available_realisations`` repeatedly is free
after the first call per process). Reproducibility: passing
``snapshot=path`` replays a saved inventory and asserts every listed file
still exists.

JSON dicts under ``/work/uo1075/.../CMIP6/{ssp}/cmip6_{ssp}_{var}_dict.json``
are known incomplete and never consulted.
"""

import datetime as dt
import glob
import json
import os
import subprocess
import threading

import numpy as np
import xarray as xr


# region CONFIG ----------------------------------------------------------------

SEARCH_ROOTS = (
    '/work/uo1075/m300817/teu_amoc/data/CMIP6',
    '/work/bu1431/T_EU_AMOC/CMIP6',
)

SCENARIOS = ('historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585')

# Local cache suffix per ESGF variable_id; ``'amoc'`` is an alias for ``amoc26``.
VAR_SUFFIX = {'tas': 'tas', 'amoc26': 'amoc26', 'amoc': 'amoc26'}

# Authoritative lowercase-directory → ESGF source_id mapping.
LOWER_TO_ESGF = {
    'access-cm2': 'ACCESS-CM2',
    'access-esm1-5': 'ACCESS-ESM1-5',
    'canesm5': 'CanESM5',
    'canesm5-1': 'CanESM5-1',
    'canesm5-canoe': 'CanESM5-CanOE',
    'cas-esm2-0': 'CAS-ESM2-0',
    'cesm2': 'CESM2',
    'cesm2-fv2': 'CESM2-FV2',
    'cesm2-waccm': 'CESM2-WACCM',
    'cesm2-waccm-fv2': 'CESM2-WACCM-FV2',
    'cnrm-cm6-1': 'CNRM-CM6-1',
    'cnrm-esm2-1': 'CNRM-ESM2-1',
    'e3sm-1-0': 'E3SM-1-0',
    'e3sm-1-1': 'E3SM-1-1',
    'e3sm-1-1-eca': 'E3SM-1-1-ECA',
    'ec-earth3': 'EC-Earth3',
    'fgoals-f3-l': 'FGOALS-f3-L',
    'fgoals-g3': 'FGOALS-g3',
    'giss-e2-1-g': 'GISS-E2-1-G',
    'giss-e2-1-g-cc': 'GISS-E2-1-G-CC',
    'giss-e2-2-g': 'GISS-E2-2-G',
    'hadgem3-gc31-ll': 'HadGEM3-GC31-LL',
    'hadgem3-gc31-mm': 'HadGEM3-GC31-MM',
    'icon-esm-lr': 'ICON-ESM-LR',
    'inm-cm4-8': 'INM-CM4-8',
    'inm-cm5-0': 'INM-CM5-0',
    'ipsl-cm6a-lr': 'IPSL-CM6A-LR',
    'miroc-es2l': 'MIROC-ES2L',
    'miroc6': 'MIROC6',
    'mpi-esm-1-2-ham': 'MPI-ESM-1-2-HAM',
    'mpi-esm1-2-hr': 'MPI-ESM1-2-HR',
    'mpi-esm1-2-lr': 'MPI-ESM1-2-LR',
    'mri-esm2-0': 'MRI-ESM2-0',
    'norcpm1': 'NorCPM1',
    'noresm2-lm': 'NorESM2-LM',
    'noresm2-mm': 'NorESM2-MM',
    'sam0-unicon': 'SAM0-UNICON',
    'ukesm1-0-ll': 'UKESM1-0-LL',
}
ESGF_TO_LOWER = {v: k for k, v in LOWER_TO_ESGF.items()}

# ``functions.get_cmip_projections`` uses a non-ESGF spelling for the two
# HadGEM3 variants ('HadGEM3-GC3-1LL' / 'HadGEM3-GC3-1MM' with an extra dash
# between the 3 and the 1). Accept it as a synonym so consumers don't need
# their own mapping.
_LEGACY_ALIASES = {
    'HadGEM3-GC3-1LL': 'HadGEM3-GC31-LL',
    'HadGEM3-GC3-1MM': 'HadGEM3-GC31-MM',
}

# Retracted realisations. Excluded by default; pass ``include_retracted=True``
# to surface them (audit-only callers, sanity checks).
# CESM2 r1–r3 i1p1f1 retracted by NCAR for a forcing-data bug.
RETRACTED = frozenset({
    ('CESM2', 'r1i1p1f1'),
    ('CESM2', 'r2i1p1f1'),
    ('CESM2', 'r3i1p1f1'),
})

SNAPSHOT_DIR = '/home/m/m300940/teu_amoc/data/cmip6_inventory_snapshots'
SNAPSHOT_LATEST = os.path.join(SNAPSHOT_DIR, 'latest.json')

_SCAN_CACHE = {}
_SCAN_LOCK = threading.Lock()
_SNAPSHOT_CHECKED = [False]
_SNAPSHOT_WRITE_LOCK = threading.Lock()

# Per-model authoritative lat/lon grid source — picked once, pinned forever.
# Every file the project opens via ``open_canonical`` is snapped to this
# grid via ``assign_coords``, eliminating the float-precision drift that
# bit us with EC-Earth3 historical (lat-coord float-precision drift).
# The chosen rea is the lowest-numbered
# real historical realisation on disk (CESM2 r1-r3 retracted → r4; GISS
# uses the staging tree, not the main CMIP6 archive). MPI-ESM1-2-LR is
# absent: it goes through MPI-GE not the CMIP6 archive, its grid is
# already self-consistent, and it's the regression engine's reference.
CANONICAL_GRID_SOURCE = {
    'EC-Earth3':       ('historical', 'r1i1p1f1', 'tas'),
    'CESM2':           ('historical', 'r4i1p1f1', 'tas'),
    'MPI-ESM1-2-HR':   ('historical', 'r1i1p1f1', 'tas'),
    'CanESM5':         ('historical', 'r1i1p2f1', 'tas'),
    'IPSL-CM6A-LR':    ('historical', 'r1i1p1f1', 'tas'),
    'HadGEM3-GC31-LL': ('historical', 'r1i1p1f3', 'tas'),
    'HadGEM3-GC31-MM': ('historical', 'r1i1p1f3', 'tas'),
    'GISS-E2-1-G':     ('historical', 'r1i1p1f2', 'tas'),  # GISS_STAGING_PATH
}

# Cache of (lat, lon) numpy arrays per model. Populated lazily on first
# canonical_grid() call.
_GRID_CACHE = {}
_GRID_CACHE_LOCK = threading.Lock()
# endregion


# region NAME RESOLUTION -------------------------------------------------------

def to_esgf(name):
    """Return the canonical ESGF source_id for ``name``.

    Accepts lowercase dir names, ESGF ids, and the legacy 'HadGEM3-GC3-1{LL,MM}'
    spelling used inside ``functions.get_cmip_projections``.
    """
    if name in _LEGACY_ALIASES:
        return _LEGACY_ALIASES[name]
    if name in ESGF_TO_LOWER:
        return name
    return LOWER_TO_ESGF.get(name.lower(), name)


def to_lower(name):
    """Return the canonical lowercase directory name for ``name``."""
    canonical = to_esgf(name)
    return ESGF_TO_LOWER.get(canonical, canonical.lower())


def _suffix(var):
    if var not in VAR_SUFFIX:
        raise ValueError(
            f"var must be one of {sorted(VAR_SUFFIX)}; got {var!r}")
    return VAR_SUFFIX[var]
# endregion


# region SCAN ------------------------------------------------------------------

def _scan_one(scenario, var):
    """Return ``{esgf_id: sorted variant_id list}`` for one (scenario, var)."""
    suffix = _suffix(var)
    out = {}
    for root in SEARCH_ROOTS:
        scen_dir = os.path.join(root, scenario)
        if not os.path.isdir(scen_dir):
            continue
        for model_dir in sorted(glob.glob(os.path.join(scen_dir, '*'))):
            model_local = os.path.basename(model_dir)
            if (not os.path.isdir(model_dir)
                    or model_local.startswith('cmip6_')):
                continue
            esgf = to_esgf(model_local)
            existing = out.setdefault(esgf, set())
            pattern = os.path.join(
                model_dir, f'{model_local}_*_{suffix}_yr.nc')
            for f in glob.glob(pattern):
                name = os.path.basename(f)
                prefix = f'{model_local}_'
                tail = f'_{suffix}_yr.nc'
                if not (name.startswith(prefix) and name.endswith(tail)):
                    continue
                rea = name[len(prefix):-len(tail)]
                if not rea:
                    continue
                existing.add(rea)
    return {m: sorted(reas) for m, reas in out.items()}


def _scan_all():
    """Full ``{var: {scenario: {model: [...]}}}`` scan, memoised per process."""
    with _SCAN_LOCK:
        if 'full' in _SCAN_CACHE:
            return _SCAN_CACHE['full']
        result = {}
        for var in ('tas', 'amoc26'):
            result[var] = {scen: _scan_one(scen, var) for scen in SCENARIOS}
        _SCAN_CACHE['full'] = result
    return result


def clear_cache():
    """Force re-scan on next call. Used by tests; rarely needed in production."""
    with _SCAN_LOCK:
        _SCAN_CACHE.clear()
    _SNAPSHOT_CHECKED[0] = False
# endregion


# region PUBLIC API ------------------------------------------------------------

def available_realisations(scenario, var, models=None, *,
                           esgf_names=True, include_retracted=False,
                           snapshot=None):
    """Return ``{model: sorted list of variant_id strings}`` on disk.

    Parameters
    ----------
    scenario : str
        ``'historical'`` | ``'ssp126'`` | ``'ssp245'`` | ``'ssp370'`` | ``'ssp585'``.
    var : str
        ``'tas'``, ``'amoc26'`` or the alias ``'amoc'``.
    models : list[str] or None
        Optional whitelist. None → every model dir found under
        ``{root}/{scenario}/`` in either search root. Names may be ESGF
        source_ids, lowercase dir names, or the legacy
        ``HadGEM3-GC3-1{LL,MM}`` spelling.
    esgf_names : bool
        True → keys are canonical ESGF source_ids ('CanESM5'); False →
        lowercase dir names ('canesm5') for compatibility with the
        ``cmip_cooling`` lowercase API.
    include_retracted : bool
        Include realisations listed in ``RETRACTED``. Default False.
    snapshot : str or None
        Optional path to a saved snapshot JSON. When given, the saved
        realisation list is returned verbatim (after verifying every file
        still exists on disk via :func:`resolve_path` — missing files raise
        ``FileNotFoundError`` rather than silently dropping).
    """
    if scenario not in SCENARIOS:
        raise ValueError(
            f"scenario must be one of {SCENARIOS}; got {scenario!r}")

    if snapshot is not None:
        return _load_snapshot_for(
            snapshot, scenario, var, models=models,
            esgf_names=esgf_names, include_retracted=include_retracted)

    suffix = _suffix(var)
    full = _scan_all()
    raw = {m: list(rs) for m, rs in full[suffix].get(scenario, {}).items()}
    raw = _apply_retracted(raw, include_retracted=include_retracted)
    raw = _select_and_rekey(raw, models=models, esgf_names=esgf_names)

    _maybe_write_snapshot()
    return raw


def resolve_path(model, scenario, variant_id, var):
    """Locate the file for ``(model, scenario, variant_id, var)`` on disk.

    Tries ``SEARCH_ROOTS`` in order. Special-cases MRI-ESM2-0 ssp126
    r1i1p1f1: the project keeps a 2015–2300 extension at
    ``{LEGACY}/ssp126/mri-esm2-0/extension_2300/`` that replaces (not
    appends to) the standard 2015–2100 cache. Lifted from
    ``cmip_cooling._resolve_path``.
    """
    suffix = _suffix(var)
    lower = to_lower(model)
    if (lower == 'mri-esm2-0' and scenario == 'ssp126'
            and variant_id == 'r1i1p1f1'):
        ext = os.path.join(
            SEARCH_ROOTS[0], 'ssp126', 'mri-esm2-0', 'extension_2300',
            f'mri-esm2-0_{variant_id}_{suffix}_yr.nc')
        if os.path.exists(ext):
            return ext
    for root in SEARCH_ROOTS:
        p = os.path.join(root, scenario, lower,
                         f'{lower}_{variant_id}_{suffix}_yr.nc')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"no {var} cache for {model}/{scenario}/{variant_id} in any of "
        f"{SEARCH_ROOTS}")


def available_pairs(scenario, var, models=None, *, esgf_names=True,
                    include_retracted=False, snapshot=None):
    """Return ``[(model, variant_id), ...]`` present for both ``scenario`` and
    ``'historical'`` under ``var``.

    Useful for envelope computations (e.g. weakening% = scen − hist) where
    both legs must come from the same rea.
    """
    scen = available_realisations(
        scenario, var, models=models, esgf_names=esgf_names,
        include_retracted=include_retracted, snapshot=snapshot)
    hist = available_realisations(
        'historical', var, models=models, esgf_names=esgf_names,
        include_retracted=include_retracted, snapshot=snapshot)
    out = []
    for m, reas in scen.items():
        hist_set = set(hist.get(m, []))
        for r in reas:
            if r in hist_set:
                out.append((m, r))
    return out


def load_snapshot(path):
    """Load a saved snapshot JSON and return its top-level dict."""
    with open(path, 'r') as f:
        return json.load(f)
# endregion


# region HELPERS ---------------------------------------------------------------

def _apply_retracted(d, include_retracted=False):
    """Drop retracted (model_esgf, variant_id) pairs from ``d`` in place."""
    if include_retracted:
        return d
    for model, reas in list(d.items()):
        d[model] = [r for r in reas if (model, r) not in RETRACTED]
    return d


def _select_and_rekey(d, models, esgf_names):
    """Subset by ``models`` whitelist and convert keys.

    ``d`` is keyed by ESGF source_id. When ``models`` is given, the returned
    keys mirror the input spelling (so a caller passing ``'HadGEM3-GC3-1LL'``
    can index the result by the same string). When ``models`` is None,
    ``esgf_names`` controls between ESGF source_ids and lowercase dir names.
    """
    if models is not None:
        out = {}
        for m in models:
            esgf = to_esgf(m)
            reas = d.get(esgf, [])
            if esgf_names:
                # Preserve input spelling: any synonym resolves to the same
                # canonical ensemble, indexed by what the caller passed.
                out[m] = reas
            else:
                out[ESGF_TO_LOWER.get(esgf, esgf.lower())] = reas
        return out
    if esgf_names:
        return d
    return {ESGF_TO_LOWER.get(m, m.lower()): reas for m, reas in d.items()}


def _load_snapshot_for(path, scenario, var, *, models, esgf_names,
                       include_retracted):
    snap = load_snapshot(path)
    suffix = _suffix(var)
    sub = snap['realisations'][suffix].get(scenario, {})
    out = {m: list(rs) for m, rs in sub.items()}
    out = _apply_retracted(out, include_retracted=include_retracted)
    out = _select_and_rekey(out, models=models, esgf_names=esgf_names)
    for m, reas in out.items():
        esgf_m = to_esgf(m) if not esgf_names else m
        for r in reas:
            resolve_path(esgf_m, scenario, r, var)
    return out


def _git_short_hash():
    try:
        sha = subprocess.check_output(
            ['/sw/spack-levante/git-2.43.7-2ofazl/bin/git',
             'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        )
        return sha.decode().strip()
    except Exception:
        return None


def _build_current_snapshot():
    """Build the full inventory snapshot (raw, pre-retracted-filter)."""
    full = _scan_all()
    return {
        'written': dt.datetime.now().isoformat(timespec='seconds'),
        'git_hash': _git_short_hash(),
        'search_roots': list(SEARCH_ROOTS),
        'retracted': sorted([list(t) for t in RETRACTED]),
        'realisations': full,
    }


def _maybe_write_snapshot():
    """Write ``latest.json`` only if the current scan differs from it.

    Process-locally guarded so only the first disk scan per process pays the
    diff cost; subsequent calls return immediately.
    """
    if _SNAPSHOT_CHECKED[0]:
        return
    with _SNAPSHOT_WRITE_LOCK:
        if _SNAPSHOT_CHECKED[0]:
            return
        _SNAPSHOT_CHECKED[0] = True
        try:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        except OSError:
            return  # snapshot dir unwritable; silent skip
        cur = _build_current_snapshot()
        old_realisations = None
        old_written = None
        if os.path.exists(SNAPSHOT_LATEST):
            try:
                with open(SNAPSHOT_LATEST, 'r') as f:
                    old = json.load(f)
                old_realisations = old.get('realisations')
                old_written = old.get('written')
            except (json.JSONDecodeError, OSError):
                old_realisations = None
        if old_realisations == cur['realisations']:
            return  # no change; don't touch latest.json
        if old_realisations is not None and old_written is not None:
            safe_ts = old_written.replace(':', '-')
            archived = os.path.join(SNAPSHOT_DIR, f'{safe_ts}.json')
            if not os.path.exists(archived):
                try:
                    os.rename(SNAPSHOT_LATEST, archived)
                except OSError:
                    pass
        with open(SNAPSHOT_LATEST, 'w') as f:
            json.dump(cur, f, indent=2, sort_keys=True)
# endregion


# region CANONICAL GRID -------------------------------------------------------

def canonical_grid(model):
    """Return ``(lat, lon)`` numpy arrays for ``model``'s authoritative grid.

    Source file pinned in :data:`CANONICAL_GRID_SOURCE`. Lazily cached; the
    first call per process opens one netCDF, subsequent calls are free.
    """
    esgf = to_esgf(model)
    if esgf in _GRID_CACHE:
        return _GRID_CACHE[esgf]
    with _GRID_CACHE_LOCK:
        if esgf in _GRID_CACHE:
            return _GRID_CACHE[esgf]
        if esgf not in CANONICAL_GRID_SOURCE:
            raise ValueError(
                f"No canonical-grid source defined for {esgf!r}. "
                f"Known models: {sorted(CANONICAL_GRID_SOURCE)}.")
        scen, rea, var = CANONICAL_GRID_SOURCE[esgf]
        path = resolve_path(esgf, scen, rea, var)
        with xr.open_dataset(path) as ds:
            _GRID_CACHE[esgf] = (ds.lat.values.copy(), ds.lon.values.copy())
    return _GRID_CACHE[esgf]


def open_canonical(path, model, *, tol=1e-3, as_dataset=False):
    """Open ``path`` and snap lat/lon to ``model``'s canonical grid.

    Returns whatever ``xr.open_dataarray`` / ``xr.open_dataset`` would
    return, with lat/lon coords overridden by the model's canonical grid.
    Cosmetic float drift (the Group A/B family, ~1e-6° magnitude) is
    re-stamped silently. A real grid difference (different resolution or
    geographic extent) larger than ``tol`` raises ``ValueError`` so the
    caller can fail loudly rather than silently mis-align data.

    ``as_dataset`` selects between ``open_dataarray`` (default; for
    single-variable yearly caches) and ``open_dataset`` (True when the
    file has multiple vars or aux dims like ``bnds`` / ``height`` that
    need dropping). Both honour ``use_cftime=True``.
    """
    if as_dataset:
        opened = xr.open_dataset(path, use_cftime=True)
    else:
        opened = xr.open_dataarray(path, use_cftime=True)
    # AMOC@26.5N files (and any other scalar-grid product) lack a 2D lat/lon
    # grid — they're 1D time series. Nothing to snap; return unchanged.
    if 'lat' not in opened.dims or 'lon' not in opened.dims:
        return opened
    can_lat, can_lon = canonical_grid(model)
    if opened.lat.size != can_lat.size or opened.lon.size != can_lon.size:
        raise ValueError(
            f"Grid-shape mismatch in {path}: got "
            f"{opened.lat.size}x{opened.lon.size}, "
            f"canonical {can_lat.size}x{can_lon.size}.")
    # Some HosMIP-pipeline files (e.g. MPI-ESM1-2-HR_u03-hos_tas_yr.nc) have
    # lat in reverse order vs the CMIP6-archive canonical. Sort first so the
    # element-wise drift check is meaningful and the data values land in the
    # canonical orientation. Same treatment for lon, although in practice
    # only lat reversal has been observed.
    if not bool(np.all(np.diff(opened.lat.values) >= 0)):
        opened = opened.sortby('lat')
    if not bool(np.all(np.diff(opened.lon.values) >= 0)):
        opened = opened.sortby('lon')
    lat_drift = float(np.max(np.abs(opened.lat.values - can_lat)))
    lon_drift = float(np.max(np.abs(opened.lon.values - can_lon)))
    if lat_drift > tol or lon_drift > tol:
        raise ValueError(
            f"Real grid mismatch in {path}: max lat drift = "
            f"{lat_drift:.2e}°, max lon drift = {lon_drift:.2e}° "
            f"(tol={tol}). Either pick a new CANONICAL_GRID_SOURCE for "
            f"{model!r} or fix the upstream regrid.")
    return opened.assign_coords(lat=can_lat, lon=can_lon)
# endregion
