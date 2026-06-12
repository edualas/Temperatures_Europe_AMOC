# region PACKAGES

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle
from matplotlib import ticker
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
plt.rcParams.update({'font.size': 12})
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.geometry import LineString, Polygon
from shapely.ops import split, unary_union
import xarray as xr
import regionmask
import statsmodels.api as sm
from scipy import stats as scipy_stats
import pickle
import colorsys
import datetime as _dt
import json as _json
import warnings as _warnings

import cmip6_inventory # type: ignore

# endregion

########################################################################################################################
# GLOBAL DEFINITIONS
########################################################################################################################

# region GLOBAL DEFINITIONS

global_plot_bg = 'white'

data_path = '/work/uo1075/m300817/teu_amoc/data/'
local_path = '/home/m/m300940/teu_amoc/data/'
# CMIP6 AMOC-weakening envelope sidecar (see compute_amoc_extent).
AMOC_EXTENT_CACHE = local_path + 'cmip6_amoc_extent.json'
# Sidecar keys for the four (aggregation, unit) envelope variants.
AMOC_EXTENT_KEYS = {
    ('realisation', 'pct'): 'bounds_pct', ('model', 'pct'): 'bounds_pct_model',
    ('realisation', 'sv'): 'bounds_sv', ('model', 'sv'): 'bounds_sv_model',
}

# CMIP6 raw-data root + per-model documented (physics, forcing) variant: only
# r{N}i1{p}{f} files are admitted; the integer N's come from disk.
cmip6_data_path = '/work/uo1075/m300817/teu_amoc/data/CMIP6/'
MODEL_PHYSICS_FORCING = {
        'HadGEM3-GC3-1LL': ('p1', 'f3'),
        'HadGEM3-GC3-1MM': ('p1', 'f3'),
        'EC-Earth3':       ('p1', 'f1'),
        'CESM2':           ('p1', 'f1'),
        'MPI-ESM1-2-HR':   ('p1', 'f1'),
        'CanESM5':         ('p2', 'f1'),
        'IPSL-CM6A-LR':    ('p1', 'f1'),
        'GISS-E2-1-G':     ('p1', 'f2'),
    }


GISS_MEMBERS = [f'r{i}i1p1f2' for i in range(1, 11)]


def compute_amoc_pi_hist_scalar(model):
    """1850–1899 CMIP6 historical ensemble-mean AMOC@26.5N (Sv).

    Module-load helper for the two scalars (``AMOC_pi_MPI``,
    ``AMOC_pi_GISS``) that flow through the conversion utilities and
    plot scaling. Reads only the per-realisation yearly AMOC files (one
    small NetCDF for MPI-ESM-LR via MPI-GE; ten for GISS-E2-1-G p1f2),
    so module import stays under ~2 s. The per-model HosMIP/CESM2
    baselines use ``get_hist_pi_baseline`` instead (those need
    ``cmip6_ctrl_data`` which is built lazily inside the loaders).
    """
    if model == 'MPI-ESM1-2-LR':
        da = xr.open_dataarray(
            data_path + 'MPI-GE/his_amoc26_yr.nc', use_cftime=True)
        return float(da.sel(time=slice('1850', '1899'))
                     .mean(['time', 'realiz']).values)
    if model == 'GISS-E2-1-G':
        hist_dir = data_path + 'CMIP6/historical/giss-e2-1-g'
        members = [f'r{i}i1p1f2' for i in range(1, 11)]
        da = xr.concat([
            xr.open_dataarray(
                f'{hist_dir}/giss-e2-1-g_{r}_amoc26_yr.nc', use_cftime=True)
            for r in members
        ], dim='realiz')
        return float(da.sel(time=slice('1850', '1899'))
                     .mean(['time', 'realiz']).values)
    raise ValueError(f'No scalar AMOC_pi defined for model {model}')


# 1850–1899 CMIP6 historical ensemble-mean AMOC, MPI-ESM1.2-LR (MPI-GE).
# Previously (piControl scalar): 19.0154 Sv. New value differs by ≈0.02 %.
AMOC_pi_MPI = compute_amoc_pi_hist_scalar('MPI-ESM1-2-LR')

# 1850–1899 CMIP6 historical ensemble-mean AMOC over the canonical 10
# GISS-E2-1-G p1f2 members. Previously (piControl scalar, Romanou et al.
# 2023 ``constants.json``): 24.3409 Sv. New value is numerically identical
# to 4 dp; the relabelling resolves the historical/piControl naming drift.
# GISS_STAGING_PATH: staging tree consumed by ``get_cmip_projections``.
# GISS_PROCESSED_PATH: per-member his + ssp245-to-2500 caches.
AMOC_pi_GISS = compute_amoc_pi_hist_scalar('GISS-E2-1-G')
GISS_COMPOSITE = ['r3i1p1f2', 'r4i1p1f2', 'r10i1p1f2']
GISS_NON_COMPOSITE = [m for m in GISS_MEMBERS if m not in GISS_COMPOSITE]
# Time periods stored along the ``time_period`` dim of ``reg_ds_giss``.
GISS_TIME_PERIODS = [('2015', '2500'), ('2101', '2300')]
GISS_STAGING_PATH = '/work/bu1431/T_EU_AMOC/CMIP6/giss-e2-1-g/cmip_proj_staging/'
GISS_PROCESSED_PATH = '/work/bu1431/T_EU_AMOC/CMIP6/giss-e2-1-g/processed/'

# NAMED SCIENTIFIC CONSTANTS
# These were previously inlined throughout the regression engine. Lifting them
# here gives the values a single source of truth. **Numerical values must
# remain unchanged from the submitted state.**
PI_WINDOW = ('1850', '1899')      # 50-yr pre-industrial baseline (MPI-GE).
FUTURE_WINDOW = ('2091', '2100')  # 10-yr end-of-century target window (MPI-GE).
PD_HIS_WINDOW = ('1991', '2014')  # historical leg of present-day reference.
PD_SSP_WINDOW = ('2015', '2020')  # ssp245 leg of present-day reference.
PD_LABEL = f'{PD_HIS_WINDOW[0]}–{PD_SSP_WINDOW[1]}'  # e.g. '1991–2020'
# 10-yr centred rolling mean applied to AMOC and TAS series prior to
# regression. Default for ``simulations_plot``, ``regression_plot``,
# ``hosmip_regression_plot``, ``cesm_regressions``.
ROLLING_WINDOW = 10


# STRING-ARG ENUMS
# Allowed values for the string-dispatch kwargs checked by ``validate_choice``
# (in HELPER FUNCTIONS). ``region`` is deliberately excluded — its allowed
# values are dynamic (per-model masks dict), and a typo there already raises a
# noisy ``KeyError`` on mask lookup.
ALLOWED_SEASONS = ('', 'djf', 'jja')
ALLOWED_VARS = ('tas', 'tmn', 'pr')
ALLOWED_HOSING_TYPES = ('all', 'all1Sv', 'constant', 'linear', '1', None)
ALLOWED_T_REF = ('pi', 'pd')
ALLOWED_PLOT_BG = ('white', 'black')

ssps = ['ssp126', 'ssp245', 'ssp370']

# regions = ['RU', 'NO', 'FR', 'SE', 'BY', 'UA', 'PL', 'AT', 'HU', 'MD', 'RO', 'LT', 'LV', 'EE', 'DE', 'BG', 'GR', 'TR', 'HR', 'CH', 'BE', 'NL', 'PT', 'ES', 'IE', 'IT', 'DK', 'GB', 'SI', 'FI', 'SK', 'CZ', 'MK', 'RS', 'XK', 'IS']

# these are sorted by effect size in range plot for both MPI-ESM and CESM2 data
regions = ['IE', 'GB', 'IS', 'NO', 'NL', 'DK', 'SE', 'BE', 'PT', 'FI', 'EE', 'FR', 'LT', 'LV', 'DE', 'LU', 'RU', 'PL', 'BY', 'ES', 'CZ', 'UA', 'MD', 'SK', 'CH', 'AT', 'RO', 'GR', 'BG', 'HU', 'AL', 'TR', 'HR', 'ME', 'RS', 'IT', 'XK', 'SI', 'BA', 'MK', 'CY']

ssp_labels = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5"
}

hosing_colors = {
    'ssp126': {
        # Exact base RGBA value:
        'ge': np.array([23/255, 60/255, 102/255]) * 2.5 if global_plot_bg == 'black' else (23/255, 60/255, 102/255),
        # Slightly darker blues for negative hosing:
        'neg01': 'midnightblue',
        # Slightly lighter / more saturated blues for positive hosing:
        '01': 'steelblue',
        '03': 'cornflowerblue',
        '05': 'deepskyblue',
        '1': 'green',
        # For linear hosing, shift a bit toward greenish tones:
        'linneg02': 'teal',
        'lin02': 'cadetblue',
        'lin06': 'mediumaquamarine',
        'lin10': 'aquamarine'
    },
    'ssp245': {
        'ge': (247/255, 148/255, 32/255),
        'neg01': 'darkorange',
        '01': 'gold',
        '03': 'yellow',
        '05': 'navajowhite',
        '1': '#d35400',
        'linneg02': 'olive',
        'lin02': 'darkkhaki',
        'lin06': 'khaki',
        'lin10': 'palegoldenrod'
    },
    'ssp370': {
        'ge': (231/255, 29/255, 37/255),
        'neg01': 'darkred',
        '01': 'orangered',
        '03': 'salmon',
        '05': 'lightsalmon',
        '1': 'green',
        'linneg02': 'mediumorchid',
        'lin02': 'hotpink',
        'lin06': 'plum',
        'lin10': 'pink'
    },
    'ssp585': {
        'ge': (149/255, 27/255, 30/255),
    }
}

hosing_names = dict(zip(['ssp126', 'ssp245', 'ssp370'], 
                        [dict(zip(['ge', 'neg01', '01', '03', '05', '1', 'linneg02', 'lin02', 'lin06', 'lin10'], 
                                  ['MPI-ESM GE SSP1-2.6', 'SSP1-2.6  -0.1Sv', 'SSP1-2.6 +0.1Sv', 'SSP1-2.6 +0.3Sv', 'SSP1-2.6 +0.5Sv', 'SSP1-2.6 +1.0Sv', 'SSP1-2.6 lin. -0.2Sv', 'SSP1-2.6 lin.+0.2Sv', 'SSP1-2.6 lin.+0.6Sv', 'SSP1-2.6 lin.+1.0Sv'])), 
                         dict(zip(['ge', 'neg01', '01', '03', '05', '1', 'linneg02', 'lin02', 'lin06', 'lin10'], 
                                  ['MPI-ESM GE SSP2-4.5', 'SSP2-4.5  -0.1Sv', 'SSP2-4.5 +0.1Sv', 'SSP2-4.5 +0.3Sv', 'SSP2-4.5 +0.5Sv', 'SSP2-4.5 +1.0Sv', 'SSP2-4.5 lin. -0.2Sv', 'SSP2-4.5 lin.+0.2Sv', 'SSP2-4.5 lin.+0.6Sv', 'SSP2-4.5 lin.+1.0Sv'])),
                         dict(zip(['ge', 'neg01', '01', '03', '05', 'linneg02', 'lin02', 'lin06', 'lin10'], 
                                  ['MPI-ESM GE SSP3-7.0', 'SSP3-7.0  -0.1Sv', 'SSP3-7.0 +0.1Sv', 'SSP3-7.0 +0.3Sv', 'SSP3-7.0 +0.5Sv', 'SSP3-7.0 lin. -0.2Sv', 'SSP3-7.0 lin.+0.2Sv', 'SSP3-7.0 lin.+0.6Sv', 'SSP3-7.0 lin.+1.0Sv'])),
                         ]))

hosing_names['ssp585'] = {}
hosing_names['ssp585']['ge'] = 'MPI-ESM GE SSP5-8.5'

hosing_symbols = {
    'ge': '●',
    '01': '▲',
    '03': '✕',
    '05': '★',
    '1': '◆',
    'neg01': '▼',
    'lin02': '▶',
    'lin06': '⬟',
    'lin10': '■',
    'linneg02': '◀',
}

hosing_markers = {
    'ge': 'o',
    '01': '^',
    '03': 'x',
    '05': '*',
    '1': 'D',
    'neg01': 'v',
    'lin02': '>',
    'lin06': 'p',
    'lin10': 's',
    'linneg02': '<',
}

hosmip_labels = ['CanESM5', 'EC-Earth3', 'CESM2', 'IPSL-CM6A-LR', 'HadGEM3-GC3-1MM', 'HadGEM3-GC3-1LL', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR']

hosmip_colors = {
    'CanESM5': 'mediumseagreen', #'mediumturquoise',
    'EC-Earth3': 'darkturquoise', #'cadetblue',
    'CESM2': 'steelblue',
    'IPSL-CM6A-LR': 'darkslateblue',
    'HadGEM3-GC3-1MM': 'mediumorchid',
    'HadGEM3-GC3-1LL': 'plum',
    'MPI-ESM1-2-HR': 'mediumvioletred',
    'MPI-ESM1-2-LR': 'brown' #'firebrick' #'crimson' 'goldenrod
}

# this is for loading seasonal CMIP data, where dimensions and variables differ between models
drop_stuff = {}
drop_stuff['MPI-ESM1-2-HR'] = {}
drop_stuff['MPI-ESM1-2-HR']['dim'] = ['bnds']
drop_stuff['MPI-ESM1-2-HR']['var'] = ['height']

drop_stuff['MPI-ESM1-2-LR'] = {}
drop_stuff['MPI-ESM1-2-LR']['dim'] = ['bnds']
drop_stuff['MPI-ESM1-2-LR']['var'] = ['height']

drop_stuff['CanESM5'] = {}
drop_stuff['CanESM5']['dim'] = ['bnds']
drop_stuff['CanESM5']['var'] = ['height']

drop_stuff['CESM2'] = {}
drop_stuff['CESM2']['dim'] = 'nbnd'

drop_stuff['IPSL-CM6A-LR'] = {}
drop_stuff['IPSL-CM6A-LR']['var'] = ['height']

drop_stuff['HadGEM3-GC3-1LL'] = {}
drop_stuff['HadGEM3-GC3-1LL']['dim'] = ['bnds']
drop_stuff['HadGEM3-GC3-1LL']['var'] = ['height']

drop_stuff['HadGEM3-GC3-1MM'] = {}
drop_stuff['HadGEM3-GC3-1MM']['dim'] = ['bnds']
drop_stuff['HadGEM3-GC3-1MM']['var'] = ['height']

drop_stuff['EC-Earth3'] = {}
drop_stuff['EC-Earth3']['dim'] = ['bnds']
drop_stuff['EC-Earth3']['var'] = ['height']

drop_stuff['GISS-E2-1-G'] = {}
drop_stuff['GISS-E2-1-G']['dim'] = ['bnds']
drop_stuff['GISS-E2-1-G']['var'] = ['height']


def get_hosmip_colors(plot_bg='white'):
    if plot_bg == 'black':
        # Lighter colors for dark background - blue to green spectrum
        return {
            'CanESM5': '#4A90E2',      # Bright blue
            'EC-Earth3': '#5BA3D4',    # Blue-cyan
            'CESM2': '#6BB6C6',        # Cyan
            'IPSL-CM6A-LR': '#7BC9B8', # Blue-green
            'HadGEM3-GC3-1MM': '#8BDCAA', # Green-cyan
            'HadGEM3-GC3-1LL': '#9BEF9C', # Light green
            'MPI-ESM1-2-HR': '#6DA9D4', # Medium blue-cyan
            'MPI-ESM1-2-LR': '#A8A8A8'  # Gray for contrast
        }
    else:  # white background
        # Darker colors for light background - blue to green spectrum
        return {
            'CanESM5': '#1E3A8A',      # Dark blue
            'EC-Earth3': 'darkturquoise',
            'CESM2': 'steelblue',
            'IPSL-CM6A-LR': '#1E9A6E', # Dark blue-green
            'HadGEM3-GC3-1MM': '#808080', #  gray for contrast '#1EBA5A', # Dark green-cyan
            'HadGEM3-GC3-1LL': '#2EDA46', # Dark green
            'MPI-ESM1-2-HR': '#1E7A82', # Dark cyan
            'MPI-ESM1-2-LR': 'blueviolet' #'brown' 
        }

# Update your existing code:
hosmip_colors = get_hosmip_colors()

country_names = {
    'RU': 'Russia',
    'NO': 'Norway', 
    'FR': 'France',
    'SE': 'Sweden',
    'BY': 'Belarus',
    'UA': 'Ukraine',
    'PL': 'Poland',
    'AT': 'Austria',
    'HU': 'Hungary',
    'MD': 'Moldova',
    'RO': 'Romania',
    'LT': 'Lithuania',
    'LV': 'Latvia',
    'EE': 'Estonia',
    'DE': 'Germany',
    'BG': 'Bulgaria',
    'GR': 'Greece',
    'TR': 'Turkey',
    'HR': 'Croatia',
    'CH': 'Switzerland',
    'BE': 'Belgium',
    'NL': 'Netherlands',
    'PT': 'Portugal',
    'ES': 'Spain',
    'IE': 'Ireland',
    'IT': 'Italy',
    'DK': 'Denmark',
    'GB': 'United Kingdom',
    'SI': 'Slovenia',
    'FI': 'Finland',
    'SK': 'Slovakia',
    'CZ': 'Czechia',
    'MK': 'North Macedonia',
    'RS': 'Serbia',
    'XK': 'Kosovo',
    'IS': 'Iceland',
    'BA': 'Bosnia and Herz.',
    'AL': 'Albania',
    'ME': 'Montenegro',
    'CY': 'Cyprus',
    'LU': 'Luxembourg',
}

country_codes = {
    'RU': 18,  # Russia
    'NO': 21,  # Norway
    'FR': 43,  # France
    'SE': 110, # Sweden
    'BY': 111, # Belarus
    'UA': 112, # Ukraine
    'PL': 113, # Poland
    'AT': 114, # Austria
    'HU': 115, # Hungary
    'MD': 116, # Moldova
    'RO': 117, # Romania
    'LT': 118, # Lithuania
    'LV': 119, # Latvia
    'EE': 120, # Estonia
    'DE': 121, # Germany
    'BG': 122, # Bulgaria
    'GR': 123, # Greece
    'TR': 124, # Turkey
    'HR': 126, # Croatia
    'CH': 127, # Switzerland
    'BE': 129, # Belgium
    'NL': 130, # Netherlands
    'PT': 131, # Portugal
    'ES': 132, # Spain
    'IE': 133, # Ireland
    'IT': 141, # Italy
    'DK': 142, # Denmark
    'GB': 143, # United Kingdom
    'SI': 150, # Slovenia
    'FI': 151, # Finland
    'SK': 152, # Slovakia
    'CZ': 153, # Czechia
    'RS': 172, # Serbia
    'XK': 174, # Kosovo
    'IS': 144, # Iceland
    'MK': 171, # North Macedonia
    # 'CN': 160, # N. Cyprus
    'ME': 173, # Montenegro
    'CY': 161, # Cyprus
    'BA': 170, # Bosnia and Herz.
    'AL': 125, # Albania
    'LU': 128, # Luxembourg
}

# endregion

########################################################################################################################
# HELPER FUNCTIONS
########################################################################################################################

def validate_choice(name, value, allowed):
    """Raise ``ValueError`` if ``value`` is not in ``allowed``.

    Used to catch silent typos in string-dispatch kwargs (``season``,
    ``var``, ``hosing``/``hos_type``, ``T_ref``, ``plot_bg``) against the
    ALLOWED_* enums in GLOBAL DEFINITIONS. Keep the error message verbose so
    an interactive user spots the typo quickly.
    """
    if value not in allowed:
        raise ValueError(
            f"{name}={value!r} not in {allowed}. "
            "Check the call site for a typo."
        )


def get_hist_pi_baseline(model, var, cmip6_ctrl_data=None,
                         multi_model_dict=None, season=''):
    """1850–1899 CMIP6 historical ensemble-mean baseline.

    Per-model baseline lookup used by ``load_regression_ds_mpi``,
    ``get_cesm_reg_ds``, ``get_hosmip_reg_ds``, ``get_giss_reg_ds`` and
    ``hosmip_regression_plot`` to anchor AMOC and tas anomalies to the
    **same** CMIP6 historical 1850–1899 ensemble mean across every loader
    (harmonised 2026-05-25). Returns a scalar (Sv) for ``var='amoc'``; a (lat, lon)
    DataArray (°C) for ``var='tas'``. ``cmip6_ctrl_data`` must already be
    built via ``get_cmip_projections``; ``multi_model_dict`` is retained in
    the signature for backward compatibility but no longer consulted.
    EC-Earth3 follows this convention too (20 historical reas auto-discovered
    on disk via the unified inventory), so the former NAHosMIP piControl
    fallback was removed (2026-05-26 a).
    """
    if var not in ('amoc', 'tas'):
        raise ValueError(f"var must be 'amoc' or 'tas'; got {var!r}")
    if cmip6_ctrl_data is None or model not in cmip6_ctrl_data:
        raise ValueError(
            f"get_hist_pi_baseline needs cmip6_ctrl_data[{model!r}].")
    ds = cmip6_ctrl_data[model]
    if var == 'amoc':
        return float(ds['amoc'].sel(scenar='his')
                     .sel(time=slice(*PI_WINDOW)).mean('time').values)
    return ds['tas'].sel(scenar='his', season=season) \
                    .sel(time=slice(*PI_WINDOW)).mean('time')


def resolve_future_window(future_window=None):
    """Normalise ``future_window`` to a tuple, defaulting to ``FUTURE_WINDOW``.

    ``FUTURE_WINDOW`` is the end-of-century target window averaged to form the
    ``AMOC_future`` / ``T_future`` fields (and everything derived: req_strength,
    req_weakening, the CMIP6 ``AMOC_extent`` envelope, the net-cooling maps). To
    run a sensitivity on that window without touching the canonical caches, every
    loader accepts ``future_window=None`` (None -> the module default) and keys
    its cache file by the window via ``fw_suffix``: the canonical window yields
    the existing no-suffix filename; any other window writes a parallel
    ``_fw{start}-{end}`` variant alongside. ``resolve_future_window`` and
    ``fw_suffix`` are the single source of truth for that convention; the window
    used is also stamped into each dataset's ``future_window`` attribute.
    """
    return FUTURE_WINDOW if future_window is None else tuple(future_window)


def fw_suffix(future_window=None):
    """Cache-filename suffix for a future window. '' for the canonical window
    so canonical filenames are byte-unchanged; ``_fw-{start}-{end}`` otherwise."""
    fw = resolve_future_window(future_window)
    return '' if fw == FUTURE_WINDOW else f'_fw-{fw[0]}-{fw[1]}'


def amoc_extent_cache_path(future_window=None):
    """Window-keyed path for the CMIP6 AMOC-extent JSON sidecar."""
    base, ext = os.path.splitext(AMOC_EXTENT_CACHE)
    return f'{base}{fw_suffix(future_window)}{ext}'


def check_diag_cache(cache, version, label):
    """Return the cached per-pixel diagnostics Dataset if it exists and is at
    least ``version``, else None (closing and announcing a recompute on a stale
    hit). Shared cache-version gate for the linearity / scenario-independence
    ``get_*`` diagnostic loaders."""
    if not os.path.exists(cache):
        return None
    cached = xr.open_dataset(cache)
    if int(cached.attrs.get('version', 0)) >= version:
        print(f'Loading precomputed {label} diagnostics...')
        return cached
    cached.close()
    print(f'Existing {label} cache is v{cached.attrs.get("version", 0)} < v{version}; recomputing.')
    return None


def reset_diagnostics(y, X, exp_id, model, cov_spec=None):
    """Ramsey RESET-style linearity diagnostic.

    Returns a dict with partial_r2 (in [0,1]; share of M0 residual variance
    absorbed by curvature), delta_r2_total (share of total y variance absorbed
    by curvature; algebraically partial_r2 * (1 - baseline_r2)), gamma2/gamma3
    (augmented-model coefficients on yhat^2 / yhat^3 — use these to plot the
    curvature the test detected), beta2_sign/beta3_sign (signs of gamma2/gamma3
    kept for legacy callers), reset_pvalue (robust F test on
    H0: gamma2=gamma3=0), baseline_r2, yhat, resid. NaN on degenerate cases.

    cov_spec=None (default) reproduces the legacy MPI behaviour:
    cov_type='cluster' with groups=exp_id, use_correction=True. Pass a tuple
    (cov_type, cov_kwds) to override — e.g. ('HAC', {'maxlags': 9}) for the
    CESM2 combined-forcing case where G=2 makes cluster-robust SE degenerate.
    """
    import statsmodels.api as _sm

    nan_result = {
        'partial_r2': np.nan,
        'delta_r2_total': np.nan,
        'gamma2': np.nan,
        'gamma3': np.nan,
        'beta2_sign': np.nan,
        'beta3_sign': np.nan,
        'reset_pvalue': np.nan,
        'baseline_r2': np.nan,
        'intercept_aug': np.nan,
        'coef_aug': np.nan,
        'yhat': np.full_like(np.asarray(y, dtype=float), np.nan),
        'resid': np.full_like(np.asarray(y, dtype=float), np.nan),
    }

    y_arr = np.asarray(y, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    exp_arr = np.asarray(exp_id)

    try:
        yhat = np.asarray(model.fittedvalues, dtype=float)
        baseline_r2 = float(model.rsquared)
        ssr0 = float(model.ssr)
        sst0 = float(model.centered_tss)
    except Exception:
        _warnings.warn('reset_diagnostics: could not read fitted model; returning NaN', RuntimeWarning)
        return nan_result

    if not np.isfinite(ssr0) or ssr0 == 0.0:
        _warnings.warn('reset_diagnostics: zero residual variance; returning NaN', RuntimeWarning)
        return nan_result
    if not np.isfinite(np.var(yhat)) or np.var(yhat) == 0.0:
        _warnings.warn('reset_diagnostics: zero fitted variance; returning NaN', RuntimeWarning)
        return nan_result

    X_aug = np.column_stack([X_arr, yhat**2, yhat**3])

    if cov_spec is None:
        _cov_type = 'cluster'
        _cov_kwds = {'groups': exp_arr, 'use_correction': True}
    else:
        _cov_type, _cov_kwds = cov_spec

    try:
        model_aug = _sm.OLS(y_arr, X_aug).fit(
            cov_type=_cov_type,
            cov_kwds=_cov_kwds,
        )
    except np.linalg.LinAlgError:
        _warnings.warn('reset_diagnostics: singular augmented design; returning NaN', RuntimeWarning)
        return nan_result
    except Exception:
        _warnings.warn('reset_diagnostics: augmented OLS failed; returning NaN', RuntimeWarning)
        return nan_result

    ssr_aug = float(model_aug.ssr)
    partial_r2 = (ssr0 - ssr_aug) / ssr0
    delta_r2_total = (ssr0 - ssr_aug) / sst0 if sst0 > 0 else np.nan
    gamma2 = float(model_aug.params[-2])
    gamma3 = float(model_aug.params[-1])
    beta2_sign = float(np.sign(gamma2))
    beta3_sign = float(np.sign(gamma3))
    # Augmented-model intercept and x-coefficient. Useful for downstream plots
    # that need to draw the augmented prediction curve in raw (x, y) space.
    intercept_aug = float(model_aug.params[0])
    coef_aug = float(model_aug.params[1]) if X_aug.shape[1] >= 4 else np.nan

    # Ramsey RESET F-test: H0: gamma2 = gamma3 = 0 on the augmented model,
    # using the cluster-robust covariance already attached to model_aug.
    # Done manually because statsmodels linear_reset has a kwarg-passing bug
    # (passes cov_kwargs instead of cov_kwds to fit, dropping the cluster
    # 'groups' and raising KeyError).
    try:
        k_aug = X_aug.shape[1]
        R = np.zeros((2, k_aug))
        R[0, -2] = 1.0
        R[1, -1] = 1.0
        f_res = model_aug.f_test(R)
        reset_pvalue = float(np.asarray(f_res.pvalue).item())
    except Exception:
        _warnings.warn('reset_diagnostics: RESET F-test failed; returning NaN p-value', RuntimeWarning)
        reset_pvalue = np.nan

    resid = y_arr - yhat

    return {
        'partial_r2': float(partial_r2),
        'delta_r2_total': float(delta_r2_total),
        'gamma2': gamma2,
        'gamma3': gamma3,
        'beta2_sign': beta2_sign,
        'beta3_sign': beta3_sign,
        'reset_pvalue': reset_pvalue,
        'baseline_r2': baseline_r2,
        'intercept_aug': intercept_aug,
        'coef_aug': coef_aug,
        'yhat': yhat,
        'resid': resid,
    }


def scenario_independence_diagnostics(y, X_base, exp_id, ssp_name, model,
                                       cov_spec=None, scenarios=None):
    """Robust Wald-F test of scenario-independence of the slope.

    Fits a fully-interacted OLS where scenario dummies and their
    interactions with x are added on top of the baseline [const, x]
    design — using the first SSP present (in the canonical order given
    by `scenarios`) as the reference category. Tests the joint null
    H0: all slope-interaction coefficients = 0 (slope equality across
    SSPs; intercepts free) via a Wald-F using the same robust
    covariance as the baseline OLS.

    Returns wald_fstat, wald_pvalue, partial_r2_indep (share of pooled-
    model residual variance absorbed by allowing scenario-varying
    slopes), delta_r2_total_indep (share of total y variance absorbed;
    algebraically partial_r2_indep * (1 - baseline_r2)), baseline_r2
    (read from the passed-in model so the sidecar cache is self-
    contained), per-SSP implied slopes (beta_{s}) and intercepts
    (intercept_{s}) for each scenario in `scenarios`, with NaN for SSPs
    not present in this pixel's data, max_pairwise_slope_diff = max -
    min of present SSPs' slopes in K/Sv (the physical-unit magnitude
    metric), and n_scenarios_present (defensive). NaN dict if fewer
    than two SSPs are present or the interacted design is singular.

    cov_spec=None (default) reproduces the legacy MPI behaviour:
    cov_type='cluster' with groups=exp_id, use_correction=True. Pass a
    tuple (cov_type, cov_kwds) to override — e.g.
    ('HAC', {'maxlags': 9}) for the CESM2 case.

    scenarios=None (default) uses the MPI canonical order
    ['ssp126','ssp245','ssp370']. Pass a list to override — e.g.
    ['ssp126','ssp585'] for CESM2. Output keys are emitted for each
    scenario in this list.

    Choice of reference category does not affect Wald-F, ΔR², partial
    R², per-SSP slopes, or max_pairwise_slope_diff — only γ labelling.

    Like reset_diagnostics, this uses statsmodels' default f_test df
    (N-based, not cluster-based G−1). A uniform G−1 correction across
    both diagnostic Wald-F tests is scoped as separate future work.
    """
    import statsmodels.api as _sm

    if scenarios is None:
        SSPS = ['ssp126', 'ssp245', 'ssp370']
    else:
        SSPS = list(scenarios)

    nan_result = {
        'wald_fstat': np.nan,
        'wald_pvalue': np.nan,
        'partial_r2_indep': np.nan,
        'delta_r2_total_indep': np.nan,
        'baseline_r2': np.nan,
        'max_pairwise_slope_diff': np.nan,
        'n_scenarios_present': 0,
    }
    for s in SSPS:
        nan_result[f'beta_{s}'] = np.nan
        nan_result[f'intercept_{s}'] = np.nan

    y_arr = np.asarray(y, dtype=float)
    X_arr = np.asarray(X_base, dtype=float)
    exp_arr = np.asarray(exp_id)
    ssp_arr = np.asarray(ssp_name)

    present = [s for s in SSPS if np.any(ssp_arr == s)]
    n_present = len(present)
    if n_present < 2:
        nan_result['n_scenarios_present'] = int(n_present)
        return nan_result

    try:
        baseline_r2 = float(model.rsquared)
        ssr0 = float(model.ssr)
        sst0 = float(model.centered_tss)
    except Exception:
        _warnings.warn('scenario_independence_diagnostics: could not read fitted baseline model; returning NaN', RuntimeWarning)
        return nan_result

    if not np.isfinite(ssr0) or ssr0 == 0.0 or sst0 <= 0:
        _warnings.warn('scenario_independence_diagnostics: degenerate baseline (zero SSR or non-positive SST); returning NaN', RuntimeWarning)
        return nan_result

    if X_arr.ndim != 2 or X_arr.shape[1] != 2:
        _warnings.warn(f'scenario_independence_diagnostics: expected X_base shape (n, 2), got {X_arr.shape}; returning NaN', RuntimeWarning)
        return nan_result
    x_col = X_arr[:, 1]

    # Reference = first present SSP (canonical order); non-ref get dummies.
    ref = present[0]
    non_ref = [s for s in present if s != ref]
    dummy_cols = [(ssp_arr == s).astype(float) for s in non_ref]
    inter_cols = [d * x_col for d in dummy_cols]
    X_int = np.column_stack([X_arr] + dummy_cols + inter_cols)

    if cov_spec is None:
        _cov_type = 'cluster'
        _cov_kwds = {'groups': exp_arr, 'use_correction': True}
    else:
        _cov_type, _cov_kwds = cov_spec

    try:
        model_int = _sm.OLS(y_arr, X_int).fit(
            cov_type=_cov_type,
            cov_kwds=_cov_kwds,
        )
    except np.linalg.LinAlgError:
        _warnings.warn('scenario_independence_diagnostics: singular interacted design; returning NaN', RuntimeWarning)
        nan_result['n_scenarios_present'] = int(n_present)
        return nan_result
    except Exception:
        _warnings.warn('scenario_independence_diagnostics: interacted OLS failed; returning NaN', RuntimeWarning)
        nan_result['n_scenarios_present'] = int(n_present)
        return nan_result

    ssr_int = float(model_int.ssr)
    partial_r2_indep = (ssr0 - ssr_int) / ssr0
    delta_r2_total_indep = (ssr0 - ssr_int) / sst0

    n_inter = len(non_ref)
    try:
        k_int = X_int.shape[1]
        R = np.zeros((n_inter, k_int))
        for i in range(n_inter):
            R[i, k_int - n_inter + i] = 1.0
        f_res = model_int.f_test(R)
        wald_fstat = float(np.asarray(f_res.fvalue).item())
        wald_pvalue = float(np.asarray(f_res.pvalue).item())
    except Exception:
        _warnings.warn('scenario_independence_diagnostics: Wald-F test failed; returning NaN p-value', RuntimeWarning)
        wald_fstat = np.nan
        wald_pvalue = np.nan

    # Implied per-SSP intercepts and slopes.
    # X_int columns: [const, x] + dummy_cols (one per non-ref) + inter_cols (one per non-ref)
    params = model_int.params
    intercept_ref = float(params[0])
    coef_ref = float(params[1])
    intercepts = {s: np.nan for s in SSPS}
    slopes = {s: np.nan for s in SSPS}
    intercepts[ref] = intercept_ref
    slopes[ref] = coef_ref
    for i, s in enumerate(non_ref):
        intercepts[s] = intercept_ref + float(params[2 + i])
        slopes[s] = coef_ref + float(params[2 + n_inter + i])

    slopes_present = [slopes[s] for s in present]
    max_pairwise_slope_diff = float(max(slopes_present) - min(slopes_present))

    out = {
        'wald_fstat': wald_fstat,
        'wald_pvalue': wald_pvalue,
        'partial_r2_indep': float(partial_r2_indep),
        'delta_r2_total_indep': float(delta_r2_total_indep),
        'baseline_r2': baseline_r2,
        'max_pairwise_slope_diff': max_pairwise_slope_diff,
        'n_scenarios_present': int(n_present),
    }
    for s in SSPS:
        out[f'beta_{s}'] = float(slopes[s]) if np.isfinite(slopes[s]) else np.nan
        out[f'intercept_{s}'] = float(intercepts[s]) if np.isfinite(intercepts[s]) else np.nan
    return out


def weighted_area_lat(ds):
    """
    Calculate the area-weighted temperature over its domain. 
    In a regular latitude/ longitude grid the grid cell area decreases towards the pole.
    We can use the cosine of the latitude as proxy for the grid cell area.
    IMPORTANT: after applying function, do .mean('lat') before 'lon'
    Taken from https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html
    
    Parameters:
    ds (xr.Dataset): Dataset containing latitude data.
    
    Returns:
    xr.Dataset: Dataset with weighted area.
    """
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"
    return ds.weighted(weights)


def make_country_masks_land_aware(ds, overlap_thresholds=(0.5, 0.3, 0.2, 0.1, 0.05, 0.0),
                                  land_mask=None, include_ipcc_regions=True,
                                  verbose=False):
    """Robust per-country (+ IPCC AR6 region) masks for grid-cell aggregation.

    Resolves the mask for each country in tiered order:
      1. centroid-cells ∩ land_mask — preferred, matches the canonical
         ``.where(centroid).where(land == 0)`` pattern.
      2. For θ in ``overlap_thresholds`` (default 0.5 → 0.3 → 0.2 →
         0.1 → 0.05 → 0.00): cells whose grid-cell-area is at least
         θ covered by the country polygon, ∩ land_mask. First non-empty
         result wins. Prefers land-dominated cells before relaxing.
      3. Last resort (country polygon intersects no land-classified cell
         on this grid — e.g. Cyprus at 2° resolution): cells overlapping
         the polygon with no land filter applied. Flagged via
         ``da.attrs['land_filter_applied'] = False``.

    Returned country masks are bool (True over the selected cells). The
    ``'LAND'`` entry uses the regionmask float convention (0 over land,
    NaN over ocean) so existing ``.where(masks['LAND']==0)`` callsites
    keep working.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Provides the target lat/lon grid.
    overlap_thresholds : tuple of float, default (0.5, 0.3, 0.2, 0.1, 0.05, 0.0)
        Overlap-fraction thresholds tried in order. Values must be
        monotonically decreasing.
    land_mask : xr.DataArray or None
        Optional pre-computed land mask where ``== 0`` denotes land.
        If None, computed via
        ``regionmask.defined_regions.natural_earth_v5_0_0.land_110``.
    include_ipcc_regions : bool, default True
        Also build the IPCC AR6 region masks (NEU, WCE, EEU, MED, EU,
        EU_EEU, EU_buffer). Replaces the role of the (now deleted)
        ``define_region_masks``.
    verbose : bool, default False
        If True, prints which fallback tier fired for each country.

    Returns
    -------
    dict
        Keys:
        - 7 IPCC AR6 region keys (when ``include_ipcc_regions``): bool masks
        - 1 key per ``country_codes`` entry: bool mask, with attrs
          ``tier`` and ``land_filter_applied``
        - ``'LAND'``: float DataArray (0 over land, NaN over ocean) — the
          raw output of ``natural_earth_v5_0_0.land_110.mask(ds)``.
    """
    from shapely.geometry import Polygon
    from shapely.prepared import prep

    if land_mask is None:
        land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(ds)
    land_bool = (land_mask == 0)

    # IPCC AR6 region masks (lifted from the former define_region_masks)
    if include_ipcc_regions:
        mask_eur = regionmask.defined_regions.ar6.land[16, 17, 18, 19].mask(ds)
        base_mask = ((mask_eur == 16) | (mask_eur == 17) | (mask_eur == 19)).copy()
        shifted_west = base_mask.roll(lon=-15, roll_coords=False)
        shifted_east = base_mask.roll(lon=5, roll_coords=False)
        mask_eu_buffer = base_mask | shifted_west | shifted_east
        ipcc_masks = {
            'NEU': mask_eur == 16,
            'WCE': mask_eur == 17,
            'EEU': mask_eur == 18,
            'MED': mask_eur == 19,
            'EU':  (mask_eur == 16) | (mask_eur == 17) | (mask_eur == 19),
            'EU_EEU': (mask_eur == 16) | (mask_eur == 17) | (mask_eur == 18) | (mask_eur == 19),
            'EU_buffer': mask_eu_buffer,
        }
    else:
        ipcc_masks = {}

    shpfilename = shpreader.natural_earth(
        resolution='10m', category='cultural', name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    polygons = {c.attributes['ISO_A2_EH']: c.geometry
                for c in reader.records()}

    lon_vals = ds.lon.values
    lat_vals = ds.lat.values
    lon_b = np.concatenate([[1.5 * lon_vals[0] - 0.5 * lon_vals[1]],
                            (lon_vals[:-1] + lon_vals[1:]) / 2,
                            [1.5 * lon_vals[-1] - 0.5 * lon_vals[-2]]])
    lat_b = np.concatenate([[1.5 * lat_vals[0] - 0.5 * lat_vals[1]],
                            (lat_vals[:-1] + lat_vals[1:]) / 2,
                            [1.5 * lat_vals[-1] - 0.5 * lat_vals[-2]]])

    centroid_all = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(ds)

    def _overlap_fractions(polygon):
        prep_poly = prep(polygon)
        out = np.zeros((len(lat_vals), len(lon_vals)), dtype=float)
        for j in range(len(lat_vals)):
            for i in range(len(lon_vals)):
                cell = Polygon([
                    (lon_b[i], lat_b[j]),
                    (lon_b[i+1], lat_b[j]),
                    (lon_b[i+1], lat_b[j+1]),
                    (lon_b[i], lat_b[j+1]),
                ])
                if prep_poly.intersects(cell):
                    out[j, i] = polygon.intersection(cell).area / cell.area
        return out

    def _mk_da(arr, template):
        return xr.DataArray(arr.astype(bool),
                            dims=template.dims, coords=template.coords)

    out = dict(ipcc_masks)
    template = land_bool
    for country, code in country_codes.items():
        centroid_mask = (centroid_all == code)
        combined = (centroid_mask & land_bool)
        if bool(combined.any()):
            da = combined
            da.attrs['land_filter_applied'] = True
            da.attrs['tier'] = 'centroid'
            out[country] = da
            continue

        polygon = polygons.get(country)
        if polygon is None:
            if verbose:
                print(f"    {country}: no polygon found")
            da = _mk_da(np.zeros_like(centroid_mask.values, dtype=bool), template)
            da.attrs['land_filter_applied'] = True
            da.attrs['tier'] = 'empty'
            out[country] = da
            continue

        frac = _overlap_fractions(polygon)
        picked = None
        picked_tier = None
        for theta in overlap_thresholds:
            overlap_mask = _mk_da(frac >= theta, template)
            # Guard against θ=0 catching cells with zero-area intersection.
            if theta == 0.0:
                overlap_mask = _mk_da(frac > 0, template)
            combined = overlap_mask & land_bool
            if bool(combined.any()):
                picked = combined
                picked_tier = f'overlap_theta={theta:.2f}_landed'
                break
        if picked is None:
            # No tier gives a land cell on this grid. Drop the land filter,
            # but preserve the tiered-threshold preference so sliver cells
            # (e.g. 1% DK-coverage corners) are only included when nothing
            # stricter survives.
            for theta in overlap_thresholds:
                overlap_mask = _mk_da(frac >= theta if theta > 0
                                      else frac > 0, template)
                if bool(overlap_mask.any()):
                    picked = overlap_mask
                    picked_tier = f'overlap_theta={theta:.2f}_no_land'
                    break
            if picked is None:
                # Polygon does not intersect any grid cell at all — return
                # the empty overlap mask so downstream NaN propagation is
                # explicit.
                picked = _mk_da(frac > 0, template)
                picked_tier = 'no_overlap'
            picked.attrs['land_filter_applied'] = False
            if verbose:
                print(f"    {country}: {picked_tier} "
                      f"({int(picked.sum())} cells; polygon does not "
                      f"intersect any land-classified cell at this grid)")
        else:
            picked.attrs['land_filter_applied'] = True
        picked.attrs['tier'] = picked_tier
        out[country] = picked
        if verbose:
            print(f"    {country}: tier={picked_tier}, cells={int(picked.sum())}")

    out['LAND'] = land_mask
    return out


def make_legend(fig, plot_ssp, hos_type, hos_strength='all', incl_ge=True, line_spacing=0.045, pos=(1.01, 1.0), col_width=0.15, plot_bg='white', markers=True, black_text=False):
    """
    Create a legend for the plot.
    
    Parameters:
    fig (matplotlib.Figure): Figure object to add the legend to.
    plot_ssp (str): SSP scenario to plot.
    hos_type (str): Hosing type: all, constant, linear.
    incl_ge (bool): Whether to include GE historical data in the legend (default is True).
    """
    i, left_out = -2, 0

    if plot_ssp != 'all':
        left_out = 0 # originally 2 but for simulation plot only 0 worked

    for (k, ssp) in enumerate(['ssp370', 'ssp245', 'ssp126', 'hist']):
        if ssp == 'hist':
            if incl_ge:
                fig.text(pos[0], pos[1], 'MPI-ESM GE historical', fontsize=10, color='k' if not plot_bg=='black' else 'white', weight='bold', va='top', ha='left', clip_on=False)
                i+=1
            else:
                i -= 1
        else:
            if plot_ssp != 'all' and ssp != plot_ssp:
                left_out += 1
                continue
            for exp in ['ge', 'neg01', '01', '03', '05', 'linneg02', 'lin02', 'lin06', 'lin10']:
                if exp == 'ge' and not incl_ge:
                    # print only scenario names
                    fig.text(pos[0], 
                            pos[1] - line_spacing*i + 4*(k-left_out-1)*line_spacing - (k-left_out)*line_spacing,
                            hosing_names[ssp][exp][11:], fontsize=10, color=hosing_colors[ssp][exp], weight='bold', va='top', ha='left', clip_on=False)
                    i += 1
                    continue
                if hos_type != 'all':
                    if hos_type == 'linear':
                        if 'lin' not in exp and exp != 'ge':
                            i += 1
                            continue
                    elif hos_type == 'constant':
                        if 'lin' in exp:
                            i += 1
                            continue
                    elif hos_type == None: # this might work only if ssp != 'all'
                        # print('k:',k, 'exp:', exp, 'i:', i, 'left_out:', left_out)
                        if exp != 'ge':
                            continue
                    else:
                        print('Unknown hosing type in legend.')
                        continue
                    if hos_strength != 'all' and exp != hos_strength:
                        continue
                col_offset = col_width if 'lin' in exp else 0.
                col_offset = 0 if hos_type == 'linear' else col_offset
                vert_shift = 4*(k-left_out)*line_spacing if 'lin' in exp else 4 * (k-left_out-1) * line_spacing
                fig.text(pos[0] + col_offset,
                        pos[1] - line_spacing*i + vert_shift - (k-left_out)*line_spacing,
                        (hosing_symbols[exp]+' '+hosing_names[ssp][exp][9:] if markers else hosing_names[ssp][exp][9:]) if exp!='ge' else (hosing_symbols[exp]+' '+hosing_names[ssp][exp] if markers else hosing_names[ssp][exp]), fontsize=10, color='k' if black_text else hosing_colors[ssp][exp], weight='bold', va='top', ha='left', clip_on=False)
                i += 1


def add_square(ax, lon_min, lon_max, lat_min, lat_max, colour='white'):
    rectangle = Polygon([(lon_min, lat_min), (lon_min, lat_max), (lon_max, lat_max), (lon_max, lat_min)])
    ax.add_geometries([rectangle], crs=ccrs.PlateCarree(), facecolor=colour, edgecolor=colour, zorder=4)


def get_custom_cmap():

    orig_cmap = plt.get_cmap('ocean_r')
    n_colors = 256

    replace_frac = 0.4
    colors = orig_cmap(np.linspace(0.04, (1-replace_frac), n_colors - 100))

    end_color = np.array([[0.8, 0.2, 0.8, 1.0]])  # pink for strong cooling

    transition = np.linspace(colors[-1], end_color.flatten(), 100)
    colors = np.vstack([colors, transition])

    custom_cmap = LinearSegmentedColormap.from_list('ocean_r_purple', colors)
    custom_cmap.set_over((0.95, 0.3, 0.95, 1.0))  # Brighter magenta for extend triangle

    return custom_cmap

########################################################################################################################
# MPI-ESM ANALYSIS
########################################################################################################################

# HELPER FUNCTIONS FOR MPI-ESM ANALYSIS

def split_ds_exper_dims(ds):
    if np.shape(ds.exper) == ():
        scenar = str(ds.exper.values).split('_')[0]
        hosing = str(ds.exper.values).split('_')[1][3:-2]
        # For 0-dim: drop exper and assign scalar coords directly
        return ds.drop_vars('exper').assign_coords(scenar=scenar, hosing=hosing)
    else:
        scenar = [val.split('_')[0] for val in ds.exper.values]
        hosing = [val.split('_')[1][3:-2] for val in ds.exper.values]
        ds = ds.assign_coords(
            scenar=('exper', scenar),
            hosing=('exper', hosing)
        )
        return ds.set_index(exper=['scenar', 'hosing']).unstack('exper')


def convert_strength_to_weakening(strength):
    return (AMOC_pi_MPI - strength) / AMOC_pi_MPI * 100


def convert_weakening_to_strength(weakening):
    return - weakening/100 * AMOC_pi_MPI + AMOC_pi_MPI


def add_weakening_pct_overlay(fig, ax, plot_bg='white', weak_min_pct=-30, weak_max_pct=40,
                              label='AMOC weakening additional to MPI-GE [w.r.t. 1850–1899]',
                              drop_parent_xaxis=False):
    """Thin AMOC-weakening-% ruler beneath a regression panel whose data x-axis is
    in Sv deviation (e.g. regression_plot output). Binding invariant: both the Sv
    axis and this %-axis end on round ticks. Since 1% = AMOC_pi_MPI/100 Sv maps no
    round Sv to a round %, the ruler is an independent fig.add_axes spanning only
    [weak_min_pct, weak_max_pct], positioned by fraction inside the panel's Sv xlim
    (sv = -pct*AMOC_pi_MPI/100). Source design: Fig1 panel c. With
    drop_parent_xaxis=True the parent's own x-axis is stripped so only this ruler
    shows (used where regression_plot left an orphaned top Sv axis)."""
    fg = 'black' if plot_bg != 'black' else 'white'
    if drop_parent_xaxis:
        ax.set_xlabel('')
        ax.tick_params(axis='x', which='both', top=False, bottom=False,
                       labeltop=False, labelbottom=False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    pos = ax.get_position()
    xlim = ax.get_xlim()  # Sv; may be inverted (e.g. (+6, -8))
    sv_at_min = -weak_min_pct * AMOC_pi_MPI / 100
    sv_at_max = -weak_max_pct * AMOC_pi_MPI / 100
    frac_left = (sv_at_min - xlim[0]) / (xlim[1] - xlim[0])
    frac_right = (sv_at_max - xlim[0]) / (xlim[1] - xlim[0])

    new_height = 0.005
    weakax = fig.add_axes([pos.x0 + frac_left * pos.width, pos.y0 - new_height,
                           (frac_right - frac_left) * pos.width, new_height])
    weakax.set_xlim(weak_min_pct, weak_max_pct)
    weakax.set_xmargin(0)
    ticks = list(range(weak_min_pct, weak_max_pct + 1, 10))
    weakax.set_xticks(ticks)
    weakax.set_xticklabels([f"{t}%" for t in ticks])
    weakax.spines['bottom'].set_bounds(weak_min_pct, weak_max_pct)
    weakax.spines['top'].set_visible(False)
    weakax.spines['left'].set_visible(False)
    weakax.spines['right'].set_visible(False)
    weakax.spines['bottom'].set_visible(True)
    weakax.set_yticks([])
    weakax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False)
    weakax.set_xlabel(label, labelpad=5, fontsize=10, color=fg)
    weakax.set_facecolor('none')
    return weakax


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256, darken=1.0):
    
    new_colors = cmap(np.linspace(minval, maxval, n))
    
    # Darken: multiply RGB channels by darken factor (keep alpha unchanged)
    new_colors[:, :3] = new_colors[:, :3] * darken
    new_colors = np.clip(new_colors, 0, 1)
    
    truncated_cmap = LinearSegmentedColormap.from_list('trunc({},{:.2f},{:.2f})'.format(cmap.name, minval, maxval), new_colors)
    
    return truncated_cmap


def mpi_diagnostics_scan(data_dict, var, fields, sentinel_key, compute_kw):
    """Shared per-pixel scan loop for the MPI-ESM linearity / scenario-
    independence diagnostics (the MPI twin of ``cesm_diagnostics_scan``).

    Iterates season × lat × lon, calls ``regression_plot`` with the requested
    ``compute_*`` flag, and populates ``fields`` from ``regression_plot.last_call``
    into an xr.Dataset. ``sentinel_key`` must be present in ``last_call`` for the
    pixel to count (gates against cross-contamination when regression_plot bails
    out before setting diagnostic keys); ocean pixels are skipped.
    """
    seasons = ['', 'djf', 'jja'] if var == 'tas' else ['']
    nlat = data_dict[seasons[0]][var]['ssphos'].sizes['lat']
    nlon = data_dict[seasons[0]][var]['ssphos'].sizes['lon']

    arr_shape = (len(seasons), nlat, nlon)
    data_vars = {name: (['season', 'lat', 'lon'], np.full(arr_shape, np.nan)) for name in fields}
    coords = {
        'season': seasons,
        'lat': data_dict[seasons[0]][var]['ssphos'].lat.values,
        'lon': data_dict[seasons[0]][var]['ssphos'].lon.values,
    }
    out = xr.Dataset(data_vars=data_vars, coords=coords)

    for s_i, s in enumerate(seasons):
        print(f'  season={s!r}')
        ssphos_var = data_dict[s][var]['ssphos']
        i = 0
        for lat_i in range(nlat):
            for lon_i in range(nlon):
                if np.sum(~np.isnan(ssphos_var.isel(lat=lat_i, lon=lon_i)[var].values)) == 0:
                    i += 1
                    continue
                regression_plot.last_call = {}
                regression_plot(data_dict, season=s, var=var, lat=lat_i, lon=lon_i,
                                no_plots=True, **compute_kw)
                lc = regression_plot.last_call
                if sentinel_key in lc:
                    for name in fields:
                        out[name].values[s_i, lat_i, lon_i] = lc[name]
                i += 1
                if i % 100 == 0:
                    print(f'    {i}/{nlat*nlon} pixels done')
    return out


def get_linearity_diagnostics_mpi(data_dict=None, var='tas', recompute=False):
    """Per-pixel Ramsey RESET diagnostics over MPI-ESM. Returns a Dataset with
    dims (season, lat, lon) and variables: partial_r2, delta_r2_total, gamma2,
    gamma3, beta2_sign, beta3_sign, reset_pvalue, baseline_r2. Cached to
    data/linearity_diagnostics_mpi_{var}.nc.
    """
    LINEARITY_CACHE_VERSION = 2
    validate_choice('var', var, ALLOWED_VARS)
    cache = local_path + f'linearity_diagnostics_mpi_{var}.nc'
    if not recompute:
        cached = check_diag_cache(cache, LINEARITY_CACHE_VERSION, 'MPI-ESM linearity')
        if cached is not None:
            return cached

    print('Calculating MPI-ESM linearity diagnostics (per-pixel RESET)...')
    if data_dict is None:
        data_dict = load_mpi_esm_data(eur_only=True)

    fields = ['partial_r2', 'delta_r2_total', 'gamma2', 'gamma3',
              'beta2_sign', 'beta3_sign', 'reset_pvalue', 'baseline_r2']
    out = mpi_diagnostics_scan(
        data_dict, var=var, fields=fields, sentinel_key='partial_r2',
        compute_kw={'compute_linearity': True},
    )
    out.attrs['version'] = LINEARITY_CACHE_VERSION
    out.to_netcdf(cache)
    print(f'Saved MPI-ESM linearity diagnostics to {cache}.')
    return out


def get_scenario_independence_diagnostics_mpi(data_dict=None, var='tas', recompute=False):
    """Per-pixel cluster-robust Wald-F test of slope equality across SSPs
    in MPI-ESM. Returns a Dataset with dims (season, lat, lon) and ten
    variables: wald_fstat, wald_pvalue, partial_r2_indep,
    delta_r2_total_indep, baseline_r2, beta_ssp126, beta_ssp245,
    beta_ssp370, max_pairwise_slope_diff, n_scenarios_present. Cached to
    data/scenario_independence_diagnostics_mpi_{var}.nc.

    Mirrors get_linearity_diagnostics_mpi's loop and caching structure.
    Per-SSP intercepts are kept in regression_plot.last_call (panel-f
    line drawing) but not cached. baseline_r2 is duplicated from the
    linearity cache so this figure's column a is self-contained.
    """
    SCENARIO_INDEPENDENCE_CACHE_VERSION = 1
    validate_choice('var', var, ALLOWED_VARS)
    cache = local_path + f'scenario_independence_diagnostics_mpi_{var}.nc'
    if not recompute:
        cached = check_diag_cache(cache, SCENARIO_INDEPENDENCE_CACHE_VERSION,
                                  'MPI-ESM scenario-independence')
        if cached is not None:
            return cached

    print('Calculating MPI-ESM scenario-independence diagnostics (per-pixel Wald-F)...')
    if data_dict is None:
        data_dict = load_mpi_esm_data(eur_only=True)

    fields = ['wald_fstat', 'wald_pvalue', 'partial_r2_indep',
              'delta_r2_total_indep', 'baseline_r2',
              'beta_ssp126', 'beta_ssp245', 'beta_ssp370',
              'max_pairwise_slope_diff', 'n_scenarios_present']
    out = mpi_diagnostics_scan(
        data_dict, var=var, fields=fields, sentinel_key='wald_pvalue',
        compute_kw={'compute_scenario_independence': True},
    )
    out.attrs['version'] = SCENARIO_INDEPENDENCE_CACHE_VERSION
    out.to_netcdf(cache)
    print(f'Saved MPI-ESM scenario-independence diagnostics to {cache}.')
    return out


# LOADING MPI-ESM DATA

def load_mpi_esm_data(eur_only=True, load_daily=False):

    print("Loading MPI-ESM data...")

    his_amoc_yr = xr.open_dataarray(data_path+"MPI-GE/his_amoc26_yr.nc", use_cftime=True).to_dataset(name='AMOC_strength').drop_vars('lev')
    his_tas_yr = xr.open_dataarray(data_path+"MPI-GE/his_tas_yr.nc", use_cftime=True).to_dataset(name='tas') - 273.15
    his_tas_djf = xr.open_dataset(data_path+"MPI-GE/his_tas_djf.nc", use_cftime=True).drop_vars(['height', 'lat_bnds', 'lon_bnds']) - 273.15
    his_tas_jja = xr.open_dataset(data_path+"MPI-GE/his_tas_jja.nc", use_cftime=True).drop_vars(['height', 'lat_bnds', 'lon_bnds']) - 273.15

    ssp_amoc_yr = xr.open_dataarray(data_path+"MPI-GE/ssp_amoc26_yr.nc", use_cftime=True).to_dataset(name='AMOC_strength').drop_vars('lev')
    ssp_tas_yr = xr.open_dataarray(data_path+"MPI-GE/ssp_tas_yr.nc", use_cftime=True).to_dataset(name='tas') - 273.15
    ssp_tas_djf = xr.open_dataset(data_path+"MPI-GE/ssp_tas_djf.nc", use_cftime=True).drop_vars(['height', 'lat_bnds', 'lon_bnds']) - 273.15
    ssp_tas_jja = xr.open_dataset(data_path+"MPI-GE/ssp_tas_jja.nc", use_cftime=True).drop_vars(['height', 'lat_bnds', 'lon_bnds']) - 273.15

    ssphos_amoc_yr = xr.open_dataarray(data_path+"ssphos/ssphos_amoc_yr.nc", use_cftime=True).to_dataset(name='AMOC_strength')
    ssphos_tas_yr = xr.open_dataarray(data_path+"ssphos/ssphos_tas_yr.nc", use_cftime=True).to_dataset(name='tas') - 273.15
    ssphos_tas_djf = xr.open_dataarray(data_path+"ssphos/ssphos_tas_djf.nc", use_cftime=True).to_dataset(name='tas') - 273.15
    ssphos_tas_jja = xr.open_dataarray(data_path+"ssphos/ssphos_tas_jja.nc", use_cftime=True).to_dataset(name='tas') - 273.15

    his_pr_yr = xr.open_dataarray(data_path+"MPI-GE/his_pr_yr.nc", use_cftime=True).to_dataset(name='pr') * 3600 * 24  # convert from kg m-2 s-1 to mm d-1
    ssp_pr_yr = xr.open_dataarray(data_path+"MPI-GE/ssp_pr_yr.nc", use_cftime=True).to_dataset(name='pr') * 3600 * 24  # convert from kg m-2 s-1 to mm d-1
    ssphos_pr_yr = xr.open_dataarray(data_path+"ssphos/ssphos_pr_yr.nc", use_cftime=True).to_dataset(name='pr') * 3600 * 24  # convert from kg m-2 s-1 to mm d-1

    his_tmn_yr = xr.open_dataarray(data_path+"MPI-GE/his_tmn_year.nc", use_cftime=True).to_dataset(name='tmn') - 273.15
    ssp_tmn_yr = xr.open_dataarray(data_path+"MPI-GE/ssp_tmn_year.nc", use_cftime=True).to_dataset(name='tmn') - 273.15
    ssphos_tmn_yr_ssp245 = xr.open_dataarray(data_path+"ssphos/tmn_ssp245_grc1Sv_year.nc", use_cftime=True).to_dataset(name='tmn') - 273.15
    ssphos_tmn_yr_ssp126 = xr.open_dataarray(data_path+"ssphos/tmn_ssp126_grc1Sv_year.nc", use_cftime=True).to_dataset(name='tmn') - 273.15
    ssphos_tmn_yr = xr.concat([ssphos_tmn_yr_ssp126, ssphos_tmn_yr_ssp245], dim='exper')

    # Monthly tmn (eager load - reasonable size)
    ssphos_tmn_mon = xr.open_dataarray(data_path+"ssphos/tmn_ssp245_grc1Sv_mon.nc", use_cftime=True).to_dataset(name='tmn') - 273.15

    # Daily tas (lazy load via dask - large file)
    if load_daily:
        ssphos_tas_day = xr.open_dataarray(data_path+"ssphos/ssphos_tas_day.nc", chunks={'time': 365}, use_cftime=True).to_dataset(name='tas') - 273.15

    # split hosing datasets into scenarios and hosing types
    ssphos_amoc_yr = split_ds_exper_dims(ssphos_amoc_yr)
    ssphos_tas_yr = split_ds_exper_dims(ssphos_tas_yr)
    ssphos_tas_djf = split_ds_exper_dims(ssphos_tas_djf)
    ssphos_tas_jja = split_ds_exper_dims(ssphos_tas_jja)
    ssphos_pr_yr = split_ds_exper_dims(ssphos_pr_yr)
    ssphos_tmn_yr = split_ds_exper_dims(ssphos_tmn_yr)
    ssphos_tmn_mon = split_ds_exper_dims(ssphos_tmn_mon)
    if load_daily:
        ssphos_tas_day = split_ds_exper_dims(ssphos_tas_day)

    # make grids align
    ssp_tas_yr.coords['lon'] = (ssp_tas_yr.coords['lon'] + 180) % 360 - 180
    ssp_tas_yr = ssp_tas_yr.sortby(ssp_tas_yr.lon)
    ssp_tas_djf.coords['lon'] = (ssp_tas_djf.coords['lon'] + 180) % 360 - 180
    ssp_tas_djf = ssp_tas_djf.sortby(ssp_tas_djf.lon)
    ssp_tas_jja.coords['lon'] = (ssp_tas_jja.coords['lon'] + 180) % 360 - 180
    ssp_tas_jja = ssp_tas_jja.sortby(ssp_tas_jja.lon)

    his_tas_yr.coords['lon'] = (his_tas_yr.coords['lon'] + 180) % 360 - 180
    his_tas_yr = his_tas_yr.sortby(his_tas_yr.lon)
    his_tas_djf.coords['lon'] = (his_tas_djf.coords['lon'] + 180) % 360 - 180
    his_tas_djf = his_tas_djf.sortby(his_tas_djf.lon)
    his_tas_jja.coords['lon'] = (his_tas_jja.coords['lon'] + 180) % 360 - 180
    his_tas_jja = his_tas_jja.sortby(his_tas_jja.lon)

    his_pr_yr.coords['lon'] = (his_pr_yr.coords['lon'] + 180) % 360 - 180
    his_pr_yr = his_pr_yr.sortby(his_pr_yr.lon)
    ssp_pr_yr.coords['lon'] = (ssp_pr_yr.coords['lon'] + 180) % 360 - 180
    ssp_pr_yr = ssp_pr_yr.sortby(ssp_pr_yr.lon)
    ssphos_pr_yr = ssphos_pr_yr.sortby(ssphos_pr_yr.lat)

    ssphos_tas_yr = ssphos_tas_yr.sortby(ssphos_tas_yr.lat)
    ssphos_tas_djf = ssphos_tas_djf.sortby(ssphos_tas_djf.lat)
    ssphos_tas_jja = ssphos_tas_jja.sortby(ssphos_tas_jja.lat)
    
    his_tmn_yr.coords['lon'] = (his_tmn_yr.coords['lon'] + 180) % 360 - 180
    his_tmn_yr = his_tmn_yr.sortby(his_tmn_yr.lat)
    ssp_tmn_yr.coords['lon'] = (ssp_tmn_yr.coords['lon'] + 180) % 360 - 180
    ssp_tmn_yr = ssp_tmn_yr.sortby(ssp_tmn_yr.lat)

    ssphos_tmn_yr.coords['lon'] = (ssphos_tmn_yr.coords['lon'] + 180) % 360 - 180
    ssphos_tmn_yr = ssphos_tmn_yr.sortby(ssphos_tmn_yr.lat)

    ssphos_tmn_mon.coords['lon'] = (ssphos_tmn_mon.coords['lon'] + 180) % 360 - 180
    ssphos_tmn_mon = ssphos_tmn_mon.sortby(ssphos_tmn_mon.lat)

    if load_daily:
        ssphos_tas_day.coords['lon'] = (ssphos_tas_day.coords['lon'] + 180) % 360 - 180
        ssphos_tas_day = ssphos_tas_day.sortby(ssphos_tas_day.lat)

    masks_dict = make_country_masks_land_aware(ssp_tas_yr)

    data_dict = {}
    data_dict[''] = {}
    data_dict['djf'] = {}
    data_dict['jja'] = {}
    data_dict['masks'] = masks_dict

    data_dict['']['amoc'] = {}
    data_dict['']['amoc']['his'] = his_amoc_yr
    data_dict['']['amoc']['ssp'] = ssp_amoc_yr
    data_dict['']['amoc']['ssphos'] = ssphos_amoc_yr

    data_dict['']['tas'] = {}
    data_dict['djf']['tas'] = {}
    data_dict['jja']['tas'] = {}

    data_dict['']['tas']['his'] = his_tas_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else his_tas_yr
    data_dict['']['tas']['ssp'] = ssp_tas_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssp_tas_yr
    data_dict['']['tas']['ssphos'] = ssphos_tas_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssphos_tas_yr

    data_dict['djf']['tas']['his'] = his_tas_djf.where(masks_dict['EU_buffer'], drop=True) if eur_only else his_tas_djf
    data_dict['djf']['tas']['ssp'] = ssp_tas_djf.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssp_tas_djf
    data_dict['djf']['tas']['ssphos'] = ssphos_tas_djf.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssphos_tas_djf

    data_dict['jja']['tas']['his'] = his_tas_jja.where(masks_dict['EU_buffer'], drop=True) if eur_only else his_tas_jja
    data_dict['jja']['tas']['ssp'] = ssp_tas_jja.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssp_tas_jja
    data_dict['jja']['tas']['ssphos'] = ssphos_tas_jja.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssphos_tas_jja

    data_dict['']['pr'] = {}
    data_dict['']['pr']['his'] = his_pr_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else his_pr_yr
    data_dict['']['pr']['ssp'] = ssp_pr_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssp_pr_yr
    data_dict['']['pr']['ssphos'] = ssphos_pr_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssphos_pr_yr

    data_dict['']['tmn'] = {}
    data_dict['']['tmn']['his'] = his_tmn_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else his_tmn_yr
    data_dict['']['tmn']['ssp'] = ssp_tmn_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssp_tmn_yr
    data_dict['']['tmn']['ssphos'] = ssphos_tmn_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssphos_tmn_yr    #.expand_dims(['hosing'])

    data_dict['mon'] = {}
    if load_daily:
        data_dict['day'] = {}

    data_dict['mon']['tmn'] = {}
    data_dict['mon']['tmn']['ssphos'] = ssphos_tmn_mon.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssphos_tmn_mon

    if load_daily:
        data_dict['day']['tas'] = {}
        data_dict['day']['tas']['ssphos'] = ssphos_tas_day.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssphos_tas_day

    return data_dict

# TRANSFORMING MPI-ESM DATA

def get_regression_ds(data_dict, season='', var='tas', future_window=None):

    validate_choice('season', season, ALLOWED_SEASONS)
    validate_choice('var', var, ALLOWED_VARS)
    fw = resolve_future_window(future_window)

    print(f"Calculating {'annual' if season=='' else season} MPI-ESM regression coefficients...")

    static_ds = data_dict[season][var]['ssphos'].isel(time=0, drop=True).copy(deep=True).drop_vars(var)

    reg_ds = static_ds.assign(
        coef_ensmean=(['lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, scenar=0, hosing=0, drop=True).sizes.values()), np.nan)),
        ste_ensmean=(['lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, scenar=0, hosing=0, drop=True).sizes.values()), np.nan)), # standard error of regression coefficient
        rsq_ensmean=(['lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, scenar=0, hosing=0, drop=True).sizes.values()), np.nan)), # R^2 of regression
        AMOC_future=(['scenar'], np.full(tuple(static_ds.isel(lon=0, lat=0, realiz=0, hosing=0, drop=True).sizes.values()), np.nan)),
        AMOC_pi=([], np.full(tuple(static_ds.isel(lon=0, lat=0, realiz=0, hosing=0, scenar=0, drop=True).sizes.values()), np.nan)),
        T_future=(['scenar', 'lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, hosing=0, drop=True).sizes.values()), np.nan)),
        T_pi=(['lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, scenar=0, hosing=0, drop=True).sizes.values()), np.nan)),
        T_pd_245=(['lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, scenar=0, hosing=0, drop=True).sizes.values()), np.nan)),
        req_strength_pi=(['scenar', 'lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, hosing=0, drop=True).sizes.values()), np.nan)),
        req_strength_pd=(['scenar', 'lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, hosing=0, drop=True).sizes.values()), np.nan)),
        req_weakening_pi=(['scenar', 'lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, hosing=0, drop=True).sizes.values()), np.nan)),
        req_weakening_pd=(['scenar', 'lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, hosing=0, drop=True).sizes.values()), np.nan)),
    )

    reg_ds.AMOC_pi.values = AMOC_pi_MPI
    reg_ds.T_pi.values[:, :] = data_dict[season][var]['his'].mean(dim='realiz').sel(time=slice(*PI_WINDOW)).mean(dim='time')[var]

    i=0
    for lat in np.arange(len(data_dict[season][var]['ssphos'].lat)):
        for lon in np.arange(len(data_dict[season][var]['ssphos'].lon)):
            reg_quadruple = regression_plot(data_dict, season=season, var=var, lat=lat, lon=lon, no_plots=True)
            reg_ds.coef_ensmean.values[lat, lon] = reg_quadruple[1]
            reg_ds.ste_ensmean.values[lat, lon] = reg_quadruple[2]
            reg_ds.rsq_ensmean.values[lat, lon] = reg_quadruple[3]
            i += 1
            if i % 100 == 0:
                print(f"{i}/{len(data_dict[season][var]['ssphos'].lat)*len(data_dict[season][var]['ssphos'].lon)} regression coefficients calculated.")

    for ssp_i in reg_ds.scenar.values:
        reg_ds.AMOC_future.loc[ssp_i] = data_dict['']['amoc']['ssp'].mean(dim='realiz').sel(time=slice(*fw)).mean(dim='time').sel(scenar=ssp_i).AMOC_strength
        ssp_index = list(reg_ds.scenar.values).index(ssp_i)
        reg_ds.T_future.values[ssp_index, :, :] = data_dict[season][var]['ssp'].mean(dim='realiz').sel(time=slice(*fw)).mean(dim='time').sel(scenar=ssp_i)[var]
        reg_ds.T_pd_245.values[:, :] = xr.concat([data_dict[season][var]['his'].mean(dim='realiz').sel(time=slice(*PD_HIS_WINDOW)), data_dict[season][var]['ssp'].mean(dim='realiz').sel(time=slice(*PD_SSP_WINDOW), scenar='ssp245')], dim='time').mean(dim='time')[var]
        reg_ds.req_strength_pi.loc[ssp_i].values[:, :] = reg_ds.AMOC_future.loc[ssp_i] - (reg_ds.T_future.loc[ssp_i] - reg_ds.T_pi) / reg_ds.coef_ensmean.values
        reg_ds.req_strength_pd.loc[ssp_i].values[:, :] = reg_ds.AMOC_future.loc[ssp_i] - (reg_ds.T_future.loc[ssp_i] - reg_ds.T_pd_245) / reg_ds.coef_ensmean.values
        reg_ds.req_weakening_pi.loc[ssp_i].values[:, :] = convert_strength_to_weakening(reg_ds.req_strength_pi.loc[ssp_i].values)
        reg_ds.req_weakening_pd.loc[ssp_i].values[:, :] = convert_strength_to_weakening(reg_ds.req_strength_pd.loc[ssp_i].values)

    reg_ds.attrs['future_window'] = list(fw)
    return reg_ds


def load_regression_ds_mpi(data_dict=None, var='tas', ens_mean=True, recompute=False, future_window=None):
    validate_choice('var', var, ALLOWED_VARS)
    fw = resolve_future_window(future_window)
    cache_path = local_path + f'reg_ds_mpi_{var}{fw_suffix(future_window)}.nc'
    if os.path.exists(cache_path) and not recompute:
        print("Loading precomputed MPI-ESM regression dataset...")
        reg_ds_mpi = xr.open_dataset(cache_path)
    else:
        print("Calculating MPI-ESM regression dataset...")
        if data_dict is None:
            data_dict = load_mpi_esm_data(eur_only=True)
        if var == 'tas':
            reg_ds_seasonal = [get_regression_ds(data_dict, season=s, var=var, future_window=future_window) for s in ['', 'djf', 'jja']]
            reg_ds_mpi = xr.concat([ds.assign_coords(season=s) for ds, s in zip(reg_ds_seasonal, ['', 'djf', 'jja'])], dim='season')
        else:
            reg_ds_mpi = get_regression_ds(data_dict, season='', var=var, future_window=future_window)
        reg_ds_mpi.attrs['future_window'] = list(fw)
        reg_ds_mpi.to_netcdf(cache_path)
        print(f"Saved MPI-ESM regression dataset to {cache_path}.")
    return reg_ds_mpi.mean(dim='realiz') if ens_mean else reg_ds_mpi


# PLOTTING MPI-ESM DATA

def simulations_plot(data_dict, ssp='all', hos_type='all', season='', var='tas', region='EU', window=10, plot_bg='white', ext_axs=None):

    validate_choice('season', season, ALLOWED_SEASONS)
    validate_choice('var', var, ALLOWED_VARS)
    validate_choice('hos_type', hos_type, ALLOWED_HOSING_TYPES)
    validate_choice('plot_bg', plot_bg, ALLOWED_PLOT_BG)

    # always compare against all-year AMOC strength
    his_amoc_yr = data_dict['']['amoc']['his']
    ssp_amoc_yr = data_dict['']['amoc']['ssp']
    ssphos_amoc_yr = data_dict['']['amoc']['ssphos']
    # potentially seasonal variable
    his_var = data_dict[season][var]['his']
    ssp_var = data_dict[season][var]['ssp']
    ssphos_var = data_dict[season][var]['ssphos']
    # season-invariant masks
    masks_eur_dict = data_dict['masks']
    mask_land = data_dict['masks']['LAND']

    his_years = his_amoc_yr.time.dt.year
    ssp_years = ssp_amoc_yr.time.dt.year

    plt.style.use('default')
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    if ext_axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
        fig.subplots_adjust(hspace=0.2)
    else:
        axs = ext_axs
        plt.rcParams.update({'font.size': 10})
    axs[1].sharex(axs[0])


    amoc = xr.concat([his_amoc_yr, ssp_amoc_yr.sel(scenar='ssp245')], dim='time').rolling(time=window, center=True).mean('time').AMOC_strength
    axs[0].plot(his_years,
                amoc.mean(dim='realiz')[:len(his_years)], c='k' if not plot_bg=='black' else 'white', lw=2.5, clip_on=True)
    axs[0].fill_between(his_years,
                        amoc.mean(dim='realiz')[:len(his_years)] - amoc.std(dim='realiz')[:len(his_years)],
                        amoc.mean(dim='realiz')[:len(his_years)] + amoc.std(dim='realiz')[:len(his_years)],
                        alpha=0.4, linewidth=0, color='k' if not plot_bg=='black' else 'white', clip_on=True)

    tas_eur =  weighted_area_lat(xr.concat([his_var, ssp_var.sel(scenar='ssp245')], dim='time').where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon').rolling(time=window, center=True).mean("time")[var]
    his_tas_eur_1850_1899 = weighted_area_lat(his_var.where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon')[var].mean(dim='realiz').sel(time=slice('1850', '1899')).mean(dim='time')
    axs[1].plot(his_years,
                tas_eur.mean(dim='realiz')[:len(his_years)] - his_tas_eur_1850_1899, 
                c='k' if not plot_bg=='black' else 'white', lw=2.5, clip_on=True)
    axs[1].fill_between(his_years,
                        tas_eur.mean(dim='realiz')[:len(his_years)] - his_tas_eur_1850_1899 - tas_eur.std(dim='realiz')[:len(his_years)],
                        tas_eur.mean(dim='realiz')[:len(his_years)] - his_tas_eur_1850_1899 + tas_eur.std(dim='realiz')[:len(his_years)],
                        alpha=0.4, linewidth=0, color='k' if not plot_bg=='black' else 'white', clip_on=True)

    for ssp_i in ssphos_amoc_yr.scenar.values:
        if ssp != 'all' and ssp_i != ssp:
            continue
        for hos_i in ssphos_amoc_yr.hosing.values:
            if hos_type != 'all1Sv':
                if hos_type == 'all':
                    if hos_i == '1':
                        continue
                elif hos_type == 'linear':
                    if 'lin' not in hos_i:
                        continue
                elif hos_type == 'constant':
                    if 'lin' in hos_i:
                        continue
                    if hos_i == '1':
                        continue
                elif hos_type == '1':
                    if hos_i != '1':
                        continue
                elif hos_type == None:
                    continue
                else:
                    print('Unknown hosing type.')
                    continue
            if hos_i == '1' and ssp_i != 'ssp245':
                continue
            amoc = xr.concat([his_amoc_yr, ssphos_amoc_yr.sel(scenar=ssp_i, hosing=hos_i)], dim='time').rolling(time=window, center=True).mean('time').isel(time=slice(0, -198)).AMOC_strength  # after including 1.0 Sv, ds goes until 2298
            axs[0].plot(ssp_years,
                    amoc.mean(dim='realiz')[-len(ssp_years):], color=hosing_colors[ssp_i][hos_i], lw=1.5, alpha=0.98, clip_on=False)
            axs[0].plot(ssp_years[-((window+1)//2)]+1.5,
                amoc.mean(dim='realiz')[-((window+1)//2)],
                marker=hosing_markers[hos_i], color=hosing_colors[ssp_i][hos_i], markersize=6, clip_on=False)

            if hos_i != '1' or var != 'tmn':
                tas_eur = weighted_area_lat(xr.concat([his_var.mean(dim='realiz'), ssphos_var.sel(scenar=ssp_i, hosing=hos_i).mean(dim='realiz')], dim='time').where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon').rolling(time=window, center=True).mean('time').isel(time=slice(0, -198))[var]   # after including 1.0 Sv, ds goes until 2298
            else: # separate treatment for 1.0 Sv hosing tmn because there is no ensemble, just one realisation
                tas_eur = weighted_area_lat(xr.concat([his_var.mean(dim='realiz'), ssphos_var.sel(scenar=ssp_i, hosing=hos_i)], dim='time').where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon').rolling(time=window, center=True).mean('time').isel(time=slice(0, -198))[var]   # after including 1.0 Sv, ds goes until 2298
            his_tas_eur_1850_1899 = weighted_area_lat(his_var.where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon')[var].mean(dim='realiz').sel(time=slice('1850', '1899')).mean(dim='time')
            axs[1].plot(ssp_years,
                    tas_eur[-len(ssp_years):] - his_tas_eur_1850_1899, color=hosing_colors[ssp_i][hos_i], lw=1.5, alpha=0.98, clip_on=False)
            axs[1].plot(ssp_years[-((window+1)//2)]+1.5,
                tas_eur[-((window+1)//2)] - his_tas_eur_1850_1899,
                marker=hosing_markers[hos_i], color=hosing_colors[ssp_i][hos_i], markersize=6, clip_on=False)
            
        amoc = xr.concat([his_amoc_yr, ssp_amoc_yr.sel(scenar=ssp_i)], dim='time').rolling(time=window, center=True).mean('time').AMOC_strength
        axs[0].plot(ssp_years,
                    amoc.mean(dim='realiz')[-len(ssp_years):], color=hosing_colors[ssp_i]['ge'], lw=2.5, clip_on=False)
        axs[0].fill_between(ssp_years,
                            amoc.mean(dim='realiz')[-len(ssp_years):] - amoc.std(dim='realiz')[-len(ssp_years):],
                            amoc.mean(dim='realiz')[-len(ssp_years):] + amoc.std(dim='realiz')[-len(ssp_years):],
                            alpha=0.4, linewidth=0, color=hosing_colors[ssp_i]['ge'], clip_on=False, zorder=3)
        axs[0].plot(ssp_years[-((window+1)//2)]+1.5,
                amoc.mean(dim='realiz')[-((window+1)//2)],
                marker='o', color=hosing_colors[ssp_i]['ge'], markersize=6, label=f"{ssp_i}", clip_on=False, zorder=3)

        tas_eur = weighted_area_lat(xr.concat([his_var, ssp_var.sel(scenar=ssp_i)], dim='time').where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon').rolling(time=window, center=True).mean("time")[var]
        his_tas_eur_1850_1899 = weighted_area_lat(his_var.where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon')[var].mean(dim='realiz').sel(time=slice('1850', '1899')).mean(dim='time')
        axs[1].plot(ssp_years,
                    tas_eur.mean(dim='realiz')[-len(ssp_years):] - his_tas_eur_1850_1899,
                    color=hosing_colors[ssp_i]['ge'], lw=2.5, clip_on=False)
        axs[1].fill_between(ssp_years,
                            tas_eur.mean(dim='realiz')[-len(ssp_years):] - his_tas_eur_1850_1899 - tas_eur.std(dim='realiz')[-len(ssp_years):],
                            tas_eur.mean(dim='realiz')[-len(ssp_years):] - his_tas_eur_1850_1899 + tas_eur.std(dim='realiz')[-len(ssp_years):],
                            alpha=0.4, linewidth=0, color=hosing_colors[ssp_i]['ge'], clip_on=False)
        axs[1].plot(ssp_years[-((window+1)//2)]+1.5,
                tas_eur.mean(dim='realiz')[-((window+1)//2)] - his_tas_eur_1850_1899,
                marker='o', color=hosing_colors[ssp_i]['ge'], markersize=5, clip_on=False)


    ############################ formatting

    axs[0].set_ylabel('AMOC strength [Sv]')
    axs[0].set_ylim(8, 22)
    axs[1].set_xlabel('')
    axs[1].set_ylabel((season.upper()+' ' if season else '')+rf'$\Delta \mathrm{{T}}_\mathrm{{{region}}}$ [°C] w.r.t. 1850–1899' if var=='tas' else rf'$\Delta \mathrm{{Pr}}_\mathrm{{{region}}} \left[ \frac{{\mathrm{{mm}}}}{{\mathrm{{d}}}} \right]$ ' if var=='pr' else rf'$\Delta \mathrm{{TMn}}_\mathrm{{{region}}}$ [°C]' if var=='tmn' else 'fill var label', labelpad=5)
    axs[1].set_xlim(2000, 2100)
    if var == 'tas':
        axs[1].set_ylim(1, 6)
    elif var == 'pr':
        axs[1].set_ylim(-0.25, 0.2)

    axs[0].spines['top'].set_visible(False), axs[1].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False)

    # axs[1].yaxis.tick_right(), axs[1].yaxis.set_label_position('right')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines["left"].set_position(("axes", -0.02))
    axs[0].spines["left"].set_position(("axes", -0.02))

    axs[1].set_xticks(np.arange(2000, 2101, 25))

    if ext_axs is not None:
        return

    primary_subset_min = 7.606163386737876
    primary_subset_max = 19.01540846684469
    primary_position = axs[0].get_position()

    # Calculate the new vertical extent for the secondary axis
    total_height = primary_position.height
    new_y0 = primary_position.y0 + (primary_subset_min - axs[0].get_ylim()[0]) / (axs[0].get_ylim()[1] - axs[0].get_ylim()[0]) * total_height
    new_height = (primary_subset_max - primary_subset_min) / (axs[0].get_ylim()[1] - axs[0].get_ylim()[0]) * total_height

    secax = fig.add_axes([0.9,  # x-position
                        new_y0,  # starting vertical position
                        0.02,  # Width of the secondary axis
                        new_height])  # height of secondary axis

    secax.set_ylabel('AMOC weakening [%]', labelpad=5)

    secax.spines['top'].set_visible(False)
    secax.spines['left'].set_visible(False)
    secax.spines['bottom'].set_visible(False)
    secax.spines['right'].set_visible(True)

    secax.yaxis.tick_right()
    secax.yaxis.set_label_position('right')
    secax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    secondary_subset_min = convert_strength_to_weakening(primary_subset_max)  # Note: max maps to min
    secondary_subset_max = convert_strength_to_weakening(primary_subset_min)  # Note: min maps to max

    secax.set_ylim(secondary_subset_max, secondary_subset_min)
    secondary_ticks = np.linspace(secondary_subset_max, secondary_subset_min, 7)

    secax.set_yticks(secondary_ticks)
    secax.set_yticklabels([f"{tick:.0f}" for tick in secondary_ticks])

    secax.set_facecolor('none')

    make_legend(fig, ssp, hos_type, line_spacing=0.015, pos=(0.18, 0.62), col_width=0.12, plot_bg=plot_bg)

    plt.savefig(f'../plots/simulations_ssp-{ssp}_hos-{hos_type}_window-{window}_{region}_bg-{plot_bg}.png', dpi=400, bbox_inches='tight', transparent=True if plot_bg=='black' else False)


def regression_plot(data_dict, lat=None, lon=None, region=None, var='tas', ssp='all', hos_type='all', hos_strength='all', window=10, regressions=True, combined_reg=True, no_plots=False, plot_bg='white', season='', ext_ax=None, xlim=(-8, 6), ylim=(-2.5, 1.5), equation_y_pos=0.95, equation_y_spacing=0.075, lag=0, compute_linearity=False, compute_scenario_independence=False, weakening_xaxis=True, simple_eqs=True, eqs_x=0.0):

    validate_choice('season', season, ALLOWED_SEASONS)
    validate_choice('var', var, ALLOWED_VARS)
    validate_choice('hos_type', hos_type, ALLOWED_HOSING_TYPES)
    validate_choice('plot_bg', plot_bg, ALLOWED_PLOT_BG)
    if not isinstance(lag, (int, np.integer)):
        raise ValueError(f"lag must be an integer, got {lag!r}")

    # always compare against all-year AMOC strength
    ssp_amoc_yr = data_dict['']['amoc']['ssp']
    ssphos_amoc_yr = data_dict['']['amoc']['ssphos'].isel(time=slice(0, -198))
    # potentially seasonal variable
    ssp_var = data_dict[season][var]['ssp']
    ssphos_var = data_dict[season][var]['ssphos'].isel(time=slice(0, -198))
    # season-invariant masks
    masks_eur_dict = data_dict['masks']
    mask_land = data_dict['masks']['LAND']

    if not no_plots:
        plt.style.use('default')
        if plot_bg == 'black':
            plt.style.use('dark_background')
            plt.rcParams['axes.facecolor'] = '#191919'
            plt.rcParams['figure.facecolor'] = '#191919'
        if ext_ax is None:
            fig, ax = plt.subplots(figsize=(9, 6))
        else:
            ax = ext_ax

    is_latlon, is_region, is_ocean = False, False, False
    if lat != None and lon != None:
        is_latlon = True
        if np.sum(~np.isnan(ssphos_var.isel(lat=lat, lon=lon)[var].values))==0:
            is_ocean = True
    elif region != None:
        is_region = True
    else:
        print('No lat/lon or region provided.')
        return

    combined_x = []
    combined_y = []
    combined_exp_id = []
    combined_ssp_name = []
    combined_hos_name = []

    eq_gap = 0.14  # horizontal label→equation gap (axes-frac); keeps SSP & combined rows aligned

    # for ssp_i in ssp_amoc_yr.scenar.values: # need to drop ssp585
    for ssp_id, ssp_i in enumerate(['ssp126', 'ssp245', 'ssp370']):
        ssp_x = []
        ssp_y = []
        ssp_exp_id = []
        
        if ssp != 'all' and ssp_i != ssp:
            continue

        for exp_id, exp in enumerate(np.concat([[ssphos_amoc_yr.hosing.values[-1]], ssphos_amoc_yr.hosing.values[:-1]])):
            if hos_type != 'all1Sv':
                if hos_type == 'all':
                    if exp == '1':
                        continue                    
                elif hos_type == 'linear':
                    if 'lin' not in exp:
                        continue
                elif hos_type == 'constant':
                    if 'lin' in exp:
                        continue
                    if exp == '1':
                        continue
                elif hos_type == '1':
                    if exp != '1':
                        continue
                else:
                    print('Unknown hosing type.')
                    continue
                if hos_strength != 'all' and exp != hos_strength:
                    continue
            if is_ocean:
                continue

            if exp == '1' and ssp_i != 'ssp245':
                continue
            
            x = ssphos_amoc_yr.sel(scenar=ssp_i, hosing=exp).rolling(time=window, center=True).mean("time").mean(dim='realiz').AMOC_strength.values - ssp_amoc_yr.sel(scenar=ssp_i).rolling(time=window, center=True).mean('time').mean(dim='realiz').AMOC_strength.values
            if is_latlon:
                if exp == '1' and var=='tmn':
                    y = ssphos_var.sel(scenar=ssp_i, hosing=exp).isel(lat=lat, lon=lon).rolling(time=window, center=True).mean("time")[var].values - ssp_var.sel(scenar=ssp_i).isel(lat=lat, lon=lon).rolling(time=window, center=True).mean("time").mean(dim='realiz')[var].values
                else:
                    y = ssphos_var.sel(scenar=ssp_i, hosing=exp).isel(lat=lat, lon=lon).rolling(time=window, center=True).mean("time").mean(dim='realiz')[var].values - ssp_var.sel(scenar=ssp_i).isel(lat=lat, lon=lon).rolling(time=window, center=True).mean("time").mean(dim='realiz')[var].values
            elif is_region:
                if exp == '1' and var=='tmn':
                    y = weighted_area_lat(ssphos_var.sel(scenar=ssp_i, hosing=exp).where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon').rolling(time=window, center=True).mean("time")[var].values - weighted_area_lat(ssp_var.sel(scenar=ssp_i).where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon').rolling(time=window, center=True).mean("time").mean(dim='realiz')[var].values
                else:
                    y = weighted_area_lat(ssphos_var.sel(scenar=ssp_i, hosing=exp).where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon').rolling(time=window, center=True).mean("time").mean(dim='realiz')[var].values - weighted_area_lat(ssp_var.sel(scenar=ssp_i).where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon').rolling(time=window, center=True).mean("time").mean(dim='realiz')[var].values
            else:
                print('No lat/lon or region provided.')

            # Apply lag: align x at year t-lag with y at year t.
            # lag>0: T responds `lag` years after AMOC. lag<0: T leads AMOC by |lag| years (placebo).
            if lag > 0:
                x = x[:-lag]
                y = y[lag:]
            elif lag < 0:
                x = x[-lag:]
                y = y[:lag]

            # Filter out NaN values
            mask = ~np.isnan(x) & ~np.isnan(y)
            x = x[mask]
            y = y[mask]
            
            ssp_x.extend(x)
            ssp_y.extend(y)
            ssp_exp_id.extend([exp_id]*len(x))
            combined_x.extend(x)
            combined_y.extend(y)
            combined_exp_id.extend([9*ssp_id + exp_id]*len(x))
            combined_ssp_name.extend([ssp_i]*len(x))
            combined_hos_name.extend([exp]*len(x))
        
            if not no_plots:
                ax.scatter(x, y,
                            label=hosing_names[ssp_i][exp], c=hosing_colors[ssp_i][exp],
                            marker=hosing_markers[exp], s=25, clip_on=False)

        if not is_ocean:
            # Perform linear regression using statsmodels
            x_with_const = sm.add_constant(ssp_x)
            model = sm.OLS(ssp_y, x_with_const).fit(cov_type='cluster', cov_kwds={'groups': ssp_exp_id, 'use_correction': True})

            # Manual df correction: use G-1 degrees of freedom for cluster-robust inference
            G_ssp = len(set(ssp_exp_id))
            df_cluster_ssp = G_ssp - 1
            t_crit_ssp = scipy_stats.t.ppf(0.975, df=df_cluster_ssp)
            t_stats_ssp = model.params / model.bse
            p_values_ssp = 2 * scipy_stats.t.sf(np.abs(t_stats_ssp), df=df_cluster_ssp)

            # DEBUG prints
            # print(f"\n=== {ssp_i} per-SSP regression ===")
            # print(f"  N={len(ssp_y)}, G={G_ssp}, df_cluster={df_cluster_ssp}")
            # print(f"  statsmodels df_resid={model.df_resid} (should be {df_cluster_ssp})")
            # print(f"  t_crit(0.975, df={df_cluster_ssp}) = {t_crit_ssp:.4f}")
            # for i_param, name in enumerate(['const', 'x1']):
            #     print(f"  {name}: coef={model.params[i_param]:.4f}, se={model.bse[i_param]:.4f}, "
            #           f"t={t_stats_ssp[i_param]:.3f}, "
            #           f"p_corrected={p_values_ssp[i_param]:.6f} (was {model.pvalues[i_param]:.6f}), "
            #           f"CI=[{model.params[i_param] - t_crit_ssp*model.bse[i_param]:.4f}, "
            #           f"{model.params[i_param] + t_crit_ssp*model.bse[i_param]:.4f}]")

            x_range = np.linspace(xlim[0], xlim[1], 200)
            x_range_with_const = sm.add_constant(x_range)
            y_pred = model.predict(x_range_with_const)
            if regressions and not no_plots:
                ax.plot(x_range, y_pred,
                        color=hosing_colors[ssp_i]['ge'], lw=1.5, linestyle='-')

            # Calculate confidence intervals manually with corrected df
            pred_se = model.get_prediction(x_range_with_const).se
            lower_bound = y_pred - t_crit_ssp * pred_se
            upper_bound = y_pred + t_crit_ssp * pred_se
            if regressions and not no_plots:
                ax.fill_between(x_range, lower_bound, upper_bound,
                                color=hosing_colors[ssp_i][exp], alpha=0.4, linewidth=0)

            # Plot regression equation and coefficients with corrected p-values
            coef = model.params[1]
            intercept = model.params[0]
            p_value = p_values_ssp[1]
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            else:
                significance = ''
            if is_latlon:
                # equation = fr"$\Delta T_{{lat{lat}\_lon{lon}}}$ = {coef:.2f}{significance} $\Delta AMOC$"+(" +" if intercept>0 else " ") +f"{intercept:.2f}"
                var_label = rf'$\hat{{\Delta T}}_{{\text{{AMOC}},{ssp_i}}}(\mathrm{{lat}}\,{lat},\,\mathrm{{lon}}\,{lon})$' if var=='tas' else rf'$\hat{{\Delta Pr}}_{{\text{{AMOC}},{ssp_i}}}(\mathrm{{lat}}\,{lat},\,\mathrm{{lon}}\,{lon})$' if var=='pr' else 'fill var label'
            else:
                # equation = fr"$\Delta T_{{{region}}}$ = {coef:.2f}{significance} $\Delta AMOC$"+(" +" if intercept>0 else " ") +f"{intercept:.2f}"
                var_label = rf'$\hat{{\Delta T}}_{{\text{{AMOC}},{ssp_i}}}({region})$' if var=='tas' else rf'$\hat{{\Delta Pr}}_{{\text{{AMOC}},{ssp_i}}}({region})$' if var=='pr' else rf'$\hat{{\Delta TMn}}_{{\text{{AMOC}},{ssp_i}}}({region})$' if var == 'tmn' else 'fill var label'
            if weakening_xaxis:
                coef_for_display = -coef * AMOC_pi_MPI / 10.0
                slope_units = rf'°C/10% $\Delta \mathrm{{AMOC}}_{{\mathrm{{add.}},{ssp_i}}}$'
                slope_frac = r'$\frac{{}^{\circ}\mathrm{C}}{10\%}$'
            else:
                coef_for_display = coef
                slope_units = rf'°C/Sv $\Delta \mathrm{{AMOC}}_{{\mathrm{{add.}},{ssp_i}}}$'
                slope_frac = r'$\frac{{}^{\circ}\mathrm{C}}{\mathrm{Sv}}$'
            coef_display = f'{coef_for_display:.2f}' if var=='tas' or var=='tmn' else f'{coef_for_display:.3f}' if var=='pr' else f'{coef_for_display}'
            isign = '+' if intercept >= 0 else '-'
            imag = f'{abs(intercept):.2f}' if var=='tas' or var=='tmn' else f'{abs(intercept):.3f}' if var=='pr' else f'{abs(intercept)}'
            if simple_eqs:
                equation = fr"$\hat{{y}}$ = {coef_display}$^{{{{\text{{{significance}}}}}}}$ {slope_frac} $\cdot$ x {isign} {imag} $^{{\circ}}$C"
            else:
                equation = fr"{var_label} = {coef_display}$^{{{{\text{{{significance}}}}}}}${slope_units} {isign} {imag} $^{{\circ}}$C"
            if regressions and not no_plots:
                ax.text(eqs_x, equation_y_pos, f"{hosing_names[ssp_i]['ge'][11:]}:",
                        transform=ax.transAxes, fontsize=10, verticalalignment='baseline', color=hosing_colors[ssp_i]['ge'])
                ax.text(eqs_x + eq_gap, equation_y_pos, equation,
                        transform=ax.transAxes, fontsize=10, verticalalignment='baseline', color=hosing_colors[ssp_i]['ge'])
            equation_y_pos -= equation_y_spacing

    if not is_ocean:
        # Combine both datasets for a third regression line
        combined_x = np.array(combined_x)
        combined_y = np.array(combined_y)

        combined_x_with_const = sm.add_constant(combined_x)
        combined_model = sm.OLS(combined_y, combined_x_with_const).fit(cov_type='cluster', cov_kwds={'groups': combined_exp_id, 'use_correction': True})

        # Manual df correction for combined model
        G_combined = len(set(combined_exp_id))
        df_cluster_combined = G_combined - 1
        t_crit_combined = scipy_stats.t.ppf(0.975, df=df_cluster_combined)
        t_stats_combined = combined_model.params / combined_model.bse
        p_values_combined = 2 * scipy_stats.t.sf(np.abs(t_stats_combined), df=df_cluster_combined)

        # DEBUG prints
        # print(f"\n=== Combined regression ===")
        # print(f"  N={len(combined_y)}, G={G_combined}, df_cluster={df_cluster_combined}")
        # print(f"  statsmodels df_resid={combined_model.df_resid} (should be {df_cluster_combined})")
        # print(f"  t_crit(0.975, df={df_cluster_combined}) = {t_crit_combined:.4f}")
        # for i_param, name in enumerate(['const', 'x1']):
        #     print(f"  {name}: coef={combined_model.params[i_param]:.4f}, se={combined_model.bse[i_param]:.4f}, "
        #           f"t={t_stats_combined[i_param]:.3f}, "
        #           f"p_corrected={p_values_combined[i_param]:.6f} (was {combined_model.pvalues[i_param]:.6f}), "
        #           f"CI=[{combined_model.params[i_param] - t_crit_combined*combined_model.bse[i_param]:.4f}, "
        #           f"{combined_model.params[i_param] + t_crit_combined*combined_model.bse[i_param]:.4f}]")

        combined_x_range = np.linspace(xlim[0], xlim[1], 200)
        combined_x_range_with_const = sm.add_constant(combined_x_range)
        combined_y_pred = combined_model.predict(combined_x_range_with_const)
        if (regressions and combined_reg) and not no_plots:
            ax.plot(combined_x_range, combined_y_pred,
                    color='black' if not plot_bg=='black' else 'white', lw=1.5, linestyle='-')

        # Calculate confidence intervals manually with corrected df
        combined_pred_se = combined_model.get_prediction(combined_x_range_with_const).se
        combined_lower_bound = combined_y_pred - t_crit_combined * combined_pred_se
        combined_upper_bound = combined_y_pred + t_crit_combined * combined_pred_se
        if (regressions and combined_reg) and not no_plots:
            ax.fill_between(combined_x_range, combined_lower_bound, combined_upper_bound,
                            color='black' if not plot_bg=='black' else 'white', alpha=0.4, linewidth=0)

        # Plot combined regression equation and coefficients with corrected p-values
        combined_coef = combined_model.params[1]
        combined_intercept = combined_model.params[0]
        combined_p_value = p_values_combined[1]
        combined_ste = combined_model.bse[1]
        combined_rsq = combined_model.rsquared
        if combined_p_value < 0.001:
            combined_significance = '***'
        elif combined_p_value < 0.01:
            combined_significance = '**'
        elif combined_p_value < 0.05:
            combined_significance = '*'
        else:
            combined_significance = ''
        if is_latlon:
            # combined_equation = fr"$\Delta T_{{lat{lat}\_lon{lon}}}$ = {combined_coef:.2f}{combined_significance} $\Delta AMOC$"+(" +" if intercept>0 else " ") +f"{combined_intercept:.2f}"
            var_label = rf'$\hat{{\Delta T}}_{{\text{{AMOC}}}}(\mathrm{{lat}}\,{lat},\,\mathrm{{lon}}\,{lon})$' if var=='tas' else rf'$\hat{{\Delta Pr}}_{{\text{{AMOC}}}}(\mathrm{{lat}}\,{lat},\,\mathrm{{lon}}\,{lon})$' if var=='pr' else 'fill var label'
        else:
            # combined_equation = fr"$\Delta T_{{{region}}}$ = {combined_coef:.2f}{combined_significance} $\Delta AMOC$"+(" +" if intercept>0 else " ") +f"{combined_intercept:.2f}"
            var_label = rf'$\hat{{\Delta T}}_{{\text{{AMOC}}}}({region})$' if var=='tas' else rf'$\hat{{\Delta Pr}}_{{\text{{AMOC}}}}({region})$' if var=='pr' else rf'$\hat{{\Delta TMn}}_{{\text{{AMOC}}}}({region})$' if var == 'tmn' else 'fill var label'
        if weakening_xaxis:
            combined_coef_for_display = -combined_coef * AMOC_pi_MPI / 10.0
            slope_units = r'°C/10% $\Delta \mathrm{AMOC}_\mathrm{add.}$'
            slope_frac = r'$\frac{{}^{\circ}\mathrm{C}}{10\%}$'
        else:
            combined_coef_for_display = combined_coef
            slope_units = r'°C/Sv $\Delta \mathrm{AMOC}_\mathrm{add.}$'
            slope_frac = r'$\frac{{}^{\circ}\mathrm{C}}{\mathrm{Sv}}$'
        combined_coef_display = f'{combined_coef_for_display:.2f}' if var=='tas' or var=='tmn' else f'{combined_coef_for_display:.3f}' if var=='pr' else f'{combined_coef_for_display}'
        isign = '+' if combined_intercept >= 0 else '-'
        imag = f'{abs(combined_intercept):.2f}' if var=='tas' or var=='tmn' else f'{abs(combined_intercept):.3f}' if var=='pr' else f'{abs(combined_intercept)}'
        if simple_eqs:
            combined_equation = fr"$\hat{{y}}$ = {combined_coef_display}$^{{{{\text{{{combined_significance}}}}}}}$ {slope_frac} $\cdot$ x {isign} {imag} $^{{\circ}}$C"
        else:
            combined_equation = fr"{var_label} = {combined_coef_display}$^{{{{\text{{{combined_significance}}}}}}}${slope_units} {isign} {imag} $^{{\circ}}$C"
        if (regressions and combined_reg) and not no_plots:
            ax.text(eqs_x, equation_y_pos, "All SSPs:",
                    transform=ax.transAxes, fontsize=10, verticalalignment='baseline', color='black' if not plot_bg=='black' else 'white')
            ax.text(eqs_x + eq_gap, equation_y_pos, combined_equation,
                    transform=ax.transAxes, fontsize=10, verticalalignment='baseline', color='black' if not plot_bg=='black' else 'white')

        # Set last_call before any early plotting return so ext_ax callers
        # (e.g. FigSXX panel b/c) still see the diagnostics.
        regression_plot.last_call = {
            'n_combined': int(len(combined_y)),
            'g_combined': int(G_combined),
            'lag': int(lag),
            'sigma_resid': float(np.sqrt(combined_model.mse_resid)),
            'sxx': float(np.sum((combined_x - combined_x.mean())**2)),
        }
        if compute_linearity:
            _diag = reset_diagnostics(combined_y, combined_x_with_const, combined_exp_id, combined_model)
            regression_plot.last_call.update({
                'partial_r2': _diag['partial_r2'],
                'delta_r2_total': _diag['delta_r2_total'],
                'gamma2': _diag['gamma2'],
                'gamma3': _diag['gamma3'],
                'beta2_sign': _diag['beta2_sign'],
                'beta3_sign': _diag['beta3_sign'],
                'reset_pvalue': _diag['reset_pvalue'],
                'baseline_r2': _diag['baseline_r2'],
                'combined_yhat': _diag['yhat'],
                'combined_resid': _diag['resid'],
                'combined_x': np.asarray(combined_x),
                'combined_y': np.asarray(combined_y),
                'combined_exp_id': np.asarray(combined_exp_id),
                'combined_coef': float(combined_coef),
                'combined_intercept': float(combined_intercept),
                'combined_ste': float(combined_ste),
                'combined_intercept_aug': _diag['intercept_aug'],
                'combined_coef_aug': _diag['coef_aug'],
                'combined_ssp_name': np.asarray(combined_ssp_name),
                'combined_hos_name': np.asarray(combined_hos_name),
            })

        if compute_scenario_independence:
            _diag_si = scenario_independence_diagnostics(
                combined_y, combined_x_with_const, combined_exp_id,
                np.asarray(combined_ssp_name), combined_model,
            )
            # Per-point arrays and pooled-fit coefficients are also
            # populated here so the figure script can run with only
            # compute_scenario_independence=True (no need to also pass
            # compute_linearity=True for panel d/e/f drawing).
            regression_plot.last_call.update({
                'wald_fstat': _diag_si['wald_fstat'],
                'wald_pvalue': _diag_si['wald_pvalue'],
                'partial_r2_indep': _diag_si['partial_r2_indep'],
                'delta_r2_total_indep': _diag_si['delta_r2_total_indep'],
                'baseline_r2': _diag_si['baseline_r2'],
                'beta_ssp126': _diag_si['beta_ssp126'],
                'beta_ssp245': _diag_si['beta_ssp245'],
                'beta_ssp370': _diag_si['beta_ssp370'],
                'intercept_ssp126': _diag_si['intercept_ssp126'],
                'intercept_ssp245': _diag_si['intercept_ssp245'],
                'intercept_ssp370': _diag_si['intercept_ssp370'],
                'max_pairwise_slope_diff': _diag_si['max_pairwise_slope_diff'],
                'n_scenarios_present': _diag_si['n_scenarios_present'],
                'combined_x': np.asarray(combined_x),
                'combined_y': np.asarray(combined_y),
                'combined_exp_id': np.asarray(combined_exp_id),
                'combined_coef': float(combined_coef),
                'combined_intercept': float(combined_intercept),
                'combined_ste': float(combined_ste),
                'combined_ssp_name': np.asarray(combined_ssp_name),
                'combined_hos_name': np.asarray(combined_hos_name),
            })

    if not no_plots:
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        if weakening_xaxis:
            # Sv stays as the data axis; move it to the top. The bottom
            # %-weakening axis is added as an independent overlay by the
            # caller (Fig1.make_figure) so its endpoints land on round
            # tick values — matching the panel-a treatment.
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.set_xlabel("Deviation from MPI-GE AMOC strength [Sv]", labelpad=4)
        else:
            ax.set_xlabel("Deviation from MPI-GE AMOC strength [Sv]")
        if is_latlon:
            ax.set_ylabel((season.upper()+' ' if season else '')+f"Deviation from GE ensemble mean lat{lat}_lon{lon} temperature [°C]" if var=='tas' else rf'Deviation from MPI-GE $\Delta \mathrm{{Pr}}_\mathrm{{lat{lat}lon{lon}}} \left[ \frac{{\mathrm{{mm}}}}{{\mathrm{{d}}}} \right]$ ' if var=='pr' else rf'Deviation from MPI-GE $\Delta \mathrm{{TMn}}_\mathrm{{lat{lat}lon{lon}}}$ [°C]' if var=='tmn' else 'fill var label')
        else:
            ax.set_ylabel((season.upper()+' ' if season else '')+rf"Deviation from MPI-GE $\Delta \mathrm{{T}}_\mathrm{{{region}}}$ [°C]" if var=='tas' else rf'Deviation from MPI-GE $\Delta \mathrm{{Pr}}_\mathrm{{{region}}} \left[ \frac{{\mathrm{{mm}}}}{{\mathrm{{d}}}} \right]$ ' if var=='pr' else rf'Deviation from MPI-GE $\Delta \mathrm{{TMn}}_\mathrm{{{region}}}$ [°C]' if var=='tmn' else 'fill var label')
        ax.axhline(0, color='black' if not plot_bg=='black' else 'white', lw=1., ls='--')
        ax.axvline(0, color='black' if not plot_bg=='black' else 'white', lw=1., ls='--')
        if not weakening_xaxis:
            ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines["left"].set_position(("axes", -0.02))
        if weakening_xaxis:
            # Flip the data axis so weaker AMOC sits on the right, matching
            # the bottom %-axis convention (-30% on left, +40% on right).
            # Also hide the bottom spine — the independent %-overlay built
            # in Fig1.make_figure provides the visible bottom axis line.
            ax.invert_xaxis()
            ax.spines['bottom'].set_visible(False)

        if ext_ax is not None:
            return

        if is_latlon:
            plt.title(f"Location: {ssphos_var.lat.isel(lat=lat).values:.2f}°N, {ssphos_var.lon.isel(lon=lon).values:.2f}°E", loc='left', fontsize=12)

        make_legend(fig, ssp, hos_type, hos_strength=hos_strength, line_spacing=0.023, pos=(0.65, 0.58), col_width=0.12, plot_bg=plot_bg, incl_ge=False, markers=True)

        fig.savefig(f'../plots/regression-{regressions}_ssp-{ssp}_hos-{hos_type}_strength-{hos_strength}_window-{window}_region-{region}_lat-{lat}_lon-{lon}_bg-{plot_bg}_{season}.png', dpi=400, bbox_inches='tight', transparent=True if plot_bg=='black' else False)

    if not is_ocean:
        return np.round(combined_intercept, 4), np.round(combined_coef, 4), np.round(combined_ste, 5), np.round(combined_rsq, 4)
    else:
        return np.nan, np.nan, np.nan, np.nan
    


def net_cooling_point_map(reg_ds, ssp, T_ref='pi', season='', title=False, plot_bg='white', ext_ax=None, custom_cmap=None, weakening_primary=False, show_cmip_range_contours=False, amoc_extent=None):

    validate_choice('T_ref', T_ref, ALLOWED_T_REF)
    validate_choice('season', season, ALLOWED_SEASONS)
    validate_choice('plot_bg', plot_bg, ALLOWED_PLOT_BG)
    # CMIP6 weakening envelope: default to the canonical module global; a
    # variant future-window run passes its own window-keyed extent here.
    _ext = AMOC_extent if amoc_extent is None else amoc_extent

    mid_gray = '#b0b0b0' # '#b0b0b0', "#989898"

    plt.style.use('default')
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    if ext_ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.Robinson(central_longitude=8)})
    else:
        ax = ext_ax
    ax.spines['geo'].set_visible(False)
    ax.set_rasterized(True)
    ax.coastlines(color='black' if not plot_bg=='black' else 'white', linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS, edgecolor='black' if not plot_bg=='black' else 'white', linewidth=.5)
    ax.add_feature(cfeature.LAND, facecolor=mid_gray)
    ax.add_feature(cfeature.OCEAN, facecolor='white' if not plot_bg=='black' else '#191919', zorder=2) # standard ocean: #A6CAE0

    base_cmap = cm.YlGnBu # options: PuBu, YlGnBu, cool
    mod_cmap = truncate_colormap(base_cmap, minval=0.17, maxval=1.0, darken=1.) # darkens for darken < 1, otherwise brightens
    cmap = mod_cmap if custom_cmap is None else custom_cmap

    if weakening_primary:
        levels = np.linspace(0, 100, 11)
        cmap = cmap.reversed()
        cmap.set_under(mid_gray)
        cmap.set_over(mid_gray)
    else:
        upper_bound = 18
        levels = np.linspace(0, upper_bound, 10)
        cmap.set_under(mid_gray)
        cmap.set_over(mid_gray)

    norm = BoundaryNorm(levels, ncolors=plt.get_cmap('PuBu').N, clip=False)

    if T_ref == 'pi':
        if 'season' in reg_ds.dims:
            data = reg_ds.req_weakening_pi.sel(scenar=ssp, season=season) if weakening_primary else reg_ds.req_strength_pi.sel(scenar=ssp, season=season)
        else:
            data = reg_ds.req_weakening_pi.sel(scenar=ssp) if weakening_primary else reg_ds.req_strength_pi.sel(scenar=ssp)
            if season != '':
                print('Seasonal net cooling not available!')
    else:
        if 'season' in reg_ds.dims:
            data = reg_ds.req_weakening_pd.sel(scenar=ssp, season=season) if weakening_primary else reg_ds.req_strength_pd.sel(scenar=ssp, season=season)
        else:
            data = reg_ds.req_weakening_pd.sel(scenar=ssp) if weakening_primary else reg_ds.req_strength_pd.sel(scenar=ssp)
            if season != '':
                print('Seasonal net cooling not available!')

    contour = ax.contourf(
        reg_ds.lon, reg_ds.lat, data,
        levels=levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='neither'
    )

    if show_cmip_range_contours:
        weakening_var = reg_ds[f'req_weakening_{T_ref}']
        if 'season' in reg_ds.dims:
            weakening_field = weakening_var.sel(scenar=ssp, season=season)
        else:
            weakening_field = weakening_var.sel(scenar=ssp)
        for lw, color in [(2.5, 'white'), (0.8, 'black')]:
            ax.contour(
                reg_ds.lon, reg_ds.lat, weakening_field,
                levels=[_ext[ssp][1], _ext[ssp][0]],
                colors=color, linewidths=lw,
                transform=ccrs.PlateCarree(),
                zorder=1,
                linestyles='--' if color=='black' else '-',
            )

    #  w/o Iceland: -9.5
    ax.set_extent([-18.5, 38.5, 34.8, 72], crs=ccrs.PlateCarree())
    
    # add_square(ax, 39.5, 65,  64.5, 75.5, colour='#191919' if plot_bg=='black' else 'white') # cut off North Russia where previously there was no data
    add_square(ax, -13.5, 0,  60, 75.5, colour='#191919' if plot_bg=='black' else 'white') # get rid of Nordic sea small islands
    add_square(ax, -30, -18,  68, 75, colour='#191919' if plot_bg=='black' else 'white') # get rid of Greenland

    if ext_ax is not None:
        return contour

    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.08, aspect=30, extend='neither')

    if weakening_primary:
        cbar.ax.invert_yaxis()
        cbar.set_ticks(list(range(0, 101, 10)))
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0f}%"))
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')
        cbar.set_label('AMOC weakening', fontsize=12, labelpad=2, loc='bottom')
        secax = cbar.ax.secondary_yaxis('left', functions=(convert_weakening_to_strength, convert_strength_to_weakening))
        secax.set_ylabel('AMOC strength [Sv]', fontsize=12.8, labelpad=0, loc='bottom')
        secax.set_yticks(list(range(0, 19, 2)))
        secax.set_yticklabels([f'{s} Sv' for s in range(0, 19, 2)])
    else:
        cbar.set_label('AMOC strength [Sv]', fontsize=12, labelpad=2, loc='bottom')
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.yaxis.set_ticks_position('left')
        secax = cbar.ax.secondary_yaxis('right', functions=(convert_strength_to_weakening, convert_weakening_to_strength))
        secax.set_ylabel('AMOC weakening', fontsize=12.8, labelpad=0, loc='bottom')
        secax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        secax.set_yticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])

    if weakening_primary:
        rect_y = 1 - _ext[ssp][0] / 100
        rect_height = (_ext[ssp][0] - _ext[ssp][1]) / 100
        text_y = 1 - (_ext[ssp][0] + _ext[ssp][1]) / 200
    else:
        rect_y = (1 - _ext[ssp][0]/100) / (1 - convert_strength_to_weakening(upper_bound)/100)
        rect_height = ((_ext[ssp][0]-_ext[ssp][1])/100) / (1 - convert_strength_to_weakening(upper_bound)/100)
        text_y = (1 - (_ext[ssp][0] + _ext[ssp][1]) / 200) / (1 - convert_strength_to_weakening(upper_bound)/100)

    rect = Rectangle(
        (-3.2, rect_y),
        1.2,                   # width of the patch
        rect_height, # height in normalized coords
        transform=cbar.ax.transAxes,
        color=hosing_colors[ssp]['ge'],
        alpha=1.0,
        clip_on=False
    )
    cbar.ax.add_patch(rect)

    # Add vertical text inside the rectangle
    cbar.ax.text(
        -2.55,  # x position
        text_y,
        "2100 CMIP6 range",
        color='white',
        fontsize=10,
        rotation=90,
        va='center',
        ha='center',
        transform=cbar.ax.transAxes,
        zorder=10
    )

    fig.savefig(f'../plots/net_cooling_maps_{ssp}_{T_ref}_bg-{plot_bg}.png', dpi=400, bbox_inches='tight', transparent=True if plot_bg=='black' else False)

    T_ref_str = 'preindustrial' if T_ref == 'pi' else 'present-day'
    if title:
            plt.title(f'Compensating warming since {T_ref_str} under {ssp}')


def plot_regression_coefficients(reg_ds, season='', var='tas', std_error=False, ste_relative=False, plot_mode='contourf', plot_bg='white', custom_cmap=cm.ocean_r, colorbar_v_max=0.7, ext_ax=None, savefig=True):

    if var =='tas':
        reg_ds = reg_ds.sel(season=season)
    else:
        if season != '':
            print('Seasonal regression coefficients only available for tas.')
            return

    # plot_mode: 'contourf' (default), 'gridded' (pcolormesh, ocean NaN-masked),
    # 'gridded_ocean_overlay' (pcolormesh, ocean hidden by polygon),
    # 'gridded_with_ocean' (pcolormesh, ocean values shown)
    gridded = plot_mode != 'contourf'
    mask_ocean = plot_mode == 'gridded'
    overlay_ocean = plot_mode in ('contourf', 'gridded_ocean_overlay')

    plt.style.use('default')
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    if ext_ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.Robinson(central_longitude=8)})
    else:
        ax = ext_ax
    ax.spines['geo'].set_visible(False)
    ax.set_rasterized(True)
    ax.coastlines(color='black' if not plot_bg=='black' else 'white', linewidth=.5, zorder=3)
    ax.add_feature(cfeature.BORDERS, edgecolor='black' if not plot_bg=='black' else 'white', linewidth=.5)
    ax.add_feature(cfeature.LAND, facecolor='white' if not plot_bg=='black' else '#191919')
    ax.add_feature(cfeature.OCEAN, facecolor='white' if not plot_bg=='black' else '#191919', zorder=2 if overlay_ocean else 0)

    # truncated colormap and values
    v_max = colorbar_v_max
    if std_error:
        cmap = cm.viridis
        levels = np.linspace(0.0, v_max, 9)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    else:
        if var == 'tas':
            trunc_cmap = truncate_colormap(custom_cmap, maxval=(v_max/1.5))
            levels = np.linspace(0.0, v_max, int(10*v_max)+1)
            norm = BoundaryNorm(levels, ncolors=trunc_cmap.N, clip=False)
        elif var == 'pr':
            trunc_cmap = truncate_colormap(cm.BrBG_r, minval=0.3)
            levels = np.linspace(-v_max*2/5, v_max, int(100*v_max)+1)
            norm = BoundaryNorm(levels, ncolors=trunc_cmap.N, clip=False)

    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(reg_ds) if mask_ocean else None
    if std_error:
        if ste_relative:
            # Relative standard error: ste / coef * 100 (in %)
            plot_data = reg_ds.ste_ensmean / reg_ds.coef_ensmean * 100
        else:
            # Absolute standard error: scale from °C/Sv to °C per 10% weakening
            plot_data = reg_ds.ste_ensmean * AMOC_pi_MPI / 10
        plot_cmap = cmap
    else:
        plot_data = AMOC_pi_MPI/10 * reg_ds.coef_ensmean
        plot_cmap = trunc_cmap

    if mask_ocean:
        plot_data = plot_data.where(land == 0)
    if gridded:
        contour = ax.pcolormesh(reg_ds.lon, reg_ds.lat, plot_data, transform=ccrs.PlateCarree(),
                                cmap=plot_cmap, norm=norm, shading='auto')
    elif std_error:
        contour = ax.contourf(reg_ds.lon, reg_ds.lat, plot_data, transform=ccrs.PlateCarree(),
                              cmap=plot_cmap, levels=levels, extend='max')
    else:
        contour = ax.contourf(reg_ds.lon, reg_ds.lat, plot_data, transform=ccrs.PlateCarree(),
                              cmap=plot_cmap, levels=levels, norm=norm, extend='neither')

    ax.set_extent([-18.5, 38.5, 34.8, 72], crs=ccrs.PlateCarree())

    # add_square(ax, 39.5, 65,  64.5, 75.5, colour='#191919' if plot_bg=='black' else 'white') # cut off North Russia where previously there was no data
    if plot_mode != 'gridded_with_ocean':  # keep islands/Greenland visible when showing ocean values
        add_square(ax, -13.5, 0,  60, 75.5, colour='#191919' if plot_bg=='black' else 'white') # get rid of Nordic sea small islands
        add_square(ax, -30, -18,  68, 75, colour='#191919' if plot_bg=='black' else 'white') # get rid of Greenland

    if ext_ax is not None:
        return contour

    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.02, aspect=30, extend='max' if std_error else 'neither')
    if var == 'tas':
        if std_error:
            if ste_relative:
                cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{np.round(x, 1)}%"))
            else:
                cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{np.round(x, 4)}°C"))
        else:
            cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{np.round(x, 2)}°C"))
    elif var == 'pr':
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.2f} mm/d"))
    if std_error:
        if ste_relative:
            cbar.set_label('Relative standard error [%]', fontsize=14)
        else:
            cbar.set_label('Standard error per 10% AMOC weakening [°C]', fontsize=14)
    else:
        cbar.set_label('Cooling per 10% AMOC weakening' if var=='tas' else 'Drying per 10% AMOC weakening' if var=='pr' else 'fill in var description', fontsize=14)

    if std_error:
        ste_type = 'relative' if ste_relative else 'absolute'
        plt.title(f'Regression coefficient {ste_type} standard errors for MPI-ESM', fontsize=14)
    else:
        plt.title(f'Regression coefficients for MPI-ESM', fontsize=14)

    if savefig and ext_ax is None:
        if std_error:
            ste_type = 'relative' if ste_relative else 'absolute'
            fig.savefig(f'../plots/regression_std_errors_{ste_type}_EU_bg-{plot_bg}_mode-{plot_mode}.pdf', dpi=400, bbox_inches='tight', transparent=True if plot_bg=='black' else False)
        else:
            fig.savefig(f'../plots/regression_coefficients_EU_bg-{plot_bg}_mode-{plot_mode}.pdf', dpi=400, bbox_inches='tight', transparent=True if plot_bg=='black' else False)

########################################################################################################################
# MULTI-MODEL ANALYSIS
########################################################################################################################

# HELPER FUNCTIONS FOR MULTI-MODEL ANALYSIS

# CMIP6 realisation discovery (shared by get_cmip_projections + get_gwl_diagnostic_data).

def model_root(model):
    return GISS_STAGING_PATH if model == 'GISS-E2-1-G' else cmip6_data_path


def lower_model_id(model):
    """Lowercased CMIP source_id used in directory/file names (HadGEM dropping the dash)."""
    return {'HadGEM3-GC3-1MM': 'hadgem3-gc31-mm',
            'HadGEM3-GC3-1LL': 'hadgem3-gc31-ll'}.get(model, model.lower())


def parse_int_rea(variant_id, physics, forcing):
    """Return integer N from ``r{N}i1{physics}{forcing}``, else None."""
    prefix = 'r'
    suffix = f'i1{physics}{forcing}'
    if not (variant_id.startswith(prefix) and variant_id.endswith(suffix)):
        return None
    try:
        return int(variant_id[len(prefix):-len(suffix)])
    except ValueError:
        return None


def discover_int_reas(model, scenario):
    """Return sorted integer realisations on disk for (model, scenario).

    Intersects amoc26 ∩ tas (rea must have both vars) and filters to
    the model's documented (physics, forcing) variant. Then verifies
    each candidate exists at the loader's exact path (under
    ``model_root(model)``) — necessary because for GISS-E2-1-G the
    loader uses a dedicated staging tree (``GISS_STAGING_PATH``)
    rather than the inventory's canonical roots, so the inventory
    may admit reas (e.g. uo1075 r11) that the staging tree omits.
    """
    p, f = MODEL_PHYSICS_FORCING[model]
    amoc_reas = cmip6_inventory.available_realisations(
        scenario, 'amoc26', models=[model])[model]
    tas_reas = set(cmip6_inventory.available_realisations(
        scenario, 'tas', models=[model])[model])

    scen_dir = 'historical' if scenario == 'his' else scenario
    model_lower = lower_model_id(model)
    loader_dir = model_root(model) + f'{scen_dir}/{model_lower}/'

    ns = []
    for v in amoc_reas:
        if v not in tas_reas:
            continue
        n = parse_int_rea(v, p, f)
        if n is None:
            continue
        amoc_p = loader_dir + f'{model_lower}_{v}_amoc26_yr.nc'
        tas_p = loader_dir + f'{model_lower}_{v}_tas_yr.nc'
        if os.path.exists(amoc_p) and os.path.exists(tas_p):
            ns.append(n)
    return sorted(set(ns))


def cesm_diagnostics_scan(boot_data, multi_model_dict, masks, var, fields,
                            sentinel_key, compute_kw):
    """Shared per-pixel scan loop for CESM linearity / scenario-independence.

    Combined-forcing case: pooled ssp126 + ssp585, mirroring the MPI scans
    almost line-for-line. The CESM2 pooled regression in cesm_regressions(...,
    combined_reg=True) is fit with HAC covariance (maxlags=window-1) because G=2
    makes cluster-robust SE degenerate.

    Iterates lat × lon × season, calls cesm_regressions with the requested
    compute_* flag, reads cesm_regressions.last_call, populates `fields`
    into an xr.Dataset. `sentinel_key` is a field that must be present in
    last_call for the pixel to count as successful (gates against silent
    skips). Skips ocean pixels where neither scenario carries finite tas.
    """
    seasons = ['', 'djf', 'jja'] if var == 'tas' else ['']
    # Boot's season coord is hardcoded to ['']. Skip seasonal slices that
    # would error out; the figure renders only the available seasons.
    available_seasons = [s for s in seasons
                         if s in list(boot_data.season.values.astype(str))]
    if len(available_seasons) < len(seasons):
        missing = sorted(set(seasons) - set(available_seasons))
        print(f'  CESM diagnostics: skipping seasons {missing} '
              f'(not in boot_data.season); will emit NaN.')

    nlat = boot_data.sizes['lat']
    nlon = boot_data.sizes['lon']
    arr_shape = (len(seasons), nlat, nlon)
    data_vars = {name: (['season', 'lat', 'lon'], np.full(arr_shape, np.nan))
                 for name in fields}
    coords = {
        'season': seasons,
        'lat': boot_data.lat.values,
        'lon': boot_data.lon.values,
    }
    out = xr.Dataset(data_vars=data_vars, coords=coords)

    for s_i, s in enumerate(seasons):
        if s not in available_seasons:
            continue
        print(f'  season={s!r}')
        # Cheap per-pixel "has any data" mask: collapse time/scenar/type.
        tas_any = boot_data.tas.sel(season=s).isel(time=0).isel(
            scenar=0, type=0)
        coverage = ~np.isnan(tas_any.values)
        i = 0
        for lat_i in range(nlat):
            for lon_i in range(nlon):
                if not coverage[lat_i, lon_i]:
                    i += 1
                    continue
                cesm_regressions.last_call = {}
                try:
                    cesm_regressions(boot_data, multi_model_dict, masks,
                                     ssp='all', lat=lat_i, lon=lon_i,
                                     region=None, combined_reg=True,
                                     no_plots=True, season=s,
                                     **compute_kw)
                except Exception:
                    i += 1
                    continue
                lc = cesm_regressions.last_call
                if sentinel_key in lc:
                    for name in fields:
                        if name in lc:
                            val = lc[name]
                            try:
                                out[name].values[s_i, lat_i, lon_i] = val
                            except (TypeError, ValueError):
                                pass
                i += 1
                if i % 100 == 0:
                    print(f'    {i}/{nlat*nlon} pixels done')

    return out


def get_linearity_diagnostics_cesm(boot_data=None, multi_model_dict=None,
                                   masks=None, var='tas', recompute=False):
    """Per-pixel Ramsey RESET diagnostics over CESM2 (combined ssp126+ssp585).

    Returns an xr.Dataset with dims (season, lat, lon) and variables:
    partial_r2, delta_r2_total, gamma2, gamma3, beta2_sign, beta3_sign,
    reset_pvalue, baseline_r2. Cached to
    data/linearity_diagnostics_cesm_{var}.nc. Uses HAC covariance
    (maxlags=window-1).
    """
    LINEARITY_CACHE_VERSION_CESM = 1
    validate_choice('var', var, ALLOWED_VARS)
    cache = local_path + f'linearity_diagnostics_cesm_{var}.nc'
    if not recompute:
        cached = check_diag_cache(cache, LINEARITY_CACHE_VERSION_CESM, 'CESM2 linearity')
        if cached is not None:
            return cached

    print('Calculating CESM2 linearity diagnostics (per-pixel RESET, HAC SE)...')
    if multi_model_dict is None or masks is None:
        multi_model_dict, masks = get_full_multi_model_dict()
    if boot_data is None:
        _, _, boot_data, _ = get_other_studies_data(masks)

    fields = ['partial_r2', 'delta_r2_total', 'gamma2', 'gamma3',
              'beta2_sign', 'beta3_sign', 'reset_pvalue', 'baseline_r2']
    out = cesm_diagnostics_scan(
        boot_data, multi_model_dict, masks, var=var, fields=fields,
        sentinel_key='partial_r2',
        compute_kw={'compute_linearity': True},
    )
    out.attrs['version'] = LINEARITY_CACHE_VERSION_CESM
    out.to_netcdf(cache)
    print(f'Saved CESM2 linearity diagnostics to {cache}.')
    return out


def get_scenario_independence_diagnostics_cesm(boot_data=None,
                                               multi_model_dict=None,
                                               masks=None, var='tas',
                                               recompute=False):
    """Per-pixel Wald-F test of ssp126-vs-ssp585 slope equality in CESM2.

    Returns an xr.Dataset with dims (season, lat, lon) and variables:
    wald_fstat, wald_pvalue, partial_r2_indep, delta_r2_total_indep,
    baseline_r2, beta_ssp126, beta_ssp585, max_pairwise_slope_diff,
    n_scenarios_present. Cached to
    data/scenario_independence_diagnostics_cesm_{var}.nc. Uses HAC
    covariance (maxlags=window-1). With G=2 scenarios the test reduces
    to a single interaction term (1-df F = t²) — power is lower than
    the MPI 3-scenario case.
    """
    SCENARIO_INDEPENDENCE_CACHE_VERSION_CESM = 1
    validate_choice('var', var, ALLOWED_VARS)
    cache = local_path + f'scenario_independence_diagnostics_cesm_{var}.nc'
    if not recompute:
        cached = check_diag_cache(cache, SCENARIO_INDEPENDENCE_CACHE_VERSION_CESM,
                                  'CESM2 scenario-independence')
        if cached is not None:
            return cached

    print('Calculating CESM2 scenario-independence diagnostics (per-pixel Wald-F, HAC SE)...')
    if multi_model_dict is None or masks is None:
        multi_model_dict, masks = get_full_multi_model_dict()
    if boot_data is None:
        _, _, boot_data, _ = get_other_studies_data(masks)

    fields = ['wald_fstat', 'wald_pvalue', 'partial_r2_indep',
              'delta_r2_total_indep', 'baseline_r2',
              'beta_ssp126', 'beta_ssp585',
              'max_pairwise_slope_diff', 'n_scenarios_present']
    out = cesm_diagnostics_scan(
        boot_data, multi_model_dict, masks, var=var, fields=fields,
        sentinel_key='wald_pvalue',
        compute_kw={'compute_scenario_independence': True},
    )
    out.attrs['version'] = SCENARIO_INDEPENDENCE_CACHE_VERSION_CESM
    out.to_netcdf(cache)
    print(f'Saved CESM2 scenario-independence diagnostics to {cache}.')
    return out


def get_linearity_diagnostics_giss(time_period=('2101', '2300'),
                                   amoc=None, tas=None,
                                   masks=None, var='tas', recompute=False):
    """Per-pixel RESET linearity test over GISS-E2-1-G, pooled across the
    seven non-composite ssp245 members within ``time_period``.

    GISS has G=1 scenario (ssp245), so only the linearity (RESET) test extends;
    scenario-independence is structurally undefined. The pooled fit uses
    member-minus-composite deviations with HAC SE (maxlags=window-1), matching
    the standing GISS convention (fit_member_minus_composite,
    giss_compute_gridded_regression). The cache filename embeds the time-period
    window because results depend on it.

    Returns an xr.Dataset with dims (season, lat, lon) and the same eight
    variables as the CESM linearity cache: partial_r2, delta_r2_total,
    gamma2, gamma3, beta2_sign, beta3_sign, reset_pvalue, baseline_r2.
    Cached to data/linearity_diagnostics_giss_tp-{start}-{end}_{var}.nc.

    Uses HAC covariance with maxlags=ROLLING_WINDOW-1. Loops over the
    canonical season list ['', 'djf', 'jja']; ``amoc`` is always annual
    and the per-season ``tas`` is loaded from
    ``load_giss_member_amoc_tas`` inside the loop.
    """
    LINEARITY_CACHE_VERSION_GISS = 1
    def _linearity_diagnostics_giss_cache_path(time_period, var='tas'):
        return (local_path + f'linearity_diagnostics_giss_'
                f'tp-{time_period[0]}-{time_period[1]}_{var}.nc')

    validate_choice('var', var, ALLOWED_VARS)
    cache = _linearity_diagnostics_giss_cache_path(time_period, var=var)
    if not recompute:
        cached = check_diag_cache(cache, LINEARITY_CACHE_VERSION_GISS,
                                  f'GISS linearity ({time_period[0]}-{time_period[1]})')
        if cached is not None:
            return cached

    print(f'Calculating GISS-E2-1-G linearity diagnostics (per-pixel RESET, HAC SE) for {time_period[0]}-{time_period[1]}...')
    if masks is None:
        _, masks = get_full_multi_model_dict()

    seasons = ['', 'djf', 'jja'] if var == 'tas' else ['']
    members = GISS_NON_COMPOSITE
    composite_members = GISS_COMPOSITE
    window = ROLLING_WINDOW
    hac_maxlags = max(window - 1, 1)
    fields = ['partial_r2', 'delta_r2_total', 'gamma2', 'gamma3',
              'beta2_sign', 'beta3_sign', 'reset_pvalue', 'baseline_r2']

    # Initialise output dataset using the native GISS grid (90x144) from
    # the first season's tas load.
    _amoc0, _tas0 = load_giss_member_amoc_tas(season=seasons[0])
    nlat = _tas0.sizes['lat']
    nlon = _tas0.sizes['lon']
    arr_shape = (len(seasons), nlat, nlon)
    data_vars = {name: (['season', 'lat', 'lon'], np.full(arr_shape, np.nan))
                 for name in fields}
    coords = {
        'season': seasons,
        'lat': _tas0.lat.values,
        'lon': _tas0.lon.values,
    }
    out = xr.Dataset(data_vars=data_vars, coords=coords)
    out.attrs['version'] = LINEARITY_CACHE_VERSION_GISS
    out.attrs['time_period'] = f'{time_period[0]}-{time_period[1]}'
    out.attrs['cov_type'] = f'HAC, maxlags={hac_maxlags}'
    out.attrs['members_pooled'] = ', '.join(members)
    out.attrs['composite_members'] = ', '.join(composite_members)

    for s_i, s in enumerate(seasons):
        print(f'  season={s!r}')
        if s == seasons[0]:
            amoc, tas = _amoc0, _tas0
        else:
            amoc, tas = load_giss_member_amoc_tas(season=s)

        # Pre-compute composite + deviations on the full grid once; index per pixel.
        amoc_comp = amoc.sel(realiz=composite_members).mean('realiz')
        tas_comp = tas.sel(realiz=composite_members).mean('realiz')
        amoc_dev = (amoc - amoc_comp).rolling(time=window, center=True).mean()
        tas_dev_grid = (tas - tas_comp).rolling(time=window, center=True).mean()

        # Slice once to the requested window; per-pixel work is just .isel.
        amoc_dev_p = amoc_dev.sel(realiz=members, time=slice(*time_period))
        tas_dev_p = tas_dev_grid.sel(realiz=members, time=slice(*time_period))

        # Coverage mask: pixel must have some finite tas across (member, time).
        tas_any = tas_dev_p.isel(time=0).isel(realiz=0)
        coverage = ~np.isnan(tas_any.values)

        i = 0
        for lat_i in range(nlat):
            for lon_i in range(nlon):
                if not coverage[lat_i, lon_i]:
                    i += 1
                    continue
                tas_dev_pixel = tas_dev_p.isel(lat=lat_i, lon=lon_i)
                try:
                    model, x, y, clust = fit_member_minus_composite(
                        amoc_dev_p, tas_dev_pixel, members=members,
                        time_slice=time_period, add_intercept=True,
                        window=window,
                    )
                except Exception:
                    i += 1
                    continue
                X = sm.add_constant(x)
                # Member id as a compact int array (cov_spec=HAC ignores
                # groups; this is only here for parity with the helper API).
                _u, _inv = np.unique(np.asarray(clust), return_inverse=True)
                diag = reset_diagnostics(
                    y, X, _inv.astype(int), model,
                    cov_spec=('HAC', {'maxlags': hac_maxlags}),
                )
                for name in fields:
                    val = diag.get(name, np.nan)
                    try:
                        out[name].values[s_i, lat_i, lon_i] = val
                    except (TypeError, ValueError):
                        pass
                i += 1
                if i % 500 == 0:
                    print(f'    {i}/{nlat*nlon} pixels done')

    out.to_netcdf(cache)
    print(f'Saved GISS linearity diagnostics to {cache}.')
    return out


# LOADING MULTI-MODEL DATA

def load_amoc_extent_cache(aggregation='model', future_window=None, unit='pct'):
    """Return ``{ssp: (max_weak, min_weak)}`` from the JSON sidecar.

    ``unit='pct'`` (default) reads the % envelope; ``unit='sv'`` reads the
    absolute-Sverdrup envelope (``bounds_sv`` / ``bounds_sv_model``). Old
    sidecars predate the Sv keys; when they are absent the Sv bounds are
    derived on the fly from the per-realisation / per-model ``*_Sv`` means
    already stored in the sidecar, so no NetCDF reopen / recompute is forced.

    ``aggregation`` selects which envelope to read:
    - ``'model'`` (default): min/max across per-model ensemble means.
      Each model contributes one number per SSP (the mean of weakening%
      across the reas with matched hist+scen). Treats every model equally
      regardless of ensemble size; this is what the module-level
      ``AMOC_extent`` exposes and what Fig 2 / FigGiss2 / FigS15 draw.
    - ``'realisation'``: min/max across all (model, rea) pairs. Wider
      envelope; reflects the full spread of individual ensemble members.

    Raises if the cache is missing. The expected workflow after staging
    new CMIP6 data is to call ``compute_amoc_extent(recompute=True)``,
    which writes both aggregation modes to the sidecar in a single pass.
    """
    if aggregation not in ('realisation', 'model'):
        raise ValueError(
            f"aggregation must be 'realisation' or 'model'; got {aggregation!r}")
    if unit not in ('pct', 'sv'):
        raise ValueError(f"unit must be 'pct' or 'sv'; got {unit!r}")
    # ``amoc_extent_cache_path`` is defined later in the module (it needs
    # FUTURE_WINDOW); the canonical (None) branch therefore reads the constant
    # directly so the import-time ``AMOC_extent`` load does not forward-ref it.
    cache_path = AMOC_EXTENT_CACHE if future_window is None else amoc_extent_cache_path(future_window)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"AMOC_extent cache missing at {cache_path}. "
            "Run `functions.compute_amoc_extent(recompute=True)` from "
            "scripts/ to populate it from the current on-disk inventory."
        )
    with open(cache_path, 'r') as _f:
        cache = _json.load(_f)
    key = AMOC_EXTENT_KEYS[(aggregation, unit)]
    out = {}
    for ssp, entry in cache['ssps'].items():
        if key in entry:
            out[ssp] = tuple(entry[key])
        elif unit == 'sv':
            # Sidecar predates the bounds_sv* keys: derive from the stored Sv means.
            if aggregation == 'model' and entry.get('per_model'):
                sv = [m['hist_ensmean_Sv'] - m['scen_ensmean_Sv'] for m in entry['per_model']]
                out[ssp] = (max(sv), min(sv))
            elif aggregation == 'realisation' and entry.get('per_realisation'):
                sv = [r['hist_mean_Sv'] - r['scen_mean_Sv'] for r in entry['per_realisation']]
                out[ssp] = (max(sv), min(sv))
    return out


def compute_amoc_extent(recompute=False, scenarios=None, var='amoc26',
                        models=None, snapshot=None,
                        aggregation='model', future_window=None, unit='pct'):
    """Recompute the CMIP6 (max_weak%, min_weak%) envelope per SSP.

    Iterates the (model, variant_id) pairs returned by
    ``cmip6_inventory.available_pairs(scenario, var)`` — i.e. realisations
    present in both ``historical`` and the SSP under the same model — and
    computes weakening% from the AMOC@26.5N yearly caches:

        weakening% = 100 * (mean(hist, PI_WINDOW) - mean(scen, FUTURE_WINDOW)) / mean(hist, PI_WINDOW)

    Two aggregation modes (both written to the sidecar in one pass; no
    extra file opens):

    - ``'model'`` (default): bounds = min/max across per-model means.
      For each model, the ensemble means of hist and scen are computed
      over the **same** rea set (the hist ∩ scen intersection that
      ``available_pairs`` returns), then weakening% comes from those
      means. Each model contributes one number; treats every model
      equally regardless of ensemble size.
    - ``'realisation'``: bounds = min/max across all (model, rea)
      weakenings. Wider envelope; reflects per-member spread.

    Writes a JSON sidecar with per-realisation provenance, per-model
    means, and both bounds. Returns the bounds dict in the format
    selected by ``aggregation``. Does nothing if ``recompute=False`` and
    the cache exists. Pass ``snapshot=path`` to lock the realisation set
    to a saved inventory snapshot for reproducibility.

    MPI-ESM1-2-LR is excluded from the envelope by convention (no files
    staged under CMIP6/<ssp>/mpi-esm1-2-lr/; the legacy analysis_paper.ipynb
    excluded it too). Rerun ``compute_amoc_extent(recompute=True)`` after
    staging new CMIP6 realisations to refresh the sidecar.
    """
    fw = resolve_future_window(future_window)
    cache_path = AMOC_EXTENT_CACHE if future_window is None else amoc_extent_cache_path(future_window)
    if not recompute and os.path.exists(cache_path):
        return load_amoc_extent_cache(aggregation=aggregation, future_window=future_window)

    if scenarios is None:
        scenarios = ['ssp126', 'ssp245', 'ssp370']

    pi_start, pi_end = PI_WINDOW
    fu_start, fu_end = fw

    out = {'computed': _dt.datetime.now().isoformat(timespec='seconds'),
           'pi_window': list(PI_WINDOW),
           'future_window': list(fw),
           'search_roots': list(cmip6_inventory.SEARCH_ROOTS),
           'snapshot': snapshot,
           'ssps': {}}

    for ssp in scenarios:
        pairs = cmip6_inventory.available_pairs(
            ssp, var, models=models, esgf_names=True, snapshot=snapshot)
        per_rea = []
        for model, rea in pairs:
            try:
                hist_path = cmip6_inventory.resolve_path(
                    model, 'historical', rea, var)
                scen_path = cmip6_inventory.resolve_path(
                    model, ssp, rea, var)
            except FileNotFoundError as err:
                print(f"  skip {model}/{ssp}/{rea}: {err}")
                continue
            try:
                hist_da = xr.open_dataarray(hist_path, use_cftime=True)
                scen_da = xr.open_dataarray(scen_path, use_cftime=True)
                hist_mean = float(
                    hist_da.sel(time=slice(pi_start, pi_end)).mean('time').values)
                scen_mean = float(
                    scen_da.sel(time=slice(fu_start, fu_end)).mean('time').values)
            except Exception as err:
                print(f"  skip {model}/{ssp}/{rea}: open/slice failed: {err}")
                continue
            if not (np.isfinite(hist_mean) and np.isfinite(scen_mean)
                    and hist_mean != 0):
                print(f"  skip {model}/{ssp}/{rea}: non-finite or zero baseline "
                      f"(hist={hist_mean}, scen={scen_mean})")
                continue
            weakening = 100.0 * (hist_mean - scen_mean) / hist_mean
            per_rea.append({
                'model': model,
                'variant_id': rea,
                'hist_path': hist_path,
                'scen_path': scen_path,
                'hist_mean_Sv': hist_mean,
                'scen_mean_Sv': scen_mean,
                'weakening_pct': weakening,
            })
        if not per_rea:
            print(f"compute_amoc_extent: no pairs found for {ssp}; skipping")
            continue

        # Per-realisation aggregation.
        weakenings = [r['weakening_pct'] for r in per_rea]

        # Per-model aggregation. For each model: ensemble-mean hist and
        # scen across the matched-rea set, then compute weakening% from
        # those means. Same rea set on both legs by construction
        # (available_pairs intersects hist ∩ scen per model).
        per_model = {}
        for r in per_rea:
            per_model.setdefault(r['model'], []).append(r)
        per_model_summary = []
        for model, rs in per_model.items():
            hist_ens = sum(r['hist_mean_Sv'] for r in rs) / len(rs)
            scen_ens = sum(r['scen_mean_Sv'] for r in rs) / len(rs)
            if hist_ens == 0 or not np.isfinite(hist_ens):
                continue
            weak_ens = 100.0 * (hist_ens - scen_ens) / hist_ens
            per_model_summary.append({
                'model': model,
                'n_realisations': len(rs),
                'variant_ids': sorted(r['variant_id'] for r in rs),
                'hist_ensmean_Sv': hist_ens,
                'scen_ensmean_Sv': scen_ens,
                'weakening_pct': weak_ens,
            })
        model_weakenings = [m['weakening_pct'] for m in per_model_summary]

        # Absolute-Sv weakening per realisation / model (hist - scen, both Sv).
        sv_weakenings = [r['hist_mean_Sv'] - r['scen_mean_Sv'] for r in per_rea]
        sv_model_weakenings = [m['hist_ensmean_Sv'] - m['scen_ensmean_Sv']
                               for m in per_model_summary]
        entry = {
            'n_realisations': len(per_rea),
            'n_models': len(per_model_summary),
            'per_realisation': per_rea,
            'per_model': per_model_summary,
            'bounds_pct': [max(weakenings), min(weakenings)],
            'bounds_sv': [max(sv_weakenings), min(sv_weakenings)],
        }
        if model_weakenings:
            entry['bounds_pct_model'] = [max(model_weakenings),
                                         min(model_weakenings)]
        if sv_model_weakenings:
            entry['bounds_sv_model'] = [max(sv_model_weakenings),
                                        min(sv_model_weakenings)]
        out['ssps'][ssp] = entry

    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    with open(cache_path, 'w') as _f:
        _json.dump(out, _f, indent=2, sort_keys=True)

    key = AMOC_EXTENT_KEYS[(aggregation, unit)]
    return {ssp: tuple(entry[key])
            for ssp, entry in out['ssps'].items()
            if key in entry}


try:
    AMOC_extent = load_amoc_extent_cache()
except FileNotFoundError as _err:
    _warnings.warn(
        f"{_err} AMOC_extent will be unavailable until then.",
        stacklevel=2,
    )
    AMOC_extent = None


def get_amoc_extent(future_window=None, aggregation='model', recompute=False, unit='pct'):
    """Window-keyed CMIP6 weakening envelope ``{ssp: (max_weak, min_weak)}``.

    The canonical window returns the module-global ``AMOC_extent`` loaded at
    import (or recomputes it if missing / ``recompute``); any other window
    loads — or computes and caches — the window-suffixed sidecar. This is the
    accessor the figures use so a sensitivity run never touches the canonical
    sidecar. ``unit='sv'`` returns the absolute-Sverdrup envelope instead of %;
    the module-global fast path is %-only, so Sv reads go through the cache."""
    if fw_suffix(future_window) == '':
        if recompute:
            return compute_amoc_extent(recompute=recompute, aggregation=aggregation, unit=unit)
        if unit == 'pct' and AMOC_extent is not None:
            return AMOC_extent
        return load_amoc_extent_cache(aggregation=aggregation, unit=unit)
    return compute_amoc_extent(recompute=recompute, aggregation=aggregation,
                               future_window=future_window, unit=unit)


def load_hosmip_data(apply_ocean_mask=False):
    """
    apply_ocean_mask only changes things for preindustrial mean tas calculation.
    """

    data_path = '/work/uo1075/m300817/hosing/nahosmip/post/'
    pi_data_path = '/work/uo1075/m300817/teu_amoc/data/CMIP6/picontrol/'
    local_path = '../data/'
    use_local = False

    def _open_canonical(path, model, as_dataset=False):
        """Open ``path`` and snap lat/lon to ``model``'s canonical
        CMIP6-archive grid via ``cmip6_inventory.open_canonical``. Used
        across this loader for hosing tas/amoc and piControl tas/amoc
        files alike — the same snap eliminates the HosMIP-pipeline vs
        CMIP6-archive coord drift (EC-Earth3 lat-coord float-precision
        drift).

        ``as_dataset=True`` returns an ``xr.Dataset`` (for files with aux
        vars like ``bnds`` / ``height`` that need dropping); default
        returns a DataArray. AMOC@26.5N files (1D, no lat/lon) pass
        through unchanged inside ``open_canonical``.

        MPI-ESM1-2-LR has no CMIP6-archive entry in our pipeline
        (it's loaded via MPI-GE with a different path layout) and so
        no canonical-grid registry entry. Its HosMIP/piControl grids
        are themselves canonical for MPI-LR — pass through unchanged.
        """
        if model == 'MPI-ESM1-2-LR':
            return (xr.open_dataset(path, use_cftime=True) if as_dataset
                    else xr.open_dataarray(path, use_cftime=True))
        return cmip6_inventory.open_canonical(
            path, model, as_dataset=as_dataset)

    def _open_hos(filename, model, as_dataset=False):
        """Convenience for hosing files under ``data_path`` /
        ``local_path``. See :func:`_open_canonical` for the snap."""
        path = (local_path if use_local else data_path) + filename
        return _open_canonical(path, model, as_dataset=as_dataset)

    print('Loading HOSMIP data...')

    print('Loading CanESM5 data...')
    # CanESM5
    canesm_03_amoc = _open_hos("CanESM5_u03-hos_amoc26_yr.nc", 'CanESM5').drop_vars('lev').to_dataset(name='amoc')

    canesm_03_tas = _open_hos("CanESM5_u03-hos_tas_yr.nc", 'CanESM5').to_dataset(name='tas') - 273.15
    canesm_03_tas_djf = _open_hos("CanESM5_u03-hos_tas_djf.nc", 'CanESM5', as_dataset=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15
    canesm_03_tas_jja = _open_hos("CanESM5_u03-hos_tas_jja.nc", 'CanESM5', as_dataset=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15

    canesm_03_tas.coords['lon'] = (canesm_03_tas.coords['lon'] + 180) % 360 - 180
    canesm_03_tas = canesm_03_tas.sortby(canesm_03_tas.lon)

    canesm_03_tas_djf.coords['lon'] = (canesm_03_tas_djf.coords['lon'] + 180) % 360 - 180
    canesm_03_tas_djf = canesm_03_tas_djf.sortby(canesm_03_tas_djf.lon)

    canesm_03_tas_jja.coords['lon'] = (canesm_03_tas_jja.coords['lon'] + 180) % 360 - 180
    canesm_03_tas_jja = canesm_03_tas_jja.sortby(canesm_03_tas_jja.lon)

    canesm_masks = make_country_masks_land_aware(canesm_03_tas)

    canesm_03_tas = canesm_03_tas.where(canesm_masks['EU_buffer'], drop=True)
    canesm_03_tas_djf = canesm_03_tas_djf.where(canesm_masks['EU_buffer'], drop=True)
    canesm_03_tas_jja = canesm_03_tas_jja.where(canesm_masks['EU_buffer'], drop=True)

    print('Loading CESM2 data...')
    # CESM2
    cesm_03_amoc = _open_hos("CESM2_u03-hos_amoc26_yr.nc", 'CESM2').drop_vars(['moc_z', 'moc_components']).to_dataset(name='amoc')

    cesm_03_tas = _open_hos("CESM2_u03-hos_tas_yr.nc", 'CESM2').to_dataset(name='tas') - 273.15
    cesm_03_tas_djf = _open_hos("CESM2_u03-hos_tas_djf.nc", 'CESM2', as_dataset=True).tas.to_dataset(name='tas') - 273.15
    cesm_03_tas_jja = _open_hos("CESM2_u03-hos_tas_jja.nc", 'CESM2', as_dataset=True).tas.to_dataset(name='tas') - 273.15

    cesm_03_tas.coords['lon'] = (cesm_03_tas.coords['lon'] + 180) % 360 - 180
    cesm_03_tas = cesm_03_tas.sortby(cesm_03_tas.lon)

    cesm_03_tas_djf.coords['lon'] = (cesm_03_tas_djf.coords['lon'] + 180) % 360 - 180
    cesm_03_tas_djf = cesm_03_tas_djf.sortby(cesm_03_tas_djf.lon)

    cesm_03_tas_jja.coords['lon'] = (cesm_03_tas_jja.coords['lon'] + 180) % 360 - 180
    cesm_03_tas_jja = cesm_03_tas_jja.sortby(cesm_03_tas_jja.lon)

    cesm_masks = make_country_masks_land_aware(cesm_03_tas)

    cesm_03_tas = cesm_03_tas.where(cesm_masks['EU_buffer'], drop=True)
    cesm_03_tas_djf = cesm_03_tas_djf.where(cesm_masks['EU_buffer'], drop=True)
    cesm_03_tas_jja = cesm_03_tas_jja.where(cesm_masks['EU_buffer'], drop=True)

    print('Loading EC-Earth3 data...')
    # EC-Earth3
    ecearth_03_amoc = _open_hos("EC-Earth3_u03-hos_amoc26_yr.nc", 'EC-Earth3').drop_vars(['lev', 'sector']).to_dataset(name='amoc')

    ecearth_03_tas = _open_hos("EC-Earth3_u03-hos_tas_yr.nc", 'EC-Earth3').to_dataset(name='tas') - 273.15
    ecearth_03_tas_djf = _open_hos("EC-Earth3_u03-hos_tas_djf.nc", 'EC-Earth3', as_dataset=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15
    ecearth_03_tas_jja = _open_hos("EC-Earth3_u03-hos_tas_jja.nc", 'EC-Earth3', as_dataset=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15

    ecearth_03_tas.coords['lon'] = (ecearth_03_tas.coords['lon'] + 180) % 360 - 180
    ecearth_03_tas = ecearth_03_tas.sortby(ecearth_03_tas.lon)

    ecearth_03_tas_djf.coords['lon'] = (ecearth_03_tas_djf.coords['lon'] + 180) % 360 - 180
    ecearth_03_tas_djf = ecearth_03_tas_djf.sortby(ecearth_03_tas_djf.lon)

    ecearth_03_tas_jja.coords['lon'] = (ecearth_03_tas_jja.coords['lon'] + 180) % 360 - 180
    ecearth_03_tas_jja = ecearth_03_tas_jja.sortby(ecearth_03_tas_jja.lon)

    ecearth_masks = make_country_masks_land_aware(ecearth_03_tas)

    ecearth_03_tas = ecearth_03_tas.where(ecearth_masks['EU_buffer'], drop=True)
    ecearth_03_tas_djf = ecearth_03_tas_djf.where(ecearth_masks['EU_buffer'], drop=True)
    ecearth_03_tas_jja = ecearth_03_tas_jja.where(ecearth_masks['EU_buffer'], drop=True)

    print('Loading HadGEM3-GC3-1LL data...')
    # HadGEM3-GC3-1LL
    hadgem1ll_03_amoc = _open_hos("HadGEM3-GC3-1LL_u03-hos_amoc26_yr.nc", 'HadGEM3-GC3-1LL').drop_vars(['nav_lat', 'nav_lon', 'depthw', 'time_centered']).to_dataset(name='amoc')

    hadgem1ll_03_tas = _open_hos("HadGEM3-GC3-1LL_u03-hos_tas_yr.nc", 'HadGEM3-GC3-1LL').to_dataset(name='tas') - 273.15
    hadgem1ll_03_tas_djf = _open_hos("HadGEM3-GC3-1LL_u03-hos_tas_djf.nc", 'HadGEM3-GC3-1LL', as_dataset=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15
    hadgem1ll_03_tas_jja = _open_hos("HadGEM3-GC3-1LL_u03-hos_tas_jja.nc", 'HadGEM3-GC3-1LL', as_dataset=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15

    hadgem1ll_03_tas.coords['lon'] = (hadgem1ll_03_tas.coords['lon'] + 180) % 360 - 180
    hadgem1ll_03_tas = hadgem1ll_03_tas.sortby(hadgem1ll_03_tas.lon)

    hadgem1ll_03_tas_djf.coords['lon'] = (hadgem1ll_03_tas_djf.coords['lon'] + 180) % 360 - 180
    hadgem1ll_03_tas_djf = hadgem1ll_03_tas_djf.sortby(hadgem1ll_03_tas_djf.lon)

    hadgem1ll_03_tas_jja.coords['lon'] = (hadgem1ll_03_tas_jja.coords['lon'] + 180) % 360 - 180
    hadgem1ll_03_tas_jja = hadgem1ll_03_tas_jja.sortby(hadgem1ll_03_tas_jja.lon)

    hadgem1ll_masks = make_country_masks_land_aware(hadgem1ll_03_tas)

    hadgem1ll_03_tas = hadgem1ll_03_tas.where(hadgem1ll_masks['EU_buffer'], drop=True)
    hadgem1ll_03_tas_djf = hadgem1ll_03_tas_djf.where(hadgem1ll_masks['EU_buffer'], drop=True)
    hadgem1ll_03_tas_jja = hadgem1ll_03_tas_jja.where(hadgem1ll_masks['EU_buffer'], drop=True)

    print('Loading HadGEM3-GC3-1MM data...')
    # HadGEM3-GC3-1MM
    hadgem1mm_03_amoc = _open_hos("HadGEM3-GC3-1MM_u03-hos_amoc26_yr.nc", 'HadGEM3-GC3-1MM').drop_vars(['nav_lat', 'nav_lon', 'depthw', 'time_centered']).to_dataset(name='amoc')

    hadgem1mm_03_tas = _open_hos("HadGEM3-GC3-1MM_u03-hos_tas_yr.nc", 'HadGEM3-GC3-1MM').to_dataset(name='tas') - 273.15
    hadgem1mm_03_tas_djf = _open_hos("HadGEM3-GC3-1MM_u03-hos_tas_djf.nc", 'HadGEM3-GC3-1MM', as_dataset=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15
    hadgem1mm_03_tas_jja = _open_hos("HadGEM3-GC3-1MM_u03-hos_tas_jja.nc", 'HadGEM3-GC3-1MM', as_dataset=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15

    hadgem1mm_03_tas.coords['lon'] = (hadgem1mm_03_tas.coords['lon'] + 180) % 360 - 180
    hadgem1mm_03_tas = hadgem1mm_03_tas.sortby(hadgem1mm_03_tas.lon)

    hadgem1mm_03_tas_djf.coords['lon'] = (hadgem1mm_03_tas_djf.coords['lon'] + 180) % 360 - 180
    hadgem1mm_03_tas_djf = hadgem1mm_03_tas_djf.sortby(hadgem1mm_03_tas_djf.lon)

    hadgem1mm_03_tas_jja.coords['lon'] = (hadgem1mm_03_tas_jja.coords['lon'] + 180) % 360 - 180
    hadgem1mm_03_tas_jja = hadgem1mm_03_tas_jja.sortby(hadgem1mm_03_tas_jja.lon)

    hadgem1mm_masks = make_country_masks_land_aware(hadgem1mm_03_tas)

    hadgem1mm_03_tas = hadgem1mm_03_tas.where(hadgem1mm_masks['EU_buffer'], drop=True)
    hadgem1mm_03_tas_djf = hadgem1mm_03_tas_djf.where(hadgem1mm_masks['EU_buffer'], drop=True)
    hadgem1mm_03_tas_jja = hadgem1mm_03_tas_jja.where(hadgem1mm_masks['EU_buffer'], drop=True)

    print('Loading IPSL-CM6A-LR data...')
    # IPSL-CM6A-LR
    ipsl_03_amoc = _open_hos("IPSL-CM6A-LR_u03-hos_amoc26_yr.nc", 'IPSL-CM6A-LR').drop_vars(['nav_lat', 'nav_lon', 'olevel']).to_dataset(name='amoc')

    ipsl_03_tas = _open_hos("IPSL-CM6A-LR_u03-hos_tas_yr.nc", 'IPSL-CM6A-LR').to_dataset(name='tas') - 273.15
    ipsl_03_tas_djf = _open_hos("IPSL-CM6A-LR_u03-hos_tas_djf.nc", 'IPSL-CM6A-LR', as_dataset=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15
    ipsl_03_tas_jja = _open_hos("IPSL-CM6A-LR_u03-hos_tas_jja.nc", 'IPSL-CM6A-LR', as_dataset=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15

    ipsl_03_tas.coords['lon'] = (ipsl_03_tas.coords['lon'] + 180) % 360 - 180
    ipsl_03_tas = ipsl_03_tas.sortby(ipsl_03_tas.lon)

    ipsl_03_tas_djf.coords['lon'] = (ipsl_03_tas_djf.coords['lon'] + 180) % 360 - 180
    ipsl_03_tas_djf = ipsl_03_tas_djf.sortby(ipsl_03_tas_djf.lon)

    ipsl_03_tas_jja.coords['lon'] = (ipsl_03_tas_jja.coords['lon'] + 180) % 360 - 180
    ipsl_03_tas_jja = ipsl_03_tas_jja.sortby(ipsl_03_tas_jja.lon)

    ipsl_masks = make_country_masks_land_aware(ipsl_03_tas)

    ipsl_03_tas = ipsl_03_tas.where(ipsl_masks['EU_buffer'], drop=True)
    ipsl_03_tas_djf = ipsl_03_tas_djf.where(ipsl_masks['EU_buffer'], drop=True)
    ipsl_03_tas_jja = ipsl_03_tas_jja.where(ipsl_masks['EU_buffer'], drop=True)

    print('Loading MPI-ESM1-2-HR data...')
    # MPI-ESM1-2-HR
    mpiesmhr_03_amoc = _open_hos("MPI-ESM1-2-HR_u03-hos_amoc26_yr.nc", 'MPI-ESM1-2-HR').drop_vars(['depth_2']).to_dataset(name='amoc')

    mpiesmhr_03_tas = _open_hos("MPI-ESM1-2-HR_u03-hos_tas_yr.nc", 'MPI-ESM1-2-HR').to_dataset(name='tas') - 273.15
    mpiesmhr_03_tas_djf = _open_hos("MPI-ESM1-2-HR_u03-hos_tas_djf.nc", 'MPI-ESM1-2-HR', as_dataset=True).tas.to_dataset(name='tas') - 273.15
    mpiesmhr_03_tas_jja = _open_hos("MPI-ESM1-2-HR_u03-hos_tas_jja.nc", 'MPI-ESM1-2-HR', as_dataset=True).tas.to_dataset(name='tas') - 273.15

    mpiesmhr_03_tas.coords['lon'] = (mpiesmhr_03_tas.coords['lon'] + 180) % 360 - 180
    mpiesmhr_03_tas = mpiesmhr_03_tas.sortby(mpiesmhr_03_tas.lon)

    mpiesmhr_03_tas_djf.coords['lon'] = (mpiesmhr_03_tas_djf.coords['lon'] + 180) % 360 - 180
    mpiesmhr_03_tas_djf = mpiesmhr_03_tas_djf.sortby(mpiesmhr_03_tas_djf.lon)

    mpiesmhr_03_tas_jja.coords['lon'] = (mpiesmhr_03_tas_jja.coords['lon'] + 180) % 360 - 180
    mpiesmhr_03_tas_jja = mpiesmhr_03_tas_jja.sortby(mpiesmhr_03_tas_jja.lon)

    mpiesmhr_masks = make_country_masks_land_aware(mpiesmhr_03_tas)

    mpiesmhr_03_tas = mpiesmhr_03_tas.where(mpiesmhr_masks['EU_buffer'], drop=True)
    mpiesmhr_03_tas_djf = mpiesmhr_03_tas_djf.where(mpiesmhr_masks['EU_buffer'], drop=True)
    mpiesmhr_03_tas_jja = mpiesmhr_03_tas_jja.where(mpiesmhr_masks['EU_buffer'], drop=True)

    print('Loading MPI-ESM1-2-LR data...')
    # MPI-ESM1-2-LR
    mpiesmlr_03_amoc = _open_hos("MPI-ESM1-2-LR_u03-hos_amoc26_yr.nc", 'MPI-ESM1-2-LR').drop_vars(['depth_2']).to_dataset(name='amoc')

    mpiesmlr_03_tas = _open_hos("MPI-ESM1-2-LR_u03-hos_tas_yr.nc", 'MPI-ESM1-2-LR').to_dataset(name='tas') - 273.15
    mpiesmlr_03_tas_djf = _open_hos("MPI-ESM1-2-LR_u03-hos_tas_djf.nc", 'MPI-ESM1-2-LR', as_dataset=True).tas.to_dataset(name='tas') - 273.15
    mpiesmlr_03_tas_jja = _open_hos("MPI-ESM1-2-LR_u03-hos_tas_jja.nc", 'MPI-ESM1-2-LR', as_dataset=True).tas.to_dataset(name='tas') - 273.15

    mpiesmlr_03_tas.coords['lon'] = (mpiesmlr_03_tas.coords['lon'] + 180) % 360 - 180
    mpiesmlr_03_tas = mpiesmlr_03_tas.sortby(mpiesmlr_03_tas.lon)

    mpiesmlr_03_tas_djf.coords['lon'] = (mpiesmlr_03_tas_djf.coords['lon'] + 180) % 360 - 180
    mpiesmlr_03_tas_djf = mpiesmlr_03_tas_djf.sortby(mpiesmlr_03_tas_djf.lon)

    mpiesmlr_03_tas_jja.coords['lon'] = (mpiesmlr_03_tas_jja.coords['lon'] + 180) % 360 - 180
    mpiesmlr_03_tas_jja = mpiesmlr_03_tas_jja.sortby(mpiesmlr_03_tas_jja.lon)

    mpiesmlr_masks = make_country_masks_land_aware(mpiesmlr_03_tas)

    mpiesmlr_03_tas = mpiesmlr_03_tas.where(mpiesmlr_masks['EU_buffer'], drop=True)
    mpiesmlr_03_tas_djf = mpiesmlr_03_tas_djf.where(mpiesmlr_masks['EU_buffer'], drop=True)
    mpiesmlr_03_tas_jja = mpiesmlr_03_tas_jja.where(mpiesmlr_masks['EU_buffer'], drop=True)

    amoc_data= {
        'CanESM5': canesm_03_amoc,
        'EC-Earth3': ecearth_03_amoc,
        'CESM2': cesm_03_amoc,
        'MPI-ESM1-2-LR': mpiesmlr_03_amoc,
        'HadGEM3-GC3-1MM': hadgem1mm_03_amoc,
        'HadGEM3-GC3-1LL': hadgem1ll_03_amoc,
        'MPI-ESM1-2-HR': mpiesmhr_03_amoc,
        'IPSL-CM6A-LR': ipsl_03_amoc
    }

    tas_data = {}
    tas_data[''] = {
        'CanESM5': canesm_03_tas,
        'EC-Earth3': ecearth_03_tas,
        'CESM2': cesm_03_tas,
        'MPI-ESM1-2-LR': mpiesmlr_03_tas,
        'HadGEM3-GC3-1MM': hadgem1mm_03_tas,
        'HadGEM3-GC3-1LL': hadgem1ll_03_tas,
        'MPI-ESM1-2-HR': mpiesmhr_03_tas,
        'IPSL-CM6A-LR': ipsl_03_tas
    }
    tas_data['djf'] = {
        'CanESM5': canesm_03_tas_djf,
        'EC-Earth3': ecearth_03_tas_djf,
        'CESM2': cesm_03_tas_djf,
        'MPI-ESM1-2-LR': mpiesmlr_03_tas_djf,
        'HadGEM3-GC3-1MM': hadgem1mm_03_tas_djf,
        'HadGEM3-GC3-1LL': hadgem1ll_03_tas_djf,
        'MPI-ESM1-2-HR': mpiesmhr_03_tas_djf,
        'IPSL-CM6A-LR': ipsl_03_tas_djf
    }
    tas_data['jja'] = {
        'CanESM5': canesm_03_tas_jja,
        'EC-Earth3': ecearth_03_tas_jja,
        'CESM2': cesm_03_tas_jja,
        'MPI-ESM1-2-LR': mpiesmlr_03_tas_jja,
        'HadGEM3-GC3-1MM': hadgem1mm_03_tas_jja,
        'HadGEM3-GC3-1LL': hadgem1ll_03_tas_jja,
        'MPI-ESM1-2-HR': mpiesmhr_03_tas_jja,
        'IPSL-CM6A-LR': ipsl_03_tas_jja
    }

    masks = {
        'CanESM5': canesm_masks,
        'EC-Earth3': ecearth_masks,
        'CESM2': cesm_masks,
        'MPI-ESM1-2-LR': mpiesmlr_masks,
        'HadGEM3-GC3-1MM': hadgem1mm_masks,
        'HadGEM3-GC3-1LL': hadgem1ll_masks,
        'MPI-ESM1-2-HR': mpiesmhr_masks,
        'IPSL-CM6A-LR': ipsl_masks
    }

    data_cutoffs = {}
    data_cutoffs[''] = {
        'CanESM5': [0, 0],
        'EC-Earth3': [0, 0],
        'CESM2': [0, 1],
        'MPI-ESM1-2-LR': [0, 0],
        'HadGEM3-GC3-1MM': [0, 1],
        'HadGEM3-GC3-1LL': [45, 0],
        'MPI-ESM1-2-HR': [0, 0],
        'IPSL-CM6A-LR': [0, 50]
    }
    data_cutoffs['jja'] = {
        'CanESM5': [0, 0],
        'EC-Earth3': [0, 0],
        'CESM2': [0, 28],
        'MPI-ESM1-2-LR': [0, 0],
        'HadGEM3-GC3-1MM': [0, 1],
        'HadGEM3-GC3-1LL': [45, 0],
        'MPI-ESM1-2-HR': [0, 0],
        'IPSL-CM6A-LR': [0, 50]
    }
    data_cutoffs['djf'] = {
        'CanESM5': [0, 0],
        'EC-Earth3': [0, 0],
        'CESM2': [0, 28],
        'MPI-ESM1-2-LR': [0, 0],
        'HadGEM3-GC3-1MM': [0, 1],
        'HadGEM3-GC3-1LL': [45, 0],
        'MPI-ESM1-2-HR': [0, 0],
        'IPSL-CM6A-LR': [0, 50]
    }

    pi_data_path = '/work/uo1075/m300817/teu_amoc/data/CMIP6/picontrol/'
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

    hosmip_realiz_dict = {
        'MPI-ESM1-2-HR': 'r1i1p1f1', 
        'MPI-ESM1-2-LR': 'r1i1p1f1',
        'CanESM5': 'r1i1p1f1',
        'CESM2': 'r1i1p1f1',
        'HadGEM3-GC31-LL': 'r1i1p1f1',
        'HadGEM3-GC31-MM': 'r1i1p1f1',
        'IPSL-CM6A-LR': 'r1i2p1f1',
        'EC-Earth3': 'r1i1p1f1'
        }

    hosmip_amoc_pi_data = {}
    hosmip_tas_pi_data = {}
    for model in hosmip_realiz_dict.keys():
        print('Loading preindustrial data from '+model)
        
        hosmip_amoc_pi_data[model] = _open_canonical(
            pi_data_path+f'{model.lower()}/{model.lower()}_{hosmip_realiz_dict[model]}_amoc26_yr.nc',
            model).drop_vars('lev' if model != 'IPSL-CM6A-LR' else 'olevel').to_dataset(name='amoc')

        for season in ['', 'djf', 'jja']:
            if season == '':
                raw_tas_data = _open_canonical(
                    pi_data_path+f'{model.lower()}/{model.lower()}_{hosmip_realiz_dict[model]}_tas_yr.nc',
                    model).to_dataset(name='tas') - 273.15

                if model in ['CESM2', 'CanESM5', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM']:
                    new_time = xr.cftime_range(start=str(raw_tas_data.time.values[0]), freq='YS', periods=raw_tas_data.sizes['time'], calendar='proleptic_gregorian')
                    raw_tas_data = raw_tas_data.assign_coords(time=new_time)

                raw_tas_data.coords['lon'] = (raw_tas_data.coords['lon'] + 180) % 360 - 180
                raw_tas_data = raw_tas_data.sortby(raw_tas_data.lon)

                raw_tas_data.coords['season'] = season
                raw_tas_data = raw_tas_data.expand_dims('season', axis=-1)

                all_season_tas_data = raw_tas_data
                
                mask_land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(raw_tas_data)
            else:
                raw_tas_data = _open_canonical(
                    pi_data_path+f'{model.lower()}/tas_{season}/{model.lower()}_{hosmip_realiz_dict[model]}_tas_{season}.nc',
                    model, as_dataset=True) - 273.15

                model_key = 'HadGEM3-GC3-1LL' if model == 'HadGEM3-GC31-LL' else ('HadGEM3-GC3-1MM' if model == 'HadGEM3-GC31-MM' else model)
                if 'dim' in drop_stuff[model_key].keys():
                    raw_tas_data = raw_tas_data.drop_dims(drop_stuff[model_key]['dim'])
                if 'var' in drop_stuff[model_key].keys():
                    raw_tas_data = raw_tas_data.drop_vars(drop_stuff[model_key]['var'])

                raw_tas_data.coords['lon'] = (raw_tas_data.coords['lon'] + 180) % 360 - 180
                raw_tas_data = raw_tas_data.sortby(raw_tas_data.lon)

                raw_tas_data.coords['season'] = season
                raw_tas_data = raw_tas_data.expand_dims('season', axis=-1)

                all_season_tas_data = xr.concat([all_season_tas_data, raw_tas_data], dim='season')

            hosmip_tas_pi_data[model] = all_season_tas_data.where(mask_land==0, drop=True) if apply_ocean_mask else all_season_tas_data

    # rename HadGEM model keys
    hosmip_amoc_pi_data['HadGEM3-GC3-1LL'] = hosmip_amoc_pi_data.pop('HadGEM3-GC31-LL')
    hosmip_amoc_pi_data['HadGEM3-GC3-1MM'] = hosmip_amoc_pi_data.pop('HadGEM3-GC31-MM')
    hosmip_tas_pi_data['HadGEM3-GC3-1LL'] = hosmip_tas_pi_data.pop('HadGEM3-GC31-LL')
    hosmip_tas_pi_data['HadGEM3-GC3-1MM'] = hosmip_tas_pi_data.pop('HadGEM3-GC31-MM')

    return amoc_data, hosmip_amoc_pi_data, tas_data, hosmip_tas_pi_data, masks, data_cutoffs


def build_hosmip_dataset(amoc_data, hosmip_amoc_pi_data, tas_data, hosmip_tas_pi_data, masks, data_cutoffs):

    multi_model_dict = {}

    print('Building combined HosMIP dataset...')

    for model in hosmip_labels:
        print('Processing model: ', model)

        for season in ['', 'djf', 'jja']:

            amoc = amoc_data[model] if data_cutoffs[season][model][0] == 0 else amoc_data[model].isel(time=slice(0, -data_cutoffs[season][model][0]))
            tas = tas_data[season][model] if data_cutoffs[season][model][1] == 0 else tas_data[season][model].isel(time=slice(0, -data_cutoffs[season][model][1]))

            masks_ds = xr.concat([masks[model][k] for k in masks[model].keys()], 
                                dim=pd.Index(list(masks[model].keys()), name='region'))
            masks_ds = masks_ds.where(masks[model]['EU_buffer'], drop=True)

            amoc_tas = xr.merge([amoc, tas])
            amoc_tas.coords['season'] = season
            amoc_tas.coords['scenar'] = 'pi'
            amoc_tas.coords['type'] = 'hosing'

            amoc_tas = amoc_tas.expand_dims(['scenar', 'season', 'type'], axis=[-1, -2, -3])

            if season == '':
                amoc_tas_all = amoc_tas
            else:
                amoc_tas_all = xr.merge([amoc_tas_all, amoc_tas])

        ctrl_amoc = hosmip_amoc_pi_data[model].mean(dim='time')
        ctrl_tas = hosmip_tas_pi_data[model].where(masks[model]['EU_buffer'], drop=True).mean(dim='time')

        model_ctrl = xr.merge([ctrl_amoc, ctrl_tas])

        model_ctrl.coords['scenar'] = 'pi'
        model_ctrl.coords['type'] = 'control'
        model_ctrl = model_ctrl.expand_dims(['scenar', 'type'], axis=[-1, -2])

        amoc_tas_all = xr.merge([amoc_tas_all, model_ctrl]) 

        amoc_tas_masks = xr.merge([amoc_tas_all, masks_ds])

        multi_model_dict[model] = amoc_tas_masks
    
    return multi_model_dict


def get_other_studies_data(masks):

    print('Loading Bellomo & Mehling (2024) data...')

    data_f4_tas = xr.open_dataset(local_path + 'Bellomo2024_plotting_data/data_tas_fig4.nc')

    data_f4_tas.coords['lon'] = (data_f4_tas.coords['lon'] + 180) % 360 - 180
    data_f4_tas = data_f4_tas.sortby(data_f4_tas.lon)

    tas_4x = data_f4_tas.tas_4x.where(masks['EC-Earth3']['EU'], drop=True).drop_vars('height')
    tas_hosing = data_f4_tas.tas_hosing.where(masks['EC-Earth3']['EU'], drop=True).drop_vars('height')

    tas_4x = tas_4x.assign_coords(scenar='ghg', type='diff')
    tas_hosing = tas_hosing.assign_coords(scenar='pi', type='diff')

    tas_diff = xr.concat([tas_4x, tas_hosing], dim='scenar')
    tas_diff = tas_diff.expand_dims('type', axis=-1)

    tas_diff = tas_diff.transpose('lat', 'lon', 'scenar', 'type')
    tas_diff = tas_diff.assign_coords(season='')
    tas_diff = tas_diff.expand_dims('season', axis=-3)
    tas_diff = tas_diff.transpose('lat', 'lon', 'season', 'scenar', 'type')

    bm_data = tas_diff.to_dataset(name='tas')

    amoc_4x = xr.DataArray(8.2)
    amoc_hosing = xr.DataArray(10.2)

    amoc_4x = amoc_4x.assign_coords(scenar='ghg', type='diff')
    amoc_hosing = amoc_hosing.assign_coords(scenar='pi', type='diff')

    amoc_diff = xr.concat([amoc_4x, amoc_hosing], dim='scenar')
    amoc_diff = amoc_diff.expand_dims('type', axis=-1)

    amoc_diff = amoc_diff.assign_coords(season='')
    amoc_diff = amoc_diff.expand_dims('season', axis=-3)
    amoc_diff = amoc_diff.transpose('season', 'scenar', 'type')

    bm_data['amoc'] = amoc_diff
    
    ########################################

    print('Loading van Westen & Baatsten (2025) data...')

    tas_vwb_600_pi = xr.open_dataset(local_path + 'RenevanWesten-AMOC-TEMP-Extremes-458bcf4/Data/CESM_0600_PI/Atmosphere/TEMP_2m_Europe_month_1-1.nc')
    amoc_vwb_600_pi = xr.open_dataset(local_path + 'RenevanWesten-AMOC-TEMP-Extremes-458bcf4/Data/CESM_0600_PI/Ocean/AMOC_transport_depth_0-1000m.nc')
    tas_vwb_1500_pi = xr.open_dataset(local_path + 'RenevanWesten-AMOC-TEMP-Extremes-458bcf4/Data/CESM_1500_PI/Atmosphere/TEMP_2m_Europe_month_1-1.nc')
    amoc_vwb_1500_pi = xr.open_dataset(local_path + 'RenevanWesten-AMOC-TEMP-Extremes-458bcf4/Data/CESM_1500_PI/Ocean/AMOC_transport_depth_0-1000m.nc')

    tas_vwb_600_45 = xr.open_dataset(local_path + 'RenevanWesten-AMOC-TEMP-Extremes-458bcf4/Data/CESM_0600_RCP45/Atmosphere/TEMP_2m_Europe_month_1-1.nc')
    amoc_vwb_600_45 = xr.open_dataset(local_path + 'RenevanWesten-AMOC-TEMP-Extremes-458bcf4/Data/CESM_0600_RCP45/Ocean/AMOC_transport_depth_0-1000m.nc')
    tas_vwb_1500_45 = xr.open_dataset(local_path + 'RenevanWesten-AMOC-TEMP-Extremes-458bcf4/Data/CESM_1500_RCP45/Atmosphere/TEMP_2m_Europe_month_1-1.nc')
    amoc_vwb_1500_45 = xr.open_dataset(local_path + 'RenevanWesten-AMOC-TEMP-Extremes-458bcf4/Data/CESM_1500_RCP45/Ocean/AMOC_transport_depth_0-1000m.nc')

    mean_amoc_vwb_600_pi = amoc_vwb_600_pi.Transport.sel(time=slice(1001, 1100)).mean()
    mean_amoc_vwb_600_45 = amoc_vwb_600_45.Transport.sel(time=slice(2400, 2500)).mean()
    mean_amoc_vwb_1500_pi = amoc_vwb_1500_pi.Transport.sel(time=slice(1901, 2000)).mean()
    mean_amoc_vwb_1500_45 = amoc_vwb_1500_45.Transport.sel(time=slice(2400, 2500)).mean()

    vwb_mask = make_country_masks_land_aware(tas_vwb_600_45.mean(dim='time').TEMP_2m)

    tas_600_pi = tas_vwb_600_pi.TEMP_2m.mean(dim='time').where(vwb_mask['EU_buffer'], drop=True)
    tas_600_45 = tas_vwb_600_45.TEMP_2m.mean(dim='time').where(vwb_mask['EU_buffer'], drop=True)
    tas_1500_pi = tas_vwb_1500_pi.TEMP_2m.mean(dim='time').where(vwb_mask['EU_buffer'], drop=True)
    tas_1500_45 = tas_vwb_1500_45.TEMP_2m.mean(dim='time').where(vwb_mask['EU_buffer'], drop=True)

    tas_600_pi = tas_600_pi.assign_coords(scenar='pi', type='control')
    tas_600_45 = tas_600_45.assign_coords(scenar='ghg', type='control')
    tas_1500_pi = tas_1500_pi.assign_coords(scenar='pi', type='hosing')
    tas_1500_45 = tas_1500_45.assign_coords(scenar='ghg', type='hosing')

    tas_control = xr.concat([tas_600_pi, tas_600_45], dim='scenar')
    tas_hosing = xr.concat([tas_1500_pi, tas_1500_45], dim='scenar')
    vwb_data_tas = xr.concat([tas_control, tas_hosing], dim='type')
    vwb_data_tas = vwb_data_tas.transpose('lat', 'lon', 'scenar', 'type')
    vwb_data = vwb_data_tas.to_dataset(name='tas')

    amoc_600_pi = mean_amoc_vwb_600_pi.assign_coords(scenar='pi', type='control')
    amoc_600_45 = mean_amoc_vwb_600_45.assign_coords(scenar='ghg', type='control')
    amoc_1500_pi = mean_amoc_vwb_1500_pi.assign_coords(scenar='pi', type='hosing')
    amoc_1500_45 = mean_amoc_vwb_1500_45.assign_coords(scenar='ghg', type='hosing')

    amoc_control = xr.concat([amoc_600_pi, amoc_600_45], dim='scenar')
    amoc_hosing = xr.concat([amoc_1500_pi, amoc_1500_45], dim='scenar')
    vwb_data_amoc = xr.concat([amoc_control, amoc_hosing], dim='type')
    vwb_data_amoc = vwb_data_amoc.transpose('scenar', 'type')
    vwb_data['amoc'] = vwb_data_amoc

    vwb_data = vwb_data.assign_coords(season='')
    vwb_data = vwb_data.expand_dims('season', axis=-3)

    vwb_masks_ds = xr.concat([vwb_mask[k] for k in vwb_mask.keys()],
                    dim=pd.Index(list(vwb_mask.keys()), name='region'))
    vwb_masks_ds = vwb_masks_ds.where(vwb_mask['EU_buffer'], drop=True)

    vwb_data['mask'] = vwb_masks_ds

    ########################################

    print('Loading Boot et al. (2024) data...')

    boot_exps = ['CTL_126', 'CTL_585', 'HOS_126', 'HOS_585']
    boot_data_dict = {}

    for i, exp in enumerate(boot_exps):
        # AMOC strength data (MOC)
        amoc = xr.open_dataset(f"/work/uo1075/m300817/teu_amoc/data/CESM2_hosing/{exp}/{exp}_amoc26_yr.nc", use_cftime=True)

        ctrl_or_hos = 'control' if exp[:3] == 'CTL' else 'hosing_05'
        amoc = amoc.assign_coords(scenar=f'ssp{exp[4:]}', type=ctrl_or_hos if ctrl_or_hos=='control' else 'hosing')

        boot_data_dict[exp+'_amoc'] = amoc.MOC.drop_vars(['transport_reg', 'moc_comp', 'lat_aux_grid', 'moc_z'])

        # Surface Air Temperature at Reference Height (TREFHT)
        tas = xr.open_dataset(local_path + f'Boot2024/TREFHT_{ctrl_or_hos}_{exp[4:]}.nc', use_cftime=True).TREFHT - 273.15
        tas.coords['lon'] = (tas.coords['lon'] + 180) % 360 - 180
        tas = tas.sortby(tas.lon)
        
        tas = tas.where(masks['CESM2']['EU_buffer'], drop=True)

        tas = tas.assign_coords(scenar=f'ssp{exp[4:]}', type=ctrl_or_hos if ctrl_or_hos=='control' else 'hosing')
        
        boot_data_dict[exp+'_tas'] = tas.resample(time='1YS').mean(dim='time').isel(time=slice(1, None))

    boot_tas_control = xr.concat([boot_data_dict['CTL_126_tas'], boot_data_dict['CTL_585_tas']], dim='scenar')
    boot_tas_hosing = xr.concat([boot_data_dict['HOS_126_tas'], boot_data_dict['HOS_585_tas']], dim='scenar')
    boot_tas = xr.concat([boot_tas_control, boot_tas_hosing], dim='type')
    boot_tas = boot_tas.transpose('lat', 'lon', 'time', 'scenar', 'type')

    boot_tas = boot_tas.assign_coords(season='')
    boot_tas = boot_tas.expand_dims('season', axis=-3)
    boot_tas = boot_tas.transpose('lat', 'lon', 'time', 'season', 'scenar', 'type')
    # boot_tas = boot_tas
    boot_data = boot_tas.to_dataset(name='tas')

    boot_amoc_control = xr.concat([boot_data_dict['CTL_126_amoc'], boot_data_dict['CTL_585_amoc']], dim='scenar')
    boot_amoc_hosing = xr.concat([boot_data_dict['HOS_126_amoc'], boot_data_dict['HOS_585_amoc']], dim='scenar')
    boot_amoc = xr.concat([boot_amoc_control, boot_amoc_hosing], dim='type')
    boot_amoc = boot_amoc.transpose('time', 'scenar', 'type')

    boot_amoc = boot_amoc.assign_coords(season='')
    boot_amoc = boot_amoc.expand_dims('season', axis=-3)
    boot_amoc = boot_amoc.transpose('time', 'season', 'scenar', 'type')
    boot_data['amoc'] = boot_amoc

    ########################################

    print('Loading Liu et al. (2020) data...')

    liu_hist = xr.open_dataset(local_path + 'Liu2020/b40.20th.track1.1deg.ensm.cam2.h0.TS.1961-1980.annclim.nc').TS.drop_vars(['time']).squeeze()
    liu_85 = xr.open_dataset(local_path + 'Liu2020/b40.rcp8_5.1deg.ensm.cam2.h0.TS.2061-2080.annclim.nc').TS.drop_vars(['time']).squeeze()
    liu_85_stable = xr.open_dataset(local_path + 'Liu2020/b40.rcp8_5.1deg.ensm.dhos6.cam2.h0.TS.2061-2080.annclim.nc').TS.drop_vars(['time']).squeeze()

    liu_hist_ip = liu_hist.interp_like(liu_85)

    amoc_pi = xr.DataArray(25.84686852) # 1850-1900
    amoc_85 = xr.DataArray(25.19321251) # 2061-2080
    amoc_85_stable = xr.DataArray(19.09885025) # 2061-2080

    liu_hist_ip = liu_hist_ip.assign_coords(scenar='his', type='control')
    liu_85 = liu_85.assign_coords(scenar='ghg', type='control')
    liu_85_stable = liu_85_stable.assign_coords(scenar='ghg', type='hosing')

    ghg_both = xr.concat([liu_85, liu_85_stable], dim='type')
    liu_data = xr.concat([liu_hist_ip.expand_dims('type', axis=-1), ghg_both], dim='scenar')
    liu_data = liu_data.transpose('lat', 'lon', 'scenar', 'type')
    liu_data = liu_data.to_dataset(name='tas')

    amoc_pi = amoc_pi.assign_coords(scenar='pi', type='control')
    amoc_85 = amoc_85.assign_coords(scenar='ghg', type='control')
    amoc_85_stable = amoc_85_stable.assign_coords(scenar='ghg', type='hosing')

    ghg_amoc = xr.concat([amoc_85, amoc_85_stable], dim='type')
    liu_data = liu_data.reindex(scenar=['pi', 'his', 'ghg'])
    liu_data['amoc'] = xr.concat([amoc_pi.expand_dims('type', axis=-1), ghg_amoc], dim='scenar')

    liu_data = liu_data.assign_coords(season='')
    liu_data = liu_data.expand_dims('season', axis=-3)

    liu_data['tas'] = liu_data['tas'].transpose('lat', 'lon', 'season', 'scenar', 'type')

    liu_data.coords['lon'] = (liu_data.coords['lon'] + 180) % 360 - 180
    liu_data = liu_data.sortby(liu_data.lon)

    liu_mask = make_country_masks_land_aware(liu_data.tas.sel(scenar='his', type='control', season=''))

    liu_masks_ds = xr.concat([liu_mask[k] for k in liu_mask.keys()], 
                    dim=pd.Index(list(liu_mask.keys()), name='region'))
    liu_masks_ds = liu_masks_ds.where(liu_mask['EU_buffer'], drop=True)
    liu_masks_ds = liu_masks_ds.drop_vars(['scenar', 'type', 'season'])

    liu_data = liu_data.transpose('lat', 'lon', 'season', 'scenar', 'type')

    liu_data['tas'] = liu_data['tas'].where(liu_masks_ds.sel(region='EU_buffer'), drop=True)

    liu_data['mask'] = liu_masks_ds

    ########################################

    return bm_data, vwb_data, boot_data, liu_data


def get_full_multi_model_dict():

    if os.path.exists(local_path + 'multi_model_dict.pkl'):
        print("Loading precomputed multi_model_dict...")
        with open(local_path + 'multi_model_dict.pkl', 'rb') as f:
            multi_model_dict = pickle.load(f)
        print("Loading precomputed HosMIP masks...")
        with open(local_path + 'hosmip_masks.pkl', 'rb') as f:
            masks = pickle.load(f)
    else:
        print("Precomputed multi_model_dict not found. Recomputing...")

        amoc_data, hosmip_amoc_pi_data, tas_data, hosmip_tas_pi_data, masks, data_cutoffs = load_hosmip_data()
        multi_model_dict = build_hosmip_dataset(amoc_data, hosmip_amoc_pi_data, tas_data, hosmip_tas_pi_data, masks, data_cutoffs)
        bm_data, vwb_data, boot_data, liu_data = get_other_studies_data(masks)

        multi_model_dict['CCSM4'] = liu_data
        multi_model_dict['CESM1'] = vwb_data
        multi_model_dict['EC-Earth3'] = xr.merge([multi_model_dict['EC-Earth3'], bm_data])
        multi_model_dict['CESM2'] = xr.merge([multi_model_dict['CESM2'], boot_data])

        print("Saving multi_model_dict to ../data/multi_model_dict.pkl")
        with open(local_path + 'multi_model_dict.pkl', 'wb') as f:
            pickle.dump(multi_model_dict, f)

        print("Saving HosMIP masks to ../data/hosmip_masks.pkl")
        with open(local_path + 'hosmip_masks.pkl', 'wb') as f:
            pickle.dump(masks, f)

    # GISS-E2-1-G masks come from the dedicated bu1431 cache (built on the GISS
    # native 90x144 grid; same EU/NEU/WCE/MED + country coverage as the HosMIP
    # models). The pickled masks live on the model's native lon=[1.25,358.75]
    # (0-360 convention); get_cmip_projections wraps loaded tas/amoc to
    # lon=[-180,180]. We wrap the masks here to match so the where() crop and
    # the plotter's per-country _select() both align — otherwise the western
    # hemisphere (IE, GB, IS, ...) silently drops out (cells dropped on data
    # side, but country masks still nominally cover them at lon ~ 350°).
    if 'GISS-E2-1-G' not in masks:
        giss_masks_path = GISS_PROCESSED_PATH + 'giss-e2-1-g_masks.pkl'
        if os.path.exists(giss_masks_path):
            with open(giss_masks_path, 'rb') as f:
                giss_masks = pickle.load(f)
            giss_masks = {
                k: v.assign_coords(
                    lon=(((v.lon + 180) % 360) - 180)).sortby('lon')
                for k, v in giss_masks.items()}
            masks['GISS-E2-1-G'] = giss_masks

    return multi_model_dict, masks


def get_cmip_projections(masks, data_dict=None, extra_ssps=None):
    # ``extra_ssps``: optional list of SSP labels beyond the module-level
    # ``ssps`` (default 3-SSP paper set, e.g. ``['ssp585']``). Realisations
    # for every SSP — paper-set and extras alike — come from
    # ``cmip6_inventory.available_realisations``; auto-discovery is the
    # only source of truth.

    if data_dict is None:
        data_dict = load_mpi_esm_data(eur_only=True)

    cmip6_ctrl_data = {}
    model_names = ['MPI-ESM1-2-HR', 'CESM2', 'CanESM5', 'IPSL-CM6A-LR',
                'HadGEM3-GC3-1MM', 'HadGEM3-GC3-1LL', 'EC-Earth3',
                'GISS-E2-1-G']


    _extras = [s for s in (extra_ssps or []) if s not in ssps]
    local_ssps = list(ssps) + _extras


    realiz_ids = {}
    for model in model_names:
        physics, _f = MODEL_PHYSICS_FORCING[model]
        realiz_ids[model] = {'physics': physics, 'missing': []}
        for ssp_i in local_ssps:
            ns = discover_int_reas(model, ssp_i)
            realiz_ids[model][ssp_i] = ns
            if not ns:
                realiz_ids[model]['missing'].append(ssp_i)
        # Historical
        realiz_ids[model]['his'] = discover_int_reas(model, 'historical')
        if not realiz_ids[model]['his']:
            realiz_ids[model]['missing'].append('his')

    # Load SSP scenarios first (they share time coordinates 2015-2100)
    for model in model_names:
        for ssp_i in local_ssps:
            for season in ['', 'djf', 'jja']:
                if ssp_i in realiz_ids[model]['missing']:
                    continue
                print(f"Loading {model} {ssp_i} {'annual' if season=='' else season} data...")

                physics = realiz_ids[model]['physics']
                model_lower = lower_model_id(model)
                f_number = MODEL_PHYSICS_FORCING[model][1]

                data_path = model_root(model) + f'{ssp_i}/{model_lower}/'

                # For extra SSPs (ssp585) and for models that ship without
                # seasonal subsets (e.g. GISS-E2-1-G — only annual files are
                # staged), skip the seasonal branch silently. The seasonal
                # slot stays NaN for those (model, scenar) combinations,
                # which Fig3/FigS14/FigS15 honour because their plotters
                # explicitly check season=='' before drawing.
                if season != '' and not os.path.isdir(data_path + f'tas_{season}'):
                    # GISS seasonal scenmip is not staged on bu1431 (no
                    # transform applied, just a different file layout); fall
                    # back to the uo1075 location where seasonalised yearly
                    # files for ssp126/ssp245/ssp370 already exist as real
                    # files. Only this model+season needs the fallback; other
                    # misses still skip silently.
                    if model == 'GISS-E2-1-G':
                        uo_data_path = (f'/work/uo1075/m300817/teu_amoc/data/'
                                        f'CMIP6/{ssp_i}/{model_lower}/')
                        if os.path.isdir(uo_data_path + f'tas_{season}'):
                            data_path = uo_data_path
                        else:
                            print(f"  (seasonal {season} dir absent for "
                                  f"{model} {ssp_i} on bu1431 and uo1075 "
                                  f"— skipping)")
                            continue
                    else:
                        print(f"  (seasonal {season} dir absent for "
                              f"{model} {ssp_i} — skipping)")
                        continue

                if season == '':
                    new_cmip_tas_data = xr.concat([
                        (cmip6_inventory.open_canonical(data_path+f'{model_lower}_r{r}i1{physics}{f_number}_tas_yr.nc', model).to_dataset(name='tas') - 273.15).assign_coords(scenar=ssp_i, season=season) for r in realiz_ids[model][ssp_i]
                    ], dim='realiz')

                    if model in ['CESM2', 'CanESM5', 'HadGEM3-GC3-1LL', 'HadGEM3-GC3-1MM', 'GISS-E2-1-G']:
                        new_cmip_tas_data = new_cmip_tas_data.assign_coords(time=cmip6_ctrl_data['MPI-ESM1-2-HR'].time.values)
                else:
                    # Filter realisations to those with seasonal files present
                    # — annual ensembles can outgrow seasonal ones when new
                    # raw data is staged before the seasonal split is rerun
                    # (e.g. CESM2 ssp126 r4/r11 added 2026-05-24 without
                    # corresponding tas_djf / tas_jja files).
                    _seasonal_realiz = [
                        r for r in realiz_ids[model][ssp_i]
                        if os.path.isfile(
                            data_path + f'tas_{season}/{model_lower}_r{r}i1{physics}{f_number}_tas_{season}.nc')
                    ]
                    _missing_seasonal = sorted(set(realiz_ids[model][ssp_i]) - set(_seasonal_realiz))
                    if _missing_seasonal:
                        print(f"  ({model} {ssp_i} {season}: seasonal files "
                              f"missing for r{_missing_seasonal}; loading "
                              f"{len(_seasonal_realiz)}/{len(realiz_ids[model][ssp_i])} members)")
                    if not _seasonal_realiz:
                        continue
                    new_cmip_tas_data = xr.concat([
                        (cmip6_inventory.open_canonical(data_path+f'tas_{season}/{model_lower}_r{r}i1{physics}{f_number}_tas_{season}.nc', model, as_dataset=True) - 273.15).assign_coords(scenar=ssp_i, season=season) for r in _seasonal_realiz
                    ], dim='realiz')
                    if 'dim' in drop_stuff[model].keys():
                        new_cmip_tas_data = new_cmip_tas_data.drop_dims(drop_stuff[model]['dim'])
                    if 'var' in drop_stuff[model].keys():
                        new_cmip_tas_data = new_cmip_tas_data.drop_vars(drop_stuff[model]['var'])

                new_cmip_tas_data.coords['lon'] = (new_cmip_tas_data.coords['lon'] + 180) % 360 - 180
                new_cmip_tas_data = new_cmip_tas_data.sortby(new_cmip_tas_data.lon)

                new_cmip_tas_data = new_cmip_tas_data.where(masks[model]['EU_buffer'], drop=True)

                if model not in cmip6_ctrl_data.keys():
                    cmip6_ctrl_data[model] = new_cmip_tas_data.mean(dim='realiz')
                else:
                    cmip6_ctrl_data[model]['tas'].loc[dict(scenar=ssp_i, season=season)] = new_cmip_tas_data.tas.mean(dim='realiz')

                if season != '':
                    continue

                new_cmip_amoc_data = xr.concat([
                    cmip6_inventory.open_canonical(data_path+f'{model_lower}_r{r}i1{physics}{f_number}_amoc26_yr.nc', model).drop_vars('lev' if model != 'IPSL-CM6A-LR' else 'olevel').to_dataset(name='amoc') for r in realiz_ids[model][ssp_i]
                ], dim='realiz')
                if model in ['CESM2', 'CanESM5', 'HadGEM3-GC3-1LL', 'HadGEM3-GC3-1MM', 'GISS-E2-1-G']:
                    # Historical [50:] slice for CESM2 ssp126 was needed when
                    # MPI-HR carried hist+ssp (251 yr) and CESM2 ssp126 amoc
                    # started at 2015 (86 yr). Both are 86 yr (2015–2100) now,
                    # so we always align on the full MPI-HR time axis.
                    new_cmip_amoc_data = new_cmip_amoc_data.assign_coords(time=cmip6_ctrl_data['MPI-ESM1-2-HR'].time.values)

                if 'amoc' not in cmip6_ctrl_data[model].data_vars:
                    cmip6_ctrl_data[model]['amoc'] = new_cmip_amoc_data.amoc.mean(dim='realiz')
                    cmip6_ctrl_data[model] = cmip6_ctrl_data[model].expand_dims('scenar', axis=-1).reindex(scenar=local_ssps)
                    cmip6_ctrl_data[model]['tas'] = cmip6_ctrl_data[model]['tas'].expand_dims('season', axis=-1).reindex(season=['', 'djf', 'jja'])
                else:
                    cmip6_ctrl_data[model]['amoc'].loc[dict(scenar=ssp_i)] = new_cmip_amoc_data.amoc.mean(dim='realiz')

    # Historical (1850-2015) loaded separately, then outer-concatenated onto the SSP data.
    mpi_hr_his_time = None

    for model in model_names:
        if 'his' in realiz_ids[model]['missing']:
            continue
        print(f"Loading {model} his annual data...")

        physics = realiz_ids[model]['physics']
        model_lower = lower_model_id(model)

        data_path = model_root(model) + f'historical/{model_lower}/'
        physics_num = '1' if physics == 'p1' else '2'
        f_num = MODEL_PHYSICS_FORCING[model][1][1:]  # 'f3' -> '3'

        # Load historical TAS data
        new_cmip_tas_data = xr.concat([
            (cmip6_inventory.open_canonical(data_path+f'{model_lower}_r{r}i1p{physics_num}f{f_num}_tas_yr.nc', model).to_dataset(name='tas') - 273.15).assign_coords(scenar='his', season='') for r in realiz_ids[model]['his']
        ], dim='realiz')

        new_cmip_tas_data.coords['lon'] = (new_cmip_tas_data.coords['lon'] + 180) % 360 - 180
        new_cmip_tas_data = new_cmip_tas_data.sortby(new_cmip_tas_data.lon)
        new_cmip_tas_data = new_cmip_tas_data.where(masks[model]['EU_buffer'], drop=True)

        # Load historical AMOC data
        new_cmip_amoc_data = xr.concat([
            cmip6_inventory.open_canonical(data_path+f'{model_lower}_r{r}i1p{physics_num}f{f_num}_amoc26_yr.nc', model).drop_vars('lev' if model != 'IPSL-CM6A-LR' else 'olevel').to_dataset(name='amoc') for r in realiz_ids[model]['his']
        ], dim='realiz')

        # Store MPI-ESM1-2-HR time coords for calendar alignment
        if model == 'MPI-ESM1-2-HR':
            mpi_hr_his_time = new_cmip_tas_data.time.values

        # Align time coordinates for models with different calendars
        if model in ['CESM2', 'CanESM5', 'HadGEM3-GC3-1LL', 'HadGEM3-GC3-1MM', 'GISS-E2-1-G']:
            new_cmip_tas_data = new_cmip_tas_data.assign_coords(time=mpi_hr_his_time)
            new_cmip_amoc_data = new_cmip_amoc_data.assign_coords(time=mpi_hr_his_time)

        # Build historical dataset with proper dimensions
        his_tas = new_cmip_tas_data.tas.mean(dim='realiz').expand_dims('scenar').assign_coords(scenar=['his'])
        his_tas = his_tas.expand_dims('season').assign_coords(season=[''])
        his_amoc = new_cmip_amoc_data.amoc.mean(dim='realiz').expand_dims('scenar').assign_coords(scenar=['his'])

        # Historical seasonal tas (djf, jja). Same path convention as the
        # SSP seasonal block above. Per-realisation seasonal files may not
        # all exist for newer ensemble members; filter to those present
        # (mirrors the SSP-seasonal fallback added 2026-05-25 (c)).
        # Time coords reuse the annual historical axis so concat works.
        his_seasonal_tas_dict = {}
        # Seasonal historical tas for HosMIP-7 + GISS is written to bu1431
        # by scripts/process_seasonal_tas.py (uo1075 is read-only). Look
        # there as a fallback when the canonical uo1075 mirror is absent.
        _BU1431_HIST_ROOT = '/work/bu1431/T_EU_AMOC/CMIP6/historical/'
        for _season in ('djf', 'jja'):
            _seas_dir = data_path + f'tas_{_season}/'
            if not os.path.isdir(_seas_dir):
                _bu_seas_dir = f'{_BU1431_HIST_ROOT}{model_lower}/tas_{_season}/'
                if os.path.isdir(_bu_seas_dir):
                    _seas_dir = _bu_seas_dir
                else:
                    print(f"  ({model} his {_season}: dir absent on uo1075+bu1431 — leaving NaN)")
                    continue
            _seas_realiz = [
                r for r in realiz_ids[model]['his']
                if os.path.isfile(_seas_dir + f'{model_lower}_r{r}i1p{physics_num}f{f_num}_tas_{_season}.nc')
            ]
            _missing = sorted(set(realiz_ids[model]['his']) - set(_seas_realiz))
            if _missing:
                print(f"  ({model} his {_season}: per-realisation files missing "
                      f"for r{_missing}; loading "
                      f"{len(_seas_realiz)}/{len(realiz_ids[model]['his'])} members)")
            if not _seas_realiz:
                continue
            _seas_data = xr.concat([
                (cmip6_inventory.open_canonical(_seas_dir + f'{model_lower}_r{r}i1p{physics_num}f{f_num}_tas_{_season}.nc', model, as_dataset=True) - 273.15).assign_coords(scenar='his', season=_season)
                for r in _seas_realiz
            ], dim='realiz')
            # Apply the same per-model attribute scrubbing that the SSP
            # seasonal block does (drop_stuff cleanup). Guard against
            # absent dims/vars — bu1431 seasonal-historical files
            # produced by process_seasonal_tas.py omit lat_bnds/lon_bnds
            # and the 'bnds' dim, whereas the uo1075 SSP-seasonal files
            # carry them. Either form is now tolerated. GISS isn't in
            # drop_stuff at all (no scrubbing needed); guarded too.
            _model_drop = drop_stuff.get(model, {})
            if 'dim' in _model_drop:
                _dims_to_drop = [d for d in _model_drop['dim']
                                 if d in _seas_data.dims]
                if _dims_to_drop:
                    _seas_data = _seas_data.drop_dims(_dims_to_drop)
            if 'var' in _model_drop:
                _vars_to_drop = [v for v in _model_drop['var']
                                 if v in _seas_data.variables]
                if _vars_to_drop:
                    _seas_data = _seas_data.drop_vars(_vars_to_drop)
            _seas_data.coords['lon'] = (_seas_data.coords['lon'] + 180) % 360 - 180
            _seas_data = _seas_data.sortby(_seas_data.lon)
            _seas_data = _seas_data.where(masks[model]['EU_buffer'], drop=True)
            # Calendar align (mirror annual branch above).
            if model in ['CESM2', 'CanESM5', 'HadGEM3-GC3-1LL', 'HadGEM3-GC3-1MM', 'GISS-E2-1-G']:
                _seas_data = _seas_data.assign_coords(time=mpi_hr_his_time)
            _his_seas_tas = _seas_data.tas.mean(dim='realiz').expand_dims('scenar').assign_coords(scenar=['his'])
            his_seasonal_tas_dict[_season] = _his_seas_tas

        # Pad his_tas to all three seasons; fill seasonal slots with loaded
        # data where available, leave as the annual-shaped NaN placeholder
        # otherwise (matches the existing season axis layout).
        his_tas_full = his_tas.reindex(season=['', 'djf', 'jja'])
        for _season, _data in his_seasonal_tas_dict.items():
            his_tas_full.loc[dict(season=_season)] = _data.isel(scenar=0)

        his_ds = xr.Dataset({'tas': his_tas_full, 'amoc': his_amoc})

        # Concatenate historical with SSP data using join='outer' to handle
        # different time coordinates. Both legs are on the model's canonical
        # grid (via ``cmip6_inventory.open_canonical`` at load time), so the
        # lat/lon coords are bit-identical and the outer concat unions only
        # the time and scenar dims.
        cmip6_ctrl_data[model] = xr.concat([cmip6_ctrl_data[model], his_ds], dim='scenar', join='outer')

    # add MPI-ESM1-2-LR data (SSP scenarios)
    # Skip any extra SSPs not present in the MPI-GE data_dict's scenar coord
    _mpi_lr_available = set(data_dict['']['tas']['ssp'].scenar.values.tolist())
    _mpi_lr_ssps = [s for s in local_ssps if s in _mpi_lr_available]
    for ssp_i in _mpi_lr_ssps:
        for season in ['', 'djf', 'jja']:
            # Per-season availability (seasonal files may lack ssp585)
            if ssp_i in _extras and season != '':
                _seas_available = set(
                    data_dict[season]['tas']['ssp'].scenar.values.tolist())
                if ssp_i not in _seas_available:
                    print(f"  (MPI-ESM1-2-LR {ssp_i} {season} not in "
                          f"data_dict — skipping)")
                    continue
            print(f"Loading MPI-ESM1-2-LR {ssp_i} {'annual' if season=='' else season} data...")
            if 'MPI-ESM1-2-LR' not in cmip6_ctrl_data.keys():
                cmip6_ctrl_data['MPI-ESM1-2-LR'] = data_dict[season]['tas']['ssp'].sel(scenar=ssp_i).mean(dim='realiz').assign_coords(season=season)
                cmip6_ctrl_data['MPI-ESM1-2-LR']['amoc'] = data_dict['']['amoc']['ssp'].AMOC_strength.sel(scenar=ssp_i).mean(dim='realiz')
                cmip6_ctrl_data['MPI-ESM1-2-LR'] = cmip6_ctrl_data['MPI-ESM1-2-LR'].expand_dims('scenar', axis=-1).reindex(scenar=_mpi_lr_ssps)
                cmip6_ctrl_data['MPI-ESM1-2-LR']['tas'] = cmip6_ctrl_data['MPI-ESM1-2-LR']['tas'].expand_dims('season', axis=-1).reindex(season=['', 'djf', 'jja'])
            else:
                if season != '':
                    cmip6_ctrl_data['MPI-ESM1-2-LR']['tas'].loc[dict(scenar=ssp_i, season=season)] = data_dict[season]['tas']['ssp'].tas.sel(scenar=ssp_i).assign_coords(time=data_dict['']['tas']['ssp'].time.values).mean(dim='realiz')
                    continue
                else:
                    cmip6_ctrl_data['MPI-ESM1-2-LR']['tas'].loc[dict(scenar=ssp_i, season=season)] = data_dict[season]['tas']['ssp'].tas.sel(scenar=ssp_i).mean(dim='realiz')
                cmip6_ctrl_data['MPI-ESM1-2-LR']['amoc'].loc[dict(scenar=ssp_i)] = data_dict['']['amoc']['ssp'].AMOC_strength.sel(scenar=ssp_i).mean(dim='realiz')

    # add MPI-ESM1-2-LR historical data (annual + seasonal tas; amoc annual)
    print("Loading MPI-ESM1-2-LR his annual data...")
    his_tas_annual = data_dict['']['tas']['his'].tas.mean(dim='realiz')
    his_tas = (his_tas_annual
               .expand_dims('scenar').assign_coords(scenar=['his'])
               .expand_dims('season').assign_coords(season=[''])
               .reindex(season=['', 'djf', 'jja']))
    # Wire MPI-GE seasonal historical (his_tas_djf.nc / his_tas_jja.nc
    # already present in data_dict via load_mpi_esm_data). Time axis is
    # aligned to the annual axis so the outer scenar concat unions cleanly.
    for _season in ('djf', 'jja'):
        if 'his' in data_dict.get(_season, {}).get('tas', {}):
            print(f"Loading MPI-ESM1-2-LR his {_season} data...")
            _seas = data_dict[_season]['tas']['his'].tas.mean(dim='realiz')
            _seas = _seas.assign_coords(time=his_tas_annual.time.values)
            his_tas.loc[dict(season=_season)] = _seas
    his_amoc = data_dict['']['amoc']['his'].AMOC_strength.mean(dim='realiz').expand_dims('scenar').assign_coords(scenar=['his'])
    his_ds = xr.Dataset({'tas': his_tas, 'amoc': his_amoc})

    # Concatenate historical with SSP data using join='outer' to handle different time coordinates
    cmip6_ctrl_data['MPI-ESM1-2-LR'] = xr.concat([cmip6_ctrl_data['MPI-ESM1-2-LR'], his_ds], dim='scenar', join='outer')

    return cmip6_ctrl_data


def get_gwl_diagnostic_data(data_dict=None, recompute=False):
    """Ensemble-mean global tas + ensemble-mean AMOC per (model, scenar).

    Diagnostic loader for the global-warming-level overview plot
    (scripts/gwl_overview.py). NOT part of the canonical pipeline; the
    realiz-id table is duplicated from ``get_cmip_projections`` and may
    drift — if you change member coverage there, refresh the cache by
    re-running this with ``recompute=True``.

    Returns
    -------
    dict[str, xr.Dataset]
        Keyed by model. Each Dataset has dims ``(scenar, time)`` and
        variables ``tas_global`` (K, area-weighted global mean) and
        ``amoc`` (Sv). ``scenar`` always contains ``his``, ``ssp126``,
        ``ssp245``, ``ssp370``; missing combinations are NaN-filled.
        Both fields are ensemble means across the available realisations.

    Cached to ``data/gwl_diagnostic.pkl``.
    """
    cache_path = local_path + 'gwl_diagnostic.pkl'
    if (not recompute) and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            out = pickle.load(f)
        print(f'Loaded GWL diagnostic data from {cache_path}')
        return out

    print('Computing GWL diagnostic data (ensemble-mean global tas + AMOC)...')

    # --- AMOC: reuse canonical loader (already ensemble-mean). ---
    _, masks = get_full_multi_model_dict()
    if data_dict is None:
        data_dict = load_mpi_esm_data(eur_only=False)
    cmip_ens_dict = get_cmip_projections(masks, data_dict)


    file_models = ['MPI-ESM1-2-HR', 'CESM2', 'CanESM5', 'IPSL-CM6A-LR',
                   'HadGEM3-GC3-1MM', 'HadGEM3-GC3-1LL', 'EC-Earth3',
                   'GISS-E2-1-G']


    realiz_ids = {}
    for model in file_models:
        physics, _f = MODEL_PHYSICS_FORCING[model]
        realiz_ids[model] = {'physics': physics, 'missing': []}
        for ssp_i in ['ssp126', 'ssp245', 'ssp370']:
            ns = discover_int_reas(model, ssp_i)
            realiz_ids[model][ssp_i] = ns
            if not ns:
                realiz_ids[model]['missing'].append(ssp_i)
        realiz_ids[model]['his'] = discover_int_reas(model, 'historical')
        if not realiz_ids[model]['his']:
            realiz_ids[model]['missing'].append('his')

    def _load_global_tas_one(path):
        da = xr.open_dataarray(path, use_cftime=True) - 273.15
        da.coords['lon'] = (da.coords['lon'] + 180) % 360 - 180  # wrap to [-180, 180]
        da = da.sortby(da.lon)
        return weighted_area_lat(da).mean(('lat', 'lon'))

    out = {}
    scenars = ['his', 'ssp126', 'ssp245', 'ssp370']

    # MPI-HR historical/ssp time grids align cftime calendars across models (as in get_cmip_projections).
    mpi_hr_his_time = None
    mpi_hr_ssp_time = None

    for model in file_models:
        physics = realiz_ids[model]['physics']
        model_lower = lower_model_id(model)
        fnum = MODEL_PHYSICS_FORCING[model][1]
        per_scenar = {}

        # SSPs
        for ssp_i in ['ssp126', 'ssp245', 'ssp370']:
            members = realiz_ids[model][ssp_i]
            if not members or ssp_i in realiz_ids[model].get('missing', []):
                per_scenar[ssp_i] = None
                continue
            print(f'  global tas: {model} {ssp_i} ({len(members)} members)')
            data_path_ssp = model_root(model) + f'{ssp_i}/{model_lower}/'
            per_member = []
            for r in members:
                fn = f'{data_path_ssp}{model_lower}_r{r}i1{physics}{fnum}_tas_yr.nc'
                per_member.append(_load_global_tas_one(fn).assign_coords(realiz=r))
            arr = xr.concat(per_member, dim='realiz')
            if model in ['CESM2', 'CanESM5', 'HadGEM3-GC3-1LL', 'HadGEM3-GC3-1MM', 'GISS-E2-1-G'] and mpi_hr_ssp_time is not None:
                arr = arr.assign_coords(time=mpi_hr_ssp_time)
            elif model == 'MPI-ESM1-2-HR' and ssp_i == 'ssp126':
                mpi_hr_ssp_time = arr.time.values
            per_scenar[ssp_i] = arr.mean('realiz')

        # Historical
        if realiz_ids[model]['his']:
            members = realiz_ids[model]['his']
            print(f'  global tas: {model} his ({len(members)} members)')
            data_path_his = model_root(model) + f'historical/{model_lower}/'
            per_member = []
            for r in members:
                fn = f'{data_path_his}{model_lower}_r{r}i1{physics}{fnum}_tas_yr.nc'
                per_member.append(_load_global_tas_one(fn).assign_coords(realiz=r))
            arr = xr.concat(per_member, dim='realiz')
            if model == 'MPI-ESM1-2-HR':
                mpi_hr_his_time = arr.time.values
            if model in ['CESM2', 'CanESM5', 'HadGEM3-GC3-1LL', 'HadGEM3-GC3-1MM', 'GISS-E2-1-G'] and mpi_hr_his_time is not None:
                arr = arr.assign_coords(time=mpi_hr_his_time)
            per_scenar['his'] = arr.mean('realiz')
        else:
            per_scenar['his'] = None

        out[model] = per_scenar

    # MPI-ESM1-2-LR via data_dict (already loaded with eur_only=False).
    print('  global tas: MPI-ESM1-2-LR (MPI-GE)')
    mpi_lr = {}
    his_tas_full = data_dict['']['tas']['his'].tas  # (realiz, time, lat, lon)
    ssp_tas_full = data_dict['']['tas']['ssp'].tas  # (realiz, time, lat, lon, scenar)
    mpi_lr['his'] = weighted_area_lat(his_tas_full).mean(('lat', 'lon')).mean('realiz')
    for ssp_i in ['ssp126', 'ssp245', 'ssp370']:
        if ssp_i in ssp_tas_full.scenar.values:
            mpi_lr[ssp_i] = weighted_area_lat(ssp_tas_full.sel(scenar=ssp_i)).mean(('lat', 'lon')).mean('realiz')
        else:
            mpi_lr[ssp_i] = None
    out['MPI-ESM1-2-LR'] = mpi_lr

    # Package per-model Datasets with (scenar, time) layout, NaN-fill missing.
    # Collapse time to calendar-free mid-year datetime64 so the pickle round-trips
    # regardless of which cftime subclasses the underlying NetCDFs used (mixed
    # calendars across his vs ssp on a single model crashes CFTimeIndex on reload).
    def _to_year_dt64(arr):
        if np.issubdtype(arr.time.dtype, np.datetime64):
            years = pd.to_datetime(arr.time.values).year
        else:
            years = np.array([t.year for t in arr.time.values])
        new_time = pd.to_datetime([f'{int(y)}-07-01' for y in years]).values
        return arr.assign_coords(time=new_time)

    packaged = {}
    for model, per_scenar in out.items():
        tas_arrays = []
        for sc in scenars:
            arr = per_scenar.get(sc)
            if arr is None:
                continue
            arr = _to_year_dt64(arr)
            tas_arrays.append(arr.assign_coords(scenar=sc).expand_dims('scenar'))
        if not tas_arrays:
            continue
        tas_concat = xr.concat(tas_arrays, dim='scenar', join='outer')
        tas_concat = tas_concat.reindex(scenar=scenars)

        if model in cmip_ens_dict and 'amoc' in cmip_ens_dict[model].data_vars:
            amoc = _to_year_dt64(cmip_ens_dict[model]['amoc']).reindex(scenar=scenars)
        else:
            amoc = xr.full_like(tas_concat, np.nan)

        ds = xr.Dataset({'tas_global': tas_concat, 'amoc': amoc})
        packaged[model] = ds

    with open(cache_path, 'wb') as f:
        pickle.dump(packaged, f)
    print(f'Saved GWL diagnostic data to {cache_path}')
    return packaged


# TRANSFORMING MULTI-MODEL DATA

def net_cooling_point(coefs, T_2100, AMOC_2100, AMOC_pi_MPI, AMOC_metric='rel'):
    '''
        Mainly useful for quadratic regressions (quantile regressions). Not necessary for main (linear) analyses in paper.
        Calculate the net cooling point based on regression coefficients and temperature at 2100.
        Parameters:
            coefs (dict): Dictionary containing regression coefficients. Keys: 'coef_lin', coef_quad'. If linear regression, coef_quad should be 0.
            T_2100 (float): Temperature at 2100.
            AMOC_2100 (float): AMOC strength at 2100.
            AMOC_pi_MPI (float): Pre-industrial AMOC strength from MPI-ESM1.2-LR.
            AMOC_metric (str): Relative AMOC weakening in % ('rel') or absolute AMOC strength in Sv ('abs'). Default is 'rel'.
    '''

    quad_solution = np.roots([
        coefs['coef_quad'], 
        coefs['coef_lin'], 
        T_2100
        ]) if not np.isnan(coefs['coef_quad']) else np.nan

    if np.iscomplex(quad_solution).any():
        return np.nan
    elif coefs['coef_quad']==0 and coefs['coef_lin'] >= 0:
        return 110 if AMOC_metric=='rel' else -1
    else:
        if coefs['coef_quad'] < 0:
                if AMOC_metric == 'rel':
                    return (1 - AMOC_2100 / AMOC_pi_MPI) * 100 + np.max(quad_solution)
                elif AMOC_metric == 'abs':
                    return AMOC_2100 - AMOC_pi_MPI/100 * np.max(quad_solution)
                else:
                    raise ValueError("AMOC_metric must be 'rel' or 'abs'.")
        elif coefs['coef_quad'] >= 0:
                if AMOC_metric == 'rel':
                    return (1 - AMOC_2100 / AMOC_pi_MPI) * 100 + np.min(quad_solution)
                elif AMOC_metric == 'abs':
                    return AMOC_2100 - AMOC_pi_MPI/100 * np.min(quad_solution)
                else:
                    raise ValueError("AMOC_metric must be 'rel' or 'abs'.")


def get_hosmip_reg_ds(hosmip_ref_pi=True, recompute=False, data_dict=None, future_window=None):

    fw = resolve_future_window(future_window)
    cache_path = local_path + f'hosmip_reg_ds_dict{fw_suffix(future_window)}.pkl'
    if recompute or not os.path.exists(cache_path):

        print('Recomputing HosMIP regression dataset...')

        if data_dict is None:
            data_dict = load_mpi_esm_data(eur_only=True)
        multi_model_dict, masks = get_full_multi_model_dict()
        cmip6_ctrl_data = get_cmip_projections(masks, data_dict)

        hosmip_reg_ds_dict = {}
        for model in hosmip_labels:
            static_reg_ds = multi_model_dict[model].isel(time=0, drop=True).copy(deep=True).drop_vars(['tas', 'amoc', 'mask', 'region', 'type']) if 'time' in list(multi_model_dict[model].coords) else multi_model_dict[model].copy(deep=True).drop_vars(['tas', 'amoc', 'mask', 'region', 'type'])

            static_reg_ds = static_reg_ds.assign_coords(scenar=list(set([str(s) for s in static_reg_ds.scenar.values]+ssps)))

            reg_ds = static_reg_ds.assign(
                lin_coef_hosmip=(['lat', 'lon', 'season'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),
                ste_hosmip=(['lat', 'lon', 'season'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),  # standard error of regression coefficient
                rsq_hosmip=(['lat', 'lon', 'season'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),  # R^2 of regression
                pvalue_hosmip=(['lat', 'lon', 'season'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),  # p-value of regression coefficient
                AMOC_future=(['scenar'], np.full(tuple(static_reg_ds.isel(lat=0, lon=0, season=0).sizes.values()), np.nan)),
                AMOC_pi=([], np.full(tuple(static_reg_ds.isel(lat=0, lon=0, scenar=0, season=0).sizes.values()), np.nan)),
                T_future=(['lat', 'lon', 'season', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
                T_pi=(['lat', 'lon', 'season'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),
                T_pd_245=(['lat', 'lon', 'season'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),
                req_strength_pi=(['lat', 'lon', 'season', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
                req_strength_pd=(['lat', 'lon', 'season', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
                req_weakening_pi=(['lat', 'lon', 'season', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
                req_weakening_pd=(['lat', 'lon', 'season', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
            )

            hosmip_reg_ds_dict[model] = reg_ds

        # Set the harmonised 1850–1899 CMIP6 historical ensemble-mean baselines
        # BEFORE the regression loop. Doing it here (rather than after) lets
        # hosmip_regression_plot consume them as the % denominator and tas
        # anchor, so the cached slopes are intrinsically in K-per-(% relative
        # to the harmonised AMOC_pi) — no post-hoc rescale needed.
        for model in hosmip_labels:
            hosmip_reg_ds_dict[model].AMOC_pi.values = get_hist_pi_baseline(
                model, 'amoc', cmip6_ctrl_data=cmip6_ctrl_data,
                multi_model_dict=multi_model_dict)
            for _season in ['', 'djf', 'jja']:
                # T_pi from get_hist_pi_baseline shares the EU-subset
                # canonical grid with the reg_ds T_pi slot (both ultimately
                # come from cmip6_ctrl_data which is loaded via
                # ``cmip6_inventory.open_canonical``). Direct
                # xarray-aligned assignment is safe.
                hosmip_reg_ds_dict[model].T_pi.loc[:, :, _season] = (
                    get_hist_pi_baseline(
                        model, 'tas', cmip6_ctrl_data=cmip6_ctrl_data,
                        multi_model_dict=multi_model_dict, season=_season))

        for model in hosmip_labels:
            for season_idx, season in enumerate(['', 'djf', 'jja']):
                print(f"Calculating {'annual' if season=='' else season} regression coefficients for {model}...")
                i=0
                for lat in np.arange(len(hosmip_reg_ds_dict[model].lat)):
                    for lon in np.arange(len(hosmip_reg_ds_dict[model].lon)):

                        model_reg_coefs = hosmip_regression_plot(
                            multi_model_dict, season=season, region=None,
                            lat=lat, lon=lon, only_model=model,
                            hosmip_ref_pi=hosmip_ref_pi,
                            hosmip_reg_ds_dict=hosmip_reg_ds_dict,
                            single_model_linear=True, quantile_reg=False,
                            no_plots=True)

                        hosmip_reg_ds_dict[model].lin_coef_hosmip.values[lat, lon, season_idx] = model_reg_coefs['model_coef_lin']
                        hosmip_reg_ds_dict[model].ste_hosmip.values[lat, lon, season_idx] = model_reg_coefs['model_ste_lin']
                        hosmip_reg_ds_dict[model].rsq_hosmip.values[lat, lon, season_idx] = model_reg_coefs['model_rsq']
                        hosmip_reg_ds_dict[model].pvalue_hosmip.values[lat, lon, season_idx] = model_reg_coefs['model_pvalue_lin']
                        i += 1

                        if i % 100 == 0:
                            print(f"{i}/{len(hosmip_reg_ds_dict[model].lat)*len(hosmip_reg_ds_dict[model].lon)} regression coefficients calculated.")

        for model in hosmip_labels:
            for _season in ['', 'djf', 'jja']:
                hosmip_reg_ds_dict[model].T_pd_245.loc[:, :, _season] = xr.concat([
                    cmip6_ctrl_data[model]['tas'].sel(scenar='his', season=_season).sel(time=slice(*PD_HIS_WINDOW)),
                    cmip6_ctrl_data[model]['tas'].sel(scenar='ssp245', season=_season).sel(time=slice(*PD_SSP_WINDOW)),
                ], dim='time').mean(dim='time').values

            for ssp_i in ssps:
                print(f"Calculating required AMOC changes for {model} under {ssp_i}...")

                if np.isnan(cmip6_ctrl_data[model]['amoc'].sel(scenar=ssp_i).sel(time=slice(*fw)).mean(dim='time').values):
                    print(f'Skipping model {model} for scenario {ssp_i} due to missing AMOC data.')
                    continue

                hosmip_reg_ds_dict[model].AMOC_future.loc[ssp_i] = cmip6_ctrl_data[model]['amoc'].sel(scenar=ssp_i).sel(time=slice(*fw)).mean(dim='time').values
                hosmip_reg_ds_dict[model].T_future.loc[:, :, :, ssp_i] = cmip6_ctrl_data[model]['tas'].sel(scenar=ssp_i).sel(time=slice(*fw)).mean(dim='time').values

                hosmip_reg_ds_dict[model].req_strength_pi.loc[:, :, :, ssp_i] = hosmip_reg_ds_dict[model].AMOC_future.loc[ssp_i] - (hosmip_reg_ds_dict[model].T_future.loc[:, :, :, ssp_i] - hosmip_reg_ds_dict[model].T_pi) / (hosmip_reg_ds_dict[model].lin_coef_hosmip.loc[:, :, :].where(hosmip_reg_ds_dict[model].lin_coef_hosmip.loc[:, :, :]<0)*(-100)/hosmip_reg_ds_dict[model].AMOC_pi)

                # req_strength_pd: per-season inversion using the season-aligned
                # T_pd_245 baseline, T_future, and slope. AMOC is annual at
                # 26°N (no seasonal sibling), so AMOC_pi / AMOC_2090
                # are season-invariant; only the temperature terms change.
                hosmip_reg_ds_dict[model].req_strength_pd.loc[:, :, :, ssp_i] = hosmip_reg_ds_dict[model].AMOC_future.loc[ssp_i] - (hosmip_reg_ds_dict[model].T_future.loc[:, :, :, ssp_i] - hosmip_reg_ds_dict[model].T_pd_245) / (hosmip_reg_ds_dict[model].lin_coef_hosmip.loc[:, :, :].where(hosmip_reg_ds_dict[model].lin_coef_hosmip.loc[:, :, :]<0)*(-100)/hosmip_reg_ds_dict[model].AMOC_pi)

                hosmip_reg_ds_dict[model].req_weakening_pi.loc[:, :, :, ssp_i] = (hosmip_reg_ds_dict[model].AMOC_pi - hosmip_reg_ds_dict[model].req_strength_pi.loc[:, :, :, ssp_i]) / hosmip_reg_ds_dict[model].AMOC_pi * 100

                hosmip_reg_ds_dict[model].req_weakening_pd.loc[:, :, :, ssp_i] = (hosmip_reg_ds_dict[model].AMOC_pi - hosmip_reg_ds_dict[model].req_strength_pd.loc[:, :, :, ssp_i]) / hosmip_reg_ds_dict[model].AMOC_pi * 100

        # save to pickle
        for model in hosmip_reg_ds_dict:
            hosmip_reg_ds_dict[model].attrs['future_window'] = list(fw)
        with open(cache_path, 'wb') as f:
            pickle.dump(hosmip_reg_ds_dict, f)
            print(f"HosMIP regression dataset dictionary saved to '{cache_path}'.")
    else:
        # open from to pickle
        with open(cache_path, 'rb') as f:
            hosmip_reg_ds_dict = pickle.load(f)
            print(f"HosMIP regression dataset dictionary loaded from '{cache_path}'.")

    return hosmip_reg_ds_dict


def add_quadratic_to_hosmip_reg_ds(hosmip_ref_pi=True, window=10):
    """Augment the existing HosMIP regression dataset with quadratic coefficients.

    Loads ``hosmip_reg_ds_dict.pkl`` (linear fits), computes quadratic
    regressions (y = c1*x + c2*x²) for every model × season × grid point,
    adds the results as new variables, and saves as
    ``hosmip_reg_ds_dict_v2.pkl``.

    New variables per model Dataset
    -------------------------------
    quad_c1_hosmip : (lat, lon, season)
        Linear coefficient of quadratic fit (°C per % AMOC weakening).
    quad_c2_hosmip : (lat, lon, season)
        Quadratic coefficient (°C per (% AMOC weakening)²).
    quad_ste_c1_hosmip, quad_ste_c2_hosmip : (lat, lon, season)
        Standard errors (HAC / Newey-West).
    quad_rsq_hosmip : (lat, lon, season)
        R² of quadratic fit.
    quad_pvalue_c1_hosmip, quad_pvalue_c2_hosmip : (lat, lon, season)
        p-values of c1 and c2.
    """
    # Load existing linear dataset
    hosmip_reg_ds_dict = get_hosmip_reg_ds(recompute=False)

    # Load multi-model dict for the regression function
    multi_model_dict, _masks = get_full_multi_model_dict()

    # Add quadratic variables to each model's Dataset
    for model in hosmip_labels:
        ds = hosmip_reg_ds_dict[model]
        shape_3d = (len(ds.lat), len(ds.lon), 3)  # 3 seasons
        for vname in ('quad_c1_hosmip', 'quad_c2_hosmip',
                      'quad_ste_c1_hosmip', 'quad_ste_c2_hosmip',
                      'quad_rsq_hosmip',
                      'quad_pvalue_c1_hosmip', 'quad_pvalue_c2_hosmip'):
            ds[vname] = (['lat', 'lon', 'season'],
                         np.full(shape_3d, np.nan))

    # Compute quadratic regressions
    for model in hosmip_labels:
        ds = hosmip_reg_ds_dict[model]
        for season_idx, season in enumerate(['', 'djf', 'jja']):
            print(f"Computing quadratic regression for {model} "
                  f"({'annual' if season == '' else season})...")
            n_done = 0
            n_total = len(ds.lat) * len(ds.lon)
            for lat in np.arange(len(ds.lat)):
                for lon in np.arange(len(ds.lon)):
                    res = hosmip_regression_plot(
                        multi_model_dict, season=season, region=None,
                        lat=lat, lon=lon, only_model=model,
                        hosmip_ref_pi=hosmip_ref_pi,
                        hosmip_reg_ds_dict=hosmip_reg_ds_dict,
                        single_model_linear=False,
                        quantile_reg=False, no_plots=True,
                        window=window)

                    ds.quad_c1_hosmip.values[lat, lon, season_idx] = \
                        res['model_coef_lin']
                    ds.quad_c2_hosmip.values[lat, lon, season_idx] = \
                        res['model_coef_quad']
                    ds.quad_ste_c1_hosmip.values[lat, lon, season_idx] = \
                        res['model_ste_lin']
                    ds.quad_ste_c2_hosmip.values[lat, lon, season_idx] = \
                        res['model_ste_quad']
                    ds.quad_rsq_hosmip.values[lat, lon, season_idx] = \
                        res['model_rsq']
                    ds.quad_pvalue_c1_hosmip.values[lat, lon, season_idx] = \
                        res['model_pvalue_lin']
                    ds.quad_pvalue_c2_hosmip.values[lat, lon, season_idx] = \
                        res['model_pvalue_quad']

                    n_done += 1
                    if n_done % 100 == 0:
                        print(f"  {n_done}/{n_total} grid points done.")

    # Save augmented dataset
    cache_path = local_path + 'hosmip_reg_ds_dict_v2.pkl'
    with open(cache_path, 'wb') as f:
        pickle.dump(hosmip_reg_ds_dict, f)
    print(f"Augmented HosMIP regression dataset saved to '{cache_path}'.")

    return hosmip_reg_ds_dict


def get_cesm_reg_ds(recompute=False, data_dict=None, multi_model_dict=None, masks=None, cmip6_ctrl_data=None, future_window=None):

    fw = resolve_future_window(future_window)
    cache_path = local_path + f'reg_ds_cesm{fw_suffix(future_window)}.nc'
    if recompute or not os.path.exists(cache_path):
        print('Recomputing CESM2 regression dataset...')

        # load the necessary datasets
        print('############################################################')
        print('Loading necessary datasets...')
        print('############################################################')
        multi_model_dict, masks = get_full_multi_model_dict() if multi_model_dict is None or masks is None else (multi_model_dict, masks)
        cmip6_ctrl_data = get_cmip_projections(masks, data_dict=(data_dict if data_dict!=None else None)) if cmip6_ctrl_data is None else cmip6_ctrl_data
        bm_data, vwb_data, boot_data, liu_data = get_other_studies_data(masks)

        print('############################################################')
        print('Calculating CESM2 regression coefficients...')
        print('############################################################')
        static_reg_ds = multi_model_dict['CESM2'].isel(time=0, drop=True).copy(deep=True).drop_vars(['tas', 'amoc', 'mask', 'region', 'type', 'season'])
        static_reg_ds = static_reg_ds.assign_coords(scenar=ssps+['ssp585'])

        reg_ds_cesm = static_reg_ds.assign(
            coef_ensmean=(['lat', 'lon'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),
            ste_ensmean=(['lat', 'lon'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)), # standard error of regression coefficient
            rsq_ensmean=(['lat', 'lon'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)), # R^2 of regression
            coef_ensmean_intercept=(['lat', 'lon'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),
            ste_ensmean_intercept=(['lat', 'lon'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)), # standard error of regression coefficient
            rsq_ensmean_intercept=(['lat', 'lon'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)), # R^2 of regression
            coef_ssp=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
            ste_ssp=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),  # standard error for SSP-specific regressions
            rsq_ssp=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),  # R^2 for SSP-specific regressions
            AMOC_future=(['scenar'], np.full(tuple(static_reg_ds.isel(lat=0, lon=0).sizes.values()), np.nan)),
            AMOC_pi=([], np.full(tuple(static_reg_ds.isel(lat=0, lon=0, scenar=0).sizes.values()), np.nan)),
            T_future=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
            T_pi=(['lat', 'lon'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),
            T_pd_245=(['lat', 'lon'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),
            req_strength_pi=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
            req_strength_pd=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
            req_weakening_pi=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
            req_weakening_pd=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
        )

        # Set the harmonised 1850–1899 CMIP6 historical baselines BEFORE the
        # regression loop, so cesm_regressions runs against the same AMOC_pi
        # that ends up in reg_ds_cesm.AMOC_pi. Cached slopes are then in
        # K-per-(% relative to the harmonised AMOC_pi) — no post-hoc rescale.
        reg_ds_cesm.AMOC_pi.values = get_hist_pi_baseline(
            'CESM2', 'amoc', cmip6_ctrl_data=cmip6_ctrl_data,
            multi_model_dict=multi_model_dict)
        # T_pi shares lat/lon with the cache by construction (see HosMIP
        # loader comment). Direct .values assignment after a positional
        # transpose-free fetch.
        reg_ds_cesm.T_pi.values[:, :] = get_hist_pi_baseline(
            'CESM2', 'tas', cmip6_ctrl_data=cmip6_ctrl_data,
            multi_model_dict=multi_model_dict, season='').values
        _cesm_amoc_pi_for_regression = float(reg_ds_cesm.AMOC_pi.values)

        i=0
        for lat in np.arange(len(static_reg_ds.lat)):
            for lon in np.arange(len(static_reg_ds.lon)):
                reg_quadruples = cesm_regressions(boot_data, multi_model_dict, masks, ssp='all', lat=lat, lon=lon, region=None, combined_reg=True, no_plots=True, add_combined_intercept=False, cesm_amoc_pi_override=_cesm_amoc_pi_for_regression)
                reg_quadruples_intercept = cesm_regressions(boot_data, multi_model_dict, masks, ssp='all', lat=lat, lon=lon, region=None, combined_reg=True, no_plots=True, add_combined_intercept=True, cesm_amoc_pi_override=_cesm_amoc_pi_for_regression)
                reg_ds_cesm.coef_ensmean.values[lat, lon] = reg_quadruples['combined'][1]
                reg_ds_cesm.ste_ensmean.values[lat, lon] = reg_quadruples['combined'][2]
                reg_ds_cesm.rsq_ensmean.values[lat, lon] = reg_quadruples['combined'][3]
                reg_ds_cesm.coef_ensmean_intercept.values[lat, lon] = reg_quadruples_intercept['combined'][1]
                reg_ds_cesm.ste_ensmean_intercept.values[lat, lon] = reg_quadruples_intercept['combined'][2]
                reg_ds_cesm.rsq_ensmean_intercept.values[lat, lon] = reg_quadruples_intercept['combined'][3]
                # SSP-specific: tuple is (coef, ste, rsq)
                reg_ds_cesm.coef_ssp.values[lat, lon, 0] = reg_quadruples['ssp126'][0]
                reg_ds_cesm.ste_ssp.values[lat, lon, 0] = reg_quadruples['ssp126'][1]
                reg_ds_cesm.rsq_ssp.values[lat, lon, 0] = reg_quadruples['ssp126'][2]
                reg_ds_cesm.coef_ssp.values[lat, lon, 3] = reg_quadruples['ssp585'][0]
                reg_ds_cesm.ste_ssp.values[lat, lon, 3] = reg_quadruples['ssp585'][1]
                reg_ds_cesm.rsq_ssp.values[lat, lon, 3] = reg_quadruples['ssp585'][2]
                i += 1

                if i % 100 == 0:
                    print(f"{i}/{len(static_reg_ds.lat)*len(static_reg_ds.lon)} regression coefficients calculated.")

        # AMOC_pi and T_pi were set before the regression loop (above) so the
        # cached slopes are intrinsically consistent with reg_ds_cesm.AMOC_pi.
        # reg_ds_cesm.T_pd_245.loc[:, :] = xr.concat([.sel(time=slice('2000', '2014')),
        #                                             cmip6_ctrl_data['CESM2']['tas'].sel(scenar='ssp245', season='').sel(time=slice('2015', '2029'))], dim='time').mean(dim='time').values
        reg_ds_cesm.T_pd_245.loc[:, :] = xr.concat([cmip6_ctrl_data['CESM2']['tas'].sel(scenar='his', season='').sel(time=slice(*PD_HIS_WINDOW)), cmip6_ctrl_data['CESM2']['tas'].sel(scenar='ssp245', season='').sel(time=slice(*PD_SSP_WINDOW))], dim='time').mean(dim='time').values

        for ssp_i in ssps:
            reg_ds_cesm.AMOC_future.loc[ssp_i] = cmip6_ctrl_data['CESM2']['amoc'].sel(scenar=ssp_i).sel(time=slice(*fw)).mean(dim='time').values
            reg_ds_cesm.T_future.loc[:, :, ssp_i] = cmip6_ctrl_data['CESM2']['tas'].sel(scenar=ssp_i, season='').sel(time=slice(*fw)).mean(dim='time').values

            reg_ds_cesm.req_strength_pi.loc[:, :, ssp_i] = reg_ds_cesm.AMOC_future.loc[ssp_i] - (reg_ds_cesm.T_future.loc[:, :, ssp_i] - reg_ds_cesm.T_pi) / (reg_ds_cesm.coef_ensmean*(-100)/reg_ds_cesm.AMOC_pi)
            reg_ds_cesm.req_strength_pd.loc[:, :, ssp_i] = reg_ds_cesm.AMOC_future.loc[ssp_i] - (reg_ds_cesm.T_future.loc[:, :, ssp_i] - reg_ds_cesm.T_pd_245) / (reg_ds_cesm.coef_ensmean*(-100)/reg_ds_cesm.AMOC_pi)
            reg_ds_cesm.req_weakening_pi.loc[:, :, ssp_i] = (reg_ds_cesm.AMOC_pi - reg_ds_cesm.req_strength_pi.loc[:, :, ssp_i]) / reg_ds_cesm.AMOC_pi * 100
            reg_ds_cesm.req_weakening_pd.loc[:, :, ssp_i] = (reg_ds_cesm.AMOC_pi - reg_ds_cesm.req_strength_pd.loc[:, :, ssp_i]) / reg_ds_cesm.AMOC_pi * 100

        reg_ds_cesm.attrs['future_window'] = list(fw)
        print(f'Saving CESM2 regression dataset to {cache_path}')
        reg_ds_cesm.to_netcdf(cache_path)
    else:
        print(f'Loading CESM2 regression dataset from {cache_path}')
        reg_ds_cesm = xr.open_dataset(cache_path)
    return reg_ds_cesm


def load_giss_member_amoc_tas(season=''):
    """Load 10 GISS ssp245 members (his + ssp245-to-2500) on the native
    90x144 grid. Returns (amoc, tas) DataArrays with a ``realiz``
    dim of length 10.

    For ``season=''`` (annual): historical tas comes from the uo1075
    legacy yearly cache, future tas from the bu1431 ``_tas_yr_to2500.nc``
    extension. For ``season in ('djf','jja')``: historical tas comes
    from the bu1431 historical-leg seasonal staging
    (``tas_{season}/giss-e2-1-g_{r}_tas_{season}.nc``, written by
    ``process_seasonal_tas.py``), and future tas from the bu1431 SSP
    seasonal extension (``_tas_{season}_to2500.nc``, written by
    ``process_giss_seasonal_to2500.py``).

    AMOC at 26°N has no seasonal sibling — seasonal regression
    pairs seasonal tas with annual AMOC. Returned ``amoc`` is therefore
    identical regardless of ``season``.

    Single source of truth for the historical+future concatenation;
    used by get_giss_reg_ds and get_giss_panel_data.
    """
    validate_choice('season', season, ALLOWED_SEASONS)
    hist_dir = '/work/uo1075/m300817/teu_amoc/data/CMIP6/historical/giss-e2-1-g'
    hist_seasonal_dir = ('/work/bu1431/T_EU_AMOC/CMIP6/historical/'
                         'giss-e2-1-g')
    amoc_list, tas_list = [], []
    for r in GISS_MEMBERS:
        # AMOC: always annual.
        amoc_h = xr.open_dataarray(
            f'{hist_dir}/giss-e2-1-g_{r}_amoc26_yr.nc', use_cftime=True)
        amoc_f = xr.open_dataarray(
            f'{GISS_PROCESSED_PATH}giss-e2-1-g_{r}_amoc26_yr_to2500.nc',
            use_cftime=True)
        # tas: route through season-specific paths.
        if season == '':
            tas_h_path = f'{hist_dir}/giss-e2-1-g_{r}_tas_yr.nc'
            tas_f_path = f'{GISS_PROCESSED_PATH}giss-e2-1-g_{r}_tas_yr_to2500.nc'
        else:
            tas_h_path = (f'{hist_seasonal_dir}/tas_{season}/'
                          f'giss-e2-1-g_{r}_tas_{season}.nc')
            tas_f_path = (f'{GISS_PROCESSED_PATH}giss-e2-1-g_{r}_'
                          f'tas_{season}_to2500.nc')
        tas_h = xr.open_dataarray(tas_h_path, use_cftime=True)
        tas_f = xr.open_dataarray(tas_f_path, use_cftime=True)
        amoc_list.append(xr.concat(
            [amoc_h, amoc_f.sel(time=slice('2015', None))], dim='time'
        ).assign_coords(realiz=r))
        tas_list.append(xr.concat(
            [tas_h, tas_f.sel(time=slice('2015', None))], dim='time'
        ).assign_coords(realiz=r))
    amoc = xr.concat(amoc_list, dim='realiz')
    tas = xr.concat(tas_list, dim='realiz')
    return amoc.drop_vars('height', errors='ignore'), tas.drop_vars('height', errors='ignore')


def fit_member_minus_composite(amoc_dev, tas_dev, members, time_slice,
                                add_intercept=True, window=10):
    """HAC-OLS on Δtas vs ΔAMOC, where Δ = member − composite.

    Each (member, year) point is one observation, pooled into one flat
    sample. Covariance: HAC with maxlags=window-1, matching the
    Boot et al. (2024) treatment in hosmip_regression_plot. Cluster-
    robust SE is not appropriate here — every member shares the same
    external forcing, so within-member dependence is autocorrelation,
    not a treatment shock.
    """
    xs, ys, clusters = [], [], []
    for r in members:
        x = amoc_dev.sel(realiz=r, time=slice(*time_slice)).values
        y = tas_dev.sel(realiz=r, time=slice(*time_slice)).values
        keep = np.isfinite(x) & np.isfinite(y)
        xs.append(x[keep]); ys.append(y[keep]); clusters.append(np.full(int(keep.sum()), r))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    clust = np.concatenate(clusters)
    X = sm.add_constant(x) if add_intercept else x
    model = sm.OLS(y, X).fit(
        cov_type='HAC',
        cov_kwds={'maxlags': max(window - 1, 1)},
    )
    return model, x, y, clust


def get_giss_panel_data(masks=None, recompute=False,
                        regions=('EU', 'NEU', 'WCE', 'MED'),
                        time_slice=('2015', '2500'),
                        window=ROLLING_WINDOW, season=''):
    """Build (or load) the per-region GISS member-minus-composite deviation
    panel used by hosmip_regression_plot's giss_data block.

    Pools (non_composite_member × year) into a flat ``point`` sample.
    AMOC deviation is scalar (no region dim); T deviation gets one
    column per region.

    ``season`` selects which tas series is loaded (``''`` annual,
    ``'djf'`` / ``'jja'``); AMOC stays annual. Caches are per-season:
    ``reg_ds_giss_panel.nc`` for annual (back-compat), ``reg_ds_giss_panel_djf.nc``
    / ``reg_ds_giss_panel_jja.nc`` for seasonal.

    Dataset layout::

        coords:
            point   (flat (non_composite_member, year) index)
            region  [EU, NEU, WCE, MED]
            member(point), year(point)
        vars:
            amoc_dev_sv(point)         ΔAMOC in Sv  (member − composite)
            amoc_dev_pct(point)        = amoc_dev_sv / (-AMOC_pi_GISS) * 100
            t_dev(point, region)       ΔT_region in K (cos-lat land-only mean)

    Cache is invalidated when the requested time_window /
    rolling_window / signal / regions do not match the saved attrs.

    `masks` must come from get_full_multi_model_dict() (longitude-
    wrapped to [-180, 180]); raw hosmip_masks.pkl will not work.
    """
    validate_choice('season', season, ALLOWED_SEASONS)
    panel_suffix = f'_{season}' if season else ''
    panel_path = local_path + f'reg_ds_giss_panel{panel_suffix}.nc'
    cache_signal = (f'member - composite (ssp245, season={season or "annual"})')
    if (not recompute) and os.path.exists(panel_path):
        ds = xr.open_dataset(panel_path)
        if (ds.attrs.get('time_window') == f'{time_slice[0]}-{time_slice[1]}'
                and int(ds.attrs.get('rolling_window', -1)) == window
                and ds.attrs.get('signal') == cache_signal
                and set(ds.region.values.tolist()) >= set(regions)
                and ds.attrs.get('lon_wrap_fixed') == 'True'):
            print(f'Loading GISS panel data from {panel_path}')
            return ds
        ds.close()
        print('GISS panel cache config mismatch (or pre-2026-05-29 (b) '
              'lon-wrap bug), recomputing')

    if masks is None:
        _, masks = get_full_multi_model_dict()
    assert 'GISS-E2-1-G' in masks, (
        'masks must come from get_full_multi_model_dict (lon-wrapped); '
        f'available keys: {sorted(masks.keys())}')

    print(f'Building GISS panel data (member − composite, per region, '
          f'season={season or "annual"})...')
    amoc, tas = load_giss_member_amoc_tas(season=season)
    giss_masks = masks['GISS-E2-1-G']

    # GISS masks are on [-180, 180] but raw tas is on [0, 360].
    # Wrap tas to match the mask
    # convention; otherwise western-hemisphere cells silently drop. This
    # affected EU/NEU/WCE/MED region means in every prior cache write
    # (Portugal, Ireland, western WCE were missing) — invalidates the
    # existing reg_ds_giss_panel*.nc caches.
    tas = tas.assign_coords(
        lon=(((tas.lon + 180) % 360) - 180)).sortby('lon')

    t_region = {
        r: weighted_area_lat(
            tas.where(giss_masks[r]).where(giss_masks['LAND'] == 0)
        ).mean(('lat', 'lon'))
        for r in regions
    }
    amoc_comp = amoc.sel(realiz=GISS_COMPOSITE).mean('realiz')
    t_region_comp = {r: t_region[r].sel(realiz=GISS_COMPOSITE).mean('realiz')
                     for r in regions}
    amoc_dev = (amoc - amoc_comp).rolling(time=window, center=True).mean()
    t_region_dev = {r: (t_region[r] - t_region_comp[r])
                    .rolling(time=window, center=True).mean()
                    for r in regions}
    amoc_dev = amoc_dev.sel(realiz=GISS_NON_COMPOSITE,
                            time=slice(*time_slice))
    t_region_dev = {r: arr.sel(realiz=GISS_NON_COMPOSITE,
                               time=slice(*time_slice))
                    for r, arr in t_region_dev.items()}

    xs, ys_r, mem_l, yr_l = [], {r: [] for r in regions}, [], []
    for m in GISS_NON_COMPOSITE:
        x_m = amoc_dev.sel(realiz=m).values
        yr_vals = amoc_dev.sel(realiz=m).time.dt.year.values
        ys_m = {r: t_region_dev[r].sel(realiz=m).values for r in regions}
        keep = np.isfinite(x_m)
        for r in regions:
            keep = keep & np.isfinite(ys_m[r])
        xs.append(x_m[keep])
        for r in regions:
            ys_r[r].append(ys_m[r][keep])
        mem_l.append(np.full(int(keep.sum()), m))
        yr_l.append(yr_vals[keep])
    amoc_dev_sv = np.concatenate(xs)
    amoc_dev_pct = amoc_dev_sv / (-AMOC_pi_GISS) * 100
    t_dev = np.stack([np.concatenate(ys_r[r]) for r in regions], axis=1)
    mem_flat = np.concatenate(mem_l)
    yr_flat = np.concatenate(yr_l).astype(int)

    # Sign-convention sanity: r1 is known weaker than the (r3,r4,r10)
    # composite, so its median ΔAMOC %-weakening must come out positive.
    r1_med = float(np.nanmedian(amoc_dev_pct[mem_flat == 'r1i1p1f2']))
    assert r1_med > 0, (
        f'Unexpected sign convention: median(amoc_dev_pct[r1]) = '
        f'{r1_med:.2f}; expected > 0 since r1 is weaker than the '
        f'(r3,r4,r10) composite.')

    ds = xr.Dataset(
        data_vars={
            'amoc_dev_sv':  (('point',), amoc_dev_sv,
                {'units': 'Sv',
                 'long_name': 'ΔAMOC (member − composite)'}),
            'amoc_dev_pct': (('point',), amoc_dev_pct,
                {'units': '%',
                 'long_name': 'ΔAMOC as % weakening relative to AMOC_pi_GISS'}),
            't_dev':        (('point', 'region'), t_dev,
                {'units': 'K',
                 'long_name': 'ΔT_region (member − composite), '
                              'cos-lat-weighted land-only mean'}),
            'member':       (('point',), mem_flat),
            'year':         (('point',), yr_flat),
        },
        coords={'region': list(regions)},
        attrs={
            'model': 'GISS-E2-1-G',
            'reference': 'Romanou et al. 2023, JCLI (long-run ssp245 ensemble)',
            'signal': cache_signal,
            'members_pooled': ', '.join(GISS_NON_COMPOSITE),
            'composite_members': ', '.join(GISS_COMPOSITE),
            'time_window': f'{time_slice[0]}-{time_slice[1]}',
            'rolling_window': window,
            'amoc_pi_giss_sv': float(AMOC_pi_GISS),
            'cov_type_when_fit': 'HAC, maxlags=window-1',
            'lon_wrap_fixed': 'True',
        },
    )
    ds = ds.drop_vars('height', errors='ignore')
    print(f'Saving GISS panel data to {panel_path}: sizes={dict(ds.sizes)}')
    ds.to_netcdf(panel_path)
    return ds


def giss_compute_gridded_regression(amoc_anom, tas_anom, members, time_slice,
                                      window, composite_members):
    """Per-cell HAC-OLS on (member - composite) deviations.

    Returns an xr.Dataset on (lat, lon) with slope_K_per_Sv, ste_K_per_Sv,
    rsq, n_obs. Pools (year x non-composite-member) into one regression
    per grid cell. Used both by the full-grid GISS Fig2 cache (per time
    window) and by get_giss_reg_ds (one call per GISS_TIME_PERIODS entry).

    Covariance: HAC, maxlags=window-1. Same convention as
    fit_member_minus_composite and the Boot et al. (2024) overlay in
    hosmip_regression_plot. Cluster-robust SE is not appropriate here
    because every member shares the same external forcing.
    """
    tas_comp = tas_anom.sel(realiz=composite_members).mean('realiz')
    amoc_comp = amoc_anom.sel(realiz=composite_members).mean('realiz')
    tas_dev_s = (tas_anom - tas_comp).rolling(time=window, center=True).mean()
    amoc_dev_s = (amoc_anom - amoc_comp).rolling(time=window, center=True).mean()
    tas_p = tas_dev_s.sel(time=slice(*time_slice), realiz=members)
    amoc_p = amoc_dev_s.sel(time=slice(*time_slice), realiz=members)

    x_2d = amoc_p.transpose('time', 'realiz').values
    y_4d = tas_p.transpose('time', 'realiz', 'lat', 'lon').values
    T, R = x_2d.shape
    _, _, nlat, nlon = y_4d.shape
    # Pool by stacking members along the time axis so HAC's maxlags
    # treats each member's autocorrelation correctly within its block;
    # the resulting flat sample has length T*R.
    x_flat = x_2d.transpose(1, 0).reshape(-1)
    y_flat = y_4d.transpose(1, 0, 2, 3).reshape(T * R, nlat, nlon)

    slope = np.full((nlat, nlon), np.nan, dtype=np.float32)
    ste = np.full_like(slope, np.nan, dtype=np.float32)
    rsq = np.full_like(slope, np.nan, dtype=np.float32)
    n_obs = np.zeros((nlat, nlon), dtype=np.int32)

    print(f'  GISS gridded fit {time_slice[0]}-{time_slice[1]}: '
          f'{nlat * nlon} cells, pool size T*R={T * R}', flush=True)
    x_finite = np.isfinite(x_flat)
    hac_maxlags = max(window - 1, 1)
    for i in range(nlat):
        if i % 10 == 0:
            print(f'    row {i}/{nlat}', flush=True)
        for j in range(nlon):
            y_ij = y_flat[:, i, j]
            keep = x_finite & np.isfinite(y_ij)
            if keep.sum() < 30:
                continue
            try:
                m = sm.OLS(y_ij[keep], sm.add_constant(x_flat[keep])).fit(
                    cov_type='HAC',
                    cov_kwds={'maxlags': hac_maxlags})
                slope[i, j] = m.params[1]
                ste[i, j] = m.bse[1]
                rsq[i, j] = m.rsquared
                n_obs[i, j] = int(m.nobs)
            except Exception:
                pass

    return xr.Dataset(
        data_vars={
            'slope_K_per_Sv': (('lat', 'lon'), slope,
                {'units': 'K Sv-1',
                 'long_name': 'OLS slope of ΔT on ΔAMOC'}),
            'ste_K_per_Sv':   (('lat', 'lon'), ste,
                {'units': 'K Sv-1',
                 'long_name': f'HAC SE (maxlags={hac_maxlags})'}),
            'rsq':            (('lat', 'lon'), rsq,
                {'long_name': 'OLS R²'}),
            'n_obs':          (('lat', 'lon'), n_obs,
                {'long_name': 'pooled (member, year) count per cell'}),
        },
        coords={'lat': tas_anom.lat, 'lon': tas_anom.lon},
    )


def giss_load_or_compute_gridded(amoc_anom, tas_anom, members=None,
                                   composite_members=None, time_slice=None,
                                   window=ROLLING_WINDOW, force_recompute=False,
                                   season=''):
    """Return the GISS full-grid gridded regression Dataset for ``time_slice``.

    Uses an existing bu1431 cache (giss-e2-1-g_reg_ds_tas_{start}-{end}_
    window-{w}[_{season}].nc) if present and its attrs match the requested
    config; otherwise computes and writes one. Used by Fig2 (gridded map
    of scaling factors) and by get_giss_reg_ds.

    The annual cache has no ``_{season}`` suffix (back-compat with the
    existing on-disk file). Seasonal caches add ``_djf`` / ``_jja``.
    The returned Dataset always carries a singleton ``season`` dim whose
    label matches the requested ``season`` so callers can concat across
    seasons without ambiguity.
    """
    def _giss_gridded_cache_path(time_slice, window=ROLLING_WINDOW, season=''):
        suffix = f'_{season}' if season else ''
        return (f'{GISS_PROCESSED_PATH}giss-e2-1-g_reg_ds_tas_'
                f'{time_slice[0]}-{time_slice[1]}_window-{window}{suffix}.nc')

    validate_choice('season', season, ALLOWED_SEASONS)
    if members is None:
        members = GISS_NON_COMPOSITE
    if composite_members is None:
        composite_members = GISS_COMPOSITE
    if time_slice is None:
        time_slice = ('2015', '2500')
    cache_path = _giss_gridded_cache_path(time_slice, window, season=season)
    expected_cov = f'HAC, maxlags={max(window - 1, 1)}'
    if (not force_recompute) and os.path.exists(cache_path):
        ds = xr.open_dataset(cache_path)
        if (ds.attrs.get('regression_window') == f'{time_slice[0]}-{time_slice[1]}'
                and int(ds.attrs.get('rolling_window', -1)) == window
                and ds.attrs.get('cov_type') == expected_cov):
            print(f'gridded regression: using cache {cache_path}')
            ds = ds.drop_vars('height', errors='ignore')
            return ds.expand_dims(season=[season]) if 'season' not in ds.dims else ds
        ds.close()
        print(f'gridded regression: cache config mismatch (cov_type or window), recomputing')

    print(f'gridded regression: computing → {cache_path}')
    ds = giss_compute_gridded_regression(
        amoc_anom, tas_anom, members=members, time_slice=time_slice,
        window=window, composite_members=composite_members)
    ds = ds.drop_vars('height', errors='ignore')
    ds.attrs.update({
        'regression_window': f'{time_slice[0]}-{time_slice[1]}',
        'rolling_window': window,
        'members_pooled': ', '.join(members),
        'composite_members': ', '.join(composite_members),
        'amoc_pi_giss_sv': float(AMOC_pi_GISS),
        'cov_type': f'HAC, maxlags={max(window - 1, 1)}',
        'season': season,
    })
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    ds.to_netcdf(cache_path)
    print(f'wrote {cache_path}')
    return ds.expand_dims(season=[season])


def giss_regressions(amoc, tas, masks=None, region='EU', lat=None, lon=None,
                     time_period=('2101', '2300'), season='',
                     window=ROLLING_WINDOW, members=None, composite_members=None,
                     no_plots=True, ext_ax=None, plot_bg='white',
                     add_combined_intercept=True, compute_linearity=False,
                     xlim=None, low_ylim=None):
    """GISS-E2-1-G analogue of ``cesm_regressions``.

    Fits the pooled member-minus-composite regression for either a region
    (cos-lat-weighted land-only mean over ``masks['GISS-E2-1-G'][region]``)
    or a single (lat, lon) pixel. Pools across the seven non-composite
    members over the requested ``time_period`` window. Uses HAC covariance
    with ``maxlags = window - 1`` (the standing GISS convention; see
    ``fit_member_minus_composite`` and ``giss_compute_gridded_regression``).

    ``amoc, tas`` should be the raw outputs of
    ``load_giss_member_amoc_tas(season=season)`` — ``amoc`` is always
    annual; ``tas`` is the season the user asked for. Composite deviations
    are computed internally.

    When ``compute_linearity=True`` the function also calls
    ``reset_diagnostics(..., cov_spec=('HAC', {'maxlags': window-1}))``
    and adds the RESET diagnostic keys to ``giss_regressions.last_call``.

    ``last_call`` keys mirror ``cesm_regressions.last_call`` so the
    figure script can read either with identical consumer code:
    ``combined_x``, ``combined_y``, ``combined_scenario_id`` (all zeros
    here — G=1; kept for parallelism), ``combined_scenario_name``
    (``'ssp245'`` everywhere), ``combined_member_id``,
    ``combined_intercept``, ``combined_coef``, ``combined_ste``,
    ``combined_rsq``, ``n_combined``, ``window``.

    Returns a small dict {'combined': (intercept, coef, ste, rsq)} for
    symmetry with the other model wrappers; the rich payload lives on
    ``last_call``.
    """
    validate_choice('season', season, ALLOWED_SEASONS)
    validate_choice('plot_bg', plot_bg, ALLOWED_PLOT_BG)
    if members is None:
        members = GISS_NON_COMPOSITE
    if composite_members is None:
        composite_members = GISS_COMPOSITE

    is_latlon = (lat is not None) and (lon is not None)
    is_region = (not is_latlon) and (region is not None)
    if not (is_latlon or is_region):
        raise ValueError('giss_regressions: pass region=... or (lat, lon)')

    # Composite + deviations on AMOC (scalar timeseries per member).
    amoc_comp = amoc.sel(realiz=composite_members).mean('realiz')
    amoc_dev = (amoc - amoc_comp).rolling(time=window, center=True).mean()

    # Composite + deviations on tas for the requested region or pixel.
    if is_latlon:
        tas_target = tas.isel(lat=lat, lon=lon)
    else:
        if masks is None:
            raise ValueError('giss_regressions: masks required for region=...')
        giss_masks = masks['GISS-E2-1-G']
        # GISS masks live on [-180, 180] but raw tas is on [0, 360].
        # Wrap tas to match
        # the mask convention before .where() — otherwise western-
        # hemisphere cells (e.g. 'IS' Iceland at lon -21:-16) silently
        # drop with all-NaN results.
        tas_wrapped = tas.assign_coords(
            lon=(((tas.lon + 180) % 360) - 180)).sortby('lon')
        tas_target = weighted_area_lat(
            tas_wrapped.where(giss_masks[region]).where(giss_masks['LAND'] == 0)
        ).mean(('lat', 'lon'))
    tas_comp = tas_target.sel(realiz=composite_members).mean('realiz')
    tas_dev = (tas_target - tas_comp).rolling(time=window, center=True).mean()

    # Pooled HAC OLS on the seven non-composite members within time_period.
    try:
        model, x, y, clust = fit_member_minus_composite(
            amoc_dev, tas_dev, members=members, time_slice=time_period,
            add_intercept=True, window=window,
        )
        _fit_ok = bool(np.isfinite(model.params).all())
    except Exception:
        _fit_ok = False
    if not _fit_ok:
        # Populate last_call with a fully-keyed failure state so consumers
        # can detect via np.size(combined_x)==0 or np.isnan(combined_coef)
        # without KeyError (common for small region masks like 'IS' on the
        # native 90×144 GISS grid where weighted_area_lat returns all-NaN).
        giss_regressions.last_call = {
            'combined_x': np.array([], dtype=float),
            'combined_y': np.array([], dtype=float),
            'combined_scenario_id': np.array([], dtype=int),
            'combined_scenario_name': np.array([], dtype='<U10'),
            'combined_member_id': np.array([], dtype=int),
            'combined_coef': np.nan,
            'combined_intercept': np.nan,
            'combined_ste': np.nan,
            'combined_rsq': np.nan,
            'n_combined': 0,
            'window': int(window),
            'time_period': f'{time_period[0]}-{time_period[1]}',
            'fit_ok': False,
        }
        return {'combined': (np.nan, np.nan, np.nan, np.nan)}

    combined_intercept = float(model.params[0])
    combined_coef = float(model.params[1])
    combined_ste = float(model.bse[1])
    combined_rsq = float(model.rsquared)
    combined_x_with_const = sm.add_constant(x)

    # Encode scenario+member ids analogously to cesm_regressions. With G=1
    # the scenario id collapses to a single value; the member id provides
    # the within-scenario grouping but is NOT used as a cluster (HAC, not
    # cluster-robust).
    n = len(y)
    combined_scenario_id = np.zeros(n, dtype=int)
    combined_scenario_name = np.array(['ssp245'] * n)
    # Map member string -> int id for compactness (HAC ignores; useful for
    # downstream consumers that may want to group/colour by member).
    _u, _inv = np.unique(np.asarray(clust), return_inverse=True)
    combined_member_id = _inv.astype(int)

    giss_regressions.last_call = {
        'combined_x': np.asarray(x),
        'combined_y': np.asarray(y),
        'combined_scenario_id': combined_scenario_id,
        'combined_scenario_name': combined_scenario_name,
        'combined_member_id': combined_member_id,
        'combined_coef': combined_coef,
        'combined_intercept': combined_intercept,
        'combined_ste': combined_ste,
        'combined_rsq': combined_rsq,
        'n_combined': int(n),
        'window': int(window),
        'time_period': f'{time_period[0]}-{time_period[1]}',
        'fit_ok': True,
    }

    _hac_spec = ('HAC', {'maxlags': max(window - 1, 1)})

    if compute_linearity:
        _diag = reset_diagnostics(
            y, combined_x_with_const, combined_member_id, model,
            cov_spec=_hac_spec,
        )
        giss_regressions.last_call.update({
            'partial_r2': _diag['partial_r2'],
            'delta_r2_total': _diag['delta_r2_total'],
            'gamma2': _diag['gamma2'],
            'gamma3': _diag['gamma3'],
            'beta2_sign': _diag['beta2_sign'],
            'beta3_sign': _diag['beta3_sign'],
            'reset_pvalue': _diag['reset_pvalue'],
            'baseline_r2': _diag['baseline_r2'],
            'combined_yhat': _diag['yhat'],
            'combined_resid': _diag['resid'],
            'combined_intercept_aug': _diag['intercept_aug'],
            'combined_coef_aug': _diag['coef_aug'],
        })

    # Optional plotting block — kept lean; matches cesm_regressions style.
    if not no_plots:
        if ext_ax is not None:
            ax = ext_ax
        else:
            plt.style.use('default')
            if plot_bg == 'black':
                plt.style.use('dark_background')
                plt.rcParams['axes.facecolor'] = '#191919'
                plt.rcParams['figure.facecolor'] = '#191919'
            _, ax = plt.subplots(figsize=(9, 6))

        fg = 'black' if plot_bg != 'black' else 'white'
        # Colour points by member, with all members sharing the ssp245
        # marker convention. Use a single accent colour from the project
        # palette for visual consistency with the CESM panel d.
        try:
            cl = hosing_colors['ssp245']['ge']
        except KeyError:
            cl = '#999999'
        ax.scatter(x, y, s=15, c=[cl], alpha=0.5,
                   label=f'GISS-E2-1-G ssp245 (members − composite)')

        # Linear fit line + 95% CI band over the data range.
        x_range = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
        x_range_with_const = sm.add_constant(x_range)
        y_pred = model.predict(x_range_with_const)
        conf = model.get_prediction(x_range_with_const).conf_int(alpha=0.05)
        ax.plot(x_range, y_pred, color=fg, lw=1.5, ls='-')
        ax.fill_between(x_range, conf[:, 0], conf[:, 1], color=fg, alpha=0.2)

        p_value = model.pvalues[1]
        sig = '***' if p_value < 0.001 else ('**' if p_value < 0.01
              else ('*' if p_value < 0.05 else ''))
        # Simple ŷ = a·x + b form (Fig1 simple_eqs). GISS x is in Sv, so the
        # slope unit is °C/Sv (not °C/10% as in the %-axis CESM/MPI panels).
        slope_frac = r'$\frac{{}^{\circ}\mathrm{C}}{\mathrm{Sv}}$'
        isign = '+' if combined_intercept >= 0 else '-'
        eq = fr"$\hat{{y}}$ = {combined_coef:.3f}$^{{{{\text{{{sig}}}}}}}$ {slope_frac} $\cdot$ x {isign} {abs(combined_intercept):.2f} $^{{\circ}}$C"
        ax.text(0.05, 0.10, f'Combined: {eq}', transform=ax.transAxes,
                fontsize=12, va='top', color=fg)

        ax.axhline(0, color=fg, lw=1.0, ls='--')
        ax.axvline(0, color=fg, lw=1.0, ls='--')
        ax.set_xlabel('AMOC deviation from composite [Sv]')
        if is_latlon:
            ax.set_ylabel(f'Dev. from composite lat={lat}, lon={lon} temp. [°C]')
        else:
            ax.set_ylabel(f'Dev. from composite {region} temp. [°C]')
        if xlim is not None:
            ax.set_xlim(xlim)
        if low_ylim is not None:
            ax.set_ylim(low_ylim, abs(low_ylim) * 0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=10, frameon=False, loc='center left')

    return {'combined': (np.round(combined_intercept, 4),
                         np.round(combined_coef, 5),
                         np.round(combined_ste, 5),
                         np.round(combined_rsq, 4))}


def get_giss_reg_ds(recompute=False, masks=None, cmip6_ctrl_data=None, future_window=None):
    """Build (or load) the GISS-E2-1-G gridded regression dataset for Fig3.

    Per-cell cluster-robust OLS slope of (ΔT, ΔAMOC) deviations from the
    3-member composite mean (r3, r4, r10), pooled over the 7 non-composite
    members and the years in each ``GISS_TIME_PERIODS`` window. Single
    SSP source for the slope (ssp245 long-run extension), but cross-SSP
    projection fields (T_future, AMOC_future) pulled from
    ``get_cmip_projections``, so ``req_weakening_{pi,pd}`` can be
    computed for ssp126/ssp245/ssp370 against the same slope.

    Dataset layout mirrors ``reg_ds_cesm`` with two extras: a
    ``time_period`` dim for the selectable fit window, and a singleton
    ``season`` dim that leaves room for djf/jja regressions later.

    Covariance: HAC, maxlags=window-1 (see
    ``giss_compute_gridded_regression``). Point estimate equals the
    OLS slope; reported SE uses HAC because within-member dependence
    is autocorrelation (every member shares the same ssp245 forcing),
    not a treatment shock.
    """
    fw = resolve_future_window(future_window)
    cache_path = local_path + f'reg_ds_giss{fw_suffix(future_window)}.nc'
    if (not recompute) and os.path.exists(cache_path):
        print(f'Loading GISS-E2-1-G regression dataset from {cache_path}')
        return xr.open_dataset(cache_path)

    print('Recomputing GISS-E2-1-G regression dataset...')
    if masks is None:
        _, masks = get_full_multi_model_dict()
    if cmip6_ctrl_data is None:
        cmip6_ctrl_data = get_cmip_projections(masks)

    src = cmip6_ctrl_data['GISS-E2-1-G']
    period_labels = [f'{a}-{b}' for (a, b) in GISS_TIME_PERIODS]
    AMOC_2090 = src.amoc.sel(time=slice(*fw)).mean('time')    # (scenar,)
    AMOC_pi = xr.DataArray(AMOC_pi_GISS)

    # ---- Per-season build: regression on seasonal tas vs annual AMOC.
    # AMOC at 26°N has no seasonal sibling; the same annual ΔAMOC drives
    # all three seasonal ΔT regressions.
    # For T_pi / T_pd_245 / T_future, the annual path uses
    # cmip6_ctrl_data['GISS-E2-1-G'].tas (already cropped to EU_buffer and
    # realisation-meaned by get_cmip_projections). For seasonal we derive
    # those fields directly from load_giss_member_amoc_tas(season=...)
    # output and then crop to the canonical EU_buffer grid — GISS ssp245
    # seasonal yearly files are not staged in the
    # /work/uo1075/.../ssp245/giss-e2-1-g/tas_{djf,jja}/ convention that
    # get_cmip_projections expects, so its seasonal ssp245 slot stays NaN.
    per_season_ds = []
    for _season in ['', 'djf', 'jja']:
        print(f'  GISS regression: season={_season or "annual"}')
        amoc_s, tas_s = load_giss_member_amoc_tas(season=_season)
        per_period = [
            giss_load_or_compute_gridded(
                amoc_s, tas_s, time_slice=ts, season=_season).isel(season=0)
            for ts in GISS_TIME_PERIODS
        ]
        grid_full = xr.concat(
            per_period, dim=pd.Index(period_labels, name='time_period'))
        coef_full = grid_full.slope_K_per_Sv
        ste_full  = grid_full.ste_K_per_Sv
        rsq_full  = grid_full.rsq

        # Reproject coef/ste/rsq onto the cropped grid via lon-wrap + .sel.
        coef_full = coef_full.assign_coords(
            lon=(((coef_full.lon + 180) % 360) - 180)).sortby('lon')
        ste_full = ste_full.assign_coords(
            lon=(((ste_full.lon + 180) % 360) - 180)).sortby('lon')
        rsq_full = rsq_full.assign_coords(
            lon=(((rsq_full.lon + 180) % 360) - 180)).sortby('lon')
        coef_c = coef_full.sel(lat=src.lat, lon=src.lon)
        ste_c  = ste_full.sel(lat=src.lat, lon=src.lon)
        rsq_c  = rsq_full.sel(lat=src.lat, lon=src.lon)

        if _season == '':
            # Annual: use cmip6_ctrl_data (cropped to EU_buffer + realiz-mean).
            T_pi = src.tas.sel(scenar='his', season=_season).sel(
                time=slice(*PI_WINDOW)).mean('time')                   # (lat, lon)
            T_pd_245 = xr.concat([
                src.tas.sel(scenar='his',    season=_season).sel(time=slice(*PD_HIS_WINDOW)),
                src.tas.sel(scenar='ssp245', season=_season).sel(time=slice(*PD_SSP_WINDOW)),
            ], dim='time').mean('time')                                # (lat, lon)
            T_2090 = src.tas.sel(season=_season).sel(
                time=slice(*fw)).mean('time')               # (scenar, lat, lon)
        else:
            # Seasonal: T_pi / T_pd_245 derived from the long-run ssp245
            # pooled-member tas (tas_s, full-grid 90×144 in K — only ssp245
            # has the long-run extension). T_2090 reads from src.tas instead,
            # which now carries seasonal scenmip for all three SSPs after the
            # 2026-05-27 GISS seasonal loader fallback. Slope (coef_c) is
            # treated as scenario-independent across the pipeline.
            tas_em = (tas_s - 273.15).mean('realiz')
            tas_em = tas_em.assign_coords(
                lon=(((tas_em.lon + 180) % 360) - 180)).sortby('lon')
            tas_em = tas_em.sel(lat=src.lat, lon=src.lon)
            T_pi = tas_em.sel(time=slice(*PI_WINDOW)).mean('time')      # (lat, lon)
            T_pd_245 = tas_em.sel(time=slice(PD_HIS_WINDOW[0],
                                              PD_SSP_WINDOW[1])).mean('time')
            T_2090 = src.tas.sel(season=_season).sel(
                time=slice(*fw)).mean('time')                # (scenar, lat, lon)

        # coef_c is K per Sv (raw regression of ΔT on ΔAMOC) — MPI convention,
        # not CESM's K per %-weakening. See pre-existing comment in the
        # annual-only version of this loader prior to 2026-05-26.
        delta_T_pi = T_2090.sel(scenar=ssps) - T_pi
        delta_T_pd = T_2090.sel(scenar=ssps) - T_pd_245
        req_strength_pi = AMOC_2090.sel(scenar=ssps) - delta_T_pi / coef_c
        req_strength_pd = AMOC_2090.sel(scenar=ssps) - delta_T_pd / coef_c
        req_weakening_pi = (AMOC_pi_GISS - req_strength_pi) / AMOC_pi_GISS * 100
        req_weakening_pd = (AMOC_pi_GISS - req_strength_pd) / AMOC_pi_GISS * 100

        ds_s = xr.Dataset(
            data_vars={
                'coef_ensmean':     coef_c,
                'ste_ensmean':      ste_c,
                'rsq_ensmean':      rsq_c,
                'T_pi':             T_pi,
                'T_pd_245':         T_pd_245,
                'T_future':      T_2090.sel(scenar=ssps),
                'req_strength_pi':  req_strength_pi,
                'req_strength_pd':  req_strength_pd,
                'req_weakening_pi': req_weakening_pi,
                'req_weakening_pd': req_weakening_pd,
            },
        ).expand_dims(season=[_season])
        per_season_ds.append(ds_s)

    reg_ds_giss = xr.concat(per_season_ds, dim='season')
    # AMOC fields are season-invariant — assign once outside the season loop.
    reg_ds_giss['AMOC_future'] = AMOC_2090.sel(scenar=ssps)
    reg_ds_giss['AMOC_pi'] = AMOC_pi
    reg_ds_giss.attrs.update({
        'model': 'GISS-E2-1-G',
        'reference': 'Romanou et al. 2023, JCLI (long-run ssp245 ensemble)',
        'members_pooled': ', '.join(GISS_NON_COMPOSITE),
        'composite_members': ', '.join(GISS_COMPOSITE),
        'rolling_window': ROLLING_WINDOW,
        'time_periods': ', '.join(period_labels),
        'amoc_pi_giss_sv': float(AMOC_pi_GISS),
        'cov_type': f'HAC, maxlags={max(ROLLING_WINDOW - 1, 1)}',
        'ssp126_ensemble': '5 NCCS post-erratum members (r1-r5); '
                           'r6-r10 have no ssp126 simulation. ssp245/ssp370 use 10 members.',
        'seasonal_note': ('Seasonal regressions pair seasonal tas with annual '
                          'AMOC26 — AMOC has no DJF/JJA sibling by design.'),
    })

    reg_ds_giss = reg_ds_giss.drop_vars('height', errors='ignore')

    reg_ds_giss.attrs['future_window'] = list(fw)
    print(f'Saving GISS-E2-1-G regression dataset to {cache_path}')
    reg_ds_giss.to_netcdf(cache_path)
    return reg_ds_giss


# PLOTTING MULTI-MODEL DATA

def hosmip_regression_plot(multi_model_dict, window=10, region='EU', season='', no_plots=False, plot_bg='white', only_model=None, only_model_label=None, low_ylim=-16, markersize=20, quantile_reg=False, linear_95_reg=False, linear_5_reg=False, central_reg=None, central_reg_intercept=True, single_model_linear=True, single_model_intercept=False, hosmip_ref_pi=True, lat=None, lon=None, cap_x_range=100, bm_data=False, vwb_data=False, boot_data=False, liu_data=False, boot_regression=False, boot_intercept=False, giss_data=None, giss_regression=False, giss_intercept=False, giss_time_period=None, add_combined_results=False, combined_results_label=True, add_hosmip_mpi_regression=False, hosmip_reg_ds_dict=None, weakening_unit='pct', sv_xmax=None, ax=None, savefig=False):
    # When hosmip_reg_ds_dict is passed AND hosmip_ref_pi=True, the per-model
    # AMOC and tas baselines are read from the cached regression dataset
    # (1850–1899 CMIP6 historical ens-mean, EC-Earth3 = NAHosMIP piControl
    # snapshot). When None, falls back to the legacy piControl-snapshot inline
    # computation — this branch is regen-safe (used by get_hosmip_reg_ds's
    # internal call at line ~2962, where the slope is baseline-invariant so
    # the choice doesn't matter). Slopes are baseline-invariant; only the
    # plotted scatter and the y-intercept of the per-model line shift.

    validate_choice('season', season, ALLOWED_SEASONS)
    validate_choice('plot_bg', plot_bg, ALLOWED_PLOT_BG)

    if season != '':
        if not no_plots:
            print('No seasonal data for other studies.')
        bm_data, vwb_data, boot_data, liu_data = False, False, False, False

    if not no_plots:
        plt.style.use('default')
        if plot_bg == 'black':
            plt.style.use('dark_background')
            plt.rcParams['axes.facecolor'] = '#191919'
            plt.rcParams['figure.facecolor'] = '#191919'
        if ax is None:
            plt.rcParams.update({'font.size': 14,
                                 'axes.labelsize': 14,
                                 'xtick.labelsize': 14,
                                 'ytick.labelsize': 14,
                                 })
            fig, ax = plt.subplots(figsize=(9, 6))
            ext_ax = False
        else:
            ext_ax = True

    is_latlon, is_region, is_ocean = False, False, False
    if lat != None and lon != None:
        is_latlon = True
        if only_model is not None:
            if np.sum(~np.isnan(multi_model_dict[only_model].sel(type='control', season='', scenar='pi').tas.isel(time=0).values))==0:
                return {
                        '95pct_coef_lin': np.nan,
                        '95pct_coef_quad': np.nan,
                        '5pct_coef_lin': np.nan,
                        '5pct_coef_quad': np.nan,
                        'model_coef_lin': np.nan,
                        'model_coef_quad': np.nan
                        }
        else:
            print('Lat/lon only works for a single model.')
            return
    elif region != None:
        is_region = True
    else:
        print('No lat/lon or region provided.')
        return

    all_x = []
    all_y = []
    for model in hosmip_labels:
        if only_model is not None and model != only_model:
            continue
        amoc = multi_model_dict[model].sel(type='hosing', scenar='pi', season='').rolling(time=window, center=True).mean('time').amoc
        use_cached_pi = (hosmip_ref_pi and hosmip_reg_ds_dict is not None
                         and model in hosmip_reg_ds_dict)
        if use_cached_pi:
            # Harmonised hist 1850–1899 baseline (Convention A).
            amoc_base_val = float(hosmip_reg_ds_dict[model].AMOC_pi.values)
            x = (amoc.values - amoc_base_val) / amoc_base_val * (-100)
            _pi_block = amoc_base_val
        elif hosmip_ref_pi:
            amoc_base = multi_model_dict[model].sel(type='control', scenar='pi', season='').amoc.isel(time=0)
            x = (amoc.values - amoc_base.values) / amoc_base.values * (-100)
            _pi_block = float(amoc_base.values)
        else:
            amoc_base = amoc.isel(time=slice(0, 10)).mean('time')
            x = (amoc.values - amoc_base.values) / amoc_base.values * (-100)
            _pi_block = float(amoc_base.values)
        if weakening_unit == 'sv':
            # % weakening (vs this model's own PI) -> absolute Sv weakening.
            x = x / 100.0 * _pi_block
        if is_latlon:
            tas_data_unsorted = multi_model_dict[model].sel(type='hosing', scenar='pi', season=season).tas
            tas_lat_sorted = tas_data_unsorted.sortby(tas_data_unsorted.lat)
            tas = tas_lat_sorted.isel(lat=lat, lon=lon)
        else:
            tas = weighted_area_lat(multi_model_dict[model].sel(type='hosing', scenar='pi', season=season).tas.where(
                multi_model_dict[model].mask.sel(region=region)).where(multi_model_dict[model].mask.sel(region='LAND')==0)
                ).mean('lat').mean('lon')
        tas_rolling = tas.rolling(time=window, center=True).mean('time')
        if use_cached_pi:
            # tas baseline: same season slice as the rolling series above.
            tpi_field = hosmip_reg_ds_dict[model].T_pi.sel(season=season)
            if is_latlon:
                tas_base = tpi_field.isel(lat=lat, lon=lon)
            else:
                tas_base = weighted_area_lat(tpi_field.where(
                    multi_model_dict[model].mask.sel(region=region)).where(
                    multi_model_dict[model].mask.sel(region='LAND') == 0)
                    ).mean('lat').mean('lon')
        elif hosmip_ref_pi:
            if is_latlon:
                tas_base = multi_model_dict[model].sel(type='control', scenar='pi', season=season).tas.isel(time=0, lat=lat, lon=lon)
            else:
                tas_base = weighted_area_lat(multi_model_dict[model].sel(type='control', scenar='pi', season=season).tas.isel(time=0).where(
                    multi_model_dict[model].mask.sel(region=region)).where(multi_model_dict[model].mask.sel(region='LAND')==0)
                    ).mean('lat').mean('lon')
        else:
            tas_base = tas.isel(time=slice(0, 10)).mean('time')
        y = tas_rolling.values - tas_base.values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        all_x.append(x)
        all_y.append(y)
        if not no_plots:
            if only_model and only_model_label != None:
                ax.scatter(x, y, label=only_model_label, color=hosmip_colors[model], clip_on=False, s=markersize, alpha=0.9, marker='o') # '+' '$\\bigcirc$'
            else:
                ax.scatter(x, y, label=model, color=hosmip_colors[model], clip_on=False, s=markersize, alpha=0.9, marker='o') # '+' '$\\bigcirc$'
        
    all_x_flat = np.concatenate(all_x)
    all_y_flat = np.concatenate(all_y)

    # Sort x for plotting
    sort_idx = np.argsort(all_x_flat)
    restricted_x_range = np.where((all_x_flat[sort_idx] < cap_x_range) & (all_x_flat[sort_idx] > 0))
    x_sorted = (all_x_flat[sort_idx])[restricted_x_range]
    y_sorted = (all_y_flat[sort_idx])[restricted_x_range]

    # Per-gridcell call (no_plots, single_model_*) over a NaN cell: every x
    # got filtered. Return all-NaN coefs so the caller writes NaN into the
    # cache rather than crashing in OLS / QuantReg with empty arrays.
    if no_plots and len(x_sorted) < 3:
        return {
            '95pct_coef_lin': np.nan, '95pct_coef_quad': np.nan,
            '5pct_coef_lin':  np.nan, '5pct_coef_quad':  np.nan,
            'model_coef_lin': np.nan, 'model_coef_quad': np.nan,
            'model_ste_lin':  np.nan, 'model_ste_quad':  np.nan,
            'model_rsq':      np.nan,
            'model_pvalue_lin': np.nan, 'model_pvalue_quad': np.nan,
            'model_intercept_lin':         np.nan,
            'model_intercept_ste_lin':     np.nan,
            'model_intercept_pvalue_lin':  np.nan,
        }
    Xq = np.column_stack([x_sorted, x_sorted**2])
    Xq_lin = np.column_stack([x_sorted])
    # Xq = sm.add_constant(Xq)  # add intercept

    # The quantile-regression block below is only used for plot overlays
    # (5th/95th percentile lines drawn over the scatter). With no_plots=True
    # and an empty/near-empty x_sorted (e.g. a gridcell whose hist-baseline
    # tas is NaN after interp), QuantReg raises. Short-circuit before that.
    _qr_skip = no_plots or len(x_sorted) < 3
    if _qr_skip:
        res_5 = res_95 = None
        yq_5 = yq_95 = np.array([])

    if not _qr_skip:
        # 5th percentile
        mod_5 = sm.QuantReg(y_sorted, Xq_lin) if linear_5_reg else sm.QuantReg(y_sorted, Xq)
        res_5 = mod_5.fit(q=0.05)
        yq_5 = res_5.predict(Xq_lin) if linear_5_reg else res_5.predict(Xq)
        # 95th percentile
        mod_95 = sm.QuantReg(y_sorted, Xq_lin) if linear_95_reg else sm.QuantReg(y_sorted, Xq)
        res_95 = mod_95.fit(q=0.95)
        yq_95 = res_95.predict(Xq_lin) if linear_95_reg else res_95.predict(Xq)

    if central_reg == 'linear':
        X = x_sorted
        if central_reg_intercept:
            X = sm.add_constant(x_sorted)  # Add intercept column to regressor
        mod_lin = sm.OLS(y_sorted, X)
        x_range = np.linspace(0, max(x_sorted), 100)
        X_pred = sm.add_constant(x_range) if central_reg_intercept else x_range
        pred_lin = mod_lin.fit(cov_type='HAC', cov_kwds={'maxlags': window-1}).predict(X_pred)
        if not no_plots:
            ax.plot(x_range, pred_lin, color='black' if plot_bg != 'black' else 'white', lw=2, ls='-', label='Linear regression')
            if central_reg_intercept:
                ax.text(52, -10, f"y = {mod_lin.fit(cov_type='HAC', cov_kwds={'maxlags': window-1}).params[0]:.5f} + {mod_lin.fit().params[1]:.5f} x (lin. reg.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')
            else:
                ax.text(52, -10, f"y = {mod_lin.fit(cov_type='HAC', cov_kwds={'maxlags': window-1}).params[0]:.5f} x (lin. reg.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')
    elif central_reg == 'quadratic':
        # Quadratic regression, linear + quadratic term, with optional intercept
        X_quad = np.column_stack([x_sorted, x_sorted**2])
        if central_reg_intercept:
            X_quad = sm.add_constant(X_quad)  # Adds intercept as first column
        mod_quad = sm.OLS(y_sorted, X_quad)
        x_range = np.linspace(0, max(x_sorted), 100)
        X_range_quad = np.column_stack([x_range, x_range**2])
        if central_reg_intercept:
            X_range_quad = sm.add_constant(X_range_quad)
        pred_quad = mod_quad.fit(cov_type='HAC', cov_kwds={'maxlags': window-1}).predict(X_range_quad)
        if not no_plots:
            ax.plot(x_range, pred_quad, color='black' if plot_bg != 'black' else 'white', lw=2, ls='-', label='Quadratic regression')
            params = mod_quad.fit(cov_type='HAC', cov_kwds={'maxlags': window-1}).params
            if central_reg_intercept:
                ax.text(52, -10, f"y = {params[0]:.5f} + {params[1]:.5f} x + {params[2]:.5f} x² (quad. reg.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')
            else:
                ax.text(52, -10, f"y = {params[0]:.5f} x + {params[1]:.5f} x² (quad. reg.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')

    # for a single model, also linear regression
    if only_model:
        if single_model_linear:
            # Linear regression, optionally with intercept
            if single_model_intercept:
                x_with_const = sm.add_constant(x_sorted)
                mod_lin = sm.OLS(y_sorted, x_with_const)
            else:
                mod_lin = sm.OLS(y_sorted, x_sorted)
            mod_lin_fit = mod_lin.fit(cov_type='HAC', cov_kwds={'maxlags': window-1})
            x_range = np.linspace(0, max(x_sorted), 100)
            if single_model_intercept:
                print(only_model, mod_lin_fit.summary())
                x_range_pred = sm.add_constant(x_range)
                mod_lin_intercept = mod_lin_fit.params[0]
                mod_lin_coef = mod_lin_fit.params[1]
                mod_lin_intercept_ste = mod_lin_fit.bse[0]
                mod_lin_ste = mod_lin_fit.bse[1]
                mod_lin_rsq = mod_lin_fit.rsquared
                mod_lin_intercept_pvalue = mod_lin_fit.pvalues[0]
                mod_lin_pvalue = mod_lin_fit.pvalues[1]
            else:
                x_range_pred = x_range
                mod_lin_intercept = None
                mod_lin_intercept_ste = None
                mod_lin_intercept_pvalue = None
                mod_lin_coef = mod_lin_fit.params[0]
                mod_lin_ste = mod_lin_fit.bse[0]
                mod_lin_rsq = mod_lin_fit.rsquared
                mod_lin_pvalue = mod_lin_fit.pvalues[0]
            pred_lin = mod_lin_fit.predict(x_range_pred)
            if not no_plots and central_reg != False:
                ax.plot(x_range, pred_lin, color='black' if plot_bg != 'black' else 'white', lw=2, ls='-', label='Linear regression')
                if single_model_intercept:
                    ax.text(52, -10, f"y = {mod_lin_intercept:.5f} + {mod_lin_coef:.5f} x (lin. reg.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')
                else:
                    ax.text(52, -10, f"y = {mod_lin_coef:.5f} x (lin. reg.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')
        else:
            # Quadratic regression, linear + quadratic term, no intercept
            X_quad = np.column_stack([x_sorted, x_sorted**2])
            mod_quad = sm.OLS(y_sorted, X_quad)
            mod_quad_fit = mod_quad.fit(cov_type='HAC', cov_kwds={'maxlags': window-1})
            x_range = np.linspace(0, max(x_sorted), 100)
            X_range_quad = np.column_stack([x_range, x_range**2])
            pred_quad = mod_quad_fit.predict(X_range_quad)
            # Extract coefficients, standard errors and R² for quadratic regression
            coef_lin = mod_quad_fit.params[0]
            coef_quad = mod_quad_fit.params[1]
            mod_quad_ste_lin = mod_quad_fit.bse[0]
            mod_quad_ste_quad = mod_quad_fit.bse[1]
            mod_quad_rsq = mod_quad_fit.rsquared
            mod_quad_pvalue_lin = mod_quad_fit.pvalues[0]
            mod_quad_pvalue_quad = mod_quad_fit.pvalues[1]
            if not no_plots and central_reg != False:
                ax.plot(x_range, pred_quad, color='black' if plot_bg != 'black' else 'white', lw=2, ls='-', label='Quadratic regression')
                ax.text(52, -10, f"y = {coef_lin:.5f} x + {coef_quad:.5f} x² (quad. reg.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')
    if not no_plots:
        # Plot quantile regression lines
        if quantile_reg:
            ax.plot(x_sorted, yq_5, color='black' if plot_bg != 'black' else 'white', lw=2, ls=':', clip_on=True)
            ax.plot(x_sorted, yq_95, color='black' if plot_bg != 'black' else 'white', lw=2, ls=':', clip_on=True)
        # ax.fill_between(x_sorted, yq_5, yq_95, color='black' if plot_bg != 'black' else 'white', alpha=0.1, zorder=-10, clip_on=False,
                        #  label='5th-95th percentile envelope'
                        #  )

        if ax is None:
            # plot both quadratic quantile regression equations at bottom of plot
            if linear_95_reg:
                ax.text(5, -9, fr"95th percentile: $\Delta \mathrm{{T}}_\mathrm{{{region}}}$ = {res_95.params[0]:.4f} $\Delta \mathrm{{AMOC}}$", fontsize=10, color='black' if plot_bg != 'black' else 'white')
            else:
                ax.text(5, -9, f"$\Delta \mathrm{{T}}_\mathrm{{{region}}}$ = {res_95.params[0]:.4f} $\Delta \mathrm{{AMOC}}$ + {res_95.params[1]:.5f} $\Delta \mathrm{{AMOC}}$² (95th percentile quad.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')
            if linear_5_reg:
                ax.text(5, -10, f"$\Delta \mathrm{{T}}_\mathrm{{{region}}}$ = {res_5.params[0]:.4f} $\Delta \mathrm{{AMOC}}$ (5th percentile lin.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')
            else:
                ax.text(5, -10, f"5th percentile: $\Delta \mathrm{{T}}_\mathrm{{{region}}}$ = {res_5.params[0]:.4f} $\Delta \mathrm{{AMOC}}$" + (" + " if res_5.params[1]>0 else " ") + f"{res_5.params[1]:.5f} $\Delta \mathrm{{AMOC}}^2$", fontsize=10, color='black' if plot_bg != 'black' else 'white')

        if bm_data:

            Delta_AMOC_hosing = (multi_model_dict['EC-Earth3'].sel(type='diff', scenar='pi', season='').amoc.isel(time=0) / \
                                multi_model_dict['EC-Earth3'].sel(type='control', scenar='pi', season='').amoc.isel(time=0)).values * 100
            
            Delta_AMOC_4x = (multi_model_dict['EC-Earth3'].sel(type='diff', scenar='ghg', season='').amoc.isel(time=0) / \
                            multi_model_dict['EC-Earth3'].sel(type='control', scenar='pi', season='').amoc.isel(time=0)).values * 100

            if weakening_unit == 'sv':
                _ec_pi = float(multi_model_dict['EC-Earth3'].sel(type='control', scenar='pi', season='').amoc.isel(time=0).values)
                Delta_AMOC_hosing = Delta_AMOC_hosing / 100 * _ec_pi
                Delta_AMOC_4x = Delta_AMOC_4x / 100 * _ec_pi

            if is_region:
                Delta_tas_hosing = weighted_area_lat(
                    multi_model_dict['EC-Earth3'].sel(type='diff', scenar='pi', season='').tas.where(
                    multi_model_dict['EC-Earth3'].mask.sel(region=region)).where(multi_model_dict['EC-Earth3'].mask.sel(region='LAND')==0)
                    ).mean('lat').mean('lon').isel(time=0)
                Delta_tas_4x = weighted_area_lat(
                    multi_model_dict['EC-Earth3'].sel(type='diff', scenar='ghg', season='').tas.where(
                    multi_model_dict['EC-Earth3'].mask.sel(region=region)).where(multi_model_dict['EC-Earth3'].mask.sel(region='LAND')==0)
                    ).mean('lat').mean('lon').isel(time=0)
            elif is_latlon: # untested whether this works correctly (have to verify that it is fully the same grid ordering as the other EC-Earth3 data) --> should work but still untested
                Delta_tas_hosing = multi_model_dict['EC-Earth3'].sel(type='diff', scenar='pi', season='').tas.isel(lat=lat, lon=lon).isel(time=0)
                Delta_tas_4x = multi_model_dict['EC-Earth3'].sel(type='diff', scenar='ghg', season='').tas.isel(lat=lat, lon=lon).isel(time=0)
            
            ax.scatter(Delta_AMOC_hosing, Delta_tas_hosing, color=hosmip_colors['EC-Earth3'], edgecolors='k', linewidths=1.5, marker='P', s=10*markersize, label='Bellomo & Mehling\npreindustrial', zorder=10) # '$\ominus$'
            ax.scatter(Delta_AMOC_4x, Delta_tas_4x, color=hosmip_colors['EC-Earth3'], edgecolors='k', linewidths=1.5, marker='X', s=10*markersize, label='Bellomo & Mehling\n'+r'4x$\mathrm{{CO}}_2$ experiment', zorder=10) # '$\otimes$'

        if vwb_data:

            Delta_AMOC_600_1500 = (
                (multi_model_dict['CESM1'].sel(type='control', scenar='ghg', season='').amoc.values -
                multi_model_dict['CESM1'].sel(type='control', scenar='pi', season='').amoc.values) -
                (multi_model_dict['CESM1'].sel(type='hosing', scenar='ghg', season='').amoc.values -
                multi_model_dict['CESM1'].sel(type='hosing', scenar='pi', season='').amoc.values)) / \
                multi_model_dict['CESM1'].sel(type='control', scenar='pi', season='').amoc.values * 100

            if weakening_unit == 'sv':
                Delta_AMOC_600_1500 = Delta_AMOC_600_1500 / 100 * multi_model_dict['CESM1'].sel(type='control', scenar='pi', season='').amoc.values

            if is_region:
                vwb_tas_1500 = weighted_area_lat(
                    (multi_model_dict['CESM1'].sel(type='hosing', scenar='ghg', season='') - multi_model_dict['CESM1'].sel(type='hosing', scenar='pi', season='')).tas.where(
                    multi_model_dict['CESM1'].mask.sel(region=region)).where(multi_model_dict['CESM1'].mask.sel(region='LAND')==0)
                    ).mean('lat').mean('lon').values
                vwb_tas_600 = weighted_area_lat(
                    (multi_model_dict['CESM1'].sel(type='control', scenar='ghg', season='') - multi_model_dict['CESM1'].sel(type='control', scenar='pi', season='')).tas.where(
                    multi_model_dict['CESM1'].mask.sel(region=region)).where(multi_model_dict['CESM1'].mask.sel(region='LAND')==0)
                    ).mean('lat').mean('lon').values

                # marker='$\\ddagger$'
                ax.scatter(Delta_AMOC_600_1500, vwb_tas_1500 - vwb_tas_600, color=hosmip_colors['CESM2'], edgecolors='k', linewidths=1.5, marker='v', s=10*markersize, label='v. Westen & Baatsen\nRCP4.5 in 2400-2500', zorder=10, clip_on=False)

        if liu_data:

            Delta_AMOC = (multi_model_dict['CCSM4'].sel(type='control', scenar='ghg', season='').amoc.values -
                          multi_model_dict['CCSM4'].sel(type='hosing', scenar='ghg', season='').amoc.values) / \
                          multi_model_dict['CCSM4'].sel(type='control', scenar='pi', season='').amoc.values * 100

            if weakening_unit == 'sv':
                Delta_AMOC = Delta_AMOC / 100 * multi_model_dict['CCSM4'].sel(type='control', scenar='pi', season='').amoc.values

            if is_region:
                liu_tas_rcp85 = weighted_area_lat(
                    (multi_model_dict['CCSM4'].sel(type='control', scenar='ghg', season='').tas.where(
                    multi_model_dict['CCSM4'].mask.sel(region=region)
                    ).where(multi_model_dict['CCSM4'].mask.sel(region='LAND')==0))
                    ).mean('lat').mean('lon')
                liu_tas_rcp85_stable = weighted_area_lat(
                    (multi_model_dict['CCSM4'].sel(type='hosing', scenar='ghg', season='').tas.where(
                    multi_model_dict['CCSM4'].mask.sel(region=region)
                    ).where(multi_model_dict['CCSM4'].mask.sel(region='LAND')==0))
                    ).mean('lat').mean('lon')
            
                ax.scatter(Delta_AMOC, liu_tas_rcp85 - liu_tas_rcp85_stable, color=hosmip_colors['CESM2'], edgecolors='k', linewidths=1.5, marker='^', s=10*markersize, label='Liu et al. (2020)\nRCP8.5 in 2061-2080', zorder=10, clip_on=False)

        if boot_data:

            if is_region:
                boot_x = []
                boot_y = []
                for ssp_i in ['ssp126', 'ssp585']:
                    x = (multi_model_dict['CESM2'].amoc.sel(type='hosing', scenar=ssp_i, season='').rolling(time=window, center=True).mean('time').values -
                         multi_model_dict['CESM2'].amoc.sel(type='control', scenar=ssp_i, season='').rolling(time=window, center=True).mean('time').values) / (
                         -multi_model_dict['CESM2'].amoc.sel(type='control', scenar='pi', season='').isel(time=0)).values * 100

                    if weakening_unit == 'sv':
                        x = x / 100 * float(multi_model_dict['CESM2'].amoc.sel(type='control', scenar='pi', season='').isel(time=0).values)

                    y = weighted_area_lat(
                        multi_model_dict['CESM2'].tas.sel(type='hosing', scenar=ssp_i, season='').rolling(time=window, center=True).mean('time').where(
                        multi_model_dict['CESM2'].mask.sel(region=region)
                        ).where(multi_model_dict['CESM2'].mask.sel(region='LAND')==0)).mean('lat').mean('lon').values - \
                        weighted_area_lat(
                        multi_model_dict['CESM2'].tas.sel(type='control', scenar=ssp_i, season='').rolling(time=window, center=True).mean('time').where(
                        multi_model_dict['CESM2'].mask.sel(region=region)
                        ).where(multi_model_dict['CESM2'].mask.sel(region='LAND')==0)).mean('lat').mean('lon').values

                    ax.scatter(x, y, color='dodgerblue', edgecolors='k', linewidths=0.7, marker='d', s=1.5*markersize, label=f'Boot et al. (2024)\nSSP1-2.6 & SSP5-8.5\n2015 - 2100' if ssp_i=='ssp126' else '', clip_on=False)

                    mask = ~np.isnan(x) & ~np.isnan(y)
                    x = x[mask]
                    y = y[mask]
                    boot_x.append(x)
                    boot_y.append(y)

                if boot_regression:

                    boot_x_flat = np.concatenate(boot_x)
                    boot_y_flat = np.concatenate(boot_y)

                    # Sort x for plotting
                    boot_sort_idx = np.argsort(boot_x_flat)
                    boot_x_sorted = boot_x_flat[boot_sort_idx]
                    boot_y_sorted = boot_y_flat[boot_sort_idx]

                    boot_X = boot_x_sorted
                    if boot_intercept:
                        boot_X = sm.add_constant(boot_x_sorted)  # Add intercept column to regressor
                    boot_mod_lin = sm.OLS(boot_y_sorted, boot_X)
                    if weakening_unit == 'sv':
                        _cesm2_pi = float(multi_model_dict['CESM2'].amoc.sel(type='control', scenar='pi', season='').isel(time=0).values)
                        boot_x_range = np.linspace(0, 0.5 * _cesm2_pi, 100)
                    else:
                        boot_x_range = np.linspace(0, 50, 100) # use 50% as end of extrapolation range because SSP1-2.6 in CESM2 already has 49.8% implicit weakening by 2100
                    boot_X_pred = sm.add_constant(boot_x_range) if boot_intercept else boot_x_range
                    boot_pred_lin = boot_mod_lin.fit(cov_type='HAC', cov_kwds={'maxlags': window-1}).predict(boot_X_pred)
                    if not no_plots:
                        ax.plot(boot_x_range, boot_pred_lin, color='dodgerblue', lw=3, zorder=9)

        if giss_data is not None and is_region:
            # GISS-E2-1-G member-minus-composite deviations (ssp245
            # r1–r10, 7 non-composite members pooled, full long-run
            # window). Same conceptual role as the Boot block above
            # — additional CMIP6 evidence on top of the HosMIP scatter
            # — but the signal here is ensemble-internal AMOC
            # variability under a single SSP, not a hosing-vs-control
            # perturbation. See get_giss_panel_data.
            if giss_time_period is not None:
                y0, y1 = (int(s) for s in giss_time_period.split('-'))
                gd = giss_data.where(
                    (giss_data.year >= y0) & (giss_data.year <= y1),
                    drop=True,
                )
                giss_window_str = f'{y0}-{y1}'
            else:
                gd = giss_data
                giss_window_str = giss_data.attrs.get('time_window', '').replace('-', ' - ')
            giss_x = gd.amoc_dev_pct.values
            if weakening_unit == 'sv':
                # % deviation (vs AMOC_pi_GISS) -> absolute Sv weakening, sign preserved.
                giss_x = giss_x / 100 * AMOC_pi_GISS
            giss_y = gd.t_dev.sel(region=region).values
            giss_keep = np.isfinite(giss_x) & np.isfinite(giss_y)
            giss_x = giss_x[giss_keep]
            giss_y = giss_y[giss_keep]
            giss_color = hosing_colors['ssp245']['ge']
            ax.scatter(giss_x, giss_y, color=giss_color, edgecolors='k',
                       linewidths=0.2, marker='o', s=0.35 * markersize,
                       alpha=0.75,
                       label=f'Romanou et al. (2023)\nSSP2-4.5 {giss_window_str}',
                       clip_on=True, zorder=2)
            if giss_regression:
                giss_sort_idx = np.argsort(giss_x)
                gxs = giss_x[giss_sort_idx]
                gys = giss_y[giss_sort_idx]
                gX = sm.add_constant(gxs) if giss_intercept else gxs
                giss_x_range = (np.linspace(0, 0.69 * AMOC_pi_GISS, 100)
                                if weakening_unit == 'sv' else np.linspace(0, 69, 100))
                gX_pred = (sm.add_constant(giss_x_range)
                           if giss_intercept else giss_x_range)
                giss_pred_lin = sm.OLS(gys, gX).fit(
                    cov_type='HAC',
                    cov_kwds={'maxlags': max(window - 1, 1)},
                ).predict(gX_pred)
                if not no_plots:
                    ax.plot(giss_x_range, giss_pred_lin,
                            color=giss_color, lw=3, zorder=9)

        if add_combined_results:
            
            reg_ds_mpi = load_regression_ds_mpi()

            # with open('../data/dfs_countries.pkl', 'rb') as f:
            #     dfs_countries = pickle.load(f)
            
            # combined_coefs = dict(dfs_countries['']['ssp126'].coef_all_ssps_combined)
            regional_combined_coef = weighted_area_lat(
                                    reg_ds_mpi.coef_ensmean.sel(season=season).where(multi_model_dict['MPI-ESM1-2-LR'].mask.sel(region=region)).where(multi_model_dict['MPI-ESM1-2-LR'].mask.sel(region='LAND')==0)
                                    ).mean('lat').mean('lon').values

            if weakening_unit == 'sv':
                # coef_ensmean is K/Sv: x in Sv, drop the *AMOC_pi_MPI/100 factor.
                x_plot = np.linspace(0, 0.82 * AMOC_pi_MPI, 200)
                ax.plot(x_plot, x_plot * (-regional_combined_coef), color='brown', lw=3, label='This study' if combined_results_label else '', zorder=9, clip_on=True)
            else:
                x_plot = np.linspace(0, 82, 200)
                ax.plot(x_plot, x_plot * (-regional_combined_coef * AMOC_pi_MPI / 100), color='brown', lw=3, label='This study' if combined_results_label else '', zorder=9, clip_on=True)

        if add_hosmip_mpi_regression:

            hosmip_reg_ds_dict = get_hosmip_reg_ds()
            regional_hosmip_mpi_coef = weighted_area_lat(
                                    hosmip_reg_ds_dict['MPI-ESM1-2-LR'].lin_coef_hosmip.sel(season=season).where(multi_model_dict['MPI-ESM1-2-LR'].mask.sel(region=region)).where(multi_model_dict['MPI-ESM1-2-LR'].mask.sel(region='LAND')==0)
                                    ).mean('lat').mean('lon').values

            if weakening_unit == 'sv':
                # lin_coef_hosmip is K/%: x in Sv, rescale slope to K/Sv (×100/PI).
                _mpi_pi = float(hosmip_reg_ds_dict['MPI-ESM1-2-LR'].AMOC_pi.values)
                x_plot = np.linspace(0, 0.82 * _mpi_pi, 100)
                ax.plot(x_plot, x_plot * (regional_hosmip_mpi_coef * 100 / _mpi_pi), color='blueviolet', lw=3, zorder=9, clip_on=True)
            else:
                x_plot = np.linspace(0, 82, 100)
                ax.plot(x_plot, x_plot * regional_hosmip_mpi_coef, color='blueviolet', lw=3, zorder=9, clip_on=True)

        # Formatting
        ax.set_ylim(low_ylim, 0.)
        ax.set_yticks(np.arange(low_ylim, 2, 2))
        if weakening_unit == 'sv':
            _xmax = sv_xmax if sv_xmax is not None else 20
            ax.set_xlim(0, _xmax)
            ax.set_xticks(np.arange(0, _xmax + 1, 5))
            ax.set_xlabel("Additional AMOC weakening [Sv]")
        else:
            ax.set_xlim(0, 80)
            ax.set_xticks(np.arange(0, 90, 10))
            ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
            if hosmip_ref_pi:
                ax.set_xlabel("Additional AMOC weakening") # [%]
            else:
                ax.set_xlabel("Weakening wrt first 10 yrs AMOC strength [%]")
        if hosmip_ref_pi:
            # ax.set_ylabel(f"Deviation from preindustrial {region if is_region else (lat, lon)} temperature [°C]")
            ax.set_ylabel(f"$\Delta \mathrm{{T}}_\mathrm{{{region if is_region else (lat, lon)}}}$ [°C]"+(f' in {season.upper()}' if season != '' else ''))
        else:
            ax.set_ylabel(f"Deviation from first 10 yrs {region if is_region else (lat, lon)} temperature [°C]")
        ax.axhline(0, color='black' if not plot_bg=='black' else 'white', lw=1., ls='--', clip_on=False)
        ax.axvline(0, color='black' if not plot_bg=='black' else 'white', lw=1., ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines["left"].set_position(("axes", -0.0))

        if ext_ax:
            return
        
        handles, labels = ax.get_legend_handles_labels()
        num_phantoms = 7 - (2 if bm_data else 0) - (3 if liu_data and vwb_data else 2 if vwb_data or liu_data else 0) - (1 if boot_data else 0) - (1 if giss_data is not None else 0) # 7 without external data, less for each enabled overlay
        for i in range(num_phantoms):
            handles.append(plt.Line2D([0], [0], color='none', alpha=0))
            labels.append('')
        ax.legend(handles, labels, frameon=False, ncols=2, fontsize=10, loc='upper left', bbox_to_anchor=(0., 0.52))

        if savefig != False:
            plt.savefig(savefig, dpi=400, bbox_inches='tight', transparent=True if plot_bg=='black' else False)

    return {
            '95pct_coef_lin': (res_95.params[0] if res_95 is not None else np.nan),
            '95pct_coef_quad': (res_95.params[1] if (res_95 is not None and not linear_95_reg) else 0),
            '5pct_coef_lin': (res_5.params[0] if res_5 is not None else np.nan),
            '5pct_coef_quad': (res_5.params[1] if (res_5 is not None and not linear_5_reg) else 0),
            'model_coef_lin': mod_lin_coef if only_model and single_model_linear else coef_lin if only_model and not single_model_linear else None,
            'model_coef_quad': coef_quad if only_model and not single_model_linear else None,
            'model_ste_lin': mod_lin_ste if only_model and single_model_linear else mod_quad_ste_lin if only_model and not single_model_linear else None,
            'model_ste_quad': mod_quad_ste_quad if only_model and not single_model_linear else None,
            'model_rsq': mod_lin_rsq if only_model and single_model_linear else mod_quad_rsq if only_model and not single_model_linear else None,
            'model_pvalue_lin': mod_lin_pvalue if only_model and single_model_linear else mod_quad_pvalue_lin if only_model and not single_model_linear else None,
            'model_pvalue_quad': mod_quad_pvalue_quad if only_model and not single_model_linear else None,
            'model_intercept_lin': mod_lin_intercept if only_model and single_model_linear else None,
            'model_intercept_ste_lin': mod_lin_intercept_ste if only_model and single_model_linear else None,
            'model_intercept_pvalue_lin': mod_lin_intercept_pvalue if only_model and single_model_linear else None,
            }


def cesm_regressions(boot_data, multi_model_dict, masks, ssp='all', lat=None, lon=None, region='EU', window=10, regressions=True, combined_reg=True, add_combined_intercept=False, no_plots=False, plot_bg='white', low_ylim=-10, xlim=80, hosmip_cesm=False, cesm_amoc_pi_override=None, savefig=False, ext_ax=None, equation_y_pos=0.18, equation_y_spacing=0.07, compute_linearity=False, compute_scenario_independence=False, season=''):
    # cesm_amoc_pi_override: optional scalar to use as the % AMOC weakening
    # denominator (Sv). When None, defaults to the piControl snapshot from
    # multi_model_dict (legacy behaviour). When passed (e.g. the harmonised
    # 1850–1899 CMIP6 historical ensemble-mean AMOC from
    # ``get_hist_pi_baseline``), the returned slope is in K-per-(% relative
    # to that override) — used by ``get_cesm_reg_ds`` to keep cached slopes
    # internally consistent with ``reg_ds_cesm.AMOC_pi``.

    validate_choice('plot_bg', plot_bg, ALLOWED_PLOT_BG)

    is_latlon, is_region = False, False
    if lat != None and lon != None:
        is_latlon = True
    elif region != None:
        is_region = True
    else:
        print('No lat/lon or region provided.')

    if not no_plots:
        if ext_ax is not None:
            ax = ext_ax
            fig = ax.figure
        else:
            plt.style.use('default')
            if plot_bg == 'black':
                plt.style.use('dark_background')
                plt.rcParams['axes.facecolor'] = '#191919'
                plt.rcParams['figure.facecolor'] = '#191919'

            fig, ax = plt.subplots(figsize=(9, 6))

    combined_x = []
    combined_y = []
    combined_scenario_id = []
    combined_scenario_name = []
    ssp_coefs = {}
    ssp_stes = {}
    ssp_rsqs = {}

    ssp_i = None
    SCENARIOS_CESM = ['ssp126', 'ssp585']
    for ssp_idx, ssp_i in enumerate(SCENARIOS_CESM):
        if ssp != 'all' and ssp_i != ssp:
                continue

        _cesm_amoc_pi = (cesm_amoc_pi_override
                         if cesm_amoc_pi_override is not None
                         else float(multi_model_dict['CESM2'].sel(scenar='pi', type='control', season='').amoc.isel(time=0).values))
        x = (boot_data.sel(scenar=ssp_i, type='hosing', season=season).amoc.rolling(time=window, center=True).mean("time").values - boot_data.sel(scenar=ssp_i, type='control', season=season).amoc.rolling(time=window, center=True).mean("time").values) / (-_cesm_amoc_pi) * 100

        if is_latlon:
            y = boot_data.sel(scenar=ssp_i, type='hosing', season=season).tas.rolling(time=window, center=True).mean("time").isel(lat=lat, lon=lon).values - boot_data.sel(scenar=ssp_i, type='control', season=season).tas.rolling(time=window, center=True).mean("time").isel(lat=lat, lon=lon).values
        elif is_region:
            y = weighted_area_lat(boot_data.sel(scenar=ssp_i, type='hosing', season=season).tas.rolling(time=window, center=True).mean("time").where(masks['CESM2'][region]).where(masks['CESM2']['LAND']==0)).mean('lat').mean('lon').values - weighted_area_lat(boot_data.sel(scenar=ssp_i, type='control', season=season).tas.rolling(time=window, center=True).mean("time").where(masks['CESM2'][region]).where(masks['CESM2']['LAND']==0)).mean('lat').mean('lon').values
        else:
            print('No lat/lon or region provided.')

        if not no_plots:
            boot_label = f'Boot et al. (2024, CESM2): {ssp_i.upper()}'
            ax.scatter(x, y, label=boot_label[:-2]+'-'+boot_label[-2]+'.'+boot_label[-1], s=15, c=hosing_colors[ssp_i]['ge'])

        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        combined_x.extend(x)
        combined_y.extend(y)
        combined_scenario_id.extend([ssp_idx] * len(x))
        combined_scenario_name.extend([ssp_i] * len(x))

        # Linear regression for this SSP
        x_with_const = sm.add_constant(x)
        model = sm.OLS(y, x_with_const).fit(cov_type='HAC', cov_kwds={'maxlags': window-1})
        x_range = np.linspace(0, 50, 200)
        x_range_with_const = sm.add_constant(x_range)
        y_pred = model.predict(x_range_with_const)
        if regressions and not no_plots:
            ax.plot(x_range, y_pred,
                    color=hosing_colors[ssp_i]['ge'], lw=1.5, linestyle='-')
        
        # Calculate confidence intervals for the predictions
        predictions = model.get_prediction(x_range_with_const)
        conf_int = predictions.conf_int(alpha=0.05)
        lower_bound = conf_int[:, 0]
        upper_bound = conf_int[:, 1]
        if regressions and not no_plots:
            ax.fill_between(x_range, lower_bound, upper_bound,
                            color=hosing_colors[ssp_i]['ge'], alpha=0.4, linewidth=0)
        
        # Plot regression equation and coefficients with p-values
        coef = model.params[1]
        ste = model.bse[1]
        rsq = model.rsquared
        ssp_coefs[ssp_i] = coef
        ssp_stes[ssp_i] = ste
        ssp_rsqs[ssp_i] = rsq
        intercept = model.params[0]
        p_value = model.pvalues[1]
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = ''
        # Match Fig1 simple_eqs form: ŷ = a·x + b. x is already in % weakening,
        # so coef·10 is the per-10%-weakening slope (°C/10%, the paper unit).
        slope_frac = r'$\frac{{}^{\circ}\mathrm{C}}{10\%}$'
        isign = '+' if intercept >= 0 else '-'
        equation = fr"$\hat{{y}}$ = {coef * 10:.2f}$^{{{{\text{{{significance}}}}}}}$ {slope_frac} $\cdot$ x {isign} {abs(intercept):.2f} $^{{\circ}}$C"
        if regressions and not no_plots:
            ax.text(0.05, equation_y_pos,
                    f"{hosing_names[ssp_i]['ge'][11:]}: {equation}",
                    transform=ax.transAxes, fontsize=12, verticalalignment='top', color=hosing_colors[ssp_i]['ge'])
        equation_y_pos -= equation_y_spacing

    # Combine both datasets for a third regression line
    combined_x = np.array(combined_x)
    combined_y = np.array(combined_y)
    combined_scenario_id = np.array(combined_scenario_id)
    combined_scenario_name = np.array(combined_scenario_name)

    # Diagnostics (RESET, Wald-F scenario indep) require an intercept in the
    # baseline design so X_base has shape (n, 2). Force the with-const fit
    # when either diagnostic flag is set; the user-facing add_combined_intercept
    # toggle only controls the legacy plotted equation.
    _need_intercept = add_combined_intercept or compute_linearity or compute_scenario_independence
    if _need_intercept:
        combined_x_with_const = sm.add_constant(combined_x)
        combined_model = sm.OLS(combined_y, combined_x_with_const).fit(cov_type='HAC', cov_kwds={'maxlags': window-1})
    else:
        combined_model = sm.OLS(combined_y, combined_x).fit(cov_type='HAC', cov_kwds={'maxlags': window-1})
    combined_x_range = np.linspace(0, 50, 200)
    combined_x_pred = sm.add_constant(combined_x_range) if _need_intercept else combined_x_range
    combined_y_pred = combined_model.predict(combined_x_pred)
    if combined_reg and not no_plots:
        ax.plot(combined_x_range, combined_y_pred,
                color='black' if not plot_bg=='black' else 'white', lw=1.5, linestyle='-')

    # Calculate confidence intervals for the combined predictions
    combined_predictions = combined_model.get_prediction(combined_x_pred)
    combined_conf_int = combined_predictions.conf_int(alpha=0.05)
    combined_lower_bound = combined_conf_int[:, 0]
    combined_upper_bound = combined_conf_int[:, 1]
    if combined_reg and not no_plots:
        ax.fill_between(combined_x_range, combined_lower_bound, combined_upper_bound,
                        color='black' if not plot_bg=='black' else 'white', alpha=0.2)

    # Plot combined regression equation and coefficients with p-values.
    # When _need_intercept (legacy add_combined_intercept OR a diagnostic flag
    # forced the with-const fit), params[0]=intercept, params[1]=slope.
    combined_coef = combined_model.params[1] if _need_intercept else combined_model.params[0]
    if _need_intercept:
        combined_intercept = combined_model.params[0]
    combined_p_value = combined_model.pvalues[1] if _need_intercept else combined_model.pvalues[0]
    combined_ste = combined_model.bse[1] if _need_intercept else combined_model.bse[0]
    combined_rsq = combined_model.rsquared
    if combined_p_value < 0.001:
        combined_significance = '***'
    elif combined_p_value < 0.01:
        combined_significance = '**'
    elif combined_p_value < 0.05:
        combined_significance = '*'
    else:
        combined_significance = ''
    slope_frac = r'$\frac{{}^{\circ}\mathrm{C}}{10\%}$'
    if add_combined_intercept:
        isign = '+' if combined_intercept >= 0 else '-'
        combined_equation = fr"$\hat{{y}}$ = {combined_coef * 10:.2f}$^{{{{\text{{{combined_significance}}}}}}}$ {slope_frac} $\cdot$ x {isign} {abs(combined_intercept):.2f} $^{{\circ}}$C"
    else:
        combined_equation = fr"$\hat{{y}}$ = {combined_coef * 10:.2f}$^{{{{\text{{{combined_significance}}}}}}}$ {slope_frac} $\cdot$ x"
    if combined_reg and not no_plots:
        ax.text(0.05, equation_y_pos,
                f"Combined: {combined_equation}",
                transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black' if not plot_bg=='black' else 'white')


    if hosmip_cesm:
        hosmip_regression_plot(multi_model_dict, quantile_reg=False, plot_bg=plot_bg, ax=ax, region=region, window=window, only_model='CESM2', only_model_label='NAHosMIP CESM2 (preindustrial)', low_ylim=low_ylim, central_reg=False, lat=lat, lon=lon)

    if not no_plots:
        ax.set_xlim(0, xlim)
        ax.set_ylim(low_ylim, 0)
        ax.set_xlabel("Additional AMOC weakening [%]")
        if is_latlon:
            ax.set_ylabel(f"Deviation from no-hosing lat={lat}, lon={lon} temperature [°C]")
        elif is_region:
            ax.set_ylabel(fr"Dev. from no-hosing $\Delta T_{{{region}}}$ [°C]")
        ax.axhline(0, color='black' if not plot_bg=='black' else 'white', lw=1.5, ls='--')
        ax.axvline(0, color='black' if not plot_bg=='black' else 'white', lw=1.5, ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(fontsize=12, frameon=False, loc='center left')

        if savefig is not False:
            plt.savefig(savefig, dpi=400, bbox_inches='tight', transparent=True if plot_bg=='black' else False)

    if combined_reg and add_combined_intercept:
        combined_quadruple = np.round(combined_intercept, 4), np.round(combined_coef, 5), np.round(combined_ste, 5), np.round(combined_rsq, 4)
    elif combined_reg and not add_combined_intercept:
        combined_quadruple = np.nan, np.round(combined_coef, 5), np.round(combined_ste, 5), np.round(combined_rsq, 4)

    # SSP-specific tuples: (coef, ste, rsq)
    ssp126_tuple = (np.round(ssp_coefs['ssp126'], 5), np.round(ssp_stes['ssp126'], 5), np.round(ssp_rsqs['ssp126'], 4)) if ssp == 'all' or ssp == 'ssp126' else None
    ssp585_tuple = (np.round(ssp_coefs['ssp585'], 5), np.round(ssp_stes['ssp585'], 5), np.round(ssp_rsqs['ssp585'], 4)) if ssp == 'all' or ssp == 'ssp585' else None

    # last_call cache for downstream figure scripts (mirrors
    # regression_plot.last_call). Always populated when combined_reg=True so
    # FigSXX_*_cesm.py can read per-point arrays for panels e/f even without
    # the diagnostic flags. Populated keys depend on flags — diagnostic keys
    # only present when their compute_* flag is set.
    if combined_reg:
        cesm_regressions.last_call = {
            'combined_x': np.asarray(combined_x),
            'combined_y': np.asarray(combined_y),
            'combined_scenario_id': np.asarray(combined_scenario_id),
            'combined_scenario_name': np.asarray(combined_scenario_name),
            'combined_coef': float(combined_coef),
            'combined_intercept': float(combined_intercept) if _need_intercept else np.nan,
            'combined_ste': float(combined_ste),
            'combined_rsq': float(combined_rsq),
            'n_combined': int(len(combined_y)),
            'window': int(window),
        }

        # HAC cov_spec used by both diagnostics — matches the cesm_regressions
        # convention (cov_type='HAC', maxlags=window-1; G=2 → cluster-robust
        # asymptotics degenerate).
        _hac_spec = ('HAC', {'maxlags': window - 1})

        if compute_linearity:
            _diag = reset_diagnostics(
                combined_y, combined_x_with_const, combined_scenario_id,
                combined_model, cov_spec=_hac_spec,
            )
            cesm_regressions.last_call.update({
                'partial_r2': _diag['partial_r2'],
                'delta_r2_total': _diag['delta_r2_total'],
                'gamma2': _diag['gamma2'],
                'gamma3': _diag['gamma3'],
                'beta2_sign': _diag['beta2_sign'],
                'beta3_sign': _diag['beta3_sign'],
                'reset_pvalue': _diag['reset_pvalue'],
                'baseline_r2': _diag['baseline_r2'],
                'combined_yhat': _diag['yhat'],
                'combined_resid': _diag['resid'],
                'combined_intercept_aug': _diag['intercept_aug'],
                'combined_coef_aug': _diag['coef_aug'],
            })

        if compute_scenario_independence:
            _diag_si = scenario_independence_diagnostics(
                combined_y, combined_x_with_const, combined_scenario_id,
                np.asarray(combined_scenario_name), combined_model,
                cov_spec=_hac_spec, scenarios=SCENARIOS_CESM,
            )
            cesm_regressions.last_call.update({
                'wald_fstat': _diag_si['wald_fstat'],
                'wald_pvalue': _diag_si['wald_pvalue'],
                'partial_r2_indep': _diag_si['partial_r2_indep'],
                'delta_r2_total_indep': _diag_si['delta_r2_total_indep'],
                'baseline_r2': _diag_si['baseline_r2'],
                'beta_ssp126': _diag_si['beta_ssp126'],
                'beta_ssp585': _diag_si['beta_ssp585'],
                'intercept_ssp126': _diag_si['intercept_ssp126'],
                'intercept_ssp585': _diag_si['intercept_ssp585'],
                'max_pairwise_slope_diff': _diag_si['max_pairwise_slope_diff'],
                'n_scenarios_present': _diag_si['n_scenarios_present'],
            })

    return {
        'combined': combined_quadruple,
        'ssp126': ssp126_tuple,
        'ssp585': ssp585_tuple,
    }


def plot_net_cooling_ranges(dfs_countries, T_ref='pi', AMOC_metric='rel', season='', plot_bg='white', ext_ax=None, title=True):
    plt.style.use('default')
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    if ext_ax is None:
        fig, ax = plt.subplots(figsize=(10, 14))
    else:
        ax = ext_ax

    effect_sizes = {}
    for r in dfs_countries['']['ssp126'].index:
        effect_sizes[r] = dfs_countries['']['ssp245'][f'CP_pi_hosmip_5pct_{AMOC_metric}'][r]

    regions = sorted(effect_sizes.keys(), key=lambda x: effect_sizes[x], reverse=True if AMOC_metric == 'rel' else False)
    countries = [r for r in regions if len(r) == 2 and r != 'EU']

    group_centers = np.arange(len(countries))
    bar_height = 0.3

    bar_alpha = 0.7
    line_alpha = 1.0

    for (i, r) in enumerate(countries):
        
        ax.barh(group_centers[i] + bar_height, 
                dfs_countries[season]['ssp126'][f'CP_{T_ref}_hosmip_5pct_{AMOC_metric}'][r] - dfs_countries[season]['ssp126'][f'CP_{T_ref}_hosmip_95pct_{AMOC_metric}'][r],
                left = dfs_countries[season]['ssp126'][f'CP_{T_ref}_hosmip_95pct_{AMOC_metric}'][r], 
                height=bar_height * 0.8, alpha=bar_alpha, color=hosing_colors['ssp126']['ge'])
        for model in hosmip_labels:
            ax.barh(group_centers[i] + bar_height,
                    (0.4 if AMOC_metric == 'rel' else 0.08),
                    left = dfs_countries[season]['ssp126'][f'CP_{T_ref}_hosmip_'+model+f'_{AMOC_metric}'][r] - (0.2 if AMOC_metric == 'rel' else 0.04),
                    height=bar_height * 1.0, alpha=line_alpha, color=hosing_colors['ssp126']['ge'] if model=='MPI-ESM1-2-LR' else 'none' if model=='MPI-ESM1-2-HR' else 'none' if model=='CESM2' else 'none')
        if AMOC_metric == 'rel':
            ax.scatter((1 - dfs_countries[season]['ssp126'][f'CP_{T_ref}_all_ssps_combined'][r] / AMOC_pi_MPI) * 100, group_centers[i] + 1.1 * bar_height, alpha=line_alpha, color=hosing_colors['ssp126']['ge'], marker='*')
        else:
            ax.scatter(dfs_countries[season]['ssp126'][f'CP_{T_ref}_all_ssps_combined'][r], group_centers[i] + 1.1 * bar_height, alpha=line_alpha, color=hosing_colors['ssp126']['ge'], marker='*')

        ax.barh(group_centers[i], 
                dfs_countries[season]['ssp245'][f'CP_{T_ref}_hosmip_5pct_{AMOC_metric}'][r] - dfs_countries[season]['ssp245'][f'CP_{T_ref}_hosmip_95pct_{AMOC_metric}'][r],
                left = dfs_countries[season]['ssp245'][f'CP_{T_ref}_hosmip_95pct_{AMOC_metric}'][r], 
                height=bar_height * 0.8, alpha=bar_alpha, color=hosing_colors['ssp245']['ge'])
        for model in hosmip_labels:
            ax.barh(group_centers[i],
                    (0.4 if AMOC_metric == 'rel' else 0.08),
                    left = dfs_countries[season]['ssp245'][f'CP_{T_ref}_hosmip_'+model+f'_{AMOC_metric}'][r] - (0.2 if AMOC_metric == 'rel' else 0.04),
                    height=bar_height * 1.0, alpha=line_alpha, color='darkorange' if model=='MPI-ESM1-2-LR' else 'none' if model=='MPI-ESM1-2-HR' else 'none' if model=='CESM2' else 'none')
        if AMOC_metric == 'rel':
            ax.scatter((1 - dfs_countries[season]['ssp245'][f'CP_{T_ref}_all_ssps_combined'][r] / AMOC_pi_MPI) * 100, group_centers[i] + 0.1 * bar_height, alpha=line_alpha, color=hosing_colors['ssp245']['ge'], marker='*')
        else:
            ax.scatter(dfs_countries[season]['ssp245'][f'CP_{T_ref}_all_ssps_combined'][r], group_centers[i] + 0.1 * bar_height, alpha=line_alpha, color=hosing_colors['ssp245']['ge'], marker='*')

        ax.barh(group_centers[i] - bar_height, 
                dfs_countries[season]['ssp370'][f'CP_{T_ref}_hosmip_5pct_{AMOC_metric}'][r] - dfs_countries[season]['ssp370'][f'CP_{T_ref}_hosmip_95pct_{AMOC_metric}'][r],
                left = dfs_countries[season]['ssp370'][f'CP_{T_ref}_hosmip_95pct_{AMOC_metric}'][r], 
                height=bar_height * 0.8, alpha=bar_alpha, color=hosing_colors['ssp370']['ge'])
        for model in hosmip_labels:
            ax.barh(group_centers[i] - bar_height,
                    (0.4 if AMOC_metric == 'rel' else 0.08),
                    left = dfs_countries[season]['ssp370'][f'CP_{T_ref}_hosmip_'+model+f'_{AMOC_metric}'][r] - (0.2 if AMOC_metric == 'rel' else 0.04),
                    height=bar_height * 1.0, alpha=line_alpha, color='firebrick' if model=='MPI-ESM1-2-LR' else 'none' if model=='MPI-ESM1-2-HR' else 'none' if model=='CESM2' else 'none')
        if AMOC_metric == 'rel':
            ax.scatter((1 - dfs_countries[season]['ssp370'][f'CP_{T_ref}_all_ssps_combined'][r] / AMOC_pi_MPI) * 100, group_centers[i] - 0.9 * bar_height, alpha=line_alpha, color=hosing_colors['ssp370']['ge'], marker='*')
        else:
            ax.scatter(dfs_countries[season]['ssp370'][f'CP_{T_ref}_all_ssps_combined'][r], group_centers[i] - 0.9 * bar_height, alpha=line_alpha, color=hosing_colors['ssp370']['ge'], marker='*')

    # shade every second row in light gray
    for i in range(len(countries)):
        if i % 2 == 0:
            if AMOC_metric == 'rel':
                ax.add_patch(Rectangle((-20, group_centers[i] - 0.5), 120, 1, color='black', alpha=0.15, zorder=0, linewidth=0, clip_on=False))
            else:
                ax.add_patch(Rectangle((-3.5, group_centers[i] - 0.5), 23.5, 1, color='black', alpha=0.15, zorder=0, linewidth=0, clip_on=False))


    range_y = 0.98  # y position just above the plot (in axes coords)
    range_height = 0.004  # thickness of the colored range

    for idx, ssp in enumerate(ssps):
        # Get the AMOC_extent range for this SSP
        lower, upper = AMOC_extent[ssp]
        x_min = lower if AMOC_metric == 'rel' else (1 - lower / 100) * AMOC_pi_MPI
        x_max = upper if AMOC_metric == 'rel' else (1 - upper / 100) * AMOC_pi_MPI
        # Convert to axes fraction
        x_min_frac = x_min / (100 if AMOC_metric == 'rel' else 20)
        x_max_frac = x_max / (100 if AMOC_metric == 'rel' else 20)
        # Rectangle in axes coordinates
        rect = Rectangle(
            (x_min_frac, range_y - idx * 0.007),  # (x, y) in axes fraction
            x_max_frac - x_min_frac,  # width
            range_height,             # height
            transform=ax.transAxes,
            color=hosing_colors[ssp]['ge'],
            alpha=1.0,
            clip_on=False,
        )
        ax.add_patch(rect)

    arrow_start_x = 0.65 if AMOC_metric == 'rel' else 0.3      # right edge in axes fraction
    arrow_end_x = 0.55 if AMOC_metric == 'rel' else 0.4       # where the arrow points to
    arrow_y = range_y - 0.005

    # Add arrow (in axes coordinates)
    ax.annotate(
        '', xy=(arrow_end_x, arrow_y), xytext=(arrow_start_x, arrow_y),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='-|>', lw=1., color='black' if plot_bg != 'black' else 'white'),
        annotation_clip=False
    )

    # Add text box (in axes coordinates)
    ax.text(
        0.84 if AMOC_metric == 'rel' else 0.3,  # x position in axes fraction
        arrow_y, 'CMIP6 projections',
        va='center', ha='right',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', fc='white' if plot_bg != 'black' else '#222', ec='none', alpha=0.8),
        color='black' if plot_bg != 'black' else 'white',
        fontsize=10, weight='bold'
    )

    if AMOC_metric == 'rel':
        for x in [20, 40, 60, 80]:
            ax.axvline(x=x, ymin=0.03, color='black' if plot_bg != 'black' else 'white', linestyle=(0, (1, 4)), linewidth=0.8, alpha=1.0, zorder=0)

    if AMOC_metric == 'rel':
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
        ax.set_xlabel('AMOC weakening')
        ref_temperature = 'preindustrial' if T_ref == 'pi' else 'present-day'
        season_text = 'in winter ' if season == 'djf' else 'in summer ' if season == 'jja' else ''
        if title:
            ax.set_title(f'How much AMOC weakening would lead to net cooling {season_text}wrt {ref_temperature}?', fontsize=14, color='black' if plot_bg != 'black' else 'white')
        ax.xaxis.set_ticks_position('both')
        # ax.xaxis.set_label_position('bottom')  # keep label at bottom, or use 'top' for top
        ax.tick_params(axis='x', which='both', top=True, labeltop=True, bottom=True, labelbottom=True, labelsize=10)
    else:
        ax.set_xlim(0, 20)
        ax.set_xticks(np.arange(0, 21, 4))
        ax.set_xlabel('AMOC strength [Sv]')
        if title:
            ax.set_title('At which AMOC strength would net cooling emerge?', fontsize=14, color='black' if plot_bg != 'black' else 'white')

    ax.set_yticks(group_centers)
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_yticklabels(country_names[c] for c in countries)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines["left"].set_position(("axes", 0.0))
    ax.spines["bottom"].set_position(("axes", 0.03))


def plot_net_cooling_ranges_mpi_cesm(reg_ds_mpi, reg_ds_cesm, masks, hosmip_reg_ds_dict=None, hosmip_markers=False, T_ref='pi', season='', plot_bg='white', ext_ax=None, title=True, savefig=False, reg_ds_giss=None, giss_time_period='2101-2300', cmip_cooling_ds=None, cmip_cooling_decade=None, aggregate_first=True, amoc_extent=None, weakening_unit='pct', sv_xmax=None, amoc_extent_y=1.03):
    # aggregate_first=True swaps the order in which the per-region net-cooling
    # point is built: instead of area-averaging the cached per-pixel
    # req_weakening_pi (which is the non-linear quantity (AMOC_pi - (AMOC_2090 -
    # ΔT/slope))/AMOC_pi*100), it area-averages slope, T_pi and T_2090 first
    # and then computes req_weakening once on the regional scalars. With
    # req_weakening ∝ 1/slope, the two orderings disagree by a Jensen-type
    # gap that can dominate the visible HosMIP-vs-combined-forcing offset
    # in panel e even when the regional-mean slopes coincide.
    # ``aggregate_first`` works under both ``T_ref='pi'`` and ``T_ref='pd'``;
    # the only difference is which baseline tas field substitutes into the
    # regional (T_90 - T_ref) / slope expression. T_pd_245 is annual-only,
    # so seasonal aggregate-first under PD is still not implemented (see the
    # explicit check further down inside the HosMIP branch).
    plt.style.use('default')
    # weakening_unit='sv' plots the net-cooling point as absolute Sv weakening
    # below each model's own PI instead of % weakening. It is only defined for
    # the aggregate_first path (regional-scalar req_strength); the per-pixel
    # req_weakening_pi field is %-only and never used here. The CMIP-cooling
    # overlay carries %-only weakening, so it is unsupported under 'sv'.
    if weakening_unit not in ('pct', 'sv'):
        raise ValueError(f"weakening_unit must be 'pct' or 'sv'; got {weakening_unit!r}")
    if weakening_unit == 'sv':
        if not aggregate_first:
            raise NotImplementedError("weakening_unit='sv' requires aggregate_first=True")
        if cmip_cooling_ds is not None:
            raise NotImplementedError("weakening_unit='sv' with the CMIP-cooling overlay is not supported")

    def _net_cooling_x(amoc_pi, amoc_90, T_90, T_ref_, slope):
        """Net-cooling point: Sv weakening (unit='sv') or % weakening (default).

        Under 'sv', points beyond the model's own PI (>100% weakening, i.e. more
        than a full AMOC collapse) are suppressed to NaN so they are not plotted."""
        weakening_sv = amoc_pi - (amoc_90 - (T_90 - T_ref_) / slope)
        if weakening_unit != 'sv':
            return weakening_sv / amoc_pi * 100
        return weakening_sv.where(weakening_sv <= amoc_pi)

    # CMIP6 weakening envelope: canonical global by default; variant runs
    # pass their own window-keyed extent.
    _ext = AMOC_extent if amoc_extent is None else amoc_extent
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    if ext_ax is None:
        fig, ax = plt.subplots(figsize=(10, 14))
    else:
        ax = ext_ax

    # GISS is added as a marker layer on top of the MPI/CESM bars.
    # Selecting the requested fit window up front keeps the per-(region, ssp)
    # loop body unchanged; markers only render for ssps where GISS has a
    # populated req_weakening (ssp126 is single-realisation in our archive).
    if reg_ds_giss is not None:
        reg_ds_giss_sel = reg_ds_giss.sel(time_period=giss_time_period)

    # if T_ref != 'pi':
    #     raise NotImplementedError("Only T_ref='pi' is implemented in this function.")
    # if season != '':
    #     raise NotImplementedError("Only annual data (season='') is implemented in this function.")

    regions_cutoff30k = [r for r in regions if r not in ['LU', 'CY', 'XK', 'ME', 'SI', 'MK', 'AL']]

    cesm_effect_sizes = {}
    mpi_effect_sizes = {}
    for r in regions_cutoff30k:
        cesm_effect_sizes[r] = weighted_area_lat(reg_ds_cesm.req_weakening_pi.sel(scenar='ssp126').where(masks['CESM2'][r])).mean('lat').mean('lon')
        mpi_effect_sizes[r] = weighted_area_lat(reg_ds_mpi.req_weakening_pi.sel(scenar='ssp126').where(masks['MPI-ESM1-2-LR'][r])).mean('lat').mean('lon')

    # countries = sorted(cesm_effect_sizes.keys(), key=lambda x: cesm_effect_sizes[x], reverse=True)
    # countries = sorted(mpi_effect_sizes.keys(), key=lambda x: mpi_effect_sizes[x], reverse=True)
    countries = list(reversed(regions_cutoff30k))

    group_centers = np.arange(len(countries)+1)  # +1 for the EU aggregate
    bar_height = 0.25

    bar_alpha = 0.7
    line_alpha = 1.0

    vertical_offsets = {'ssp126': 1., 'ssp245': 0., 'ssp370': -1.}

    # Build custom legend handles
    legend_color = 'white' if plot_bg == 'black' else 'black'
    legend_handles = []
    legend_labels = []

    legend_handles.append(Line2D([], [], marker='*', color=legend_color, linestyle='None', markersize=8))
    legend_labels.append('MPI-ESM1.2-LR (this study)')

    if season == '':
        legend_handles.append(Line2D([], [], marker='d', color=legend_color, linestyle='None', markersize=5))
        legend_labels.append('CESM2 (Boot et al. 2024)')

    # GISS legend entry is independent of season — reg_ds_giss carries
    # seasonal req_weakening_pi/pd since the 2026-05-27 seasonal scenmip
    # wiring, so the circle marker appears in seasonal panels too.
    if reg_ds_giss is not None:
        legend_handles.append(Line2D([], [], marker='o', color=legend_color, linestyle='None', markersize=6, markeredgewidth=0))
        legend_labels.append(f'GISS-E2-1-G (Romanou et al. 2023, {giss_time_period} fit)')

    if season == '':
        # Combined forcing: thin bar (depends on CESM, which is annual-only).
        legend_handles.append(Line2D([], [], color=legend_color, alpha=bar_alpha, linewidth=5, solid_capstyle='butt'))
        legend_labels.append('Combined forcing range (MPI-ESM1.2-LR & CESM2)')

    # NAHosMIP: thin transparent bar
    hosmip_bar_h = Line2D([], [], color=legend_color, alpha=0.3, linewidth=5, solid_capstyle='butt')
    legend_handles.append(hosmip_bar_h)
    legend_labels.append('Preindustrial hosing range (NAHosMIP models)')

    # CMIP-projected cooling markers from CMIP6 ScenarioMIP runs. Visually
    # distinguished from the regression-derived net-cooling-point ranges by
    # being open (white face, coloured edge): they represent the decade in
    # which the country's decadal-mean tas drops below 1850-1899, not a
    # regression-projected threshold. Marker shape encodes the CMIP6 model.
    cmip_cooling_markers = {
        'cesm2':       's',  # square
        'mri-esm2-0':  '^',  # triangle up
        'noresm2-lm':  'p',  # pentagon
        'noresm2-mm':  'h',  # hexagon
    }
    cmip_cooling_labels = {
        'cesm2':       'CESM2',
        'mri-esm2-0':  'MRI-ESM2-0',
        'noresm2-lm':  'NorESM2-LM',
        'noresm2-mm':  'NorESM2-MM',
    }
    # cmip_cooling_decade selects which first-onset decade is overlaid:
    #   - a decade label (e.g. '2071-2080') -> only points whose first-onset
    #     decade is that decade, at the weakening % for that decade (used by
    #     the decade-matched supplementary figure);
    #   - None -> every point that cools in any decade, plotted at its own
    #     first-onset-decade weakening (the canonical Fig 3 "plot all" mode).
    # Only models with >=1 cooling point (in the selected decade, or any
    # decade when None) appear in the legend.
    cmip_cooling_models_with_data = []
    if cmip_cooling_ds is not None and season == '':
        dec_labels = [str(d) for d in cmip_cooling_ds.decade.values]
        sel_idx = dec_labels.index(cmip_cooling_decade) if cmip_cooling_decade is not None else None
        for mname in cmip_cooling_ds.model.values:
            if str(mname) not in cmip_cooling_markers:
                continue
            onset = cmip_cooling_ds.cmip_cooling_decade_idx.sel(model=mname)
            has_data = bool((onset == sel_idx).any()) if sel_idx is not None else bool((onset >= 0).any())
            if not has_data:
                continue
            cmip_cooling_models_with_data.append(str(mname))
            mkr = cmip_cooling_markers[str(mname)]
            lbl = cmip_cooling_labels[str(mname)]
            legend_handles.append(Line2D(
                [], [], marker=mkr, markerfacecolor='none',
                markeredgecolor=legend_color, color='none',
                linestyle='None', markersize=7, markeredgewidth=1.4))
            decade_tag = cmip_cooling_decade if cmip_cooling_decade is not None else 'first decade'
            legend_labels.append(f'CMIP-projected cooling ({decade_tag}) — {lbl}')

    if hosmip_markers:
        for mkr, sz, lbl in [('*', 8, 'NAHosMIP MPI-ESM1.2-LR'), ('^', 6, 'NAHosMIP MPI-ESM1.2-HR'),
                              ('d', 6, 'NAHosMIP CESM2'), ('<', 6, 'NAHosMIP HadGEM3-GC3.1-MM'),
                              ('>', 6, 'NAHosMIP HadGEM3-GC3.1-LL'), ('v', 6, 'NAHosMIP EC-Earth3'),
                              ('X', 6, 'NAHosMIP CanESM5'), ('P', 6, 'NAHosMIP IPSL-CM6A-LR'),
                              ]:
            legend_handles.append(Line2D([], [], marker=mkr, color=legend_color, linestyle='None', markersize=sz, markeredgewidth=0, alpha=bar_alpha))
            legend_labels.append(lbl)

    # LAND==0 is applied only for the EU aggregate, where we want a continental
    # land-mean diagnostic. For per-country aggregates, the country mask is
    # already land-aware via the overlap fallback in
    # make_country_masks_land_aware; further restricting to land_110 land
    # centroids would silently drop coastal cells that were deliberately
    # included via overlap (and would wipe DK on CanESM5 entirely).
    def _select(da, model, region):
        da = da.where(masks[model][region])
        if region == 'EU':
            da = da.where(masks[model]['LAND'] == 0)
        return da

    mpi_cesm_values_ssp126 = {}
    for (i, r) in enumerate(['EU']+countries):
        for ssp_i in ssps:

            # mpi_esm_value = (1 - dfs_countries[season][ssp_i][f'CP_{T_ref}_all_ssps_combined'][r] / AMOC_pi_MPI) * 100
            if T_ref not in ('pi', 'pd'):
                raise NotImplementedError("Only T_ref='pi' and T_ref='pd' are implemented in this function.")
            if aggregate_first:
                # Combined MPI: coef_ensmean is K/Sv (no unit conversion).
                amoc_pi_m  = float(reg_ds_mpi.AMOC_pi.sel(season=season).values)
                amoc_90_m  = float(reg_ds_mpi.AMOC_future.sel(scenar=ssp_i, season=season).values)
                if T_ref == 'pi':
                    T_ref_m = weighted_area_lat(_select(reg_ds_mpi.T_pi.sel(season=season), 'MPI-ESM1-2-LR', r)).mean('lat').mean('lon')
                else:
                    # reg_ds_mpi.T_pd_245 carries a season dim (size 3) even
                    # though the content is annual; .sel(season=season) reduces it.
                    T_ref_m = weighted_area_lat(_select(reg_ds_mpi.T_pd_245.sel(season=season), 'MPI-ESM1-2-LR', r)).mean('lat').mean('lon')
                T_90_m   = weighted_area_lat(_select(reg_ds_mpi.T_future.sel(scenar=ssp_i, season=season), 'MPI-ESM1-2-LR', r)).mean('lat').mean('lon')
                slope_m  = weighted_area_lat(_select(reg_ds_mpi.coef_ensmean.sel(season=season), 'MPI-ESM1-2-LR', r)).mean('lat').mean('lon')
                mpi_value = _net_cooling_x(amoc_pi_m, amoc_90_m, T_90_m, T_ref_m, slope_m)
                # Combined CESM: coef_ensmean is K/% — convert to K/Sv via (-100/AMOC_pi).
                # CESM2 reg_ds is annual-only by construction (no season dim).
                # Skip the CESM diamond + range in seasonal panels.
                if season == '':
                    amoc_pi_c = float(reg_ds_cesm.AMOC_pi.values)
                    amoc_90_c = float(reg_ds_cesm.AMOC_future.sel(scenar=ssp_i).values)
                    if T_ref == 'pi':
                        T_ref_c = weighted_area_lat(_select(reg_ds_cesm.T_pi, 'CESM2', r)).mean('lat').mean('lon')
                    else:
                        T_ref_c = weighted_area_lat(_select(reg_ds_cesm.T_pd_245, 'CESM2', r)).mean('lat').mean('lon')
                    T_90_c   = weighted_area_lat(_select(reg_ds_cesm.T_future.sel(scenar=ssp_i), 'CESM2', r)).mean('lat').mean('lon')
                    slope_c_pct = weighted_area_lat(_select(reg_ds_cesm.coef_ensmean, 'CESM2', r)).mean('lat').mean('lon')
                    slope_c = slope_c_pct * (-100) / amoc_pi_c
                    cesm_value = _net_cooling_x(amoc_pi_c, amoc_90_c, T_90_c, T_ref_c, slope_c)
                else:
                    cesm_value = np.nan
            elif T_ref == 'pi':
                mpi_value = weighted_area_lat(_select(reg_ds_mpi.req_weakening_pi.sel(scenar=ssp_i, season=season), 'MPI-ESM1-2-LR', r)).mean('lat').mean('lon')
                cesm_value = weighted_area_lat(_select(reg_ds_cesm.req_weakening_pi.sel(scenar=ssp_i), 'CESM2', r)).mean('lat').mean('lon') if season == '' else np.nan
            else:  # T_ref == 'pd'
                mpi_value = weighted_area_lat(reg_ds_mpi.req_weakening_pd.sel(scenar=ssp_i, season=season).where(masks['MPI-ESM1-2-LR'][r])).mean('lat').mean('lon')
                cesm_value = weighted_area_lat(reg_ds_cesm.req_weakening_pd.sel(scenar=ssp_i).where(masks['CESM2'][r])).mean('lat').mean('lon') if season == '' else np.nan

            if ssp_i == 'ssp126':
                mpi_value_item = mpi_value.values.item() if hasattr(mpi_value, 'values') else float(mpi_value)
                cesm_value_item = (cesm_value.values.item() if hasattr(cesm_value, 'values')
                                   else float(cesm_value))
                _finite = [v for v in (mpi_value_item, cesm_value_item) if np.isfinite(v)]
                mpi_cesm_values_ssp126[r] = min(_finite) if _finite else np.nan

            hosmip_values = []
            for model in hosmip_labels:
                if T_ref not in ('pi', 'pd'):
                    raise NotImplementedError("Only T_ref='pi' and T_ref='pd' are implemented in this function.")
                if aggregate_first:
                    # HosMIP: lin_coef_hosmip is K/% with cooling-pixel filter (<0)
                    # applied in the cached path at functions.py:3085; mirror it
                    # here before area-averaging so the regional slope is built
                    # from the same pixel set the per-pixel cache uses.
                    hr_ds = hosmip_reg_ds_dict[model]
                    amoc_pi_h = float(hr_ds.AMOC_pi.values)
                    amoc_90_h = float(hr_ds.AMOC_future.sel(scenar=ssp_i).values)
                    if T_ref == 'pi':
                        T_ref_h = weighted_area_lat(_select(hr_ds.T_pi.sel(season=season), model, r)).mean('lat').mean('lon')
                    else:
                        T_ref_h = weighted_area_lat(_select(hr_ds.T_pd_245.sel(season=season), model, r)).mean('lat').mean('lon')
                    T_90_h = weighted_area_lat(_select(hr_ds.T_future.sel(scenar=ssp_i, season=season), model, r)).mean('lat').mean('lon')
                    slope_h_pct_field = hr_ds.lin_coef_hosmip.sel(season=season)
                    slope_h_pct = weighted_area_lat(_select(slope_h_pct_field.where(slope_h_pct_field < 0), model, r)).mean('lat').mean('lon')
                    slope_h = slope_h_pct * (-100) / amoc_pi_h
                    hosmip_value = _net_cooling_x(amoc_pi_h, amoc_90_h, T_90_h, T_ref_h, slope_h)
                elif T_ref == 'pi':
                    hosmip_value = weighted_area_lat(_select(hosmip_reg_ds_dict[model].req_weakening_pi.sel(scenar=ssp_i, season=season), model, r)).mean('lat').mean('lon')
                else:  # T_ref == 'pd'
                    hosmip_value = weighted_area_lat(_select(hosmip_reg_ds_dict[model].req_weakening_pd.sel(scenar=ssp_i, season=season), model, r)).mean('lat').mean('lon')
                # if model in ['MPI-ESM1-2-LR', 'CESM2']:
                #     hosmip_values.append(hosmip_value.values.item())
                # else:
                #     continue
                hosmip_values.append(hosmip_value.values.item())
                
                if hosmip_markers:
                    hosmip_marker = 'x'
                    if model == 'MPI-ESM1-2-LR':
                        hosmip_marker = '*'
                    elif model == 'MPI-ESM1-2-HR':
                        hosmip_marker = '^'
                    elif model == 'CESM2':
                        hosmip_marker = 'd'
                    elif model == 'HadGEM3-GC3-1MM':
                        hosmip_marker = '<'
                    elif model == 'HadGEM3-GC3-1LL':
                        hosmip_marker = '>'
                    elif model == 'EC-Earth3':
                        hosmip_marker = 'v'
                    elif model == 'CanESM5':
                        hosmip_marker = 'X'
                    elif model == 'IPSL-CM6A-LR':
                        hosmip_marker = 'P'
                    else:
                        hosmip_marker = ''
                    ax.scatter(hosmip_value.values,
                            group_centers[i] + vertical_offsets[ssp_i] * bar_height, alpha=bar_alpha, color=hosing_colors[ssp_i]['ge'],
                            marker=hosmip_marker, s=(40 if hosmip_marker == '*' else 22), edgecolors='none', clip_on=True)

            _hv = [v for v in hosmip_values if np.isfinite(v)]
            if _hv:
                ax.barh(group_centers[i] - 0.01 + vertical_offsets[ssp_i] * bar_height,
                        max(_hv) - min(_hv),
                        left = min(_hv),
                        height=bar_height * 0.8, alpha=0.4, color=hosing_colors[ssp_i]['ge'], clip_on=True)

            if season == '':
                ax.barh(group_centers[i] - 0.01 + vertical_offsets[ssp_i] * bar_height,
                        mpi_value - cesm_value,
                        left = cesm_value,
                        height=bar_height * 0.8, alpha=bar_alpha, color=hosing_colors[ssp_i]['ge'])

            ax.scatter(mpi_value,
                       group_centers[i] + vertical_offsets[ssp_i] * bar_height, alpha=line_alpha, color=hosing_colors[ssp_i]['ge'], marker='*', s=50)
            if season == '':
                ax.scatter(cesm_value,
                        group_centers[i] + vertical_offsets[ssp_i] * bar_height, alpha=line_alpha, color=hosing_colors[ssp_i]['ge'], marker='d', s=25)

            # GISS marker (diamond). Drawn when reg_ds_giss is supplied,
            # for any season the dataset carries (annual + djf/jja since
            # 2026-05-26 (g) GISS seasonal pipeline). NaN values (e.g. when
            # the slope sign yields an unrealistic req_weakening for ssp126)
            # are silently skipped by matplotlib. GISS does not contribute
            # to the bar range.
            if reg_ds_giss is not None:
                if aggregate_first:
                    # GISS: coef_ensmean is K/Sv (matches MPI convention, per
                    # functions.py:3698-3702), so no unit conversion needed.
                    amoc_pi_g = float(reg_ds_giss_sel.AMOC_pi.values)
                    amoc_90_g = float(reg_ds_giss_sel.AMOC_future.sel(scenar=ssp_i).values)
                    if T_ref == 'pi':
                        T_ref_g = weighted_area_lat(_select(reg_ds_giss_sel.T_pi.sel(season=season), 'GISS-E2-1-G', r)).mean('lat').mean('lon')
                    else:
                        # reg_ds_giss.T_pd_245 carries a size-1 season dim; reduce it.
                        T_ref_g = weighted_area_lat(_select(reg_ds_giss_sel.T_pd_245.sel(season=season), 'GISS-E2-1-G', r)).mean('lat').mean('lon')
                    T_90_g = weighted_area_lat(_select(reg_ds_giss_sel.T_future.sel(scenar=ssp_i, season=season), 'GISS-E2-1-G', r)).mean('lat').mean('lon')
                    slope_g = weighted_area_lat(_select(reg_ds_giss_sel.coef_ensmean.sel(season=season), 'GISS-E2-1-G', r)).mean('lat').mean('lon')
                    giss_value = _net_cooling_x(amoc_pi_g, amoc_90_g, T_90_g, T_ref_g, slope_g)
                elif T_ref == 'pi':
                    giss_value = weighted_area_lat(_select(
                        reg_ds_giss_sel.req_weakening_pi.sel(scenar=ssp_i, season=season),
                        'GISS-E2-1-G', r)).mean('lat').mean('lon')
                else:  # T_ref == 'pd'
                    giss_value = weighted_area_lat(_select(
                        reg_ds_giss_sel.req_weakening_pd.sel(scenar=ssp_i, season=season),
                        'GISS-E2-1-G', r)).mean('lat').mean('lon')
                ax.scatter(giss_value,
                           group_centers[i] + vertical_offsets[ssp_i] * bar_height,
                           alpha=0.8, color=hosing_colors[ssp_i]['ge'],
                           marker='o', s=40, edgecolors='none', zorder=11)

            # CMIP6 first-crossing markers (one per CMIP6 model with data).
            # Drawn at the same (row, ssp-vertical-offset) position as the
            # regression markers, with open faces so they read as a
            # qualitatively different quantity. Each marker is annotated
            # with the crossing year. Models without data for this
            # (country, scenario) pair are skipped silently.
            # CMIP-projected cooling is defined against the PI baseline (the
            # decade when local decadal-mean tas first dips below the PI mean),
            # so it is conceptually incompatible with a PD-anchored panel —
            # skip the markers when T_ref='pd'. Each marker sits at the AMOC
            # weakening % of its first-onset decade (or the selected decade)
            # and is labelled with that decade.
            if cmip_cooling_ds is not None and T_ref == 'pi' and season == '' and ssp_i in cmip_cooling_ds.scenario.values and r in cmip_cooling_ds.country.values:
                for mname in cmip_cooling_ds.model.values:
                    if str(mname) not in cmip_cooling_markers:
                        continue
                    if str(mname) not in cmip_cooling_models_with_data:
                        continue
                    onset_i = int(cmip_cooling_ds.cmip_cooling_decade_idx
                                  .sel(model=mname, scenario=ssp_i, country=r))
                    if onset_i < 0:
                        continue
                    # Decade-matched mode: only points whose first-onset decade
                    # is the selected one. None: plot every point at its onset.
                    plot_idx = sel_idx if sel_idx is not None else onset_i
                    if sel_idx is not None and onset_i != sel_idx:
                        continue
                    val = float(cmip_cooling_ds.amoc_weakening_pct
                                .isel(decade=plot_idx)
                                .sel(model=mname, scenario=ssp_i, country=r))
                    if not np.isfinite(val):
                        continue
                    y_pos = group_centers[i] + vertical_offsets[ssp_i] * bar_height
                    ax.scatter(val, y_pos,
                               s=60, marker=cmip_cooling_markers[str(mname)],
                               facecolors='none',
                               edgecolors=hosing_colors[ssp_i]['ge'],
                               linewidths=1.4, alpha=line_alpha,
                               zorder=12, clip_on=True)
                    # Annotate the decade only in plot-all mode (None), where
                    # markers span different decades. In decade-matched mode
                    # every marker is the panel's decade, so the label is
                    # redundant (the panel title carries it).
                    if sel_idx is None:
                        ax.text(val + 1.2, y_pos, dec_labels[plot_idx],
                                fontsize=6, ha='left', va='center',
                                color=hosing_colors[ssp_i]['ge'],
                                zorder=13, clip_on=True)

    # Axes-fraction denominator: x-axis spans [0, 100]% or [0, _xmax] Sv.
    _xmax = (sv_xmax if sv_xmax is not None else 20) if weakening_unit == 'sv' else 100
    _den = _xmax if weakening_unit == 'sv' else 100

    # shade every second row in light gray (1.2× the axis width, clip_on=False
    # so the shading runs slightly past the right spine — scaled to the unit).
    for i in range(len(countries)+1):
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, group_centers[i] - 0.5), 1.2 * _xmax, 1, color='black', alpha=0.15, zorder=0, linewidth=0, clip_on=False))

    range_y = amoc_extent_y  # y position above the upper x-axis (in axes coords)
    range_height = 0.006  # thickness of the colored range

    for idx, ssp in enumerate(ssps):
        # Get the AMOC_extent range for this SSP (Sv bounds when unit='sv').
        lower, upper = _ext[ssp]
        x_min = lower # 100 -
        x_max = upper
        # Convert to axes fraction
        x_min_frac = x_min / _den
        x_max_frac = x_max / _den
        # Rectangle in axes coordinates
        rect = Rectangle(
            (x_min_frac, range_y + idx * 0.01),  # (x, y) in axes fraction
            x_max_frac - x_min_frac,  # width
            range_height,             # height
            transform=ax.transAxes,
            color=hosing_colors[ssp]['ge'],
            alpha=1.0,
            clip_on=False,
        )
        ax.add_patch(rect)

        # MPI-ESM projected AMOC weakening as vertical line
        mpi_amoc_strength = float(reg_ds_mpi.AMOC_future.isel(season=0).sel(scenar=ssp))
        if weakening_unit == 'sv':
            mpi_weakening_frac = (AMOC_pi_MPI - mpi_amoc_strength) / _den
        else:
            mpi_weakening_frac = convert_strength_to_weakening(mpi_amoc_strength) / _den
        base_rgb = plt.matplotlib.colors.to_rgb(hosing_colors[ssp]['ge'])
        h, l, s = colorsys.rgb_to_hls(*base_rgb)
        line_color = colorsys.hls_to_rgb(h, max(0, l * 0.7), min(1, s * 1.3))
        bar_y = range_y + idx * 0.01
        ax.plot(
            [mpi_weakening_frac, mpi_weakening_frac],
            [bar_y - 0.001, bar_y + range_height + 0.001],
            color=line_color, linewidth=2,
            transform=ax.transAxes, clip_on=False, zorder=11
        )

        # CESM2 projected AMOC weakening as vertical line (lighter sibling of MPI tick)
        cesm_amoc_strength = float(reg_ds_cesm.AMOC_future.sel(scenar=ssp))
        cesm_amoc_pi = float(reg_ds_cesm.AMOC_pi)
        if weakening_unit == 'sv':
            cesm_weakening_frac = (cesm_amoc_pi - cesm_amoc_strength) / _den
        else:
            cesm_weakening_frac = (cesm_amoc_pi - cesm_amoc_strength) / cesm_amoc_pi
        cesm_line_color = colorsys.hls_to_rgb(h, min(1, l + 0.30), min(1, s * 1.3))
        ax.plot(
            [cesm_weakening_frac, cesm_weakening_frac],
            [bar_y - 0.001, bar_y + range_height + 0.001],
            color=cesm_line_color, linewidth=2,
            transform=ax.transAxes, clip_on=False, zorder=10
        )

    arrow_start_x = 0.65      # right edge in axes fraction (0.365 when arrows on the left)
    arrow_end_x = 0.55       # where the arrow points to (0.465 when arrows on the left)
    arrow_y = range_y + 0.009

    # Add arrow (in axes coordinates)
    ax.annotate(
        '', xy=(arrow_end_x, arrow_y), xytext=(arrow_start_x, arrow_y),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='-|>', lw=1., color='black' if plot_bg != 'black' else 'white'),
        annotation_clip=False
    )

    ax.legend(legend_handles, legend_labels,
              frameon=True, bbox_to_anchor=(0.02, 0.055), loc='lower left', fontsize=10,
              title=('Annual' if season=='' else f'Seasonal ({season.upper()})') + ' net-cooling AMOC weakening w.r.t. ' + ('preindustrial' if T_ref=='pi' else 'present-day'),
              title_fontproperties={'weight': 'bold', 'size': 10})

    # Add text box (in axes coordinates)
    ax.text(
        0.66,  # x position in axes fraction (0.18 when on the left)
        arrow_y, 'CMIP6 AMOC projections',
        va='center', ha='left',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', fc='white' if plot_bg != 'black' else '#222', ec='none', alpha=0.8),
        color='black' if plot_bg != 'black' else 'white',
        fontsize=10, weight='bold'
    )

    _grid_ticks = np.arange(5, _xmax, 5) if weakening_unit == 'sv' else np.arange(10, 100, 10)
    for x in _grid_ticks:
        ax.axvline(x=x, ymin=0.03, color='black' if plot_bg != 'black' else 'white', linestyle=(0, (1, 4)), linewidth=0.8, alpha=1.0, zorder=0)

    ax.set_ylim(-1.7, len(countries)+0.5)
    if weakening_unit == 'sv':
        ax.set_xlim(0, _xmax)
        ax.set_xticks(np.arange(0, _xmax + 1, 5))
        ax.set_xlabel('AMOC weakening [Sv]')
    else:
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 110, 10))
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
        ax.set_xlabel('AMOC weakening')
    ref_temperature = 'preindustrial' if T_ref == 'pi' else 'present-day'
    season_text = 'in winter ' if season == 'djf' else 'in summer ' if season == 'jja' else ''
    if title:
        ax.set_title(f'How much AMOC weakening would lead to net cooling {season_text}wrt {ref_temperature}?', fontsize=14, color='black' if plot_bg != 'black' else 'white')
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_label_position('bottom')  # keep label at bottom, or use 'top' for top
    ax.tick_params(axis='x', which='both', top=True, labeltop=True, bottom=True, labelbottom=True, labelsize=10)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.set_yticks(group_centers)
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_yticklabels(['European mean']+[country_names[c] for c in countries])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.spines["left"].set_position(("axes", 0.0))
    ax.spines["bottom"].set_position(("axes", 0.03))

    if savefig:
        agg_tag = '_aggfirst-on' if aggregate_first else '_aggfirst-off'
        fig.savefig(f'../plots/cesm2_mpi_net_cooling_ranges_Tref-{T_ref}_season-{season}_plotbg-{plot_bg}_hosmip_markers-{hosmip_markers}{agg_tag}.png', dpi=400, bbox_inches='tight', transparent=True if plot_bg=='black' else False)


def get_mask_plot(masks, region, ext_ax=None, alpha=0.15, plot_bg='white'):

    plt.style.use('default')
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    if ext_ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.Robinson(central_longitude=8)})
    else:
        ax = ext_ax
    ax.spines['geo'].set_visible(False)
    ax.set_rasterized(True)
    ax.coastlines(resolution='50m', color='black' if not plot_bg=='black' else 'white', linewidth=.5)
    # ax.add_feature(cfeature.BORDERS, edgecolor='black' if not plot_bg=='black' else 'white', linewidth=.5)
    ax.add_feature(cfeature.LAND, facecolor='white' if not plot_bg=='black' else '#191919')
    ax.add_feature(cfeature.OCEAN, facecolor='white' if not plot_bg=='black' else '#191919', zorder=2)

    masks['HadGEM3-GC3-1MM'][region].plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='Greys',
        alpha=alpha,
        add_colorbar=False,
        rasterized=True
    )

    # ax.set_extent([-9.5, 38.5, 34.8, 72], crs=ccrs.PlateCarree())
    ax.set_extent([-11.0, 38.5, 30, 72], crs=ccrs.PlateCarree())

    # add_square(ax, 39.5, 65,  64.5, 75.5, colour='white' if not plot_bg=='black' else '#191919') # cut off North Russia where previously there was no data
    add_square(ax, -15., 0,  60, 75.5, colour='white' if not plot_bg=='black' else '#191919') # get rid of Nordic sea small islands