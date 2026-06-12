"""CMIP-projected cooling overlay dataset for Fig 3 (decadal-block means).

For each (model, scenario, country, decade) records the decadal-mean tas
anomaly vs the 1850-1899 baseline and the decadal-mean AMOC weakening (%)
from the 1850-1899 AMOC baseline. A country "reaches CMIP-projected
cooling" in the first decade whose decadal-mean tas drops below baseline;
``cmip_cooling_decade_idx`` records that first-onset decade (index into
``decade``, -1 if it never cools).

The decadal-block construction matches how the Fig 3 net-cooling-point
*ranges* are built (a fixed end-of-century-window mean), so a CMIP point
can be compared against the range for the *same* decade. "Net cooling" is
reserved for the regression-derived ranges; this dataset is the CMIP
projection overlaid on them.

Realisations are pair-concatenated (historical + scenario per member,
intersecting the rea lists across tas and amoc) and then ensemble-mean is
taken across realisations before decadal means are computed. MRI-ESM2-0
ssp126 r1i1p1f1 uses the 2300 extension file as the scenario file
(replacement, not concat — the extension is 2015-2300 self-contained).

NorESM2-LM/-MM are excluded from the default model set: both contribute a
single ensemble member, unlike CESM2 and MRI-ESM2-0. Pass ``models=`` to
include them.

Cache: data/cmip_cooling.nc.

Run from the scripts/ directory as
    python cmip_cooling.py
or interactively with VS Code cells.
"""

# %%
import json
import os
import warnings
import importlib

import numpy as np
import xarray as xr

import sys, pathlib
if "__file__" in globals():  # standalone run: ensure scripts/ is importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import functions
importlib.reload(functions)

warnings.filterwarnings('ignore', category=DeprecationWarning)


# %%
# CONFIG

import cmip6_inventory

# Search roots, retracted-rea set, and dual-root resolution all live in
# ``cmip6_inventory``. Module-level aliases below preserve the names used
# elsewhere in this script (and by any external readers grep-ing for
# CMIP6_BASE / BU1431_BASE).
CMIP6_BASE, BU1431_BASE = cmip6_inventory.SEARCH_ROOTS
SEARCH_ROOTS = cmip6_inventory.SEARCH_ROOTS

# Retracted reas live in cmip6_inventory.RETRACTED. This alias is kept for
# downstream code that may import ``CESM2_RETRACTED_REA`` directly.
CESM2_RETRACTED_REA = {rea for model, rea in cmip6_inventory.RETRACTED
                       if model == 'CESM2'}

CACHE_PATH = '/home/m/m300940/teu_amoc/data/cmip_cooling.nc'

DEFAULT_MODELS = ['cesm2', 'mri-esm2-0']
DEFAULT_SCENARIOS = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

BASELINE_SLICE = slice('1850', '1899')

# Candidate decades scanned for CMIP-projected cooling. Each is a fixed
# decadal block whose mean is compared against the per-country baseline,
# matching the construction of the Fig 3 net-cooling-point ranges
# (functions.FUTURE_WINDOW = ('2091','2100') is the last entry).
DECADES = [('2051', '2060'), ('2061', '2070'), ('2071', '2080'),
           ('2081', '2090'), ('2091', '2100')]
DECADE_LABELS = [f'{a}-{b}' for a, b in DECADES]


# %%
# HELPERS

def _list_realisations(scenario, var, models):
    """Return ``{model: sorted list of realisation strings}`` by scanning disk.

    Thin wrapper around :func:`cmip6_inventory.available_realisations`.
    Uses ``esgf_names=False`` so the returned keys match the lowercase
    directory names this module operates on. The retracted-rea filter
    (currently CESM2 r1-r3 i1p1f1) is applied inside the inventory layer.

    The JSON index files in the same CMIP6 directories are **incomplete**
    (e.g. CESM2 ssp126/ssp370 absent) so we never use them; scanning the
    filesystem via the inventory layer is the authoritative source.
    """
    return cmip6_inventory.available_realisations(
        scenario, var, models=list(models), esgf_names=False)


def _resolve_path(model, scenario, rea, var):
    """Return the actual file path for a (model, scenario, rea, var) tuple.

    Thin wrapper around :func:`cmip6_inventory.resolve_path`. Preserves the
    MRI-ESM2-0 ssp126 r1i1p1f1 extension_2300 special case (handled inside
    the inventory layer).
    """
    return cmip6_inventory.resolve_path(model, scenario, rea, var)


def _open_tas(model, scenario, rea):
    """Open a yearly tas file as a DataArray, K, lat/lon native grid."""
    p = _resolve_path(model, scenario, rea, 'tas')
    return xr.open_dataarray(p, use_cftime=True)


def _ensure_continuous_yearly(da):
    """Reindex `da` to a continuous yearly cftime axis, NaN-padding gaps.

    `_ens_mean_concat` concatenates per-rea historical + scenario, but
    some rea (e.g. CESM2 ssp126 r10 amoc, 2065-2100 only) leave a gap.
    xarray's `.rolling(time=N).mean()` is index-based and would silently
    average across the gap. Padding with NaN forces correct propagation:
    decadal-block means that span the gap return NaN.
    """
    if 'time' not in da.dims:
        return da
    years = da.time.dt.year.values
    if years.size == 0 or (np.diff(years) == 1).all():
        return da
    cls = type(da.time.values[0])
    full_years = np.arange(int(years.min()), int(years.max()) + 1)
    new_times = [cls(int(y), 1, 1) for y in full_years]
    return da.reindex(time=new_times)


def _open_amoc(model, scenario, rea):
    p = _resolve_path(model, scenario, rea, 'amoc')
    da = xr.open_dataarray(p, use_cftime=True)
    # Drop the scalar 'lev' coord (amoc26 is evaluated at max-over-lev).
    if 'lev' in da.coords:
        da = da.drop_vars('lev')
    return da


def _ens_mean_concat(model, scenario, rea_list, var):
    """Ensemble-mean of historical+scenario for the given var.

    Returns DataArray on the model's native grid (tas) or 1-D (amoc),
    spanning historical (1850-2014) followed by the scenario period.

    Per-rea historical is concatenated with per-rea scenario, then the
    ensemble mean is taken across realisations. Realisations missing from
    either period are skipped.
    """
    open_fn = _open_tas if var == 'tas' else _open_amoc

    # Build per-rea availability using _resolve_path so both search
    # roots are checked uniformly (uo1075 legacy + bu1431 mirror).
    hist_avail = []
    sce_avail = []
    for rea in rea_list:
        try:
            _resolve_path(model, 'historical', rea, var)
            hist_avail.append(rea)
        except FileNotFoundError:
            pass
        try:
            _ = open_fn(model, scenario, rea)
            sce_avail.append(rea)
        except FileNotFoundError:
            pass
    common = sorted(set(hist_avail) & set(sce_avail))
    if not common:
        return None, []

    per_rea_full = []
    for rea in common:
        h_path = _resolve_path(model, 'historical', rea, var)
        h = xr.open_dataarray(h_path, use_cftime=True)
        if 'lev' in h.coords:
            h = h.drop_vars('lev')
        s = open_fn(model, scenario, rea)
        full = xr.concat([h, s], dim='time')
        full = _ensure_continuous_yearly(full)
        per_rea_full.append(full)

    # Ensemble mean across realisations. Time axes must align;
    # if a single realisation extends further (MRI ssp126 r1) the others
    # would be padded with NaN. Use xr.concat then mean(skipna=True).
    ens = xr.concat(per_rea_full, dim='realiz').mean('realiz', skipna=True)
    return ens, common


def _country_series(tas, mask):
    """Country-area-weighted yearly tas series from a (time, lat, lon) field."""
    masked = tas.where(mask)
    return functions.weighted_area_lat(masked).mean('lat').mean('lon')


def _decadal_mean(series, dec):
    """Mean of a yearly series over a (start, end) decade block, or NaN."""
    block = series.sel(time=slice(dec[0], dec[1]))
    if block.time.size == 0:
        return np.nan
    return float(block.mean('time'))


# %%
# MAIN ROUTINE

def make_cmip_cooling_ds(recompute=False, models=None, scenarios=None,
                         verbose=True):
    """Build (or load from cache) the CMIP-projected cooling Dataset.

    Dims ``(model, scenario, country, decade)``. ``cmip_cooling_decade_idx``
    is the first decade index whose decadal-mean tas anomaly < 0 (-1 if the
    country never cools within the scanned decades).
    """
    if (not recompute) and os.path.exists(CACHE_PATH):
        if verbose:
            print(f"Loading cached CMIP-cooling dataset from {CACHE_PATH}")
        return xr.open_dataset(CACHE_PATH)

    models = list(models) if models is not None else list(DEFAULT_MODELS)
    scenarios = list(scenarios) if scenarios is not None else list(DEFAULT_SCENARIOS)
    countries = list(functions.country_codes.keys())  # 41 ISO-2 codes
    ndec = len(DECADES)

    # Per-decade arrays: (model, scenario, country, decade)
    shape4 = (len(models), len(scenarios), len(countries), ndec)
    tas_anomaly = np.full(shape4, np.nan, dtype=np.float64)
    amoc_weak_pct = np.full(shape4, np.nan, dtype=np.float64)
    amoc_strength = np.full(shape4, np.nan, dtype=np.float64)
    # Derived first-onset decade index: (model, scenario, country)
    onset_idx = np.full((len(models), len(scenarios), len(countries)), -1,
                        dtype=np.int32)

    baseline_tas_out = np.full((len(models), len(scenarios)), np.nan)
    baseline_amoc_out = np.full((len(models), len(scenarios)), np.nan)
    n_real_tas = np.zeros((len(models), len(scenarios)), dtype=np.int32)
    n_real_amoc = np.zeros((len(models), len(scenarios)), dtype=np.int32)
    mri_ext_used = np.zeros((len(models), len(scenarios)), dtype=bool)

    for mi, model in enumerate(models):
        if verbose:
            print(f"\n=== {model} ===")
        for si, sce in enumerate(scenarios):
            tas_rea = _list_realisations(sce, 'tas', [model])[model]
            amoc_rea = _list_realisations(sce, 'amoc', [model])[model]
            common = sorted(set(tas_rea) & set(amoc_rea))
            if verbose:
                print(f"  {sce}: tas={len(tas_rea)}, amoc={len(amoc_rea)}, "
                      f"common={len(common)}")
            if not common:
                continue
            tas_ens, tas_used = _ens_mean_concat(model, sce, common, 'tas')
            amoc_ens, amoc_used = _ens_mean_concat(model, sce, common, 'amoc')
            if tas_ens is None or amoc_ens is None:
                continue
            n_real_tas[mi, si] = len(tas_used)
            n_real_amoc[mi, si] = len(amoc_used)
            if model == 'mri-esm2-0' and sce == 'ssp126':
                mri_ext_used[mi, si] = True

            # Baseline AMOC and decadal-mean AMOC weakening (country-independent).
            amoc_pi = float(amoc_ens.sel(time=BASELINE_SLICE).mean('time'))
            baseline_amoc_out[mi, si] = amoc_pi
            amoc_dec = np.array([_decadal_mean(amoc_ens, d) for d in DECADES])
            if np.isfinite(amoc_pi) and amoc_pi != 0:
                weak_dec = 100.0 * (1.0 - amoc_dec / amoc_pi)
            else:
                weak_dec = np.full(ndec, np.nan)

            # Country masks once per (model, sce) on tas grid.
            masks = functions.make_country_masks_land_aware(
                tas_ens.to_dataset(name='tas'),
                include_ipcc_regions=False, verbose=False)

            # Dataset-level baseline_tas: area-weighted mean over the model's
            # full grid (informational only, not per-country).
            tas_pi_field = tas_ens.sel(time=BASELINE_SLICE).mean('time')
            baseline_tas_out[mi, si] = float(
                functions.weighted_area_lat(tas_pi_field).mean(('lat', 'lon')))

            for ci, country in enumerate(countries):
                if country not in masks:
                    continue
                mask = masks[country]
                if not bool(mask.any()):
                    continue
                tas_series = _country_series(tas_ens, mask)
                country_baseline = float(
                    tas_series.sel(time=BASELINE_SLICE).mean('time'))
                anom_dec = np.array(
                    [_decadal_mean(tas_series, d) - country_baseline
                     for d in DECADES])
                tas_anomaly[mi, si, ci, :] = anom_dec
                amoc_weak_pct[mi, si, ci, :] = weak_dec
                amoc_strength[mi, si, ci, :] = amoc_dec

                # First decade with decadal-mean anomaly below baseline.
                below = np.where(anom_dec < 0)[0]
                if below.size:
                    onset_idx[mi, si, ci] = int(below[0])

                if verbose and country in ('IE', 'IS', 'NO', 'GB') and below.size:
                    d0 = int(below[0])
                    print(f"    {country}: onset {DECADE_LABELS[d0]}, "
                          f"weakening={weak_dec[d0]:.1f}%, "
                          f"Tanom={anom_dec[d0]:.3f}K")

    ds = xr.Dataset(
        data_vars={
            'tas_anomaly': (('model', 'scenario', 'country', 'decade'), tas_anomaly),
            'amoc_weakening_pct': (('model', 'scenario', 'country', 'decade'), amoc_weak_pct),
            'amoc_strength': (('model', 'scenario', 'country', 'decade'), amoc_strength),
            'cmip_cooling_decade_idx': (('model', 'scenario', 'country'), onset_idx),
            'baseline_tas': (('model', 'scenario'), baseline_tas_out),
            'baseline_amoc': (('model', 'scenario'), baseline_amoc_out),
            'n_realisations_tas': (('model', 'scenario'), n_real_tas),
            'n_realisations_amoc': (('model', 'scenario'), n_real_amoc),
            'mri_extension_used': (('model', 'scenario'), mri_ext_used),
        },
        coords={
            'model': models,
            'scenario': scenarios,
            'country': countries,
            'decade': DECADE_LABELS,
        },
        attrs={
            'definition': (
                'Decadal-block mean of country-area-weighted tas anomaly vs '
                'the 1850-1899 50-yr mean, and decadal-mean AMOC weakening % '
                'from the 1850-1899 AMOC mean. Realisations pair-concatenated '
                '(historical+scenario), then ensemble mean across realisations '
                'before decadal means are computed.'),
            'cmip_cooling_decade_idx_def': (
                'Index into `decade` of the first decade whose tas_anomaly < 0 '
                '(first-onset CMIP-projected cooling); -1 if it never cools.'),
            'baseline_start': '1850',
            'baseline_end': '1899',
            'amoc_weakening_pct_def': (
                '100 * (1 - amoc_strength / baseline_amoc), per decade'),
            'decades': ', '.join(DECADE_LABELS),
            'default_models_note': (
                'NorESM2-LM/-MM excluded from the default (single member each); '
                'pass models= to include them.'),
            'mri_extension_note': (
                'For MRI-ESM2-0 ssp126 r1i1p1f1, the 2015-2300 extension file '
                'in extension_2300/ is used as the scenario file (replacement).'),
            'baseline_tas_note': (
                'baseline_tas is the area-weighted mean over the model\'s full '
                'grid (not per-country); informational only.'),
        },
    )

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    ds.to_netcdf(CACHE_PATH)
    if verbose:
        print(f"\nSaved {CACHE_PATH}")
    return ds


# %%
# TRAJECTORY PLOT (diagnostic)

def plot_cmip_cooling_trajectories(ds=None, save=True, plot_bg='white'):
    """Joint time-series plot of every (model, sce, country) that cools.

    Each line: country area-mean tas anomaly vs 1850-1899, centred 10-yr
    rolling (for visual smoothness only — the dataset itself uses decadal
    blocks). Colour encodes model, line style encodes scenario. A dot marks
    the centre year of the first-onset decade. Diagnostic, not a paper figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if ds is None:
        ds = make_cmip_cooling_ds()

    plt.style.use('default')
    if plot_bg == 'black':
        plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(14, 8))

    model_colors = {
        'cesm2':       '#1f77b4',
        'mri-esm2-0':  '#d62728',
        'noresm2-lm':  '#2ca02c',
        'noresm2-mm':  '#9467bd',
    }
    scenario_styles = {
        'ssp126': '-', 'ssp245': '--', 'ssp370': '-.', 'ssp585': ':',
    }
    decade_labels = [str(d) for d in ds.decade.values]
    rolling_window = 10

    cache = {}
    n_lines = 0
    for mname in ds.model.values:
        for sce in ds.scenario.values:
            onset = ds.cmip_cooling_decade_idx.sel(model=mname, scenario=sce)
            cooled = onset.where(onset >= 0, drop=True)
            if cooled.country.size == 0:
                continue
            key = (str(mname), str(sce))
            if key not in cache:
                print(f"Loading {mname} {sce} for trajectory plot...")
                tas_rea = _list_realisations(sce, 'tas', [str(mname)])[str(mname)]
                amoc_rea = _list_realisations(sce, 'amoc', [str(mname)])[str(mname)]
                common = sorted(set(tas_rea) & set(amoc_rea))
                tas_ens, _ = _ens_mean_concat(str(mname), str(sce), common, 'tas')
                if tas_ens is None:
                    continue
                cmasks = functions.make_country_masks_land_aware(
                    tas_ens.to_dataset(name='tas'),
                    include_ipcc_regions=False, verbose=False)
                cache[key] = (tas_ens, cmasks)
            tas_ens, cmasks = cache[key]

            color = model_colors.get(str(mname), '#555555')
            for c in cooled.country.values:
                mask = cmasks.get(str(c))
                if mask is None or not bool(mask.any()):
                    continue
                series = _country_series(tas_ens, mask)
                baseline = float(series.sel(time=BASELINE_SLICE).mean('time'))
                rolling = series.rolling(time=rolling_window, center=True).mean()
                anom = (rolling - baseline).values
                years = series.time.dt.year.values
                ax.plot(years, anom, color=color,
                        linestyle=scenario_styles[str(sce)],
                        alpha=0.55, linewidth=0.9, zorder=5)
                d0 = int(onset.sel(country=c))
                dec_centre = int(decade_labels[d0].split('-')[0]) + 5
                midx = np.where(years == dec_centre)[0]
                if midx.size:
                    ax.scatter(dec_centre, anom[midx[0]], color=color, s=24,
                               zorder=10, edgecolor='white', linewidth=0.5)
                finite = np.where(np.isfinite(anom))[0]
                if finite.size:
                    end = finite[-1]
                    ax.annotate(str(c), xy=(years[end], anom[end]),
                                xytext=(3, 0), textcoords='offset points',
                                fontsize=7, color=color, ha='left',
                                va='center', zorder=6)
                n_lines += 1

    ax.axhline(0, color='black' if plot_bg != 'black' else 'white',
               lw=0.8, alpha=0.6)
    # Shade the scanned decade span.
    d_start = int(decade_labels[0].split('-')[0])
    d_end = int(decade_labels[-1].split('-')[1])
    ax.axvspan(d_start, d_end, color='#ffd966', alpha=0.4, zorder=0)
    ax.set_xlabel('Year')
    ax.set_ylabel('Country area-mean ΔT vs 1850–1899 (10-yr centred rolling) [K]')
    ax.set_title(
        f'CMIP-projected cooling: country trajectories ({n_lines} lines)\n'
        'colour = model, line style = scenario; dot marks first-onset decade centre')
    ax.set_xlim(2015, 2310)
    ax.grid(True, alpha=0.3)

    model_handles = [Line2D([], [], color=c, lw=1.5, label=m)
                     for m, c in model_colors.items()]
    sty_handles = [Line2D([], [], color='black' if plot_bg != 'black' else 'white',
                          lw=1.2, linestyle=s, label=ssp)
                   for ssp, s in scenario_styles.items()]
    leg1 = ax.legend(handles=model_handles, loc='upper left', fontsize=9,
                     title='Model', frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=sty_handles, loc='upper left', fontsize=9,
              bbox_to_anchor=(0.16, 1.0), title='Scenario', frameon=True)

    if save:
        out = '../plots/cmip_cooling_trajectories'
        fig.savefig(out + '.pdf', bbox_inches='tight')
        fig.savefig(out + '.png', bbox_inches='tight', dpi=200)
        print(f"Saved {out}.pdf and {out}.png")
    return fig, ax


# %%
if __name__ == '__main__':
    ds = make_cmip_cooling_ds(recompute=False)
    plot_cmip_cooling_trajectories(ds=ds, save=True)
    print("\n=== Summary ===")
    print(ds)
    onset = ds.cmip_cooling_decade_idx
    n_total = int((onset >= 0).sum())
    print(f"\nTotal (model, sce, country) entries that cool: {n_total}")
    if n_total > 0:
        for mi, m in enumerate(ds.model.values):
            for si, s in enumerate(ds.scenario.values):
                cnt = int((onset.isel(model=mi, scenario=si) >= 0).sum())
                if cnt:
                    print(f"  {m} {s}: {cnt} countries cool")
        # Non-empty onset decades (drives the supplementary figure panels).
        present = sorted(set(int(v) for v in onset.values.ravel() if v >= 0))
        print("\nNon-empty onset decades: "
              + ", ".join(str(ds.decade.values[i]) for i in present))

# %%
