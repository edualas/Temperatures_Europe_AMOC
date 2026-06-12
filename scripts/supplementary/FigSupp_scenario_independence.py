########################################
# %%
# SCENARIO-INDEPENDENCE DIAGNOSTIC — 2×2 grid
#
# Tests whether the linear AMOC→T slope reported by the paper as a single
# pooled (all-SSPs) number is statistically and practically indistinguishable
# from per-SSP slopes — i.e. is the pooled fit justified.
#
# Method (Wooldridge convention): fully-interacted OLS
#   y = α + β·x + δ_245·d_245 + δ_370·d_370
#         + γ_{245:x}·d_245·x + γ_{370:x}·d_370·x + ε
# with cluster-robust covariance on (ssp × hosing) experiments. Wald-F
# tests the joint null γ_{245:x} = γ_{370:x} = 0 — slope equality across
# SSPs, intercepts free.
#
# Layout (2×2):
#   a) partial R²_indep map  (per-SSP slopes' share of pooled-fit residuals)
#   b) ΔR²_total_indep map   (per-SSP slopes' share of total variance)
#   c) Region-mean pooled regression scatter   (default: EU, non-rejecting)
#   d) Most-rejecting country regression scatter (selected at render time)
#
# Maps stippled at Wald p<0.05. Both scatters use regression_plot's existing
# per-SSP and pooled cluster-robust CI bands; visual envelope overlap is a
# heuristic for slope equality, the annotated Wald-F p is the formal verdict.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.rcParams.update({'font.size': 12})

import os
import importlib
import xarray as xr
import statsmodels.api as sm
import sys, pathlib
if "__file__" in globals():  # standalone run: ensure scripts/ is importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import functions
importlib.reload(functions)

########################################
# %%
# STATE-DEPENDENCE HELPERS
#
# Emulate the SSP-loop subset of functions.regression_plot locally so we can
# fold the NAHosMIP MPI-ESM PI hosing run (`multi_model_dict['MPI-ESM1-2-LR']
# .sel(type='hosing'|'control', scenar='pi')`) in as a fourth state alongside
# the three SSPs. Lives here, not in functions.py — regression_plot is on the
# publication critical path and stays untouched. Only the truly generic test
# primitive `functions.scenario_independence_diagnostics` is reused.

SCENARIO_INDEPENDENCE_CACHE_VERSION_STATE = 2
_PI_EXP_ID = 30         # non-colliding with 9*ssp_id + exp_id ≤ 26
_PI_HOSING_COLOR = '#666666'
_STATES = ['pi', 'ssp126', 'ssp245', 'ssp370']


ALLOWED_PI_BASELINES = ('hist1850_1899', 'piControl')


def _build_state_observations(data_dict, multi_model_dict, season='', var='tas',
                              region=None, lat=None, lon=None,
                              window=10, amoc_pi_denom=None,
                              pi_baseline='hist1850_1899',
                              hosmip_reg_ds_dict=None):
    """Return (x_pct, y, exp_id, state_name, info) — pooled observations across
    the three SSPs (mirror of regression_plot's SSP × hosing × ensemble-mean
    loop at functions.py:2179–2249) plus the PI hosing arm. x is in % weakening
    relative to the PI AMOC reference; y in K.

    pi_baseline:
      'hist1850_1899' (default) — read AMOC_pi / T_pi from
                                  hosmip_reg_ds_dict['MPI-ESM1-2-LR'].
      'piControl'                — read from
                                  multi_model_dict['MPI-ESM1-2-LR']
                                  .sel(type='control', scenar='pi').

    OLS slope in K/Sv is invariant under constant x shifts, so the PI
    numerator only affects intercepts; the denominator rescales β_K/% by
    PI_new/PI_old (~3 % gap between the two options for MPI-LR)."""
    if pi_baseline not in ALLOWED_PI_BASELINES:
        raise ValueError(f'pi_baseline must be in {ALLOWED_PI_BASELINES}, '
                         f'got {pi_baseline!r}')

    is_latlon = (lat is not None) and (lon is not None)
    is_region = (region is not None)
    if not (is_latlon ^ is_region):
        raise ValueError('Pass exactly one of (lat,lon) or region.')

    # ---------- PI baseline values ----------
    mpi = multi_model_dict['MPI-ESM1-2-LR']
    if pi_baseline == 'hist1850_1899':
        if hosmip_reg_ds_dict is None:
            hosmip_reg_ds_dict = functions.get_hosmip_reg_ds(recompute=False)
        rd = hosmip_reg_ds_dict['MPI-ESM1-2-LR']
        amoc_pi_value = float(rd.AMOC_pi.values)
        tpi_field = rd.T_pi.sel(season=season)
    else:  # piControl
        amoc_pi_value = float(mpi.sel(type='control', scenar='pi', season='')
                              .amoc.isel(time=0).values)
        tpi_field = mpi.sel(type='control', scenar='pi', season=season).tas.isel(time=0)

    if is_latlon:
        lat_val = float(data_dict[season][var]['ssphos'].lat.values[lat])
        lon_val = float(data_dict[season][var]['ssphos'].lon.values[lon])
        tas_pi_value = float(tpi_field.sel(lat=lat_val, lon=lon_val,
                                            method='nearest').values)
    else:
        mask_r = mpi.mask.sel(region=region)
        mask_l = mpi.mask.sel(region='LAND')
        # T_pi (hist1850_1899 branch) is on the EU subset grid; align the
        # full-grid masks to its lat/lon coords before area-averaging.
        if tpi_field.sizes.get('lat', 0) != mpi.sizes.get('lat', 0):
            mask_r = mask_r.sel(lat=tpi_field.lat, lon=tpi_field.lon, method='nearest')
            mask_l = mask_l.sel(lat=tpi_field.lat, lon=tpi_field.lon, method='nearest')
        tas_pi_value = float(functions.weighted_area_lat(
                                tpi_field.where(mask_r).where(mask_l == 0))
                             .mean('lat').mean('lon').values)

    if amoc_pi_denom is None:
        amoc_pi_denom = amoc_pi_value

    # ---------- SSP arm ----------
    ssp_amoc_yr    = data_dict['']['amoc']['ssp']
    ssphos_amoc_yr = data_dict['']['amoc']['ssphos'].isel(time=slice(0, -198))
    ssp_var        = data_dict[season][var]['ssp']
    ssphos_var     = data_dict[season][var]['ssphos'].isel(time=slice(0, -198))
    masks_eur_dict = data_dict['masks']
    mask_land      = data_dict['masks']['LAND']

    combined_x_sv, combined_y, combined_eid, combined_sn = [], [], [], []

    hosing_loop_values = np.concat([
        [ssphos_amoc_yr.hosing.values[-1]],
        ssphos_amoc_yr.hosing.values[:-1],
    ])

    for ssp_id, ssp_i in enumerate(['ssp126', 'ssp245', 'ssp370']):
        for exp_id, exp in enumerate(hosing_loop_values):
            # regression_plot's hos_type='all' default skips exp=='1' (line 2190);
            # the only ssp370+'1' combo also gets skipped (line 2211).
            if exp == '1':
                continue

            x_sv = (ssphos_amoc_yr.sel(scenar=ssp_i, hosing=exp)
                    .rolling(time=window, center=True).mean('time')
                    .mean(dim='realiz').AMOC_strength.values
                    - ssp_amoc_yr.sel(scenar=ssp_i)
                    .rolling(time=window, center=True).mean('time')
                    .mean(dim='realiz').AMOC_strength.values)

            if is_latlon:
                y = (ssphos_var.sel(scenar=ssp_i, hosing=exp).isel(lat=lat, lon=lon)
                     .rolling(time=window, center=True).mean('time')
                     .mean(dim='realiz')[var].values
                     - ssp_var.sel(scenar=ssp_i).isel(lat=lat, lon=lon)
                     .rolling(time=window, center=True).mean('time')
                     .mean(dim='realiz')[var].values)
            else:
                y = (functions.weighted_area_lat(
                        ssphos_var.sel(scenar=ssp_i, hosing=exp)
                        .where(masks_eur_dict[region]).where(mask_land == 0)
                     ).mean('lat').mean('lon')
                     .rolling(time=window, center=True).mean('time')
                     .mean(dim='realiz')[var].values
                     - functions.weighted_area_lat(
                        ssp_var.sel(scenar=ssp_i)
                        .where(masks_eur_dict[region]).where(mask_land == 0)
                     ).mean('lat').mean('lon')
                     .rolling(time=window, center=True).mean('time')
                     .mean(dim='realiz')[var].values)

            valid = ~np.isnan(x_sv) & ~np.isnan(y)
            x_sv = x_sv[valid]; y = y[valid]

            combined_x_sv.extend(x_sv)
            combined_y.extend(y)
            combined_eid.extend([9 * ssp_id + exp_id] * len(x_sv))
            combined_sn.extend([ssp_i] * len(x_sv))

    # ---------- PI hosing arm ----------
    # The PI numerator (amoc_pi_value, tas_pi_value) was resolved above based
    # on pi_baseline. The slope in K/Sv is invariant under constant x shifts,
    # so changing the PI value here only shifts intercepts; the slope effect
    # comes from the denominator below.
    mpi = multi_model_dict['MPI-ESM1-2-LR']
    amoc_pi_h = (mpi.sel(type='hosing', scenar='pi', season='')
                 .rolling(time=window, center=True).mean('time').amoc.values)

    if is_latlon:
        lat_val = float(data_dict[season][var]['ssphos'].lat.values[lat])
        lon_val = float(data_dict[season][var]['ssphos'].lon.values[lon])
        tas_pi_h = (mpi.sel(type='hosing', scenar='pi', season=season).tas
                    .sel(lat=lat_val, lon=lon_val, method='nearest')
                    .rolling(time=window, center=True).mean('time').values)
    else:
        mask_r = mpi.mask.sel(region=region)
        mask_l = mpi.mask.sel(region='LAND')
        tas_pi_h_field = mpi.sel(type='hosing', scenar='pi', season=season).tas
        tas_pi_h = (functions.weighted_area_lat(
                        tas_pi_h_field.where(mask_r).where(mask_l == 0))
                    .mean('lat').mean('lon')
                    .rolling(time=window, center=True).mean('time').values)

    x_pi_sv = np.asarray(amoc_pi_h, dtype=float) - amoc_pi_value
    y_pi    = np.asarray(tas_pi_h, dtype=float)  - tas_pi_value
    valid   = ~np.isnan(x_pi_sv) & ~np.isnan(y_pi)
    x_pi_sv = x_pi_sv[valid]; y_pi = y_pi[valid]

    combined_x_sv.extend(x_pi_sv)
    combined_y.extend(y_pi)
    combined_eid.extend([_PI_EXP_ID] * len(x_pi_sv))
    combined_sn.extend(['pi'] * len(x_pi_sv))

    # Sv → % weakening (single denominator across all states).
    x_pct = -100.0 * np.asarray(combined_x_sv, dtype=float) / amoc_pi_denom
    y_arr = np.asarray(combined_y, dtype=float)
    eid   = np.asarray(combined_eid, dtype=int)
    sn    = np.asarray(combined_sn)

    info = {
        'n_per_state': {s: int((sn == s).sum()) for s in _STATES},
        'amoc_pi_denom': float(amoc_pi_denom),
        'amoc_pi_value': float(amoc_pi_value),
        'tas_pi_value': float(tas_pi_value),
        'pi_baseline': pi_baseline,
    }
    return x_pct, y_arr, eid, sn, info


_STATE_LABELS = {
    'pi':     'PI hosing (NAHosMIP, 1.03 Sv)',
    'ssp126': 'SSP1-2.6 + hosing',
    'ssp245': 'SSP2-4.5 + hosing',
    'ssp370': 'SSP3-7.0 + hosing',
}


def _plot_state_dep_scatter(ax, x_pct, y, exp_id, state_name, diag,
                            plot_bg='white', ylim=(-2.5, 1.5), xlim=None):
    """4-state scatter + per-state lines + 95% prediction-SE bands. SSPs use
    cluster-robust SE with G-1 df; PI uses HAC (single cluster). x-axis in
    %-weakening, inverted. Pooled fit in black."""
    import scipy.stats as scipy_stats
    fg = 'black' if plot_bg != 'black' else 'white'
    state_colors = {
        'pi':     functions.get_hosmip_colors(plot_bg=plot_bg)['MPI-ESM1-2-LR'],
        'ssp126': (23/255, 60/255, 102/255),
        'ssp245': (247/255, 148/255, 32/255),
        'ssp370': (231/255, 29/255, 37/255),
    }

    if xlim is None:
        xlim = (float(np.min(x_pct)) - 2.0, float(np.max(x_pct)) + 2.0)
    x_range = np.linspace(xlim[0], xlim[1], 200)
    X_range = sm.add_constant(x_range)

    for s in _STATES:
        sel = state_name == s
        if not np.any(sel):
            continue
        ax.scatter(x_pct[sel], y[sel], s=18, alpha=0.55,
                   c=[state_colors[s]], linewidths=0, clip_on=False)
        if s == 'pi':
            m = sm.OLS(y[sel], sm.add_constant(x_pct[sel])).fit(
                cov_type='HAC', cov_kwds={'maxlags': 9})
            t_crit = scipy_stats.t.ppf(0.975, df=m.df_resid)
        else:
            G = len(set(exp_id[sel]))
            m = sm.OLS(y[sel], sm.add_constant(x_pct[sel])).fit(
                cov_type='cluster',
                cov_kwds={'groups': exp_id[sel], 'use_correction': True})
            t_crit = scipy_stats.t.ppf(0.975, df=G - 1)
        yhat = m.predict(X_range)
        pred_se = m.get_prediction(X_range).se
        ax.plot(x_range, yhat, color=state_colors[s], lw=1.4, alpha=0.95)
        ax.fill_between(x_range, yhat - t_crit * pred_se,
                        yhat + t_crit * pred_se,
                        color=state_colors[s], alpha=0.18, linewidth=0)

    pooled = sm.OLS(y, sm.add_constant(x_pct)).fit(
        cov_type='cluster', cov_kwds={'groups': exp_id, 'use_correction': True})
    t_crit_p = scipy_stats.t.ppf(0.975, df=len(set(exp_id)) - 1)
    yhat_p = pooled.predict(X_range)
    pred_se_p = pooled.get_prediction(X_range).se
    ax.plot(x_range, yhat_p, color=fg, lw=1.8)
    ax.fill_between(x_range, yhat_p - t_crit_p * pred_se_p,
                    yhat_p + t_crit_p * pred_se_p,
                    color=fg, alpha=0.18, linewidth=0)

    ax.axhline(0, color=fg, lw=1.0, linestyle='--', alpha=0.5)
    ax.axvline(0, color=fg, lw=1.0, linestyle='--', alpha=0.5)
    # Weaker-AMOC on the right, matching Fig1 panel c (x_pct>0 = weakening; no invert).
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel('AMOC weakening [%]')
    ax.set_ylabel(r'Deviation from MPI-GE $\Delta T$ [°C]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _run_state_dep_test(x_pct, y, exp_id, state_name):
    """Fit pooled baseline + run functions.scenario_independence_diagnostics
    with scenarios=['pi','ssp126','ssp245','ssp370']. Returns the diagnostic
    dict augmented with combined_intercept/combined_coef/combined_ste (K per %
    weakening)."""
    X_base = sm.add_constant(x_pct)
    baseline = sm.OLS(y, X_base).fit(
        cov_type='cluster',
        cov_kwds={'groups': exp_id, 'use_correction': True})
    diag = functions.scenario_independence_diagnostics(
        y, X_base, exp_id, state_name, baseline, scenarios=_STATES)
    diag['combined_intercept'] = float(baseline.params[0])
    diag['combined_coef']      = float(baseline.params[1])
    diag['combined_ste']       = float(baseline.bse[1])
    return diag


def get_scenario_independence_diagnostics_mpi_state(data_dict=None,
                                                    multi_model_dict=None,
                                                    hosmip_reg_ds_dict=None,
                                                    var='tas', recompute=False,
                                                    pi_baseline='hist1850_1899'):
    """Per-pixel cluster-robust Wald-F test of slope equality across 4 states
    (PI hosing + 3 SSPs) in MPI-ESM. Returns a Dataset with dims
    (season, lat, lon) and 12 variables: wald_fstat, wald_pvalue,
    partial_r2_indep, delta_r2_total_indep, baseline_r2, beta_pi,
    beta_ssp126, beta_ssp245, beta_ssp370, max_pairwise_slope_diff,
    n_scenarios_present, plus combined_coef (pooled-fit slope in K per %
    weakening). Cached to
    data/scenario_independence_diagnostics_mpi_state_{var}.nc.

    Mirrors get_scenario_independence_diagnostics_mpi's loop structure but
    swaps the underlying call from regression_plot(..., compute_scenario_
    independence=True) to a local _build_state_observations + _run_state_
    dep_test invocation that adds the PI hosing arm. regression_plot
    itself is not touched."""
    import time as _time
    functions.validate_choice('var', var, functions.ALLOWED_VARS)
    if pi_baseline not in ALLOWED_PI_BASELINES:
        raise ValueError(f'pi_baseline must be in {ALLOWED_PI_BASELINES}, '
                         f'got {pi_baseline!r}')
    suffix = '' if pi_baseline == 'hist1850_1899' else f'_pi-{pi_baseline}'
    cache = (functions.local_path
             + f'scenario_independence_diagnostics_mpi_state_{var}{suffix}.nc')
    if os.path.exists(cache) and not recompute:
        cached = xr.open_dataset(cache)
        if (int(cached.attrs.get('version', 0)) >= SCENARIO_INDEPENDENCE_CACHE_VERSION_STATE
                and cached.attrs.get('pi_baseline', 'hist1850_1899') == pi_baseline):
            print(f'Loading precomputed MPI-ESM state-dependence diagnostics (pi_baseline={pi_baseline})...')
            return cached
        cached.close()
        print(f'Existing state-dep cache mismatch '
              f'(v{cached.attrs.get("version", 0)}/{cached.attrs.get("pi_baseline", "?")} '
              f'vs target v{SCENARIO_INDEPENDENCE_CACHE_VERSION_STATE}/{pi_baseline}); recomputing.')

    print(f'Calculating MPI-ESM state-dependence diagnostics '
          f'(per-pixel Wald-F, PI hosing + 3 SSPs, pi_baseline={pi_baseline})...')
    if data_dict is None:
        data_dict = functions.load_mpi_esm_data(eur_only=True)
    if multi_model_dict is None:
        multi_model_dict, _ = functions.get_full_multi_model_dict()
    if pi_baseline == 'hist1850_1899' and hosmip_reg_ds_dict is None:
        hosmip_reg_ds_dict = functions.get_hosmip_reg_ds(recompute=False)

    seasons = ['', 'djf', 'jja'] if var == 'tas' else ['']
    nlat = data_dict[seasons[0]][var]['ssphos'].sizes['lat']
    nlon = data_dict[seasons[0]][var]['ssphos'].sizes['lon']

    fields = ['wald_fstat', 'wald_pvalue', 'partial_r2_indep',
              'delta_r2_total_indep', 'baseline_r2',
              'beta_pi', 'beta_ssp126', 'beta_ssp245', 'beta_ssp370',
              'max_pairwise_slope_diff', 'n_scenarios_present',
              'combined_coef']
    arr_shape = (len(seasons), nlat, nlon)
    data_vars = {name: (['season', 'lat', 'lon'], np.full(arr_shape, np.nan)) for name in fields}
    coords = {
        'season': seasons,
        'lat': data_dict[seasons[0]][var]['ssphos'].lat.values,
        'lon': data_dict[seasons[0]][var]['ssphos'].lon.values,
    }
    out = xr.Dataset(data_vars=data_vars, coords=coords)
    out.attrs['version'] = SCENARIO_INDEPENDENCE_CACHE_VERSION_STATE

    for s_i, s in enumerate(seasons):
        print(f'  season={s!r}')
        ssphos_var = data_dict[s][var]['ssphos']
        i = 0
        t0 = _time.time()
        for lat_i in range(nlat):
            for lon_i in range(nlon):
                if np.sum(~np.isnan(ssphos_var.isel(lat=lat_i, lon=lon_i)[var].values)) == 0:
                    i += 1
                    continue
                try:
                    x_pct, y, eid, sn, _ = _build_state_observations(
                        data_dict, multi_model_dict, season=s, var=var,
                        lat=lat_i, lon=lon_i,
                        pi_baseline=pi_baseline,
                        hosmip_reg_ds_dict=hosmip_reg_ds_dict)
                    if len(x_pct) < 20:
                        i += 1
                        continue
                    diag = _run_state_dep_test(x_pct, y, eid, sn)
                    for name in fields:
                        if name in diag and np.isfinite(diag[name]):
                            out[name].values[s_i, lat_i, lon_i] = diag[name]
                except Exception as exc:
                    # Silent skip — bad pixels remain NaN; one-off warnings would
                    # flood the log. Coverage is checked in the verification step.
                    pass
                i += 1
                if i % 200 == 0:
                    rate = i / (_time.time() - t0 + 1e-9)
                    eta_s = (nlat*nlon - i) / max(rate, 1e-9)
                    print(f'    {i}/{nlat*nlon} pixels done '
                          f'({rate:.1f} pix/s; ETA {eta_s/60:.1f} min)')
        print(f'  season={s!r} took {(_time.time()-t0)/60:.1f} min')

    out.attrs['pi_baseline'] = pi_baseline
    out.to_netcdf(cache)
    print(f'Saved MPI-ESM state-dependence diagnostics to {cache}.')
    return out


def load_scenario_independence_diagnostics_mpi_state(data_dict=None,
                                                     multi_model_dict=None,
                                                     hosmip_reg_ds_dict=None,
                                                     var='tas', recompute=False,
                                                     pi_baseline='hist1850_1899'):
    return get_scenario_independence_diagnostics_mpi_state(
        data_dict=data_dict, multi_model_dict=multi_model_dict,
        hosmip_reg_ds_dict=hosmip_reg_ds_dict,
        var=var, recompute=recompute, pi_baseline=pi_baseline)


########################################
# %%
# LOAD DATA

if __name__ == '__main__':
    data_dict = functions.load_mpi_esm_data(eur_only=True)
    si_ds = functions.get_scenario_independence_diagnostics_mpi(var='tas')

########################################
# %%
# HELPERS

# Hardcoded most-rejecting country per season for var='tas', alpha=0.05.
# Source of truth: a `recompute=True` scan over functions.country_codes. This
# dict caches the answer so the default render path doesn't pay the 1–2 min
# scan cost per season. Regenerate (and update this dict) if anything
# upstream changes: data, AMOC reference, scenario list, mask topology, the
# regression specification. Last derived 2026-05-28.
_MOST_REJECTING_COUNTRY_BY_SEASON = {
    '':    'NO',   # Norway,  Wald-p = 0.0277 (annual)
    'djf': 'AT',   # Austria, Wald-p = 0.0055
    'jja': 'NO',   # Norway,  Wald-p = 0.0006
}

# Parallel hardcoded dict for the state-dependence (PI + 3 SSPs) test.
# Backfilled 2026-05-29 from a recompute_most_rejecting=True scan over
# functions.country_codes. All three seasons reject at p ≈ 1e-16 because PI
# hosing's slope is structurally distinct from the SSPs' transient slopes —
# AMOC-footprint countries (British Isles, Iceland) dominate the ranking.
_MOST_REJECTING_COUNTRY_BY_SEASON_STATE = {
    '':    'GB',   # Great Britain, Wald-p ≈ 0 (annual)
    'djf': 'IE',   # Ireland,       Wald-p ≈ 0
    'jja': 'IS',   # Iceland,       Wald-p ≈ 0
}

# Module-level memoization for the recompute path — the (var, season, alpha)
# tuple determines the result, so calling make_figure twice for the same
# season (e.g. white+black bg) with recompute=True reuses the scan instead
# of repeating it within the same Python session.
_country_cache = {}


def find_most_rejecting_country(data_dict, var='tas', season='', alpha=0.05,
                                 fallback_to_pixel=True, si_ds=None,
                                 verbose=False, recompute=False,
                                 state_dependence=False,
                                 multi_model_dict=None,
                                 hosmip_reg_ds_dict=None,
                                 pi_baseline='hist1850_1899'):
    """Return the most-rejecting European country for this season's Wald-F test.

    When state_dependence=False (default): scans the 3-SSP scenario-indep
    test via regression_plot(..., compute_scenario_independence=True).
    When state_dependence=True: scans the 4-state test via local
    _build_state_observations + _run_state_dep_test (PI hosing + 3 SSPs).
    Requires multi_model_dict in the latter case.

    By default reads `_MOST_REJECTING_COUNTRY_BY_SEASON`(_STATE) and returns
    the hardcoded country code without scanning, saving ~1–2 min per season.
    Pass `recompute=True` to scan `functions.country_codes` (41 2-letter ISO
    codes) and pick the smallest-p country with p ≤ alpha.

    Returns:
        {'kind': 'country', 'label': code, 'wald_pvalue': p, 'extra': None}

    `wald_pvalue` is NaN on the hardcoded path. On the recompute path it
    carries the scan's measured p.

    On recompute, if no country rejects at alpha and `fallback_to_pixel=True`
    with `si_ds` supplied, falls back to the pixel with the largest
    max_pairwise_slope_diff among Wald-rejecting pixels. Final fallback:
    smallest-p country regardless of alpha.

    Recompute results are memoised by (var, season, alpha, state_dependence)
    for the session.
    """
    hardcoded = (_MOST_REJECTING_COUNTRY_BY_SEASON_STATE
                 if state_dependence else _MOST_REJECTING_COUNTRY_BY_SEASON)
    if not recompute and var == 'tas' and alpha == 0.05 \
            and season in hardcoded:
        code = hardcoded[season]
        return {'kind': 'country', 'label': code,
                'wald_pvalue': np.nan, 'extra': None}

    cache_key = (var, season, alpha, state_dependence, pi_baseline)
    if cache_key in _country_cache:
        return _country_cache[cache_key]

    if state_dependence and multi_model_dict is None:
        raise ValueError('multi_model_dict required when state_dependence=True.')

    results = []
    for code in functions.country_codes.keys():
        try:
            if state_dependence:
                x_pct, y, eid, sn, _ = _build_state_observations(
                    data_dict, multi_model_dict, season=season, var=var,
                    region=code, pi_baseline=pi_baseline,
                    hosmip_reg_ds_dict=hosmip_reg_ds_dict)
                if len(x_pct) < 20:
                    continue
                diag = _run_state_dep_test(x_pct, y, eid, sn)
                p = diag.get('wald_pvalue', np.nan)
            else:
                functions.regression_plot.last_call = {}
                functions.regression_plot(
                    data_dict, region=code, var=var, season=season,
                    no_plots=True, compute_scenario_independence=True,
                )
                lc = functions.regression_plot.last_call
                p = lc.get('wald_pvalue', np.nan)
            if np.isfinite(p):
                results.append((code, float(p)))
                if verbose:
                    print(f'    {code}: Wald-p = {p:.4f}')
        except Exception as exc:
            if verbose:
                print(f'    {code}: skipped ({exc.__class__.__name__})')

    rejecting = [(c, p) for (c, p) in results if p <= alpha]
    if rejecting:
        c, p = min(rejecting, key=lambda x: x[1])
        result = {'kind': 'country', 'label': c, 'wald_pvalue': p, 'extra': None}
        _country_cache[cache_key] = result
        return result

    if fallback_to_pixel and si_ds is not None:
        sub = si_ds.sel(season=season)
        wald_p = sub['wald_pvalue'].values
        slope_diff = sub['max_pairwise_slope_diff'].values
        mask = (wald_p < alpha) & np.isfinite(slope_diff)
        if mask.any():
            masked_diff = np.where(mask, slope_diff, np.nan)
            idx = np.unravel_index(np.nanargmax(masked_diff), masked_diff.shape)
            lat_i, lon_i = int(idx[0]), int(idx[1])
            lat_v = float(sub['lat'].values[lat_i])
            lon_v = float(sub['lon'].values[lon_i])
            result = {
                'kind': 'pixel',
                'label': f'pixel ({lat_v:.1f}°N, {lon_v:.1f}°E)',
                'wald_pvalue': float(wald_p[lat_i, lon_i]),
                'extra': (lat_i, lon_i),
            }
            _country_cache[cache_key] = result
            return result

    if results:
        c, p = min(results, key=lambda x: x[1])
        result = {'kind': 'country', 'label': c, 'wald_pvalue': p, 'extra': None}
        _country_cache[cache_key] = result
        return result
    result = {'kind': 'none', 'label': 'n/a', 'wald_pvalue': np.nan, 'extra': None}
    _country_cache[cache_key] = result
    return result


def _annotate_wald(ax, lc, fg, bg, slope_units='K/Sv', per_state_betas=None):
    """Wald-F annotation block in the panel's lower-right corner. `slope_units`
    is a label only; the underlying max_pairwise_slope_diff carries whatever
    unit the caller fit in. When `per_state_betas` is a dict {state: beta},
    one extra row is added per state (used by the 4-state state-dependence
    panel)."""
    if 'wald_pvalue' not in lc or not np.isfinite(lc.get('wald_pvalue', np.nan)):
        return
    annot = (
        f'Wald-F p          = {lc["wald_pvalue"]:.3f}\n'
        f'partial R²_indep  = {lc["partial_r2_indep"]:.4f}\n'
        f'ΔR²_total_indep   = {lc["delta_r2_total_indep"]:.5f}\n'
        f'max |Δβ|          = {lc["max_pairwise_slope_diff"]:.3f} {slope_units}'
    )
    if per_state_betas:
        lines = []
        for s in _STATES:
            if s in per_state_betas and np.isfinite(per_state_betas[s]):
                lines.append(f'β_{s:<6} = {per_state_betas[s]:+.4f} {slope_units}')
        if lines:
            annot += '\n' + '\n'.join(lines)
    ax.text(
        0.97, 0.97, annot,
        transform=ax.transAxes, fontsize=8,
        va='top', ha='right', ma='left', family='monospace',
        bbox=dict(facecolor=bg, edgecolor=fg, alpha=0.85,
                  boxstyle='round,pad=0.4'),
    )


########################################
# %%
# FIGURE FUNCTION

def make_figure(data_dict, si_ds, var='tas', region='EU', season='',
                plot_bg='white', verbose_country_scan=False,
                recompute_most_rejecting=False,
                state_dependence=False, multi_model_dict=None,
                hosmip_reg_ds_dict=None,
                pi_baseline='hist1850_1899'):
    """Render the 2×2 scenario-/state-independence figure.

    When state_dependence=False (default): tests scenario-independence across
    the 3 SSPs (ssp126/245/370) — original published path, untouched.
    When state_dependence=True: tests state-independence across PI hosing
    (NAHosMIP 1.03 Sv) + 3 SSPs (4-state Wald-F). Requires multi_model_dict
    and si_ds to be the broader cache from
    load_scenario_independence_diagnostics_mpi_state(...).

    All scaling factors reported in K per %-weakening when state_dependence
    is True; K/Sv (with display rescaling inside regression_plot) when False.
    """
    if state_dependence and multi_model_dict is None:
        raise ValueError('state_dependence=True requires multi_model_dict.')
    if state_dependence and pi_baseline == 'hist1850_1899' and hosmip_reg_ds_dict is None:
        hosmip_reg_ds_dict = functions.get_hosmip_reg_ds(recompute=False)

    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    fg = 'black' if plot_bg != 'black' else 'white'
    bg = 'white' if plot_bg != 'black' else '#191919'

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0],
        wspace=0.25, hspace=0.20,
    )
    ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=8))
    ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson(central_longitude=8))
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ------------------------------------------------------------------
    # TOP ROW: maps a (partial R²_indep) and b (ΔR²_total_indep)
    # ------------------------------------------------------------------
    si = si_ds.sel(season=season)
    partial_r2 = si['partial_r2_indep'].values
    delta_r2 = si['delta_r2_total_indep'].values
    wald_p = si['wald_pvalue'].values
    lons = si['lon'].values
    lats = si['lat'].values

    stipple = (wald_p < 0.05) & np.isfinite(wald_p)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    stipple_lon = lon_grid[stipple]
    stipple_lat = lat_grid[stipple]

    def _draw_map(ax, field, vmin, vmax, n_bins, label):
        # Discrete sequential viridis with set_under/set_over distinct
        # from the end bins. Same _draw_map idiom as
        # FigSupp_mpi_linearity.py.
        base = plt.get_cmap('viridis')
        centers = np.linspace(0.5 / n_bins, 1.0 - 0.5 / n_bins, n_bins)
        cmap = mcolors.ListedColormap(base(centers))
        cmap.set_under(base(0.0))
        cmap.set_over(base(1.0))
        bounds = np.linspace(vmin, vmax, n_bins + 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N, clip=False)

        ax.spines['geo'].set_visible(False)
        ax.set_rasterized(True)
        ax.coastlines(color=fg, linewidth=0.5, zorder=3)
        ax.add_feature(cfeature.BORDERS, edgecolor=fg, linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor=bg, zorder=2)
        pcm = ax.pcolormesh(
            lons, lats, field,
            cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto',
        )
        ax.scatter(
            stipple_lon, stipple_lat,
            s=4.5, c=fg, marker='.', linewidths=0, alpha=0.85,
            transform=ccrs.PlateCarree(), zorder=4,
        )
        ax.set_extent([-18.5, 38.5, 34.8, 72], crs=ccrs.PlateCarree())
        functions.add_square(ax, -13.5, 0, 60, 75.5, colour=bg)
        functions.add_square(ax, -30, -18, 68, 75, colour=bg)
        cbar = fig.colorbar(pcm, ax=ax, orientation='horizontal',
                            pad=0.05, aspect=30, extend='max', ticks=bounds)
        cbar.set_label(label, fontsize=10)
        return cbar

    slope_kind = 'per-state' if state_dependence else 'per-SSP'
    _draw_map(ax_a, partial_r2, vmin=0.0, vmax=0.25, n_bins=5,
              label=f"partial R² ({slope_kind} slopes' share of residuals)")
    _draw_map(ax_b, delta_r2, vmin=0.0, vmax=0.05, n_bins=5,
              label=f"ΔR²_total ({slope_kind} slopes' share of total variance)")

    season_suffix = f' ({season.upper()})' if season else ''
    contrib_label = 'State' if state_dependence else 'Scenario'
    ax_a.set_title(f'a) {contrib_label} contribution to residuals{season_suffix}', fontsize=12, loc='left')
    ax_b.set_title(f'b) {contrib_label} contribution to total variance{season_suffix}', fontsize=12, loc='left')

    stipple_label = 'state-independence' if state_dependence else 'scenario-independence'
    fig.text(
        0.5, 0.91,
        f'Stippling: Wald F-test rejects {stipple_label} at 5%.',
        fontsize=10, ha='center', style='italic', color=fg,
    )

    # ------------------------------------------------------------------
    # BOTTOM-LEFT (c): region-mean pooled regression — non-rejecting anchor
    # ------------------------------------------------------------------
    if state_dependence:
        x_pct, y, eid, sn, info_c = _build_state_observations(
            data_dict, multi_model_dict, season=season, var=var, region=region,
            pi_baseline=pi_baseline, hosmip_reg_ds_dict=hosmip_reg_ds_dict)
        diag_c = _run_state_dep_test(x_pct, y, eid, sn)
        _plot_state_dep_scatter(ax_c, x_pct, y, eid, sn, diag_c,
                                plot_bg=plot_bg, ylim=(-2.5, 1.5))
        per_state_c = {s: diag_c.get(f'beta_{s}', np.nan) for s in _STATES}
        _annotate_wald(ax_c, diag_c, fg, bg, slope_units='K/%',
                       per_state_betas=per_state_c)
    else:
        functions.regression_plot(
            data_dict, region=region, var=var, season=season,
            no_plots=False, compute_scenario_independence=True,
            ext_ax=ax_c, plot_bg=plot_bg, ylim=(-2.5, 1.5),
            equation_y_pos=0.25,  # equations bottom-left; Wald box top-right
        )
        lc_eu = functions.regression_plot.last_call
        # Mirror Fig1.py panel c: weakening-% ruler, drop the orphaned top Sv axis.
        functions.add_weakening_pct_overlay(fig, ax_c, plot_bg=plot_bg, drop_parent_xaxis=True)
        _annotate_wald(ax_c, lc_eu, fg, bg)
    anchor_label = 'anchor' if state_dependence else 'non-rejecting case'
    ax_c.set_title(f'c) {region}-mean ({anchor_label})',
                   fontsize=12, loc='left')

    # ------------------------------------------------------------------
    # BOTTOM-RIGHT (d): most-rejecting country regression
    # ------------------------------------------------------------------
    if recompute_most_rejecting:
        print(f'  Scanning for most-rejecting country (season={season!r})...')
    else:
        print(f'  Using hardcoded most-rejecting country for season={season!r} '
              f'(pass recompute_most_rejecting=True to re-derive)')
    sel = find_most_rejecting_country(
        data_dict, var=var, season=season, alpha=0.05,
        fallback_to_pixel=True, si_ds=si_ds,
        verbose=verbose_country_scan, recompute=recompute_most_rejecting,
        state_dependence=state_dependence,
        multi_model_dict=multi_model_dict,
        hosmip_reg_ds_dict=hosmip_reg_ds_dict,
        pi_baseline=pi_baseline,
    )
    if np.isfinite(sel['wald_pvalue']):
        print(f'  -> {sel["kind"]}: {sel["label"]}, Wald-p = {sel["wald_pvalue"]:.4f}')
    else:
        print(f'  -> {sel["kind"]}: {sel["label"]} (hardcoded)')

    if state_dependence:
        if sel['kind'] == 'country':
            x_pct, y, eid, sn, _ = _build_state_observations(
                data_dict, multi_model_dict, season=season, var=var,
                region=sel['label'], pi_baseline=pi_baseline,
                hosmip_reg_ds_dict=hosmip_reg_ds_dict)
            diag_d = _run_state_dep_test(x_pct, y, eid, sn)
            _plot_state_dep_scatter(ax_d, x_pct, y, eid, sn, diag_d,
                                    plot_bg=plot_bg, ylim=(-3.5, 2.0))
            title_tag = sel['label']
        elif sel['kind'] == 'pixel':
            lat_i, lon_i = sel['extra']
            x_pct, y, eid, sn, _ = _build_state_observations(
                data_dict, multi_model_dict, season=season, var=var,
                lat=lat_i, lon=lon_i, pi_baseline=pi_baseline,
                hosmip_reg_ds_dict=hosmip_reg_ds_dict)
            diag_d = _run_state_dep_test(x_pct, y, eid, sn)
            _plot_state_dep_scatter(ax_d, x_pct, y, eid, sn, diag_d,
                                    plot_bg=plot_bg, ylim=(-4.0, 2.5))
            title_tag = sel['label']
        else:
            diag_d = None
            ax_d.text(0.5, 0.5, 'no rejecting country or pixel found',
                      transform=ax_d.transAxes, ha='center', va='center',
                      fontsize=11, color=fg, style='italic')
            ax_d.set_xticks([])
            ax_d.set_yticks([])
            title_tag = 'n/a'
        if diag_d is not None:
            per_state_d = {s: diag_d.get(f'beta_{s}', np.nan) for s in _STATES}
            _annotate_wald(ax_d, diag_d, fg, bg, slope_units='K/%',
                           per_state_betas=per_state_d)
    else:
        if sel['kind'] == 'country':
            functions.regression_plot(
                data_dict, region=sel['label'], var=var, season=season,
                no_plots=False, compute_scenario_independence=True,
                ext_ax=ax_d, plot_bg=plot_bg, ylim=(-3.5, 2.0),
                equation_y_pos=0.25,  # equations bottom-left; Wald box top-right
            )
            title_tag = sel['label']
        elif sel['kind'] == 'pixel':
            lat_i, lon_i = sel['extra']
            functions.regression_plot(
                data_dict, lat=lat_i, lon=lon_i, var=var, season=season,
                no_plots=False, compute_scenario_independence=True,
                ext_ax=ax_d, plot_bg=plot_bg, ylim=(-4.0, 2.5),
                equation_y_pos=0.25,  # equations bottom-left; Wald box top-right
            )
            title_tag = sel['label']
        else:
            ax_d.text(0.5, 0.5, 'no rejecting country or pixel found',
                      transform=ax_d.transAxes, ha='center', va='center',
                      fontsize=11, color=fg, style='italic')
            ax_d.set_xticks([])
            ax_d.set_yticks([])
            title_tag = 'n/a'

        if sel['kind'] != 'none':
            lc_rej = functions.regression_plot.last_call
            # Mirror Fig1.py panel c: weakening-% ruler, drop the orphaned top Sv axis.
            functions.add_weakening_pct_overlay(fig, ax_d, plot_bg=plot_bg, drop_parent_xaxis=True)
            _annotate_wald(ax_d, lc_rej, fg, bg)
    ax_d.set_title(f'd) {title_tag} (rejecting case)',
                   fontsize=12, loc='left')

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    # Analysis type (scenario- vs state-independence) folds into the kind.
    analysis = 'state_independence' if state_dependence else 'scenario_independence'
    pi_tag = (f'_pi-{pi_baseline}'
              if state_dependence and pi_baseline != 'hist1850_1899' else '')
    savepath = (f"../plots/FigSupp_mpi_{analysis}{pi_tag}"
                f"_var-{var}_region-{region}"
                f"_season-{season or 'annual'}_plotbg-{plot_bg}")
    fig.savefig(savepath + '.png', dpi=200, bbox_inches='tight',
                transparent=True if plot_bg == 'black' else False)
    fig.savefig(savepath + '.pdf', dpi=400, bbox_inches='tight',
                transparent=True if plot_bg == 'black' else False)
    return fig, savepath


########################################
# %%
# RUN

if __name__ == '__main__':
    fig, savepath = make_figure(data_dict, si_ds, var='tas', region='EU',
                                season='djf', plot_bg='white')
    print(f'Saved {savepath}.(pdf|png)')

# %%
