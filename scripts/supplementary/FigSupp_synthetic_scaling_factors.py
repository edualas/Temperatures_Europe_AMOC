########################################
# %%
# LOAD PACKAGES

import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import optimize, stats
plt.rcParams.update({'font.size': 12})

import importlib
import sys, pathlib
if "__file__" in globals():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import functions
importlib.reload(functions)
import cmip_cooling  # type: ignore
importlib.reload(cmip_cooling)
import FigSupp_scaling_factor_correlations as sfc  # type: ignore
importlib.reload(sfc)

########################################
# %%
# SYNTHETIC COOLING SENSITIVITIES FOR CMIP6 MODELS (supplementary)
#
# The 8 HosMIP models (+ GISS under cal-full) provide joint samples of
# (cooling sensitivity s, projected weakening w, projected warming t). OLS
# of s on SSP-mean (w, t) across the calibration models (= multivariate-
# Gaussian conditioning; default predictors 'w' since 2026-09-01, default
# cal_set 'hosing') predicts "synthetic" cooling sensitivities, with 95%
# OLS prediction intervals (t on N-k dof, leverage term), for CMIP6 models
# that have paired tas+amoc26 data but no hosing experiments. Combined
# with each target's own scenario-specific (w, t), Monte Carlo draws from
# the predictive distribution give net-cooling-point ranges
# w_net = w + t/|s|, capped at 100% weakening. The per-country pool of
# best estimates is the "Synthetic CMIP6 range" drawn in Fig3 panel b.
#
# Caveats: the calibration models are hosing-capable,
# not a random CMIP6 draw (emergent-constraint-style transfer assumption);
# the per-target ranges share the same fitted relation and are NOT
# independent across models (joint statements need the shared-draw MC).

MAIN_SSPS = ('ssp126', 'ssp245', 'ssp370')
# Seasons the target (w, t) caches carry. AMOC is annual throughout, so only
# the warming side varies.
SEASONS = ('', 'djf', 'jja')

# CMIP6 models with paired tas+amoc26 (historical + >=1 main SSP) and no
# scaling factor (inventory snapshot 2026-08-13; lowercase dir names).
TARGET_MODELS = [
    'access-cm2', 'access-esm1-5', 'cesm2-waccm', 'cnrm-cm6-1',
    'cnrm-esm2-1', 'canesm5-canoe', 'fgoals-f3-l', 'inm-cm4-8',
    'inm-cm5-0', 'miroc-es2l', 'miroc6', 'mri-esm2-0',
    'noresm2-lm', 'noresm2-mm', 'ukesm1-0-ll',
]

WT_CACHE_TMPL = functions.local_path + 'cmip_synth_wt_{region}.nc'
WT_COUNTRIES_CACHE = functions.local_path + 'cmip_synth_wt_countries.nc'
RANGE_CACHE = functions.local_path + 'cmip_range_fig3.nc'
# Bumped 2026-08-31: membership B ("Synthetic CMIP6 range", best evidence per
# model), quartile/median schema (q25/q75/median + mc quartiles). Old caches
# fail the load assert and must be rebuilt. 2026-09-01: default predictors
# back to 'w' (untagged cache = pred-w; wt carries a _pred-wt tag) — no
# schema change, the predictors attr assert catches any stale-name file.
# Bumped to 3 on 2026-09-01: dT_{min,q25,median,q75,max} added — the
# model-level range of ΔT at fixed weakening levels DT_W_LEVELS (consumed by
# FigSupp_warming_at_weakening); same membership as the w_net range.
RANGE_SCHEMA_VERSION = 3
DT_W_LEVELS = (25.0, 75.0)   # fixed weakening levels [%] for the dT_* vars


########################################
# %%
# TARGET-SIDE (w, t) CACHE

def season_ens(model, scenario, members, season):
    """Seasonal tas ensemble mean over exactly ``members``.

    Membership is decided on the annual availability upstream, so a seasonal
    ensemble that does not reproduce it means a staging gap, not a smaller
    sample; assert rather than average a different set of members.
    """
    ens, used = cmip_cooling.ens_mean_concat(
        model, scenario, list(members), 'tas', season=season)
    assert ens is not None and set(used) == set(members), \
        f"{model} {scenario} {season}: seasonal members {used} != annual {list(members)}"
    return ens


def get_target_wt(recompute=False, models=None, scenarios=MAIN_SSPS, region='EU',
                  seasons=SEASONS, verbose=True):
    """Region-mean end-of-century warming and AMOC weakening for the targets.

    Per (model, scenario): ensemble-mean over common tas∩amoc realisations
    (historical concatenated with the scenario, via cmip_cooling helpers),
    w = 100*(1 - AMOC_2091-2100 / AMOC_1850-1899), t = region-mean tas
    2091-2100 minus 1850-1899. Region mask (default EU = NEU∪WCE∪MED) ∩ land
    on each model's native grid. Cached per region at WT_CACHE_TMPL.

    ``t`` and ``baseline_tas_eu`` carry a ``season`` dim; ``w``, its baseline
    and ``n_real`` do not, because AMOC at 26N has no seasonal sibling and the
    member set is decided on the annual availability.
    """
    import os
    cache_path = WT_CACHE_TMPL.format(region=region)
    if not recompute and os.path.exists(cache_path):
        if verbose:
            print(f"Loading cached target (w, t) from {cache_path}")
        ds = xr.open_dataset(cache_path)
        if 'season' not in ds.dims:
            raise ValueError(
                f"{cache_path} predates the season dim; rebuild it with "
                f"get_target_wt(recompute=True, region={region!r}).")
        # Honour explicit subset requests instead of silently returning the
        # full cached table.
        if models is not None:
            import cmip6_inventory
            ds = ds.sel(model=[cmip6_inventory.to_esgf(m) for m in models])
        if tuple(scenarios) != MAIN_SSPS:
            ds = ds.sel(scenario=list(scenarios))
        return ds

    models = list(models) if models is not None else list(TARGET_MODELS)
    scenarios = list(scenarios)
    seasons = list(seasons)
    fut = slice(*functions.FUTURE_WINDOW)
    base = cmip_cooling.BASELINE_SLICE

    shape = (len(models), len(scenarios))
    w_out = np.full(shape, np.nan)
    t_out = np.full(shape + (len(seasons),), np.nan)
    amoc_pi_out = np.full(shape, np.nan)
    tas_pi_eu_out = np.full(shape + (len(seasons),), np.nan)
    n_real_out = np.zeros(shape, dtype=np.int32)

    for mi, model in enumerate(models):
        masks_m = None
        for si, sce in enumerate(scenarios):
            tas_rea = cmip_cooling.list_realisations(sce, 'tas', [model]).get(model, [])
            amoc_rea = cmip_cooling.list_realisations(sce, 'amoc', [model]).get(model, [])
            common = sorted(set(tas_rea) & set(amoc_rea))
            if verbose:
                print(f"{model} {sce}: tas={len(tas_rea)}, amoc={len(amoc_rea)}, common={len(common)}")
            if not common:
                continue
            tas_ens, tas_used = cmip_cooling.ens_mean_concat(model, sce, common, 'tas')
            amoc_ens, amoc_used = cmip_cooling.ens_mean_concat(model, sce, common, 'amoc')
            if tas_ens is None or amoc_ens is None:
                continue
            # ens_mean_concat re-filters by per-variable historical
            # availability; w and t must come from the same members, so
            # re-intersect on the actually-used sets when they differ.
            if set(tas_used) != set(amoc_used):
                used = sorted(set(tas_used) & set(amoc_used))
                if verbose:
                    print(f"  {model} {sce}: tas/amoc member mismatch "
                          f"({len(tas_used)}/{len(amoc_used)}), re-intersecting to {len(used)}")
                if not used:
                    continue
                tas_ens, tas_used = cmip_cooling.ens_mean_concat(model, sce, used, 'tas')
                amoc_ens, amoc_used = cmip_cooling.ens_mean_concat(model, sce, used, 'amoc')
                assert set(tas_used) == set(amoc_used) == set(used), \
                    f"{model} {sce}: member sets diverge after re-intersection"
            n_real_out[mi, si] = len(tas_used)

            amoc_pi = float(amoc_ens.sel(time=base).mean('time'))
            amoc_fut = float(amoc_ens.sel(time=fut).mean('time'))
            if np.isfinite(amoc_pi) and amoc_pi != 0:
                amoc_pi_out[mi, si] = amoc_pi
                w_out[mi, si] = 100.0 * (1.0 - amoc_fut / amoc_pi)

            if masks_m is None:
                masks_m = functions.make_country_masks_land_aware(
                    tas_ens.to_dataset(name='tas'),
                    include_ipcc_regions=True, verbose=False)
                assert region in masks_m and bool(masks_m[region].any()), \
                    f"{model}: empty or unknown {region} mask"
            for zi, _s in enumerate(seasons):
                ens = tas_ens if not _s else season_ens(model, sce, tas_used, _s)
                eu = functions.weighted_area_lat(
                    ens.where(masks_m[region]).where(masks_m['LAND'] == 0)
                ).mean('lat').mean('lon')
                tas_pi_eu = float(eu.sel(time=base).mean('time'))
                tas_pi_eu_out[mi, si, zi] = tas_pi_eu
                t_out[mi, si, zi] = float(eu.sel(time=fut).mean('time')) - tas_pi_eu

    import cmip6_inventory
    ds = xr.Dataset(
        {'w': (('model', 'scenario'), w_out),
         't': (('model', 'scenario', 'season'), t_out),
         'baseline_amoc': (('model', 'scenario'), amoc_pi_out),
         'baseline_tas_eu': (('model', 'scenario', 'season'), tas_pi_eu_out),
         'n_real': (('model', 'scenario'), n_real_out)},
        coords={'model': [cmip6_inventory.to_esgf(m) for m in models],
                'scenario': scenarios, 'season': seasons},
        attrs={'baseline': '1850-1899', 'future_window': '2091-2100',
               'region': f'{region} (land only)'})
    ds.to_netcdf(cache_path)
    if verbose:
        print(f"Wrote {cache_path}")
    return ds


########################################
# %%
# MULTI-REGION TARGET (w, t) CACHE  (per-country Fig3 CMIP ranges)

def fig3_regions():
    """Panel-e row set: EU aggregate + the >=30k km² countries."""
    return ('EU',) + tuple(functions.regions_cutoff30k)


def target_wt_model_multi(model, regions, scenarios=MAIN_SSPS, seasons=SEASONS,
                          future_window=None, verbose=True):
    """One target model's (w, t) over many regions in a single data pass.

    Member logic identical to get_target_wt (common tas∩amoc realisations,
    re-intersection guard); the region loop only redoes the cheap masked
    area-mean. LAND==0 is applied for every region (build_table's eu_mean
    convention; differs from panel e's country means).
    Returns a Dataset with dims (scenario, region, season) on the warming
    side and (scenario,) on the AMOC side, plus a scalar model coord. The
    season loop sits inside the scenario loop so the masks are built once and
    AMOC is opened once per scenario. ``future_window`` defaults to the
    canonical 2091-2100 (functions.FUTURE_WINDOW); a decade override shifts
    the (w, t) evaluation window only, the baseline stays 1850-1899.
    """
    import cmip6_inventory
    regions = list(regions)
    seasons = list(seasons)
    fw = tuple(future_window) if future_window is not None else functions.FUTURE_WINDOW
    fut = slice(*fw)
    base = cmip_cooling.BASELINE_SLICE
    n_s, n_r, n_z = len(scenarios), len(regions), len(seasons)
    w = np.full(n_s, np.nan)
    amoc_pi_arr = np.full(n_s, np.nan)
    t = np.full((n_s, n_r, n_z), np.nan)
    tas_pi = np.full((n_s, n_r, n_z), np.nan)
    n_real = np.zeros(n_s, dtype=np.int32)

    masks_m = None
    for si, sce in enumerate(scenarios):
        tas_rea = cmip_cooling.list_realisations(sce, 'tas', [model]).get(model, [])
        amoc_rea = cmip_cooling.list_realisations(sce, 'amoc', [model]).get(model, [])
        common = sorted(set(tas_rea) & set(amoc_rea))
        if verbose:
            print(f"{model} {sce}: tas={len(tas_rea)}, amoc={len(amoc_rea)}, common={len(common)}")
        if not common:
            continue
        tas_ens, tas_used = cmip_cooling.ens_mean_concat(model, sce, common, 'tas')
        amoc_ens, amoc_used = cmip_cooling.ens_mean_concat(model, sce, common, 'amoc')
        if tas_ens is None or amoc_ens is None:
            continue
        if set(tas_used) != set(amoc_used):
            used = sorted(set(tas_used) & set(amoc_used))
            if verbose:
                print(f"  {model} {sce}: tas/amoc member mismatch "
                      f"({len(tas_used)}/{len(amoc_used)}), re-intersecting to {len(used)}")
            if not used:
                continue
            tas_ens, tas_used = cmip_cooling.ens_mean_concat(model, sce, used, 'tas')
            amoc_ens, amoc_used = cmip_cooling.ens_mean_concat(model, sce, used, 'amoc')
            assert set(tas_used) == set(amoc_used) == set(used), \
                f"{model} {sce}: member sets diverge after re-intersection"
        n_real[si] = len(tas_used)

        amoc_pi = float(amoc_ens.sel(time=base).mean('time'))
        amoc_fut = float(amoc_ens.sel(time=fut).mean('time'))
        if np.isfinite(amoc_pi) and amoc_pi != 0:
            amoc_pi_arr[si] = amoc_pi
            w[si] = 100.0 * (1.0 - amoc_fut / amoc_pi)

        if masks_m is None:
            masks_m = functions.make_country_masks_land_aware(
                tas_ens.to_dataset(name='tas'),
                include_ipcc_regions=True, verbose=False)
            empty = [r for r in regions
                     if r not in masks_m or not bool(masks_m[r].any())]
            assert len(empty) < len(regions), f"{model}: no usable region masks"
            if empty and verbose:
                print(f"  {model}: empty/unknown masks (left NaN): {empty}")
        for zi, _s in enumerate(seasons):
            ens = tas_ens if not _s else season_ens(model, sce, tas_used, _s)
            for ri, r in enumerate(regions):
                if r not in masks_m or not bool(masks_m[r].any()):
                    continue
                rm = functions.weighted_area_lat(
                    ens.where(masks_m[r]).where(masks_m['LAND'] == 0)
                ).mean('lat').mean('lon')
                tas_pi[si, ri, zi] = float(rm.sel(time=base).mean('time'))
                t[si, ri, zi] = float(rm.sel(time=fut).mean('time')) - tas_pi[si, ri, zi]

    return xr.Dataset(
        {'w': (('scenario',), w),
         'baseline_amoc': (('scenario',), amoc_pi_arr),
         'n_real': (('scenario',), n_real),
         't': (('scenario', 'region', 'season'), t),
         'baseline_tas': (('scenario', 'region', 'season'), tas_pi)},
        coords={'scenario': list(scenarios), 'region': regions,
                'season': seasons, 'model': cmip6_inventory.to_esgf(model)},
        attrs={'baseline': '1850-1899', 'future_window': f'{fw[0]}-{fw[1]}',
               'land_filter': 'LAND==0 for every region (build_table convention)'})


_WT_CTX = {}


def wt_worker(model):
    return target_wt_model_multi(model, **_WT_CTX)


def get_target_wt_multi(recompute=False, regions=None, scenarios=MAIN_SSPS,
                        seasons=SEASONS, future_window=None, processes=1,
                        verbose=True):
    """All targets' (w, t) over the panel-e region set; cached with region and
    season dims. A non-default ``future_window`` gets its own ``_fw{a}-{b}``
    cache sibling (raw CMIP6 needed to build — Levante only)."""
    import os
    regions = tuple(regions) if regions is not None else fig3_regions()
    fw = tuple(future_window) if future_window is not None else functions.FUTURE_WINDOW
    fw_str = f'{fw[0]}-{fw[1]}'
    cache_path = (WT_COUNTRIES_CACHE if fw == functions.FUTURE_WINDOW
                  else WT_COUNTRIES_CACHE.replace('.nc', f'_fw{fw_str}.nc'))
    if not recompute and os.path.exists(cache_path):
        ds = xr.open_dataset(cache_path)
        if 'season' not in ds.dims:
            raise ValueError(
                f"{cache_path} predates the season dim; rebuild it with "
                f"get_target_wt_multi(recompute=True).")
        assert ds.attrs.get('future_window') == fw_str, \
            f"cache at {cache_path} has future_window={ds.attrs.get('future_window')!r} (need {fw_str})"
        cached = set(str(r) for r in ds.region.values)
        assert set(regions) <= cached, \
            f"countries cache lacks regions {sorted(set(regions) - cached)}; recompute"
        if verbose:
            print(f"Loading cached multi-region target (w, t) from {cache_path}")
        ds = ds.sel(region=list(regions))
        if tuple(scenarios) != MAIN_SSPS:
            ds = ds.sel(scenario=list(scenarios))
        return ds

    # Off-Levante the raw CMIP6 roots are absent, so the realisation scan is
    # vacuous and a build would silently write an all-NaN cache (same trap as
    # cmip6_inventory._maybe_write_snapshot). Refuse up front.
    assert any(os.path.isdir(root) for root in cmip_cooling.SEARCH_ROOTS), (
        f"raw CMIP6 roots absent ({cmip_cooling.SEARCH_ROOTS}); the target "
        f"(w, t) cache can only be built on Levante")
    global _WT_CTX
    _WT_CTX = dict(regions=regions, scenarios=scenarios, seasons=seasons,
                   future_window=fw, verbose=verbose)
    if processes > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(processes) as pool:
            parts = pool.map(wt_worker, TARGET_MODELS)   # map preserves order
    else:
        parts = [target_wt_model_multi(m, regions, scenarios, seasons, fw, verbose)
                 for m in TARGET_MODELS]
    ds = xr.concat(parts, dim='model')
    ds = ds.assign_coords(model=ds.model.astype('<U32'))
    assert np.isfinite(ds.w.values).any(), \
        'all-NaN target (w, t) build — refusing to write the cache'
    tmp = cache_path + '.tmp'
    ds.to_netcdf(tmp)
    os.replace(tmp, cache_path)
    if verbose:
        print(f"Wrote {cache_path}")
    return ds


########################################
# %%
# CALIBRATION FIT + PREDICTION

def ssp_mean(d, ssps):
    vs = [d.get(s, np.nan) for s in ssps]
    vs = [v for v in vs if np.isfinite(v)]
    return float(np.mean(vs)) if vs else np.nan


def calibration_arrays(table, ssps=MAIN_SSPS):
    """Per calibration model: s, ste, SSP-mean (w, t), full-SSP-coverage flag."""
    out = {}
    for model, info in table.items():
        cov = all(np.isfinite(info['weakening'].get(s, np.nan))
                  and np.isfinite(info['warming'].get(s, np.nan)) for s in ssps)
        out[model] = {'s': info['slope'], 'ste': info['ste'],
                      'w': ssp_mean(info['weakening'], ssps),
                      't': ssp_mean(info['warming'], ssps),
                      'full_ssp': cov,
                      'color': info['color'], 'marker': info['marker']}
    return out


# The calibration configurations, as (models dropped from the s~w fit,
# models dropped from the state-dependence observations/pairs, require full
# SSP coverage). Since 2026-08-31 the default is 'hosing' (renamed from
# 'nogiss' the same day, Felix: the old name misled under pooled, where GISS
# stays in the state fit and the range — 'hosing' says positively that the
# calibration line is fit on the 8 hosing models). Every other set is defined
# RELATIVE to it, so a sweep variant differs from the default by exactly one
# knob (Felix, 2026-08-31): 'full' adds GISS back to the fit, 'nocesm2'
# additionally removes CESM2, 'consistentssp' additionally requires full SSP
# coverage. The default's GISS exclusion is FIT-ONLY (Felix, 2026-08-31):
# GISS's warm-state slope is inconsistent inside the PI-state s~w line but
# remains a legitimate warm-background observation, so the state-dependence
# fits keep GISS's warm row under every set; only 'nocesm2' removes a model
# (CESM2) from the state observations and pairs as well. The coverage
# requirement reaches the fit and the pooled/interact observations but not
# the pair set behind dummy/loglin (scoped 2026-08-20) — see
# state_dep_factors. Range membership is decoupled from cal_set entirely
# (evidence rule, 2026-08-31) — see range_for_region.
_CAL_SPEC = {'full':          ((), (), False),
             'hosing':        (('GISS-E2-1-G',), (), False),
             'nocesm2':       (('GISS-E2-1-G', 'CESM2'), ('CESM2',), False),
             'consistentssp': (('GISS-E2-1-G',), (), True)}
DEFAULT_CAL_SET = 'hosing'


def fit_calibration(cal, cal_set=DEFAULT_CAL_SET, origin=False):
    """OLS s ~ w on the chosen calibration subset; also the s ~ w + t
    sensitivity with the warming-skill statistics. Returns a dict.

    `origin=True` drops the intercept of this CROSS-MODEL line (sensitivity
    only; the default is a free intercept). Not to be confused with the
    through-origin constraint on the physical dT-dAMOC regressions, which is
    imposed everywhere already: a model projecting no weakening need not have
    zero sensitivity to weakening. Forcing it removes the fitted line's sign
    flip below w ~ 7% but roughly doubles |s| for the low-w targets."""
    exclude, _, need_full_ssp = _CAL_SPEC[cal_set]
    models = [m for m, v in cal.items()
              if m not in exclude
              and np.isfinite(v['s']) and np.isfinite(v['w']) and np.isfinite(v['t'])
              and (v['full_ssp'] or not need_full_ssp)]
    s = np.array([cal[m]['s'] for m in models])
    w = np.array([cal[m]['w'] for m in models])
    t = np.array([cal[m]['t'] for m in models])

    add = (lambda X: np.asarray(X).reshape(len(s), -1)) if origin else sm.add_constant
    fit_w = sm.OLS(s, add(w)).fit()
    fit_wt = sm.OLS(s, add(np.column_stack([w, t]))).fit()

    # Warming skill: t-coefficient test, partial corr of s and t given w, ΔR², F.
    res_s, res_t = fit_w.resid, sm.OLS(t, sm.add_constant(w)).fit().resid
    partial_r = float(np.corrcoef(res_s, res_t)[0, 1]) if len(models) > 2 else np.nan
    f_stat, f_p, _ = fit_wt.compare_f_test(fit_w)
    i_t = 1 if origin else 2          # index of the t coefficient
    return {'models': models, 's': s, 'w': w, 't': t, 'origin': origin,
            'fit_w': fit_w, 'fit_wt': fit_wt,
            'skill': {'t_coef': float(fit_wt.params[i_t]),
                      't_coef_p': float(fit_wt.pvalues[i_t]),
                      'partial_r_st_given_w': partial_r,
                      'delta_r2': float(fit_wt.rsquared - fit_w.rsquared),
                      'f_stat': float(f_stat), 'f_p': float(f_p)}}


def design_matrix(x, origin=False):
    """Predictor values (scalar, (n,), or (n, k)) -> design matrix, with a
    constant unless the fit was made through the origin."""
    x = np.atleast_1d(np.asarray(x, dtype=float)).reshape(-1, 1) \
        if np.ndim(x) <= 1 else np.asarray(x, dtype=float)
    return x if origin else sm.add_constant(x, has_constant='add')


def predict_synthetic(fit, x0, alpha=0.05, origin=False):
    """Predictive mean and (1-alpha) prediction interval of s at predictor x0
    ((n,) weakening for s~w, or (n, 2) weakening+warming for s~w+t)."""
    sf = fit.get_prediction(design_matrix(x0, origin)).summary_frame(alpha=alpha)
    return (sf['mean'].values, sf['obs_ci_lower'].values, sf['obs_ci_upper'].values)


def draw_joint(fit, x_targets, n_draws=20000, seed=0, origin=False):
    """Joint predictive draws of s for all targets: one shared (σ², β) draw
    per iteration + independent per-target residual draws. Marginals equal
    the standard predictive t on the fit's residual dof. Returns (n_targets, n_draws)."""
    rng = np.random.default_rng(seed)
    dof = int(fit.df_resid)
    sigma2 = fit.mse_resid * dof / rng.chisquare(dof, size=n_draws)
    cov_unscaled = np.asarray(fit.normalized_cov_params)
    L = np.linalg.cholesky(cov_unscaled)
    z = rng.standard_normal((n_draws, len(fit.params)))
    beta = fit.params[None, :] + np.sqrt(sigma2)[:, None] * (z @ L.T)
    X = design_matrix(x_targets, origin)
    eps = rng.standard_normal((X.shape[0], n_draws)) * np.sqrt(sigma2)[None, :]
    return X @ beta.T + eps


def json_safe(o):
    """Strict JSON: non-finite floats -> 'inf'/'-inf'/None (json.dump would
    otherwise emit bare Infinity/NaN, which strict parsers reject)."""
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, (bool, str)) or o is None:
        return o
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if np.isfinite(f) else 'inf' if f > 0 else '-inf' if f < 0 else None
    return o


def w_net_draws(s_draws, w, t):
    """Net-cooling weakening per draw; +inf where the slope draw is >= 0."""
    out = np.full_like(s_draws, np.inf)
    neg = s_draws < 0
    out[neg] = w + t / (-s_draws[neg])
    return out


########################################
# %%
# STATE-DEPENDENCE FACTORS

STATE_DEPS = ('none', 'dummy', 'pooled', 'loglin', 'interact')
BM_FIG2_FILE = functions.local_path + 'Bellomo2024_plotting_data/data_figs_2_S3.nc'


def state_dep_factors(region, *, multi_model_dict, hosmip_reg_ds_dict,
                      reg_ds_mpi, reg_ds_cesm, table, predictors='w',
                      ssps=MAIN_SSPS, origin=False, cal_set=DEFAULT_CAL_SET,
                      season=''):
    """Pair ratios rho = s_warm/s_pi, loglin gamma, pooled dummy regression.

    The calibration slopes are preindustrial-hosing (except GISS); the three
    models with both a PI and a warm-background slope give rho = s_warm/s_pi
    (EU: MPI 1.01, CESM2 0.75, EC-Earth3 0.69), which `state_dep` applies
    multiplicatively to the synthetic s draws and to the calibration models'
    direct slopes (GISS untouched, already warm). The variants consume this
    dict differently: 'dummy' draws one of the pair rhos per MC iteration,
    'pooled' uses e^delta from the log-space dummy regression below, 'loglin'
    uses exp(gamma*G) with G the region-mean realised warming.

    Slope recipes mirror the canonical Fig3a lines (functions.py bm_data /
    boot / add_combined_results / add_hosmip_mpi_regression blocks): the warm
    slopes for MPI / CESM2 / EC-Earth3 are the same quantities as the brown /
    dodgerblue / BM-4xCO2 elements there. EC-Earth3's PI side is the
    calibration-table (NAHosMIP) slope, not BM's own PI point slope — the
    factor has to map the calibration line. Annual
    only (BM diff fields have no seasonal siblings).

    `cal_set` reaches this fit through its own exclusion column (2026-08-31,
    previously the fit column did double duty): only `nocesm2` removes a model
    (CESM2) from the observations and pairs; the hosing default keeps GISS's
    warm row because that exclusion is fit-only (see _CAL_SPEC). Before
    2026-08-17 membership was cal_set-independent, which made cal-nocesm2 x
    sdep-pooled bit-identical to cal-full in panel b while panel a advertised
    the exclusion. `consistentssp`'s coverage requirement reaches only the
    pooled/interact observations (2026-08-20), scoped by Felix to the variants
    forwarded to Fig3. The pair set behind dummy/loglin therefore keeps the
    partial-coverage models; note loglin's gamma anchor is the EC-Earth3 pair,
    which consistentssp would otherwise remove. `pairs_all` in the return dict
    is the pre-exclusion pair set — range_for_region reads the measured warm
    slopes from it so an excluded model still displays at its best estimate
    (evidence rule, 2026-08-31).

    `season` (2026-08-23): MPI and CESM2 have seasonal slopes on both sides,
    so their pair rhos are computed per season. The EC-Earth3 pair rests on
    the Bellomo & Mehling 4xCO2 difference fields, which are annual-only
    (their djf/jja slots are 100% NaN), so it drops out of a seasonal call
    together with gamma and g4x. Seasonal pooled/interact fits therefore have
    3 warm observations instead of 4, and `loglin` is refused seasonally
    because its gamma anchor IS the EC-Earth3 pair."""
    def region_mean(da, mask):
        return sfc.eu_mean(da, mask.sel(region=region), mask.sel(region='LAND'))

    pairs = {}

    # MPI-ESM1-2-LR: NAHosMIP PI vs combined-forcing (this study).
    mask_mpi = multi_model_dict['MPI-ESM1-2-LR'].mask
    pairs['MPI-ESM1-2-LR'] = {
        's_pi': region_mean(hosmip_reg_ds_dict['MPI-ESM1-2-LR']
                            .lin_coef_hosmip.sel(season=season), mask_mpi),
        's_warm': -region_mean(reg_ds_mpi.coef_ensmean.sel(season=season),
                               mask_mpi) * functions.AMOC_pi_MPI / 100}

    # CESM2: NAHosMIP PI vs Boot et al. combined-forcing pooled regression
    # (area-mean of coef_ensmean == the pooled Fig3a regression by linearity).
    mask_cesm = multi_model_dict['CESM2'].mask
    pairs['CESM2'] = {
        's_pi': region_mean(hosmip_reg_ds_dict['CESM2']
                            .lin_coef_hosmip.sel(season=season), mask_cesm),
        's_warm': region_mean(reg_ds_cesm.coef_ensmean.sel(season=season), mask_cesm)}

    # EC-Earth3: Bellomo & Mehling 4xCO2 attribution against the
    # calibration-table PI slope (NOT BM's own PI point slope — harmonised
    # 2026-08-15, see the docstring). BM's PI point is
    # kept in the dict for the audit trail.
    ece = multi_model_dict['EC-Earth3']
    if not season:
        ctrl_pi = float(ece.sel(type='control', scenar='pi', season='').amoc.isel(time=0))
        s_bm = {}
        for key, scenar in (('pi', 'pi'), ('warm', 'ghg')):
            dA = float(ece.sel(type='diff', scenar=scenar, season='').amoc.isel(time=0)) \
                / ctrl_pi * 100
            dT = region_mean(ece.sel(type='diff', scenar=scenar, season='').tas.isel(time=0),
                             ece.mask)
            s_bm[key] = dT / dA
        pairs['EC-Earth3'] = {'s_pi': table['EC-Earth3']['slope'],
                              's_warm': s_bm['warm'], 's_pi_bm_internal': s_bm['pi']}

    _, exclude_sd, need_full_ssp = _CAL_SPEC[cal_set]
    pairs_all = dict(pairs)
    pairs = {m: p for m, p in pairs.items() if m not in exclude_sd}
    rho = {m: p['s_warm'] / p['s_pi'] for m, p in pairs.items()
           if np.isfinite(p['s_warm']) and np.isfinite(p['s_pi']) and p['s_pi'] != 0}

    # loglin anchor: EC-Earth realised 4xCO2 warming (tas_4x - tas_pi), the
    # same lon-wrap + EU-subset processing as get_other_studies_data. Annual
    # only, like the pair it anchors.
    g4x = gamma = np.nan
    if not season:
        bm2 = xr.open_dataset(BM_FIG2_FILE)
        bm2.coords['lon'] = (bm2.coords['lon'] + 180) % 360 - 180
        bm2 = bm2.sortby(bm2.lon)
        g4x = region_mean(bm2.tas_4x - bm2.tas_pi, ece.mask)
        rho_ece = rho.get('EC-Earth3', np.nan)
        gamma = (np.log(rho_ece) / g4x
                 if np.isfinite(rho_ece) and rho_ece > 0 and np.isfinite(g4x) and g4x > 0
                 else np.nan)

    # Pooled observations: the PI calibration slopes + the warm pair slopes +
    # GISS (warm-background by construction). Covariates are the model-level
    # SSP-mean calibration-table (w, t) — w is a property of the model, not of
    # the slope experiment, so a model's PI and warm rows share it. Membership
    # is the full cal_set spec, exclusions and the consistentssp coverage
    # requirement alike, so this fit sees exactly the s~w calibration set.
    cal = calibration_arrays(table, ssps)
    keep_m = lambda m: m not in exclude_sd and (cal[m]['full_ssp'] or not need_full_ssp)
    obs = [(m + ' (PI)', table[m]['slope'], cal[m]['w'], cal[m]['t'], 0.0)
           for m in functions.hosmip_labels if keep_m(m)]
    obs += [(m + ' (warm)', p['s_warm'], cal[m]['w'], cal[m]['t'], 1.0)
            for m, p in pairs.items() if keep_m(m)]
    if 'GISS-E2-1-G' in table and keep_m('GISS-E2-1-G'):
        obs.append(('GISS-E2-1-G (warm)', table['GISS-E2-1-G']['slope'],
                    cal['GISS-E2-1-G']['w'], cal['GISS-E2-1-G']['t'], 1.0))
    # Finiteness is the only row filter, matching fit_calibration, so both
    # fits see the same PI rows by construction (F5 fix, 2026-09-01: the old
    # additional s < 0 condition was selection on the dependent variable —
    # a finite s >= 0 row is a valid observation of the PI line; the sign is
    # handled per-draw in the w_net accounting, where s >= 0 draws map to
    # +inf / no net cooling).
    keep = [o for o in obs if np.isfinite(o[1])
            and np.isfinite(o[2]) and np.isfinite(o[3])]
    prop = fit_proportional(
        np.array([o[1] for o in keep]), np.array([o[2] for o in keep]),
        np.array([o[3] for o in keep]), np.array([o[4] for o in keep]),
        predictors=predictors, origin=origin)
    prop.update(models=[o[0] for o in keep],
                dropped=[o[0] for o in obs if o not in keep])
    if prop['dropped']:
        print(f"WARNING: state_dep_factors[{region}]: non-finite rows dropped "
              f"from the state fit: {prop['dropped']}")

    return {'pairs': pairs, 'pairs_all': pairs_all, 'rho': rho,
            'gamma': float(gamma), 'g4x': float(g4x), 'prop': prop}


def fit_proportional(s, w, t, warm, predictors='w', origin=False):
    """NLS fit of the proportional state model s = (a + b*w [+ c*t]) * rho**warm.

    The warm-background line is the PI line scaled by one factor, so the two
    share their zero-crossing in w; the unrestricted alternative (own intercept
    AND own slope) is reported as `interact_F`/`interact_p`. rho is the state
    factor: `pooled` predicts the targets on the warm line rather than
    rescaling the PI line afterwards, so the fit itself carries the correction.
    """
    X = np.column_stack([w] if predictors == 'w' else [w, t])
    n = len(s)
    k = X.shape[1] + (1 if origin else 2)   # [intercept] + slopes + rho

    def model(_idx, *theta):
        base = X @ np.asarray(theta[:-1]) if origin \
            else theta[0] + X @ np.asarray(theta[1:-1])
        return base * (1.0 + (theta[-1] - 1.0) * warm)

    lin = sm.OLS(s, X if origin else sm.add_constant(X)).fit()
    popt, pcov = optimize.curve_fit(
        model, np.arange(n), s, p0=list(lin.params) + [0.8], maxfev=40000)
    resid = s - model(None, *popt)
    df = n - k
    sigma2 = float(resid @ resid / df)

    # Unrestricted alternative: own intercept AND own slope under warm. The
    # proportional model is that fit with X.shape[1] restrictions imposed.
    # It is also state_dep='interact' in its own right, so its parameters,
    # covariance and residual scale are returned for interact_draws.
    inter = sm.OLS(s, interact_design(X, warm, origin)).fit()
    q = X.shape[1]
    F = ((resid @ resid - inter.ssr) / q) / (inter.ssr / inter.df_resid)
    return {'params': [float(p) for p in popt], 'rho': float(popt[-1]),
            'rho_se': float(np.sqrt(pcov[-1, -1])), 'cov': pcov, 'sigma2': sigma2,
            'df': int(df), 'n': int(n), 'n_warm': int(warm.sum()),
            'predictors': predictors, 'origin': origin,
            'r2': float(1 - resid @ resid / np.sum((s - s.mean()) ** 2)),
            # Centred R² for both, computed here: statsmodels reports an
            # UNcentred rsquared for no-constant fits (origin=True), which is
            # not comparable to the free-intercept value.
            'interact_r2': float(1 - inter.ssr / np.sum((s - s.mean()) ** 2)),
            'interact_params': [float(p) for p in inter.params],
            'interact_cov': np.asarray(inter.cov_params()),
            'interact_sigma2': float(inter.ssr / inter.df_resid),
            'interact_df': int(inter.df_resid),
            'prop_test_F': float(F),
            'prop_test_p': float(1 - stats.f.cdf(F, q, inter.df_resid))}


def interact_design(X, warm, origin=False):
    """Fully-interacted design: [const,] predictors, warm, predictors x warm."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    w = np.broadcast_to(np.asarray(warm, dtype=float), (X.shape[0],))[:, None]
    Xi = np.column_stack([X, w, X * w])
    return Xi if origin else np.column_stack([np.ones(len(Xi)), Xi])


def interact_ratio(prop, X):
    """Warm/PI ratio of the fully-interacted fit at predictor row X. Unlike the
    proportional fit's rho this is not a constant: the warm line has its own
    intercept and slope, so the ratio moves with w (and t). Where the PI line
    is within its own residual noise of zero the ratio is unidentified and
    diverges (F6; poles sit inside the calibration w-range in many regions,
    QA 2026-08-31) — return NaN there instead of a sign-flipping blow-up."""
    beta = np.asarray(prop['interact_params'])
    # .item(): scalar contract — raises on a multi-row X instead of silently
    # collapsing it (and works under NumPy 2, where float() on a (1,) array
    # is an error).
    pi = (interact_design(X, 0.0, prop['origin']) @ beta).item()
    floor = float(np.sqrt(prop.get('interact_sigma2', 0.0)))
    if not np.isfinite(pi) or abs(pi) < floor:
        return np.nan
    return (interact_design(X, 1.0, prop['origin']) @ beta).item() / pi


def proportional_draws(prop, X_targets, n_draws=20000, seed=0, shared_resid=False):
    """Predictive s draws on the WARM line, same structure as draw_joint:
    one shared (sigma^2, theta) draw per iteration + independent per-target
    residual draws. shared_resid=True shares the residual draw across targets
    (per-x marginals unchanged) — display-band use only, so the empirical
    percentile boundary is a smooth curve instead of MC-ragged."""
    rng = np.random.default_rng(seed + 3 * 104729)
    popt = np.asarray(prop['params'])
    df, s2_hat = prop['df'], prop['sigma2']
    sigma2 = s2_hat * df / rng.chisquare(df, size=n_draws)
    L = np.linalg.cholesky(np.asarray(prop['cov']) / s2_hat
                           + 1e-18 * np.eye(len(popt)))
    z = rng.standard_normal((n_draws, len(popt)))
    theta = popt[None, :] + np.sqrt(sigma2)[:, None] * (z @ L.T)
    n_slopes = len(popt) - (1 if prop.get('origin', False) else 2)
    X = np.atleast_2d(X_targets)
    if X.shape[1] != n_slopes:
        X = X.T
    if prop.get('origin', False):
        base = X @ theta[:, :-1].T                          # (n_targets, n_draws)
    else:
        base = theta[:, 0][None, :] + X @ theta[:, 1:-1].T
    n_eps = 1 if shared_resid else X.shape[0]
    eps = rng.standard_normal((n_eps, n_draws)) * np.sqrt(sigma2)[None, :]
    return base * theta[:, -1][None, :] + eps


def interact_draws(prop, X_targets, n_draws=20000, seed=0, shared_resid=False):
    """Predictive s draws from the FULLY-INTERACTED fit on the warm side.

    Same shared-(sigma^2, beta) structure as proportional_draws, its own rng
    offset. This is the unrestricted robustness variant: because the fit is
    saturated in the warm dummy, its PI side is exactly the PI-only OLS, so the
    preindustrial line is untouched by the warm observations (the property the
    proportional fit gives up in exchange for precision)."""
    rng = np.random.default_rng(seed + 5 * 104729)
    beta = np.asarray(prop['interact_params'])
    df, s2_hat = prop['interact_df'], prop['interact_sigma2']
    sigma2 = s2_hat * df / rng.chisquare(df, size=n_draws)
    cov = np.asarray(prop['interact_cov']) / s2_hat + 1e-18 * np.eye(len(beta))
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Rank-deficient interacted fit (under 'wt' a dropped warm row can
        # leave fewer warm observations than warm-side parameters, cf. IS):
        # statsmodels' pinv covariance is then singular. PSD eigen-factor
        # with clipped eigenvalues — the draws live in the identified
        # subspace, so the band UNDERSTATES the uncertainty of the
        # unidentified direction. Loud by design; supplementary variant only.
        print("WARNING: interact_draws: singular covariance (rank-deficient "
              "interacted fit); drawing in the identified subspace only")
        w_eig, V = np.linalg.eigh(cov)
        L = V * np.sqrt(np.clip(w_eig, 0.0, None))
    z = rng.standard_normal((n_draws, len(beta)))
    theta = beta[None, :] + np.sqrt(sigma2)[:, None] * (z @ L.T)
    k = 1 if prop['predictors'] == 'w' else 2
    X = np.atleast_2d(X_targets)
    if X.shape[1] != k:
        X = X.T
    base = interact_design(X, 1.0, prop['origin']) @ theta.T   # (n_targets, n_draws)
    n_eps = 1 if shared_resid else X.shape[0]
    return base + rng.standard_normal((n_eps, n_draws)) * np.sqrt(sigma2)[None, :]


def state_dep_rho_draws(sdf, state_dep, n_draws, seed):
    """Per-iteration multiplicative rho draws for the overlay variants, shared
    across scenarios within an iteration. None for everything except 'dummy':
    loglin's rho is deterministic per target x scenario, and pooled / interact
    are not overlays at all (their draws come from proportional_draws /
    interact_draws). The dedicated rng keeps the 'none' draw stream untouched."""
    if state_dep == 'dummy':
        rng = np.random.default_rng(seed + 104729)
        return rng.choice(np.array([sdf['rho'][m] for m in sorted(sdf['rho'])]),
                          size=n_draws)
    return None


def central_rho(sdf, state_dep, warming=None, weakening=None):
    """Point rho for the calibration models' direct slopes."""
    if state_dep == 'dummy':
        return float(np.mean(list(sdf['rho'].values())))
    if state_dep == 'pooled':
        return float(sdf['prop']['rho'])
    if state_dep == 'loglin':
        return float(np.exp(sdf['gamma'] * warming))
    if state_dep == 'interact':
        p = sdf['prop']
        return interact_ratio(p, [[weakening] if p['predictors'] == 'w'
                                  else [weakening, warming]])
    return 1.0


########################################
# %%
# LEAVE-ONE-OUT ROBUSTNESS CHECK

def loo_check(cal, ssps=MAIN_SSPS, alpha=0.05, cal_set=DEFAULT_CAL_SET,
              predictors='w'):
    """Predict each calibration model's s from a fit excluding it; compare
    with the true (hosing-derived) value. Returns rows + coverage count.
    Refits under the run's own cal_set and predictors (F8 fix, 2026-08-31;
    previously hardcoded 'full' and fit_w)."""
    exclude, _, need_full_ssp = _CAL_SPEC[cal_set]
    models = [m for m, v in cal.items()
              if m not in exclude
              and np.isfinite(v['s']) and np.isfinite(v['w']) and np.isfinite(v['t'])
              and (v['full_ssp'] or not need_full_ssp)]
    rows = []
    for m in models:
        sub = {k: v for k, v in cal.items() if k != m}
        fit = fit_calibration(sub, cal_set)[f'fit_{predictors}']
        x0 = cal[m]['w'] if predictors == 'w' else [[cal[m]['w'], cal[m]['t']]]
        mean, lo, hi = (x[0] for x in predict_synthetic(fit, x0, alpha))
        s_true, w0, t0 = cal[m]['s'], cal[m]['w'], cal[m]['t']
        rows.append({'model': m, 's_true': s_true, 's_pred': float(mean),
                     'pi_lo': float(lo), 'pi_hi': float(hi),
                     'inside': bool(lo <= s_true <= hi),
                     'w_net_true': w0 + t0 / (-s_true) if s_true < 0 else np.inf,
                     'w_net_pred': w0 + t0 / (-mean) if mean < 0 else np.inf})
    n_inside = sum(r['inside'] for r in rows)
    return rows, n_inside


########################################
# %%
# PER-COUNTRY CMIP RANGE DATASET (Fig3 panel-e replacement bars)

def range_for_region(region, *, w_mean_targets, wt_ds, multi_model_dict,
                     hosmip_reg_ds_dict, reg_ds_giss, giss_panel_ext, masks,
                     gwl_data, ssps=MAIN_SSPS, n_draws=20000, seed=0,
                     state_dep='none', reg_ds_mpi=None, reg_ds_cesm=None,
                     gamma_fallback=None, origin=False, predictors='w',
                     cal_set=DEFAULT_CAL_SET, season='', verbose=True):
    """One region's calibration fit + per-model best estimates + pooled MC.

    Best estimates: per-target unconditional median of w_net draws (inf =
    median draw never cools) + point w_net for the directly estimated models.
    Pooled MC: n_draws per target + n_draws point-mass copies per direct
    model (equal per-model weight).

    Membership (the "Synthetic CMIP6 range", evidence rule / option B, Felix
    + coauthor 2026-08-31): each range contains every model that has an
    estimate of the quantity the range describes, at its best available
    estimate, independent of cal_set (which only restricts what enters the
    estimation of synthetic cooling sensitivities). PI-state range
    (state_dep='none'): 8 hosing models + targets, GISS absent (no PI
    estimate). Warm-state ranges: MPI-ESM1-2-LR and CESM2 at their measured
    combined-forcing slopes, GISS at its own warm slope, the remaining hosing
    models (incl. EC-Earth3, whose 4xCO2 value is not an SSP-context
    estimate) at s_pi x rho, targets on the warm line / rescaled draws.
    """
    targets = list(w_mean_targets.index)
    table = sfc.build_table(multi_model_dict, hosmip_reg_ds_dict,
                            reg_ds_giss, giss_panel_ext, masks, gwl_data,
                            region=region, season=season, window=10,
                            giss_time_period='2101-2300', ssps=ssps,
                            include_giss=True, include_ccsm4_cesm1=False,
                            baseline='hist_1850_1899')
    cal = calibration_arrays(table, ssps)
    res = fit_calibration(cal, cal_set, origin=origin)
    fit = res[f'fit_{predictors}']
    # SSP-mean warming per target for this region: the second column of the
    # wt design (F3 closure, 2026-08-31). A NaN row yields NaN draws for that
    # target only; the per-ssp finiteness check drops it downstream.
    t_mean_targets = (wt_ds.t.sel(region=region, season=season).to_pandas()
                      [list(ssps)].mean(axis=1, skipna=True).reindex(targets))
    X0 = (w_mean_targets.values if predictors == 'w'
          else np.column_stack([w_mean_targets.values, t_mean_targets.values]))
    # Region-specific sub-seed: identical seeds would reuse one set of
    # standard-normal draws across all regions.
    rseed = seed + 7919 * (sorted(fig3_regions()).index(region) + 1)
    s_draws = draw_joint(fit, X0, n_draws=n_draws, seed=rseed, origin=origin)

    assert state_dep in STATE_DEPS
    assert not (state_dep == 'loglin' and season), \
        "state_dep='loglin' is annual only: its gamma anchor is the EC-Earth3 pair"
    sdf = rho_draws = None
    if state_dep != 'none':
        assert reg_ds_mpi is not None and reg_ds_cesm is not None, \
            'state_dep != none needs reg_ds_mpi and reg_ds_cesm'
        sdf = state_dep_factors(region, multi_model_dict=multi_model_dict,
                                hosmip_reg_ds_dict=hosmip_reg_ds_dict,
                                reg_ds_mpi=reg_ds_mpi, reg_ds_cesm=reg_ds_cesm,
                                table=table, predictors=predictors, ssps=ssps,
                                origin=origin, cal_set=cal_set, season=season)
        if state_dep == 'loglin' and not np.isfinite(sdf['gamma']):
            # The gamma anchor is the EC-Earth pair; where BM's 4xCO2 field has
            # no valid cells under the region mask (IS) there is no pair, so
            # fall back to the EU-aggregate gamma and record that we did.
            assert gamma_fallback is not None and np.isfinite(gamma_fallback), (
                f'{region}: loglin gamma not finite and no EU fallback supplied '
                f"(EC-Earth warm slope = {sdf['pairs']['EC-Earth3']['s_warm']})")
            sdf = dict(sdf, gamma=float(gamma_fallback), gamma_is_fallback=True)
        rho_draws = state_dep_rho_draws(sdf, state_dep, n_draws, rseed)
        if state_dep in ('pooled', 'interact'):
            # Not overlays: the targets are predicted on the fitted warm line,
            # so the integrated fit supplies the draws outright.
            draws = proportional_draws if state_dep == 'pooled' else interact_draws
            s_draws = draws(sdf['prop'], X0, n_draws=n_draws, seed=rseed)

    i_w = 0 if origin else 1          # index of the w coefficient
    out = {'fit': {'n': len(res['models']), 'origin': origin,
                   'intercept': 0.0 if origin else float(fit.params[0]),
                   'slope': float(fit.params[i_w]),
                   'slope_p': float(fit.pvalues[i_w]),
                   'r2': float(fit.rsquared),
                   'resid_std': float(np.sqrt(fit.mse_resid))},
           'ssp': {}}
    w_tab = wt_ds.w.to_pandas()  # model x scenario
    out['members'] = {}
    out['members_dT'] = {}
    for ssp in ssps:
        best, pooled, n_t, members = [], [], 0, {}
        # ΔT at fixed weakening (schema v3): dT(wl) = t0 + s·(wl - w0), the
        # same per-model (s, w0, t0) as w_net, always finite (no inf capping).
        # Targets enter at the median of their dT draws, direct models as
        # points — mirroring the w_net best-estimate rule.
        dT_best = {wl: [] for wl in DT_W_LEVELS}
        members_dT = {}
        for i, m in enumerate(targets):
            w0 = float(w_tab.loc[m, ssp])
            t0 = float(wt_ds.t.sel(model=m, scenario=ssp, region=region,
                                   season=season))
            if not (np.isfinite(w0) and np.isfinite(t0)):
                continue
            s_i = s_draws[i]
            if rho_draws is not None:
                s_i = s_i * rho_draws
            elif state_dep == 'loglin':
                s_i = s_i * np.exp(sdf['gamma'] * t0)
            wn = w_net_draws(s_i, w0, t0)
            v = float(np.percentile(wn, 50, method='lower'))
            members[str(m)] = v
            best.append(v)
            pooled.append(wn)
            n_t += 1
            members_dT[str(m)] = {}
            for wl in DT_W_LEVELS:
                dv = float(np.median(t0 + s_i * (wl - w0)))
                dT_best[wl].append(dv)
                members_dT[str(m)][wl] = dv
        for m, info in table.items():
            # Membership: evidence rule (option B), see docstring.
            w0 = info['weakening'].get(ssp, np.nan)
            t0 = info['warming'].get(ssp, np.nan)
            slope = info['slope']
            if state_dep == 'none':
                if m == 'GISS-E2-1-G':
                    continue                      # no PI estimate
            elif m in ('MPI-ESM1-2-LR', 'CESM2'):
                slope = sdf['pairs_all'][m]['s_warm']   # measured warm slope
            elif m != 'GISS-E2-1-G':
                slope = slope * central_rho(sdf, state_dep, warming=t0, weakening=w0)
            if not (np.isfinite(w0) and np.isfinite(t0) and np.isfinite(slope)):
                continue
            v = w0 + t0 / (-slope) if slope < 0 else np.inf
            members[m] = v
            best.append(v)
            pooled.append(np.full(n_draws, v))
            members_dT[m] = {}
            for wl in DT_W_LEVELS:
                dv = float(t0 + slope * (wl - w0))
                dT_best[wl].append(dv)
                members_dT[m][wl] = dv
        best = np.asarray(best)
        pooled = np.concatenate(pooled) if pooled else np.array([np.nan])
        # Model-level band: proper quartiles (linear interpolation) + the
        # even-n textbook median (mean of the lower/higher middle order
        # statistics). Any non-finite quantile (inf-inf interpolation) is
        # 'beyond cap' -> +inf, deliberately never NaN. Draw-level quantiles
        # keep method='lower' (order statistics on the inf-heavy pooled
        # sample); mc_lo/mc_hi keep their 2.5/97.5 definition.
        _capinf = lambda x: float(x) if not np.isnan(x) else float('inf')
        if best.size:
            with np.errstate(invalid='ignore'):
                q25, q75 = (_capinf(np.percentile(best, p)) for p in (25, 75))
                med = _capinf(0.5 * (np.percentile(best, 50, method='lower')
                                     + np.percentile(best, 50, method='higher')))
            wmin, wmax = float(best.min()), float(best.max())
            frac_no = float(np.mean(~(best <= 100.0)))
        else:
            q25 = q75 = med = wmin = wmax = frac_no = np.nan
        mc = np.percentile(pooled, [2.5, 25, 50, 75, 97.5], method='lower')
        dT_stats = {}
        for wl in DT_W_LEVELS:
            arr = np.asarray(dT_best[wl])
            dT_stats[wl] = ({'min': float(arr.min()),
                             'q25': float(np.percentile(arr, 25)),
                             'median': float(np.median(arr)),
                             'q75': float(np.percentile(arr, 75)),
                             'max': float(arr.max())} if arr.size else
                            {k: np.nan for k in ('min', 'q25', 'median', 'q75', 'max')})
        out['ssp'][ssp] = {
            'q25': q25, 'q75': q75, 'median': med,
            'wmin': wmin, 'wmax': wmax,
            'frac_models_no_cooling': frac_no,
            'mc_lo': float(mc[0]), 'mc_q25': float(mc[1]),
            'mc_median': float(mc[2]), 'mc_q75': float(mc[3]),
            'mc_hi': float(mc[4]),
            'p_no_cooling': float(np.mean(~(pooled <= 100.0))),
            'n_models': int(best.size), 'n_targets': n_t,
            'dT': dT_stats}
        out['members'][ssp] = members
        out['members_dT'][ssp] = members_dT
    if sdf is not None:
        out['state_dep'] = {'active': state_dep, 'rho': sdf['rho'],
                            'n_pairs': len(sdf['rho']),
                            'gamma': sdf['gamma'], 'g4x': sdf['g4x'],
                            'gamma_is_fallback': sdf.get('gamma_is_fallback', False),
                            'prop': {k: v for k, v in sdf['prop'].items()
                                     if not k.endswith('cov')}}
    if verbose:
        f = out['fit']
        print(f"{region}: n={f['n']}, slope={f['slope']:.5f} (p={f['slope_p']:.3f}), "
              f"R²={f['r2']:.2f}, resid std={f['resid_std']:.4f}"
              + (f", state_dep={state_dep}" if state_dep != 'none' else ''))
    return out


_RANGE_CTX = {}


def range_worker(region):
    return region, range_for_region(region, **_RANGE_CTX)


def range_cache_path(state_dep='none', origin=False, cal_set=DEFAULT_CAL_SET,
                     season='', predictors='w', future_window=None):
    """Defaults ('none', free intercept, hosing calibration, annual, w
    predictors, 2091-2100 window) keep the base cache path; every other
    combination gets its own file. The untagged base has re-meant twice
    (2026-08-31: cal-full → cal-hosing + pred-wt; 2026-09-01: default
    predictors back to 'w') — the schema-version and attrs asserts in
    get_cmip_range_ds refuse any mismatched file left on disk."""
    assert state_dep in STATE_DEPS
    p = (RANGE_CACHE if state_dep == 'none'
         else RANGE_CACHE.replace('.nc', f'_sdep-{state_dep}.nc'))
    if origin:
        p = p.replace('.nc', '_origin-zero.nc')
    if cal_set != DEFAULT_CAL_SET:
        p = p.replace('.nc', f'_cal-{cal_set}.nc')
    if predictors != 'w':
        p = p.replace('.nc', f'_pred-{predictors}.nc')
    if season:
        p = p.replace('.nc', f'_season-{season}.nc')
    fw = tuple(future_window) if future_window is not None else functions.FUTURE_WINDOW
    if fw != functions.FUTURE_WINDOW:
        p = p.replace('.nc', f'_fw{fw[0]}-{fw[1]}.nc')
    return p


def get_cmip_range_ds(recompute=False, *, multi_model_dict=None,
                      hosmip_reg_ds_dict=None, reg_ds_giss=None, masks=None,
                      gwl_data=None, ssps=MAIN_SSPS, n_draws=20000, seed=0,
                      state_dep='none', reg_ds_mpi=None, reg_ds_cesm=None,
                      origin=False, predictors='w', cal_set=DEFAULT_CAL_SET,
                      season='', future_window=None, processes=1, verbose=True):
    """Per-(region, scenario) Synthetic CMIP6 range dataset for Fig3 panel e.

    Cached at range_cache_path(state_dep, origin, cal_set, season, predictors,
    future_window) with a JSON sidecar of the per-region fit stats and the
    per-model member values. `cal_set` restricts the estimation (s~w[+t] fit,
    state observations per _CAL_SPEC); range membership is the evidence rule —
    see range_for_region. `season` selects the warming side throughout
    (calibration slopes, target t, GISS panel); the weakening side stays
    annual. A non-default `future_window` shifts the target (w, t) window AND
    must be matched by window-keyed reg inputs (reg_ds_mpi/cesm/giss and
    hosmip_reg_ds_dict loaded with the same future_window) — the direct
    models' w0/t0 and the warm-state slopes come from those.
    """
    import os
    assert cal_set in _CAL_SETS
    fw = tuple(future_window) if future_window is not None else functions.FUTURE_WINDOW
    fw_str = f'{fw[0]}-{fw[1]}'
    cache_path = range_cache_path(state_dep, origin, cal_set, season, predictors, fw)
    if not recompute and os.path.exists(cache_path):
        if verbose:
            print(f"Loading cached CMIP range ds from {cache_path}")
        ds = xr.open_dataset(cache_path)
        assert ds.attrs.get('schema_version') == RANGE_SCHEMA_VERSION, (
            f"cache at {cache_path} has schema_version="
            f"{ds.attrs.get('schema_version')!r} (need {RANGE_SCHEMA_VERSION}); "
            f"rebuild with recompute=True")
        assert ds.attrs.get('state_dep', 'none') == state_dep, \
            f"cache at {cache_path} has state_dep={ds.attrs.get('state_dep')!r}"
        assert ds.attrs.get('cal_set') == cal_set, \
            f"cache at {cache_path} has cal_set={ds.attrs.get('cal_set')!r}"
        assert ds.attrs.get('predictors') == predictors, \
            f"cache at {cache_path} has predictors={ds.attrs.get('predictors')!r}"
        assert ds.attrs.get('season', '') == season, \
            f"cache at {cache_path} has season={ds.attrs.get('season')!r}"
        assert int(ds.attrs.get('n_draws', -1)) == n_draws, \
            f"cache at {cache_path} has n_draws={ds.attrs.get('n_draws')!r}"
        assert int(ds.attrs.get('seed', -1)) == seed, \
            f"cache at {cache_path} has seed={ds.attrs.get('seed')!r}"
        assert ds.attrs.get('future_window') == fw_str, \
            f"cache at {cache_path} has future_window={ds.attrs.get('future_window')!r} (need {fw_str})"
        return ds

    if multi_model_dict is None or hosmip_reg_ds_dict is None:
        raise FileNotFoundError(
            f"CMIP range cache for state_dep={state_dep!r} cal_set={cal_set!r} "
            f"predictors={predictors!r} season={season!r} not found at "
            f"{cache_path}. Build it by running this script's BUILD RANGE "
            f"CACHE cell (get_cmip_range_ds(recompute=True, "
            f"state_dep={state_dep!r}, cal_set={cal_set!r}, "
            f"predictors={predictors!r}, season={season!r}, ...)), or pass "
            f"the loaded inputs to build it here.")
    if state_dep != 'none':
        assert reg_ds_mpi is not None and reg_ds_cesm is not None, \
            'state_dep != none needs reg_ds_mpi and reg_ds_cesm'
    regions = list(fig3_regions())
    wt_ds = get_target_wt_multi(recompute=False, regions=regions, future_window=fw)
    giss_panel_ext = functions.get_giss_panel_data(
        masks=masks, regions=('EU', 'NEU', 'WCE', 'MED')
        + tuple(functions.regions_cutoff30k), season=season, recompute=False)

    w_tab = wt_ds.w.to_pandas()
    w_mean = w_tab[list(ssps)].mean(axis=1, skipna=True).dropna()

    gamma_fallback = None
    if state_dep == 'loglin':
        eu_table = sfc.build_table(multi_model_dict, hosmip_reg_ds_dict, reg_ds_giss,
                                   giss_panel_ext, masks, gwl_data, region='EU',
                                   season=season, window=10,
                                   giss_time_period='2101-2300', ssps=ssps,
                                   include_giss=True, include_ccsm4_cesm1=False,
                                   baseline='hist_1850_1899')
        gamma_fallback = state_dep_factors(
            'EU', multi_model_dict=multi_model_dict,
            hosmip_reg_ds_dict=hosmip_reg_ds_dict, reg_ds_mpi=reg_ds_mpi,
            reg_ds_cesm=reg_ds_cesm, table=eu_table, ssps=ssps,
            origin=origin, predictors=predictors, cal_set=cal_set,
            season=season)['gamma']

    global _RANGE_CTX
    _RANGE_CTX = dict(w_mean_targets=w_mean, wt_ds=wt_ds,
                      multi_model_dict=multi_model_dict,
                      hosmip_reg_ds_dict=hosmip_reg_ds_dict,
                      reg_ds_giss=reg_ds_giss, giss_panel_ext=giss_panel_ext,
                      masks=masks, gwl_data=gwl_data, ssps=ssps,
                      n_draws=n_draws, seed=seed, state_dep=state_dep,
                      reg_ds_mpi=reg_ds_mpi, reg_ds_cesm=reg_ds_cesm,
                      gamma_fallback=gamma_fallback, origin=origin,
                      predictors=predictors, cal_set=cal_set, season=season,
                      verbose=verbose)
    if processes > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(processes) as pool:
            results = dict(pool.map(range_worker, regions))
    else:
        results = dict(map(range_worker, regions))

    ssp_vars = ('q25', 'q75', 'median', 'wmin', 'wmax',
                'frac_models_no_cooling', 'mc_lo', 'mc_q25', 'mc_median',
                'mc_q75', 'mc_hi', 'p_no_cooling', 'n_models', 'n_targets')
    data = {v: (('region', 'scenario'),
                np.array([[results[r]['ssp'][s][v] for s in ssps] for r in regions]))
            for v in ssp_vars}
    for st in ('min', 'q25', 'median', 'q75', 'max'):
        data[f'dT_{st}'] = (
            ('region', 'scenario', 'w_level'),
            np.array([[[results[r]['ssp'][s]['dT'][wl][st] for wl in DT_W_LEVELS]
                       for s in ssps] for r in regions]))
    for v in ('n', 'intercept', 'slope', 'slope_p', 'r2', 'resid_std'):
        data[f'fit_{v}'] = (('region',),
                            np.array([results[r]['fit'][v] for r in regions]))
    ds = xr.Dataset(data, coords={'region': regions, 'scenario': list(ssps),
                                  'w_level': list(DT_W_LEVELS)},
                    attrs={'schema_version': RANGE_SCHEMA_VERSION,
                           'n_draws': n_draws, 'seed': seed,
                           'dT_at_w': 'dT_* [K vs 1850-1899] = model-level range of '
                                      't0 + s*(w_level - w0) at fixed weakening; same '
                                      'membership and best-estimate rule as w_net',
                           'predictors': predictors, 'cal_set': cal_set,
                           'state_dep': state_dep, 'origin': str(origin),
                           'season': season, 'future_window': fw_str,
                           'membership': 'evidence rule (B, 2026-08-31): every model '
                                         'with an estimate of the range quantity, at its '
                                         'best available estimate; measured warm slopes '
                                         'for MPI-ESM1-2-LR/CESM2, GISS warm-only',
                           'best_estimate': 'target: unconditional median of MC w_net; '
                                            'direct models: point w_net (build_table conventions)',
                           'mc_pool': 'n_draws per target + n_draws point-mass copies '
                                      'per direct model (equal model weight)'})
    tmp = cache_path + '.tmp'
    ds.to_netcdf(tmp)
    os.replace(tmp, cache_path)
    sidecar_fit = {r: dict(results[r]['fit']) for r in regions}
    for r in regions:
        sidecar_fit[r]['members'] = results[r]['members']
        sidecar_fit[r]['members_dT'] = results[r]['members_dT']
    if state_dep != 'none':
        for r in regions:
            sidecar_fit[r]['state_dep'] = results[r]['state_dep']
    with open(cache_path.replace('.nc', '_fit_stats.json'), 'w') as f:
        json.dump(json_safe(sidecar_fit), f, indent=1, allow_nan=False)
    if verbose:
        print(f"Wrote {cache_path}")
    return ds


########################################
# %%
# FIGURE FUNCTION

_CAL_SETS = tuple(_CAL_SPEC)
# Short filename values for the long knob tokens (Overleaf <=150-char basename
# limit, 2026-08-31); values not listed pass through unchanged.
_CAL_TAG = {'consistentssp': 'conssp'}
_SDEP_TAG = {'interact': 'int'}


def make_figure(table=None, wt_ds=None, *,
                multi_model_dict=None, hosmip_reg_ds_dict=None,
                reg_ds_giss=None, reg_ds_giss_panel=None, masks=None, gwl_data=None,
                season='', window=10, giss_time_period='2101-2300',
                ssps=MAIN_SSPS, region='EU', baseline='hist_1850_1899',
                cal_set=DEFAULT_CAL_SET, predictors='w',
                state_dep='none', reg_ds_mpi=None, reg_ds_cesm=None,
                origin=False, n_draws=20000, seed=0,
                plot_bg='white', savedir='../plots',
                fig=None, axes=None, panel_titles=True, panel_labels=('a', 'b')):
    assert cal_set in _CAL_SETS
    assert predictors in ('w', 'wt')
    assert state_dep in STATE_DEPS
    if state_dep != 'none':
        assert not (state_dep == 'loglin' and season), \
            "state_dep='loglin' is annual only: its gamma anchor is the EC-Earth3 pair"
        assert reg_ds_mpi is not None and reg_ds_cesm is not None, \
            'state_dep != none needs reg_ds_mpi and reg_ds_cesm'

    if table is None:
        table = sfc.build_table(multi_model_dict, hosmip_reg_ds_dict,
                                reg_ds_giss, reg_ds_giss_panel, masks, gwl_data,
                                region=region, season=season, window=window,
                                giss_time_period=giss_time_period, ssps=ssps,
                                include_giss=True, include_ccsm4_cesm1=False,
                                baseline=baseline)
    if wt_ds is None:
        wt_ds = get_target_wt(recompute=False, region=region)
    else:
        wt_region = wt_ds.attrs.get('region', '')
        assert wt_region.split(' ')[0] == region, \
            f"wt_ds is for region {wt_region!r}, figure requested {region!r}"
    wt_ds = wt_ds.sel(season=season)

    cal = calibration_arrays(table, ssps)
    res = fit_calibration(cal, cal_set, origin=origin)
    fit = res['fit_w'] if predictors == 'w' else res['fit_wt']

    # State-dependence factors: computed whenever the inputs are available
    # (also for state_dep='none' runs) so every sidecar carries the
    # all-variants comparison block.
    sdf = None
    if (reg_ds_mpi is not None and reg_ds_cesm is not None
            and multi_model_dict is not None and hosmip_reg_ds_dict is not None):
        sdf = state_dep_factors(region, multi_model_dict=multi_model_dict,
                                hosmip_reg_ds_dict=hosmip_reg_ds_dict,
                                reg_ds_mpi=reg_ds_mpi, reg_ds_cesm=reg_ds_cesm,
                                table=table, predictors=predictors, ssps=ssps,
                                origin=origin, cal_set=cal_set, season=season)
    if state_dep != 'none':
        assert sdf is not None, 'state_dep needs multi_model_dict + hosmip_reg_ds_dict'
        if state_dep == 'loglin':
            assert np.isfinite(sdf['gamma']), f'{region}: loglin gamma not finite'
    rho_draws_active = (state_dep_rho_draws(sdf, state_dep, n_draws, seed)
                        if state_dep in ('dummy', 'pooled') else None)

    # Sidecar: all 3x2 variant fits + skill stats + LOO, so every quoted
    # number has a text source (never quote from the rendered figure).
    loo_rows, loo_inside = loo_check(cal, ssps, cal_set=cal_set,
                                     predictors=predictors)
    sidecar = {'predictors': predictors, 'variants': {},
               'loo': {'rows': loo_rows, 'n_inside_95pi': loo_inside,
                       'n_total': len(loo_rows)}}
    for cs in _CAL_SETS:
        r = fit_calibration(cal, cs, origin=origin)
        i_w = 0 if origin else 1
        sidecar['variants'][cs] = {
            'models': r['models'], 'n': len(r['models']), 'origin': origin,
            'intercept': 0.0 if origin else float(r['fit_w'].params[0]),
            'slope': float(r['fit_w'].params[i_w]),
            'slope_p': float(r['fit_w'].pvalues[i_w]),
            'r2': float(r['fit_w'].rsquared),
            'resid_std': float(np.sqrt(r['fit_w'].mse_resid)),
            'wt_params': [float(p) for p in r['fit_wt'].params],
            'wt_r2': float(r['fit_wt'].rsquared),
            'wt_resid_std': float(np.sqrt(r['fit_wt'].mse_resid)),
            'warming_skill': r['skill']}

    # Targets: synthetic s predicted at SSP-mean w; w_net per scenario.
    tm = [str(m) for m in wt_ds.model.values]
    w_tab = wt_ds.w.to_pandas()      # model x scenario
    t_tab = wt_ds.t.to_pandas()
    w_mean = w_tab[list(ssps)].mean(axis=1, skipna=True)
    t_mean = t_tab[list(ssps)].mean(axis=1, skipna=True)
    targets = [m for m in tm if np.isfinite(w_mean[m]) and np.isfinite(t_mean[m])]
    partial = {m: bool((w_tab.loc[m, list(ssps)].isna()
                        | t_tab.loc[m, list(ssps)].isna()).any()) for m in targets}

    X0 = w_mean[targets].values if predictors == 'w' else \
        np.column_stack([w_mean[targets].values, t_mean[targets].values])
    base_draws = draw_joint(fit, X0, n_draws=n_draws, seed=seed, origin=origin)
    s_draws = base_draws
    s_mean, s_lo, s_hi = predict_synthetic(fit, X0, origin=origin)
    if state_dep in ('pooled', 'interact'):
        draws = proportional_draws if state_dep == 'pooled' else interact_draws
        s_draws = draws(sdf['prop'], X0, n_draws=n_draws, seed=seed)
        s_mean = s_draws.mean(axis=1)
        s_lo, s_hi = np.percentile(s_draws, [2.5, 97.5], axis=1)

    per_target = {}
    for i, m in enumerate(targets):
        entry = {'w_mean': float(w_mean[m]), 's_mean': float(s_mean[i]),
                 's_pi': [float(s_lo[i]), float(s_hi[i])], 'ssp': {}}
        for ssp in ssps:
            w0, t0 = float(w_tab.loc[m, ssp]), float(t_tab.loc[m, ssp])
            if not (np.isfinite(w0) and np.isfinite(t0)):
                continue
            s_i = s_draws[i]
            if rho_draws_active is not None:
                s_i = s_i * rho_draws_active
            elif state_dep == 'loglin':
                s_i = s_i * np.exp(sdf['gamma'] * t0)
            wn = w_net_draws(s_i, w0, t0)
            frac_no = float(np.mean(~(wn <= 100.0)))
            # Unconditional quantiles over ALL draws (inf = no cooling ever);
            # quantiles beyond the cap are simply not drawn, so the plotted
            # median/bounds and the P(no-cooling) annotation refer to the
            # same distribution. method='lower' returns an order statistic
            # (finite or inf, never the NaN that linear interpolation
            # produces between two inf draws).
            q = np.percentile(wn, [2.5, 50, 97.5], method='lower')
            entry['ssp'][ssp] = {'w': w0, 't': t0, 'frac_no_cooling': frac_no,
                                 'q2.5': float(q[0]), 'median': float(q[1]),
                                 'q97.5': float(q[2])}
        per_target[m] = entry
    sidecar['targets'] = per_target

    # Calibration models' own per-scenario w_net (direct, point estimate),
    # under the same evidence rule as range_for_region (2026-08-31): GISS out
    # of the PI-state strip, measured warm slopes for MPI/CESM2 under an
    # active state_dep, rho rescale for the rest (GISS untouched).
    cal_wnet = {ssp: [] for ssp in ssps}
    for m, info in table.items():
        for ssp in ssps:
            w0, t0 = info['weakening'].get(ssp, np.nan), info['warming'].get(ssp, np.nan)
            slope = info['slope']
            if state_dep == 'none':
                if m == 'GISS-E2-1-G':
                    continue
            elif m in ('MPI-ESM1-2-LR', 'CESM2') and sdf is not None:
                slope = sdf['pairs_all'][m]['s_warm']
            elif m != 'GISS-E2-1-G':
                slope = slope * central_rho(sdf, state_dep, warming=t0, weakening=w0)
            if np.isfinite(w0) and np.isfinite(t0) and slope < 0:
                cal_wnet[ssp].append(w0 + t0 / (-slope))
    sidecar['cal_wnet'] = cal_wnet   # no longer displayed (strip removed 2026-08-31)

    # Sidecar: state-dep block on every run where the inputs allow it —
    # pair rhos, pooled regression, gamma, and the per-target per-ssp
    # comparison across ALL variants (same dedicated seeds as an active run,
    # so one 'none' run yields the full comparison as text).
    sd_block = {'active': state_dep, 'available': sdf is not None}
    if sdf is not None:
        sd_block.update({'pairs': sdf['pairs'], 'pairs_all': sdf['pairs_all'],
                         'rho': sdf['rho'],
                         'gamma': sdf['gamma'], 'g4x': sdf['g4x'],
                         'prop': {k: v for k, v in sdf['prop'].items()
                                  if not k.endswith('cov')}, 'variants': {}})
        for sd in STATE_DEPS:
            if sd == 'loglin' and not np.isfinite(sdf['gamma']):
                continue
            rd = state_dep_rho_draws(sdf, sd, n_draws, seed)
            # base_draws, never s_draws: the active run may have replaced
            # s_draws with the integrated fit's draws.
            sd_draws = base_draws
            if sd in ('pooled', 'interact'):
                draws = proportional_draws if sd == 'pooled' else interact_draws
                sd_draws = draws(sdf['prop'], X0, n_draws=n_draws, seed=seed)
            var = {}
            for i, m in enumerate(targets):
                e = {}
                for ssp in ssps:
                    w0, t0 = float(w_tab.loc[m, ssp]), float(t_tab.loc[m, ssp])
                    if not (np.isfinite(w0) and np.isfinite(t0)):
                        continue
                    if rd is not None:
                        s_i = sd_draws[i] * rd
                    elif sd == 'loglin':
                        s_i = sd_draws[i] * np.exp(sdf['gamma'] * t0)
                    else:
                        s_i = sd_draws[i]
                    wn = w_net_draws(s_i, w0, t0)
                    e[ssp] = {'median': float(np.percentile(wn, 50, method='lower')),
                              'frac_no_cooling': float(np.mean(~(wn <= 100.0)))}
                var[m] = e
            sd_block['variants'][sd] = var
    sidecar['state_dep'] = sd_block

    # ── Plot ──────────────────────────────────────────────────────────
    text_color = 'black' if plot_bg != 'black' else 'white'
    grey = '0.45' if plot_bg != 'black' else '0.65'
    if axes is None:
        plt.style.use('default')
        plt.rcParams.update({'font.size': 12})
        if plot_bg == 'black':
            plt.style.use('dark_background')
            plt.rcParams['axes.facecolor'] = '#191919'
            plt.rcParams['figure.facecolor'] = '#191919'
        fig, (ax_a, ax_b) = plt.subplots(
            1, 2, figsize=(16, 0.30 * len(targets) + 1.6),
            gridspec_kw={'width_ratios': [1, 1.35]})
    else:
        # Stacked use: draw into provided axes, styling owned by the caller.
        ax_a, ax_b = axes

    # Panel a: calibration scatter + fit + 95% prediction band + targets.
    x_hi = max(list(res['w']) + [w_mean[m] for m in targets]) * 1.12
    xg = np.linspace(0, x_hi, 200)
    if predictors == 'w':
        Xg = xg
        fit_label = (rf'OLS fit $\beta\sim\Delta$AMOC '
                     f'(n={len(res["models"])}, R²={fit.rsquared:.2f})')
    else:
        # 2-D fit shown along the calibration's own warming-given-weakening line.
        fit_tw = sm.OLS(res['t'], sm.add_constant(res['w'])).fit()
        Xg = np.column_stack([xg, fit_tw.predict(sm.add_constant(xg))])
        fit_label = (rf'OLS fit $\beta\sim(\Delta$AMOC$,\,\Delta T)$ at '
                     rf'$\widehat{{\Delta T}}(\Delta$AMOC$)$ '
                     f'(n={len(res["models"])}, R²={fit.rsquared:.2f})')
    if state_dep in ('pooled', 'interact'):
        # Integrated fits: the band belongs to the WARM line (where the targets
        # are predicted); the PI line is shown dashed for reference, and the
        # three warm pair slopes join the scatter as the observations that
        # identify the warm side.
        p = sdf['prop']
        Xgm = np.atleast_2d(Xg).T if np.ndim(Xg) == 1 else Xg
        if state_dep == 'pooled':
            gd = proportional_draws(p, Xg, n_draws=n_draws, seed=seed,
                                    shared_resid=True)
            base = (Xgm @ np.asarray(p['params'][:-1]) if origin
                    else p['params'][0] + Xgm @ np.asarray(p['params'][1:-1]))
            warm_line = base * p['rho']
            warm_lab = f"warm-background line (ρ={p['rho']:.2f}, n={p['n']})"
        else:
            gd = interact_draws(p, Xg, n_draws=n_draws, seed=seed,
                                shared_resid=True)
            beta = np.asarray(p['interact_params'])
            base = interact_design(Xgm, 0.0, origin) @ beta
            warm_line = interact_design(Xgm, 1.0, origin) @ beta
            warm_lab = (f"warm-background line (own intercept + slope, "
                        f"n={p['n']}, R²={p['interact_r2']:.2f})")
        lg, hg = np.percentile(gd, [2.5, 97.5], axis=1)
        ax_a.fill_between(xg, lg, hg, color=grey, alpha=0.18, linewidth=0,
                          label='95% prediction interval (warm)')
        ax_a.plot(xg, base, color=text_color, lw=1.0, ls='--',
                  label='preindustrial line')
        ax_a.plot(xg, warm_line, color=text_color, lw=1.6, label=warm_lab)
        for m, pr in sdf['pairs'].items():
            ax_a.scatter(cal[m]['w'], pr['s_warm'], color=cal[m]['color'],
                         marker='X', s=80, edgecolors='black', linewidths=0.6,
                         zorder=4)
        ax_a.scatter([], [], color=grey, marker='X', s=60, edgecolors='black',
                     linewidths=0.6, label='warm-background sensitivity (pair / GISS)')
    else:
        mg, lg, hg = predict_synthetic(fit, Xg, origin=origin)
        ax_a.fill_between(xg, lg, hg, color=grey, alpha=0.18, linewidth=0,
                          label='95% prediction interval')
        ax_a.plot(xg, mg, color=text_color, lw=1.2, label=fit_label)
    # Hand-tuned label offsets per region: the calibration cloud is dense
    # around w≈20-30 and its vertical arrangement shifts with the region's
    # slopes. Unlisted regions fall back to the EU offsets (the cluster's
    # relative arrangement is similar across regions; better than one
    # shared generic offset, which overprints the whole cluster).
    _lab_off = {
        'EU': {'CanESM5': (8, -9, 'left'), 'EC-Earth3': (-2, -14, 'right'),
               'CESM2': (7, 3, 'left'), 'IPSL-CM6A-LR': (2, -26, 'left'),
               'HadGEM3-GC3-1MM': (7, -4, 'left'), 'HadGEM3-GC3-1LL': (4, 9, 'left'),
               'MPI-ESM1-2-HR': (-5, 16, 'right'), 'MPI-ESM1-2-LR': (-9, -13, 'right'),
               'GISS-E2-1-G': (7, 3, 'left')},
        'IE': {'CanESM5': (8, -8, 'left'), 'EC-Earth3': (-2, -14, 'right'),
               'CESM2': (7, 3, 'left'), 'IPSL-CM6A-LR': (2, -26, 'left'),
               'HadGEM3-GC3-1MM': (9, -5, 'left'), 'HadGEM3-GC3-1LL': (4, 9, 'left'),
               'MPI-ESM1-2-HR': (-5, 16, 'right'), 'MPI-ESM1-2-LR': (-9, -13, 'right'),
               'GISS-E2-1-G': (7, 3, 'left')},
        'NEU': {'CanESM5': (-7, -4, 'right'), 'EC-Earth3': (-4, -15, 'right'),
                'CESM2': (7, 3, 'left'), 'IPSL-CM6A-LR': (-7, -13, 'right'),
                'HadGEM3-GC3-1MM': (7, -4, 'left'), 'HadGEM3-GC3-1LL': (2, 9, 'left'),
                'MPI-ESM1-2-HR': (7, -6, 'left'), 'MPI-ESM1-2-LR': (-9, -14, 'right'),
                'GISS-E2-1-G': (7, 3, 'left')},
    }
    _lab_off = _lab_off.get(region, _lab_off['EU'])
    # No SE whiskers on the calibration circles: the largest ±1 SE spans
    # 3-4% of the y-axis (about one marker diameter), invisible behind the
    # marker; the per-model ste stays in the sidecar.
    for m in res['models']:
        ax_a.scatter(cal[m]['w'], cal[m]['s'], color=cal[m]['color'],
                     marker=cal[m]['marker'], s=80, edgecolors='black',
                     linewidths=0.6, zorder=3)
        dx, dy, ha = _lab_off.get(m, (5, 4, 'left'))
        ax_a.annotate(m, xy=(cal[m]['w'], cal[m]['s']), xytext=(dx, dy),
                      textcoords='offset points', fontsize=8, color=cal[m]['color'],
                      ha=ha, zorder=5)
    # GISS has no PI slope, so it never enters res['models'] (fit-only
    # exclusion, _CAL_SPEC) — but it IS one of the warm observations behind
    # the pooled/interact state-dependence fit (state_dep_factors: 'GISS-E2-
    # 1-G (warm)' is in sdf['prop']['models'] under every cal_set the stack
    # uses). Draw it with the same warm-background 'X' marker as the three
    # PI/warm pairs so panel a shows every observation that actually informs
    # the state-dep line, not just the three that also have a PI slope.
    if state_dep in ('pooled', 'interact') and 'GISS-E2-1-G' in cal:
        ax_a.scatter(cal['GISS-E2-1-G']['w'], cal['GISS-E2-1-G']['s'],
                     color=cal['GISS-E2-1-G']['color'], marker='X', s=80,
                     edgecolors='black', linewidths=0.6, zorder=4)
        dx, dy, ha = _lab_off.get('GISS-E2-1-G', (5, 4, 'left'))
        ax_a.annotate('GISS-E2-1-G', xy=(cal['GISS-E2-1-G']['w'], cal['GISS-E2-1-G']['s']),
                      xytext=(dx, dy), textcoords='offset points', fontsize=8,
                      color=cal['GISS-E2-1-G']['color'], ha=ha, zorder=5)
    for i, m in enumerate(targets):
        mfc = 'none' if partial[m] else grey
        ax_a.errorbar(w_mean[m], s_mean[i], yerr=[[s_mean[i] - s_lo[i]], [s_hi[i] - s_mean[i]]],
                      ecolor=grey, elinewidth=1.4, capsize=2.5, zorder=2)
        ax_a.scatter(w_mean[m], s_mean[i], s=34, marker='D', facecolors=mfc,
                     edgecolors=grey, linewidths=1.0, zorder=3)
    ax_a.set_xlim(0, x_hi)
    # Fixed per-region y-limits so panel a is comparable across state_dep /
    # predictors / cal_set / origin variants of the same region; regions differ
    # by ~4x in |s|, so a single global range would squash MED. Set to cover the
    # calibration cloud, the target prediction intervals, the warm pair markers
    # and the fitted line; the 95% band clips, which is intended (it otherwise
    # drives the autoscale). Unlisted regions keep autoscale.
    y_lim = {'EU': (-0.15, 0.06), 'NEU': (-0.24, 0.09), 'WCE': (-0.14, 0.05),
             'MED': (-0.08, 0.02), 'DE': (-0.14, 0.04), 'IE': (-0.18, 0.03),
             'NO': (-0.31, 0.13)}.get(region)
    if y_lim:
        ax_a.set_ylim(*y_lim)
    ax_a.set_xlabel(rf'$\Delta$AMOC$_{{2090{{-}}2100}}$  [%, vs 1850–1899, SSP mean]')
    ax_a.set_ylabel(r'Cooling sensitivity  $\left[\frac{^\circ\mathrm{C}}{\%}\right]$')
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    # Proxy handles carrying what the old bottom footnote said: the two
    # error-bar semantics (±1 SE measured vs 95% PI predicted) and the
    # open-marker meaning (SSP-mean over incomplete scenario coverage).
    ax_a.scatter([], [], s=50, marker='o', facecolors='none',
                 edgecolors=text_color, linewidths=0.8,
                 label='preindustrial sensitivity')
    ax_a.scatter([], [], s=34, marker='D', facecolors=grey, edgecolors=grey,
                 linewidths=1.0, label='synthetic target (95% PI)')
    ax_a.scatter([], [], s=34, marker='D', facecolors='none', edgecolors=grey,
                 linewidths=1.0, label='target lacking ≥1 SSP (SSP-mean)')
    ax_a.legend(fontsize=9, loc='lower left', frameon=False)
    # Single caption identifying this row's calibration set + predictor spec
    # and the ACTUAL regression being fit and drawn (matching make_stacked_
    # figure's former per-row tag, now built here so standalone and stacked
    # renders show the same, correct thing — a separate sd_txt used to show a
    # different (state_dep-independent) PI-only R² than the one implied by
    # the state-dep-corrected line actually plotted, which read as an error).
    cal_label = _STACK_CAL_TXT.get(cal_set, cal_set)
    spec = (rf'$\beta\sim(\Delta$AMOC$,\,\Delta T_\mathrm{{{region}}})$'
            if predictors == 'wt' else r'$\beta\sim\Delta$AMOC')
    prefix = f"{cal_label},  {spec}"
    # state_dep != 'none' needs a second line: the merged prefix + fit clause
    # overflows the panel width on one line at fontsize 9 (runs into the
    # y-axis tick labels).
    if state_dep == 'pooled':
        p = sdf['prop']
        tag_txt = (f"{prefix}\n"
                  f"proportional fit (ρ={p['rho']:.2f}±{p['rho_se']:.2f}, "
                  f"n={p['n']}, {p['n_warm']} warm, R²={p['r2']:.2f})")
    elif state_dep == 'interact':
        p = sdf['prop']
        tag_txt = (f"{prefix}\n"
                  f"fully-interacted fit (n={p['n']}, {p['n_warm']} warm, "
                  f"R²={p['interact_r2']:.2f})")
    elif state_dep == 'dummy':
        tag_txt = (f"{prefix}  (n={len(res['models'])}, R²={fit.rsquared:.2f})\n"
                  "state-dep ρ (pairs): "
                  + ', '.join(f'{m} {v:.2f}' for m, v in sdf['rho'].items()))
    elif state_dep == 'loglin':
        tag_txt = (f"{prefix}  (n={len(res['models'])}, R²={fit.rsquared:.2f})\n"
                  f"loglin γ={sdf['gamma']:.3f}/K")
    else:
        tag_txt = f"{prefix}  (n={len(res['models'])}, R²={fit.rsquared:.2f})"
    ax_a.text(0.98, 0.97, tag_txt, transform=ax_a.transAxes, fontsize=9,
              color=text_color, ha='right', va='top')
    ax_a.text(0.0, 1.02, f'{panel_labels[0]} Synthetic cooling sensitivities'
              if panel_titles else panel_labels[0],
              transform=ax_a.transAxes, fontsize=14, fontweight='bold',
              color=text_color, ha='left', va='bottom')

    # Panel b: forest of w_net, one row per target, 3 offset SSP ranges.
    order = sorted(targets, key=lambda m: w_mean[m])[::-1]
    off = {ssp: dy for ssp, dy in zip(ssps, (0.26, 0.0, -0.26))}
    for yi, m in enumerate(order):
        y0 = len(order) - 1 - yi
        if yi % 2 == 0:
            ax_b.axhspan(y0 - 0.5, y0 + 0.5, color=grey, alpha=0.07, linewidth=0)
        for ssp in ssps:
            d = per_target[m]['ssp'].get(ssp)
            if d is None:
                continue
            c = functions.hosing_colors[ssp]['ge']
            y = y0 + off[ssp]
            lo, med, hi = d['q2.5'], d['median'], d['q97.5']
            if np.isfinite(lo) and lo <= 100:
                ax_b.plot([lo, min(hi, 100)], [y, y], color=c, lw=2.2,
                          solid_capstyle='butt', zorder=3)
                if med <= 100:
                    ax_b.plot(med, y, marker='o', ms=4.5, color=c,
                              mec='black', mew=0.4, zorder=4)
            ax_b.text(101.5, y, f"{d['frac_no_cooling']:.0%}", fontsize=7.5,
                      color=c, va='center', ha='left', clip_on=False)
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(-0.6, len(order) - 0.45)
    ax_b.set_yticks(range(len(order)))
    ax_b.set_yticklabels(order[::-1], fontsize=9)
    ax_b.set_xlabel(rf'{region} net-cooling AMOC weakening '
                    rf'$\Delta$AMOC$_\mathrm{{net{{-}}cooling}}$  [%, vs 1850–1899]')
    if panel_titles:
        ax_b.text(101.5, len(order) - 0.45, 'P(no net\ncooling)', fontsize=7.5,
                  color=text_color, va='bottom', ha='left', clip_on=False)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    handles = [plt.Line2D([], [], color=functions.hosing_colors[s]['ge'], lw=2.2,
                          label=sfc.SSP_TITLE[s]) for s in ssps]
    ax_b.legend(handles=handles, fontsize=9, loc='lower left', frameon=False)
    ax_b.text(0.0, 1.02, f'{panel_labels[1]} Net-cooling AMOC weakening'
              if panel_titles else panel_labels[1],
              transform=ax_b.transAxes, fontsize=14, fontweight='bold',
              color=text_color, ha='left', va='bottom')

    savepath = None
    if axes is None:
        fig.subplots_adjust(wspace=0.28)
        sdep_tag = f'_sdep-{_SDEP_TAG.get(state_dep, state_dep)}'
        sdep_tag += '_origin-zero' if origin else ''
        savepath = (f'{savedir}/FigSupp_synthetic_cooling_sensitivities'
                    f'_cal-{_CAL_TAG.get(cal_set, cal_set)}_pred-{predictors}{sdep_tag}'
                    f'_region-{region}'
                    f'_season-{season or "annual"}_window-{window}'
                    f'_base-{baseline.replace("hist_", "hist").replace("_", "-")}'
                    f'_plotbg-{plot_bg}')
        functions.check_savepath(savepath, exts=('.png', '.pdf', '_results.json'))
        fig.savefig(savepath + '.png', dpi=200, bbox_inches='tight',
                    transparent=plot_bg == 'black')
        fig.savefig(savepath + '.pdf', dpi=400, bbox_inches='tight',
                    transparent=plot_bg == 'black')
        with open(savepath + '_results.json', 'w') as f:
            json.dump(json_safe(sidecar), f, indent=1, allow_nan=False)

    # Text source for every number in the figure.
    v = sidecar['variants'][cal_set]
    print(f"\n=== s ~ {predictors} [{cal_set}] (n={int(fit.nobs)}) ===")
    print(f"params (const{', w, t' if predictors == 'wt' else ', w'}) = "
          f"{np.round(fit.params, 5).tolist()},  R²={fit.rsquared:.3f}, "
          f"resid std={np.sqrt(fit.mse_resid):.4f}")
    sk = v['warming_skill']
    print(f"warming skill: coef={sk['t_coef']:.5f} (p={sk['t_coef_p']:.3f}), "
          f"partial r(s,t|w)={sk['partial_r_st_given_w']:.3f}, "
          f"ΔR²={sk['delta_r2']:.3f}, F p={sk['f_p']:.3f}")
    print(f"LOO: {loo_inside}/{len(loo_rows)} true slopes inside their 95% PI")
    for m in order:
        e = per_target[m]
        parts = [f"{ssp}: med={d['median']:.0f}%, no-cool={d['frac_no_cooling']:.0%}"
                 for ssp, d in e['ssp'].items()]
        print(f"{m:16s} s={e['s_mean']:.3f} [{e['s_pi'][0]:.3f},{e['s_pi'][1]:.3f}]  "
              + '; '.join(parts))
    if sdf is not None:
        print(f"\n=== state-dep factors [{region}] (active: {state_dep}) ===")
        print('pair rho: ' + ', '.join(f'{m}={v:.3f}' for m, v in sdf['rho'].items())
              + f"  (mean {np.mean(list(sdf['rho'].values())):.3f})")
        p = sdf['prop']
        print(f"proportional fit: params={np.round(p['params'], 5).tolist()}, "
              f"rho={p['rho']:.3f} (se={p['rho_se']:.3f}), R²={p['r2']:.3f}, "
              f"n={p['n']} ({p['n_warm']} warm), dropped={p['dropped']}")
        print(f"  proportionality vs fully interacted: F={p['prop_test_F']:.2f}, "
              f"p={p['prop_test_p']:.3f} (interacted R²={p['interact_r2']:.3f})")
        print(f"loglin: gamma={sdf['gamma']:.4f}/K (dT_4x={sdf['g4x']:.2f} K regional, "
              f"rho_ECE={sdf['rho'].get('EC-Earth3', np.nan):.3f})")
        for sd, var in sd_block['variants'].items():
            parts = []
            for ssp in ssps:
                fr = [var[m][ssp]['frac_no_cooling'] for m in var if ssp in var[m]]
                parts.append(f"{ssp}: mean P(no-cool)={np.mean(fr):.0%}")
            print(f"variant {sd:7s} " + '; '.join(parts))
    if savepath is None:
        return fig, sidecar          # stacked use: caller owns save + sidecar
    print(f"Saved {savepath}")
    return fig, savepath


STACK_ROWS_DEFAULT = (('hosing', 'w'), ('hosing', 'wt'), ('nocesm2', 'w'))
_STACK_CAL_TXT = {'hosing': '8 NAHosMIP models', 'nocesm2': 'NAHosMIP minus CESM2',
                  'full': 'NAHosMIP + GISS', 'consistentssp': 'NAHosMIP, full-SSP only'}


def make_stacked_figure(wt_ds=None, *, rows=STACK_ROWS_DEFAULT, region='EU',
                        season='', window=10, baseline='hist_1850_1899',
                        state_dep='none',
                        n_draws=20000, seed=0, plot_bg='white', savedir='../plots',
                        **data):
    """Vertical stack of (cal_set, predictors) versions of make_figure, all
    at the same state_dep (default 'none'): full a+b row per version,
    x-labels only on the bottom row, legends and column headers only on the
    top; each row's own caption (cal_set, spec, actual fitted model + R²) is
    drawn by make_figure itself. `data` carries make_figure's data kwargs
    (state_dep != 'none' additionally needs reg_ds_mpi/reg_ds_cesm in there,
    as in make_figure itself)."""
    assert state_dep in STATE_DEPS
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    if wt_ds is None or wt_ds.attrs.get('region', '').split(' ')[0] != region:
        wt_ds = get_target_wt(recompute=False, region=region)
    n_t = wt_ds.sizes['model']
    fig, axes = plt.subplots(
        len(rows), 2, figsize=(16, len(rows) * (0.30 * n_t + 0.9) + 0.6),
        gridspec_kw={'width_ratios': [1, 1.35], 'wspace': 0.28, 'hspace': 0.15})

    sidecar = {'region': region, 'rows': []}
    for i, (cs, pred) in enumerate(rows):
        _, sc = make_figure(wt_ds=wt_ds, region=region, season=season,
                            window=window, baseline=baseline, cal_set=cs,
                            predictors=pred, state_dep=state_dep,
                            n_draws=n_draws, seed=seed, plot_bg=plot_bg,
                            fig=fig, axes=axes[i], panel_titles=(i == 0),
                            panel_labels=('abcdef'[2 * i], 'abcdef'[2 * i + 1]),
                            **data)
        # make_figure now draws its own row caption (cal_set, spec, and the
        # actual fitted model's n/R²) at this same anchor, so nothing extra
        # is drawn here.
        if i < len(rows) - 1:
            axes[i, 0].set_xlabel('')
            axes[i, 1].set_xlabel('')
        if i > 0:
            for ax in axes[i]:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
        sidecar['rows'].append({'cal_set': cs, 'predictors': pred, 'sidecar': sc})

    rows_tag = ('' if tuple(rows) == STACK_ROWS_DEFAULT else
                '_rows-' + '-'.join(f'{_CAL_TAG.get(c, c)}{p}' for c, p in rows))
    sdep_tag = '' if state_dep == 'none' else f'_sdep-{_SDEP_TAG.get(state_dep, state_dep)}'
    savepath = (f'{savedir}/FigSupp_synthetic_cooling_sensitivities_stack{rows_tag}'
                f'_region-{region}'
                f'_season-{season or "annual"}_window-{window}'
                f'_base-{baseline.replace("hist_", "hist").replace("_", "-")}'
                f'{sdep_tag}_plotbg-{plot_bg}')
    functions.check_savepath(savepath, exts=('.png', '.pdf', '_results.json'))
    fig.savefig(savepath + '.png', dpi=200, bbox_inches='tight',
                transparent=plot_bg == 'black')
    fig.savefig(savepath + '.pdf', dpi=400, bbox_inches='tight',
                transparent=plot_bg == 'black')
    with open(savepath + '_results.json', 'w') as f:
        json.dump(json_safe(sidecar), f, indent=1, allow_nan=False)
    print(f"Saved {savepath}")
    return fig, savepath


########################################
# %%
# LOAD DATA

if __name__ == '__main__':
    multi_model_dict, masks = functions.get_full_multi_model_dict()
    hosmip_reg_ds_dict = functions.get_hosmip_reg_ds(recompute=False)
    reg_ds_giss = functions.get_giss_reg_ds(recompute=False)
    # Season-keyed dict, mirroring produce_all_paper_figures: a flat annual
    # panel here would silently feed an annual GISS slope into a seasonal
    # build_table (QA finding N2, fixed 2026-08-31).
    reg_ds_giss_panel = {
        s: functions.get_giss_panel_data(masks=masks, season=s, recompute=False)
        for s in ['', 'djf', 'jja']
    }
    gwl_data = functions.get_gwl_diagnostic_data(recompute=False)
    reg_ds_mpi = functions.load_regression_ds_mpi()
    reg_ds_cesm = functions.get_cesm_reg_ds(recompute=False)


########################################
# %%
# RUN

if __name__ == '__main__':
    region = 'IE'            # any key of the land-aware masks (EU, NEU, WCE, MED, countries)
    season = ''              # '' (annual) | 'djf' | 'jja'
    window = 10
    baseline = 'hist_1850_1899'
    cal_set = 'hosing'       # 'hosing' (default)|'full'|'nocesm2'|'consistentssp'
    predictors = 'wt'         # 'w' | 'wt' (weakening + warming)
    state_dep = 'pooled'       # 'none'|'dummy'|'pooled'|'loglin'|'interact'
    origin = False           # True = cross-model line through the origin (sensitivity)
    n_draws = 20000
    seed = 0
    plot_bg = 'white'        # 'white' | 'black'

    wt_ds = get_target_wt(recompute=False, region=region)
    fig, savepath = make_figure(
        wt_ds=wt_ds,
        multi_model_dict=multi_model_dict, hosmip_reg_ds_dict=hosmip_reg_ds_dict,
        reg_ds_giss=reg_ds_giss, reg_ds_giss_panel=reg_ds_giss_panel,
        masks=masks, gwl_data=gwl_data,
        region=region, season=season, window=window, baseline=baseline,
        cal_set=cal_set, predictors=predictors,
        state_dep=state_dep, reg_ds_mpi=reg_ds_mpi, reg_ds_cesm=reg_ds_cesm,
        origin=origin, n_draws=n_draws, seed=seed, plot_bg=plot_bg,
    )


########################################
# %%
# BUILD RANGE CACHE
# Rebuild every Synthetic CMIP6 range cache consumed by
# produce_all_paper_figures (12 annual statedep x calset + 4 seasonal at the
# default calibration set). Off on a plain script run; enable interactively
# or via BUILD_RANGE_CACHES=1 after a schema/membership change.

if __name__ == '__main__':
    import os as _os
    BUILD_RANGE_CACHES = _os.environ.get('BUILD_RANGE_CACHES') == '1'
    if BUILD_RANGE_CACHES:
        _build = dict(multi_model_dict=multi_model_dict,
                      hosmip_reg_ds_dict=hosmip_reg_ds_dict,
                      reg_ds_giss=reg_ds_giss, masks=masks, gwl_data=gwl_data,
                      reg_ds_mpi=reg_ds_mpi, reg_ds_cesm=reg_ds_cesm,
                      processes=int(_os.environ.get('RANGE_PROCS', '1')))
        for _sd in ('none', 'pooled', 'interact'):
            for _cs in _CAL_SETS:
                get_cmip_range_ds(recompute=True, state_dep=_sd, cal_set=_cs,
                                  **_build)
        for _sd in ('none', 'pooled'):
            for _se in ('djf', 'jja'):
                get_cmip_range_ds(recompute=True, state_dep=_sd,
                                  cal_set=DEFAULT_CAL_SET, season=_se, **_build)

# %%
