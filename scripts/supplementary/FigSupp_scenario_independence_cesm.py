########################################
# %%
# SCENARIO-INDEPENDENCE DIAGNOSTIC (CESM2 ssp126 vs ssp585) — 2×2 grid
#
# CESM2 analogue of FigSupp_scenario_independence.py. Tests whether
# ssp126 and ssp585 are statistically distinguishable in the pooled
# AMOC→T regression. With only G=2 scenarios the joint Wald-F reduces
# to a 1-df test on the single interaction term, and covariance is
# HAC (maxlags=window-1) rather than cluster-robust (cluster-robust
# asymptotics degenerate at G=2).
#
# Layout (2×2):
#   a) partial R²_indep map  (per-SSP slopes' share of pooled-fit residuals)
#   b) ΔR²_total_indep map   (per-SSP slopes' share of total variance)
#   c) Region-mean pooled regression scatter   (default: EU)
#   d) Most-rejecting country regression scatter (selected at render time)
#
# Maps stippled at Wald p<0.05.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.rcParams.update({'font.size': 12})

import importlib
import sys, pathlib
if "__file__" in globals():  # standalone run: ensure scripts/ is importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import functions
importlib.reload(functions)

########################################
# %%
# LOAD DATA

if __name__ == '__main__':
    multi_model_dict, masks = functions.get_full_multi_model_dict()
    _, _, boot_data, _ = functions.get_other_studies_data(masks)
    si_ds = functions.get_scenario_independence_diagnostics_cesm(var='tas')

########################################
# %%
# HELPERS

# Hardcoded most-rejecting country per season for var='tas', alpha=0.05
# (CESM2 pooled ssp126+ssp585). Update by passing recompute=True.
_MOST_REJECTING_COUNTRY_BY_SEASON_CESM = {
    '': None,  # populated on first recompute=True call
    'djf': None,
    'jja': None,
}

_country_cache_cesm = {}


def find_most_rejecting_country_cesm(boot_data, multi_model_dict, masks,
                                     var='tas', season='', alpha=0.05,
                                     fallback_to_pixel=True, si_ds=None,
                                     verbose=False, recompute=False):
    """Return the most-rejecting European country for CESM2 Wald-F.

    Scans functions.country_codes by default (no useful hardcoded cache
    until the figure has been generated once); pass recompute=False once
    the dict above is populated to skip the scan.
    """
    if not recompute and var == 'tas' and alpha == 0.05 \
            and _MOST_REJECTING_COUNTRY_BY_SEASON_CESM.get(season) is not None:
        code = _MOST_REJECTING_COUNTRY_BY_SEASON_CESM[season]
        return {'kind': 'country', 'label': code,
                'wald_pvalue': np.nan, 'extra': None}

    cache_key = (var, season, alpha)
    if cache_key in _country_cache_cesm:
        return _country_cache_cesm[cache_key]

    results = []
    for code in functions.country_codes.keys():
        try:
            functions.cesm_regressions.last_call = {}
            functions.cesm_regressions(
                boot_data, multi_model_dict, masks,
                ssp='all', region=code,
                combined_reg=True, add_combined_intercept=True,
                no_plots=True, season=season,
                compute_scenario_independence=True,
            )
            lc = functions.cesm_regressions.last_call
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
        _country_cache_cesm[cache_key] = result
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
            _country_cache_cesm[cache_key] = result
            return result

    if results:
        c, p = min(results, key=lambda x: x[1])
        result = {'kind': 'country', 'label': c, 'wald_pvalue': p, 'extra': None}
        _country_cache_cesm[cache_key] = result
        return result
    result = {'kind': 'none', 'label': 'n/a', 'wald_pvalue': np.nan, 'extra': None}
    _country_cache_cesm[cache_key] = result
    return result


def _annotate_wald(ax, lc, fg, bg):
    if 'wald_pvalue' not in lc or not np.isfinite(lc.get('wald_pvalue', np.nan)):
        return
    annot = (
        f'Wald-F p          = {lc["wald_pvalue"]:.3f}\n'
        f'partial R²_indep  = {lc["partial_r2_indep"]:.4f}\n'
        f'ΔR²_total_indep   = {lc["delta_r2_total_indep"]:.5f}\n'
        f'max |Δβ|          = {lc["max_pairwise_slope_diff"]:.3f} K/% wkn'
    )
    ax.text(
        0.97, 0.23, annot,
        transform=ax.transAxes, fontsize=8,
        va='bottom', ha='right', ma='left', family='monospace',
        bbox=dict(facecolor=bg, edgecolor=fg, alpha=0.85,
                  boxstyle='round,pad=0.4'),
    )


########################################
# %%
# FIGURE FUNCTION


def make_figure(boot_data, multi_model_dict, masks, si_ds,
                var='tas', region='EU', season='', plot_bg='white',
                verbose_country_scan=False, recompute_most_rejecting=True):
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

    _draw_map(ax_a, partial_r2, vmin=0.0, vmax=0.5, n_bins=5,
              label="partial R² (ssp585 vs ssp126 slope diff, share of residuals)")
    _draw_map(ax_b, delta_r2, vmin=0.0, vmax=0.25, n_bins=5,
              label="ΔR²_total (slope diff, share of total variance)")

    season_suffix = f' ({season.upper()})' if season else ''
    ax_a.set_title(f'a) CESM2 scenario contribution (residuals){season_suffix}',
                   fontsize=12, loc='left')
    ax_b.set_title(f'b) CESM2 scenario contribution (total variance){season_suffix}',
                   fontsize=12, loc='left')

    fig.text(
        0.5, 0.91,
        'Stippling: 1-df Wald-F (HAC SE) rejects scenario-independence at 5%; '
        'CESM2 ssp126 vs ssp585.',
        fontsize=10, ha='center', style='italic', color=fg,
    )

    # ------------------------------------------------------------------
    # BOTTOM-LEFT (c): region-mean pooled regression
    # ------------------------------------------------------------------
    functions.cesm_regressions(
        boot_data, multi_model_dict, masks,
        ssp='all', region=region,
        combined_reg=True, add_combined_intercept=True,
        no_plots=False, ext_ax=ax_c, plot_bg=plot_bg, season=season,
        compute_scenario_independence=True, xlim=50, low_ylim=-8,
        equation_y_spacing=0.065,  # fraction-tall simple_eqs; stays below the Wald box
    )
    lc_eu = functions.cesm_regressions.last_call
    _annotate_wald(ax_c, lc_eu, fg, bg)
    ax_c.set_title(f'c) {region}-mean (CESM2 pooled){season_suffix}',
                   fontsize=12, loc='left')

    # ------------------------------------------------------------------
    # BOTTOM-RIGHT (d): most-rejecting country regression
    # ------------------------------------------------------------------
    if recompute_most_rejecting:
        print(f'  Scanning for most-rejecting country (CESM2, season={season!r})...')
    sel = find_most_rejecting_country_cesm(
        boot_data, multi_model_dict, masks, var=var, season=season, alpha=0.05,
        fallback_to_pixel=True, si_ds=si_ds,
        verbose=verbose_country_scan, recompute=recompute_most_rejecting,
    )
    if np.isfinite(sel['wald_pvalue']):
        print(f'  -> {sel["kind"]}: {sel["label"]}, Wald-p = {sel["wald_pvalue"]:.4f}')
    else:
        print(f'  -> {sel["kind"]}: {sel["label"]} (hardcoded)')

    if sel['kind'] == 'country':
        functions.cesm_regressions(
            boot_data, multi_model_dict, masks,
            ssp='all', region=sel['label'],
            combined_reg=True, add_combined_intercept=True,
            no_plots=False, ext_ax=ax_d, plot_bg=plot_bg, season=season,
            compute_scenario_independence=True, xlim=50, low_ylim=-8,
            equation_y_spacing=0.065,  # fraction-tall simple_eqs; stays below the Wald box
        )
        title_tag = sel['label']
    elif sel['kind'] == 'pixel':
        lat_i, lon_i = sel['extra']
        functions.cesm_regressions(
            boot_data, multi_model_dict, masks,
            ssp='all', lat=lat_i, lon=lon_i, region=None,
            combined_reg=True, add_combined_intercept=True,
            no_plots=False, ext_ax=ax_d, plot_bg=plot_bg, season=season,
            compute_scenario_independence=True,
            equation_y_spacing=0.065,  # fraction-tall simple_eqs; stays below the Wald box
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
        lc_rej = functions.cesm_regressions.last_call
        _annotate_wald(ax_d, lc_rej, fg, bg)
    ax_d.set_title(f'd) {title_tag} (most-rejecting case){season_suffix}',
                   fontsize=12, loc='left')

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    savepath = (f"../plots/FigSupp_cesm_scenario_independence_var-{var}_region-{region}"
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
    fig, savepath = make_figure(boot_data, multi_model_dict, masks, si_ds,
                                var='tas', region='EU', season='',
                                plot_bg='white')
    print(f'Saved {savepath}.(pdf|png)')

# %%
