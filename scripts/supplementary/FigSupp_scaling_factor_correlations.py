########################################
# %%
# LOAD PACKAGES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
plt.rcParams.update({'font.size': 12})

import importlib
import sys, pathlib
if "__file__" in globals():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import functions
importlib.reload(functions)

########################################
# %%
# COOLING-SENSITIVITY CORRELATIONS (supplementary)
#
# Per model: EU-mean temperature sensitivity to AMOC weakening (y, °C per %
# weakening from PI) against projected end-of-century AMOC weakening (x).
# Four panels: SSP1-2.6, SSP2-4.5, SSP3-7.0, mean across the three SSPs.
# Models: 8 HosMIP (regression slope) + GISS-E2-1-G (HAC OLS on member
# deviations) + optional CCSM4 (Liu 2020) and CESM1 (van Westen-Baatsen)
# read off Meehl et al. 2013 (J. Climate, 10.1175/JCLI-D-12-00572.1).

########################################
# %%
# MEEHL 2013 PROJECTED WEAKENING (RCP -> SSP)

MEEHL_2013_WEAKENING = {
    'CCSM4': {'ssp126': 1.5 / 25  * 100, 'ssp245': 4.0  / 25  * 100, 'ssp370': 9.0  / 25  * 100},
    'CESM1': {'ssp126': 4.5 / 24.1 * 100, 'ssp245': 8.0  / 24.1 * 100, 'ssp370': 10.5 / 24.1 * 100},
}

EXTRA_COLORS = {
    'GISS-E2-1-G': functions.hosing_colors['ssp245']['ge'],
    'CCSM4': 'goldenrod',
    'CESM1': 'darkorange',
}

EXTRA_MARKERS = {'GISS-E2-1-G': 's', 'CCSM4': '^', 'CESM1': '^'}

########################################
# %%
# LOAD DATA

if __name__ == '__main__':
    multi_model_dict, masks = functions.get_full_multi_model_dict()
    hosmip_reg_ds_dict = functions.get_hosmip_reg_ds(recompute=False)
    reg_ds_giss = functions.get_giss_reg_ds(recompute=False)
    reg_ds_giss_panel = functions.get_giss_panel_data(masks=masks, recompute=False)
    gwl_data = functions.get_gwl_diagnostic_data(recompute=False)


########################################
# %%
# HELPERS

def eu_mean(da, mask_region, mask_land):
    """Area-weighted mean over (region ∩ ocean-masked-out) on a (lat, lon) DataArray."""
    return float(functions.weighted_area_lat(
        da.where(mask_region).where(mask_land == 0)
    ).mean('lat').mean('lon').item())


def _hosmip_masks(multi_model_dict, model, region):
    m = multi_model_dict[model].mask
    return m.sel(region=region), m.sel(region='LAND')


def _giss_masks(masks, region):
    gm = masks['GISS-E2-1-G']
    return gm[region], gm['LAND']


def baseline_amoc(model, hosmip_reg_ds_dict, reg_ds_giss, baseline, gwl_data):
    """AMOC denominator for % weakening. Since the 2026-05-25 harmonisation,
    reg_ds AMOC_pi IS the hist 1850–1899 ensemble mean, so both `baseline`
    options resolve to the same single source (verified equal to the gwl_data
    hist mean to float precision, 2026-08-15). The `baseline` axis is
    therefore vestigial (kept for filename stability); gwl_data stays in the
    signature but is no longer consulted — that was a cache-drift channel."""
    ds = reg_ds_giss if model == 'GISS-E2-1-G' else hosmip_reg_ds_dict[model]
    return float(ds.AMOC_pi.item())


def _hosmip_slope(multi_model_dict, hosmip_reg_ds_dict, model, region, season, window):
    """Fig3-canonical per-model slope (°C / %) via hosmip_regression_plot."""
    res = functions.hosmip_regression_plot(
        multi_model_dict, region=region, season=season, window=window,
        no_plots=True, only_model=model, single_model_linear=True,
        single_model_intercept=False, hosmip_ref_pi=True,
        hosmip_reg_ds_dict=hosmip_reg_ds_dict, central_reg=None, ax=None,
    )
    return res['model_coef_lin'], res['model_ste_lin']


def _giss_slope(panel, region, window, giss_time_period):
    """HAC OLS (no intercept) on GISS member-minus-composite deviations."""
    if giss_time_period is not None:
        y0, y1 = (int(s) for s in giss_time_period.split('-'))
        panel = panel.where((panel.year >= y0) & (panel.year <= y1), drop=True)
    x = panel.amoc_dev_pct.values
    y = panel.t_dev.sel(region=region).values
    keep = np.isfinite(x) & np.isfinite(y)
    fit = sm.OLS(y[keep], x[keep]).fit(
        cov_type='HAC', cov_kwds={'maxlags': max(window - 1, 1)})
    return float(fit.params[0]), float(fit.bse[0])


def _liu_ccsm4(multi_model_dict, region):
    """CCSM4 (Liu 2020) slope + warming proxy from the hosing/control pair under RCP8.5.
    Sign convention follows Fig3a (`functions.py:5830`): ΔT = T(control,ghg) − T(hosing,ghg)."""
    ds = multi_model_dict['CCSM4']
    mreg, mland = ds.mask.sel(region=region), ds.mask.sel(region='LAND')
    tas_diff = (ds.tas.sel(type='control', scenar='ghg', season='') -
                ds.tas.sel(type='hosing',  scenar='ghg', season=''))
    weakening = (ds.amoc.sel(type='control', scenar='ghg', season='') -
                 ds.amoc.sel(type='hosing',  scenar='ghg', season='')) / \
                ds.amoc.sel(type='control', scenar='pi', season='') * 100
    warming = (ds.tas.sel(type='control', scenar='ghg', season='') -
               ds.tas.sel(type='control', scenar='his', season=''))
    return (eu_mean(tas_diff, mreg, mland) / float(weakening.item()),
            eu_mean(warming, mreg, mland))


def _vwb_cesm1(multi_model_dict, region):
    """CESM1 (van Westen-Baatsen) slope + warming proxy from the double-difference pair.

    Annual, so all four corners are measured (the CESM_0600_PI field comes from
    the hydroclimate record; only DJF still estimates it).
    Only reachable via include_ccsm4_cesm1=True."""
    ds = multi_model_dict['CESM1']
    mreg, mland = ds.mask.sel(region=region), ds.mask.sel(region='LAND')
    tas_diff = ((ds.tas.sel(type='hosing',  scenar='ghg', season='') -
                 ds.tas.sel(type='hosing',  scenar='pi',  season='')) -
                (ds.tas.sel(type='control', scenar='ghg', season='') -
                 ds.tas.sel(type='control', scenar='pi',  season='')))
    amoc_diff = ((ds.amoc.sel(type='control', scenar='ghg', season='') -
                  ds.amoc.sel(type='control', scenar='pi',  season='')) -
                 (ds.amoc.sel(type='hosing',  scenar='ghg', season='') -
                  ds.amoc.sel(type='hosing',  scenar='pi',  season=''))) / \
                ds.amoc.sel(type='control', scenar='pi', season='') * 100
    warming = (ds.tas.sel(type='control', scenar='ghg', season='') -
               ds.tas.sel(type='control', scenar='pi',  season=''))
    return (eu_mean(tas_diff, mreg, mland) / float(amoc_diff.item()),
            eu_mean(warming, mreg, mland))


########################################
# %%
# BUILD TABLE

def build_table(multi_model_dict, hosmip_reg_ds_dict,
                reg_ds_giss, reg_ds_giss_panel, masks, gwl_data,
                region='EU', season='', window=10,
                giss_time_period='2101-2300',
                ssps=('ssp126', 'ssp245', 'ssp370'),
                include_giss=True, include_ccsm4_cesm1=False,
                baseline='hist_1850_1899'):
    """Return dict[model] = {'slope', 'ste', 'weakening':{ssp:%}, 'warming':{ssp:°C}, 'color', 'marker'}."""
    table = {}
    for model in functions.hosmip_labels:
        slope, ste = _hosmip_slope(multi_model_dict, hosmip_reg_ds_dict,
                                   model, region, season, window)
        rd = hosmip_reg_ds_dict[model]
        amoc_base = baseline_amoc(model, hosmip_reg_ds_dict, reg_ds_giss, baseline, gwl_data)
        mreg, mland = _hosmip_masks(multi_model_dict, model, region)
        wk, wm = {}, {}
        for ssp in ssps:
            wk[ssp] = (amoc_base - float(rd.AMOC_future.sel(scenar=ssp).item())) / amoc_base * 100
            wm[ssp] = eu_mean(rd.T_future.sel(scenar=ssp, season=season)
                               - rd.T_pi.sel(season=season), mreg, mland)
        table[model] = {'slope': slope, 'ste': ste, 'weakening': wk, 'warming': wm,
                        'color': functions.hosmip_colors[model], 'marker': 'o'}

    if include_giss:
        # reg_ds_giss_panel may be a season-keyed dict (pipeline convention) or a flat Dataset.
        panel = reg_ds_giss_panel.get(season) if isinstance(reg_ds_giss_panel, dict) else reg_ds_giss_panel
        slope, ste = _giss_slope(panel, region, window, giss_time_period)
        amoc_base = baseline_amoc('GISS-E2-1-G', hosmip_reg_ds_dict, reg_ds_giss, baseline, gwl_data)
        mreg, mland = _giss_masks(masks, region)
        wk, wm = {}, {}
        for ssp in ssps:
            wk[ssp] = (amoc_base - float(reg_ds_giss.AMOC_future.sel(scenar=ssp).item())) / amoc_base * 100
            wm[ssp] = eu_mean(reg_ds_giss.T_future.sel(scenar=ssp, season=season)
                               - reg_ds_giss.T_pi.sel(season=season), mreg, mland)
        table['GISS-E2-1-G'] = {'slope': slope, 'ste': ste, 'weakening': wk, 'warming': wm,
                                'color': EXTRA_COLORS['GISS-E2-1-G'], 'marker': EXTRA_MARKERS['GISS-E2-1-G']}

    if include_ccsm4_cesm1:
        ccsm4_slope, ccsm4_warming = _liu_ccsm4(multi_model_dict, region)
        cesm1_slope, cesm1_warming = _vwb_cesm1(multi_model_dict, region)
        # Warming proxy fills only the SSP panel matching each paper's available RCP.
        table['CCSM4'] = {'slope': ccsm4_slope, 'ste': np.nan,
                         'weakening': dict(MEEHL_2013_WEAKENING['CCSM4']),
                         'warming':   {'ssp126': np.nan, 'ssp245': np.nan, 'ssp370': ccsm4_warming},
                         'color': EXTRA_COLORS['CCSM4'], 'marker': EXTRA_MARKERS['CCSM4']}
        table['CESM1'] = {'slope': cesm1_slope, 'ste': np.nan,
                         'weakening': dict(MEEHL_2013_WEAKENING['CESM1']),
                         'warming':   {'ssp126': np.nan, 'ssp245': cesm1_warming, 'ssp370': np.nan},
                         'color': EXTRA_COLORS['CESM1'], 'marker': EXTRA_MARKERS['CESM1']}

    return table


########################################
# %%
# FIGURE FUNCTION

_X_AXIS_OPTIONS = ('weakening', 'warming', 'warming_over_weakening', 'combined')
# Both baselines resolve to the hist 1850–1899 mean since the 2026-05-25
# harmonisation (the old 'piControl' label was stale — reg_ds AMOC_pi is no
# longer a piControl snapshot).
_BASE_LABEL     = {'pi': '1850–1899', 'hist_1850_1899': '1850–1899'}
SSP_TITLE      = {'ssp126': 'SSP1-2.6', 'ssp245': 'SSP2-4.5', 'ssp370': 'SSP3-7.0'}


def model_xy(info, panel, ssps, x_axis):
    """Return (x, y) for one model on one panel. `panel` is an SSP key or 'mean'.
    For x_axis in {'weakening','warming'} y is the model-level regression slope
    (constant across panels). For x_axis == 'warming_over_weakening' (literally
    "warming over weakening": y = projected warming, x = projected weakening),
    both coordinates vary per panel."""
    def _pick(d):
        if panel == 'mean':
            vs = [d.get(s, np.nan) for s in ssps]
            vs = [v for v in vs if np.isfinite(v)]
            return float(np.mean(vs)) if vs else np.nan
        return d.get(panel, np.nan)
    if x_axis == 'weakening':
        return _pick(info['weakening']), info['slope']
    if x_axis == 'warming':
        return _pick(info['warming']),   info['slope']
    return _pick(info['weakening']), _pick(info['warming'])


def make_figure(table=None, *, multi_model_dict=None, hosmip_reg_ds_dict=None,
                reg_ds_giss=None, reg_ds_giss_panel=None, masks=None, gwl_data=None,
                season='', window=10,
                giss_time_period='2101-2300',
                ssps=('ssp126', 'ssp245', 'ssp370'),
                region='EU',
                include_giss=True,
                include_ccsm4_cesm1=False,
                x_axis='weakening',
                baseline='hist_1850_1899',
                plot_bg='white',
                savedir='../plots'):
    assert x_axis in _X_AXIS_OPTIONS, x_axis
    assert baseline in _BASE_LABEL, baseline

    if table is None:
        table = build_table(multi_model_dict, hosmip_reg_ds_dict,
                            reg_ds_giss, reg_ds_giss_panel, masks, gwl_data,
                            region=region, season=season, window=window,
                            giss_time_period=giss_time_period, ssps=ssps,
                            include_giss=include_giss,
                            include_ccsm4_cesm1=include_ccsm4_cesm1,
                            baseline=baseline)

    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'
    text_color = 'black' if plot_bg != 'black' else 'white'
    grey = '0.45' if plot_bg != 'black' else '0.65'

    frac_unit  = r'$\left[\frac{^\circ\mathrm{C}}{\%}\right]$'
    x_wk_label = rf'$\Delta$AMOC$_{{2090{{-}}2100}}$  [%, vs {_BASE_LABEL[baseline]}]'
    x_wm_label = rf'$\Delta T_\mathrm{{{region},\,2090{{-}}2100}}$  [$^\circ$C, vs {_BASE_LABEL[baseline]}]'
    y_sf_label = rf'Cooling sensitivity  {frac_unit}'

    x_label = {
        'weakening':              x_wk_label,
        'warming':                x_wm_label,
        'warming_over_weakening': x_wk_label,
        'combined':               None,         # set per column below
    }[x_axis]
    y_label = x_wm_label if x_axis == 'warming_over_weakening' else y_sf_label

    panels = list(ssps) + ['mean']

    def _draw_fit(ax, panel, x_mode):
        """OLS line + 95% CI band + stats box over the 8 NAHosMIP models
        (GISS and the literature add-ons are shown but stay out of the line,
        matching the canonical hosing calibration fit). Prints the stats so
        every quoted number has a text source."""
        pts = [model_xy(table[m], panel, ssps, x_mode)
               for m in functions.hosmip_labels if m in table]
        pts = [(x, y) for x, y in pts if np.isfinite(x) and np.isfinite(y)]
        if len(pts) < 3:
            return
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        fit = sm.OLS(y, sm.add_constant(x)).fit()
        xg = np.linspace(0, x.max() * 1.1, 100)
        sf = fit.get_prediction(sm.add_constant(xg)).summary_frame(alpha=0.05)
        ax.fill_between(xg, sf['mean_ci_lower'], sf['mean_ci_upper'],
                        color=grey, alpha=0.15, linewidth=0, zorder=1)
        ax.plot(xg, sf['mean'], color=grey, lw=1.2, zorder=1)
        ax.text(0.97, 0.97,
                f"slope = {fit.params[1]:.4f} ± {fit.bse[1]:.4f}\n"
                f"R² = {fit.rsquared:.2f}, p = {fit.pvalues[1]:.3f}, n = {int(fit.nobs)}",
                transform=ax.transAxes, fontsize=8, color=text_color,
                ha='right', va='top')
        print(f"[fit {x_mode} @ {panel}] slope={fit.params[1]:.5f} "
              f"se={fit.bse[1]:.5f} R2={fit.rsquared:.3f} "
              f"p={fit.pvalues[1]:.4f} n={int(fit.nobs)} "
              f"intercept={fit.params[0]:.5f}")

    def _render_cell(ax, panel, x_mode):
        """Scatter all models on one cell using `model_xy(..., x_mode)`. No spine / title work."""
        for model, info in table.items():
            x, y = model_xy(info, panel, ssps, x_mode)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            ax.scatter(x, y, color=info['color'], marker=info['marker'],
                       s=80, edgecolors='black', linewidths=0.6, zorder=3, clip_on=False)
            if x_mode != 'warming_over_weakening' and np.isfinite(info.get('ste', np.nan)):
                ax.errorbar(x, y, yerr=info['ste'], ecolor=info['color'],
                            elinewidth=0.8, capsize=0, alpha=0.6, zorder=2)
            label = f'{model} (RCP8.5)' if panel == 'ssp370' and model in ('CCSM4', 'CESM1') else model
            ax.annotate(label, xy=(x, y), xytext=(4, 4),
                        textcoords='offset points', fontsize=8, color=info['color'])
        _draw_fit(ax, panel, x_mode)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _ssp_title(ax, panel, prefix=''):
        title = SSP_TITLE.get(panel, f'Mean across {", ".join(SSP_TITLE[s] for s in ssps)}')
        title_color = functions.hosing_colors[panel]['ge'] if panel in functions.hosing_colors else text_color
        ax.text(0.0, 1.03, f'{prefix}{title}', transform=ax.transAxes,
                fontsize=14, fontweight='bold', color=title_color, ha='left', va='bottom')

    def _extents_x(x_mode, panel=None):
        """xlim across all models. If `panel` is given, restrict to that panel; otherwise
        union across all SSPs (used for the legacy 2×2 single-x figure)."""
        xs = []
        for info in table.values():
            it = ssps if panel is None else [panel]
            for ssp in it:
                x, _ = model_xy(info, ssp, ssps, x_mode)
                if np.isfinite(x): xs.append(x)
        return (0.0, max(xs) * 1.1) if xs else (0.0, 1.0)

    if x_axis == 'combined':
        # 2 rows × 4 cols. Cols are SSPs (ssp126, ssp245, ssp370, mean).
        # Row 0 = (a) Warming setup   (y = slope, x = ΔT, SSP titles above each cell).
        # Row 1 = (b) Weakening setup (y = slope, x = ΔAMOC, SSP comes from column alignment).
        # Per-row consistent xlim; per-cell x label; shared y across the figure.
        fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey='all')
        ys = [info['slope'] for info in table.values() if np.isfinite(info['slope'])]
        y_lo, y_hi = (min(min(ys) * 1.15, -0.005), max(0.0, max(ys) * 1.05)) if ys else (-0.25, 0.0)
        axes[0, 0].set_ylim(y_lo, y_hi)
        x_lo_wm, x_hi_wm = _extents_x('warming')
        x_lo_wk, x_hi_wk = _extents_x('weakening')

        for col, panel in enumerate(panels):
            _render_cell(axes[0, col], panel, 'warming')
            _ssp_title(axes[0, col], panel)
            axes[0, col].set_xlim(x_lo_wm, x_hi_wm)
            axes[0, col].set_xlabel(x_wm_label)

            _render_cell(axes[1, col], panel, 'weakening')
            axes[1, col].set_xlim(x_lo_wk, x_hi_wk)
            axes[1, col].set_xlabel(x_wk_label)

        # a) / b) at the top-left of each row, slightly above the SSP title row.
        axes[0, 0].text(0.0, 1.18, 'a) Cooling sensitivities over projected warming',   transform=axes[0, 0].transAxes,
                        fontsize=15, fontweight='bold', color=text_color, ha='left', va='bottom')
        axes[1, 0].text(0.0, 1.03, 'b) Cooling sensitivities over projected weakening', transform=axes[1, 0].transAxes,
                        fontsize=15, fontweight='bold', color=text_color, ha='left', va='bottom')

        for r in range(2):
            axes[r, 0].set_ylabel(y_sf_label)

        fig.subplots_adjust(wspace=0.08, hspace=0.35)
        fig.text(0.5, 0.02, 'Grey line: OLS over the 8 NAHosMIP models, with the 95% CI of '
                 'the fitted line. GISS and literature estimates are shown but not fitted.',
                 fontsize=9, color=text_color, ha='center', va='top')
        savepath = (f'{savedir}/FigSupp_cooling_sensitivity_correlations'
                    f'_x-{x_axis.replace("_", "-")}'
                    f'_base-{baseline.replace("hist_", "hist").replace("_", "-")}_region-{region}'
                    f'_season-{season or "annual"}_window-{window}'
                    f'_giss-{int(include_giss)}_meehl-{int(include_ccsm4_cesm1)}'
                    f'_plotbg-{plot_bg}')
        functions.check_savepath(savepath)
        fig.savefig(savepath + '.png', dpi=200, bbox_inches='tight', transparent=plot_bg == 'black')
        fig.savefig(savepath + '.pdf', dpi=400, bbox_inches='tight', transparent=plot_bg == 'black')
        return fig, savepath

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(11, 9))
    axes_flat = axes.flatten()

    # Shared extents across panels.
    xs_all, ys_all = [], []
    for info in table.values():
        for ssp in ssps:
            x, y = model_xy(info, ssp, ssps, x_axis)
            if np.isfinite(x): xs_all.append(x)
            if np.isfinite(y): ys_all.append(y)
    x_lo, x_hi = (0.0, max(xs_all) * 1.1) if xs_all else (0.0, 1.0)
    if not ys_all:
        y_lo, y_hi = -0.25, 0.0
    elif x_axis == 'warming_over_weakening':
        y_lo, y_hi = 0.0, max(ys_all) * 1.1
    else:
        y_lo, y_hi = min(min(ys_all) * 1.15, -0.005), max(0.0, max(ys_all) * 1.05)
    axes_flat[0].set_xlim(x_lo, x_hi)
    axes_flat[0].set_ylim(y_lo, y_hi)

    for i, (ax, panel) in enumerate(zip(axes_flat, panels)):
        _render_cell(ax, panel, x_axis)
        _ssp_title(ax, panel, prefix=f'{chr(97 + i)}) ')

    # Outer labels only — sharex/sharey already strips inner tick labels.
    for ax in axes[1, :]:
        ax.set_xlabel(x_label)
    for ax in axes[:, 0]:
        ax.set_ylabel(y_label)

    fig.subplots_adjust(wspace=0.08, hspace=0.22)
    fig.text(0.5, 0.04, 'Grey line: OLS over the 8 NAHosMIP models, with the 95% CI of '
             'the fitted line. GISS and literature estimates are shown but not fitted.',
             fontsize=9, color=text_color, ha='center', va='top')

    savepath = (f'{savedir}/FigSupp_cooling_sensitivity_correlations'
                f'_x-{x_axis.replace("_", "-")}'
                f'_base-{baseline.replace("hist_", "hist").replace("_", "-")}_region-{region}'
                f'_season-{season or "annual"}_window-{window}'
                f'_giss-{int(include_giss)}_meehl-{int(include_ccsm4_cesm1)}'
                f'_plotbg-{plot_bg}')
    functions.check_savepath(savepath)
    fig.savefig(savepath + '.png', dpi=200, bbox_inches='tight',
                transparent=plot_bg == 'black')
    fig.savefig(savepath + '.pdf', dpi=400, bbox_inches='tight',
                transparent=plot_bg == 'black')
    return fig, savepath


########################################
# %%
# RUN

if __name__ == '__main__':
    season = ''
    window = 10
    giss_time_period = '2101-2300'
    ssps = ('ssp126', 'ssp245', 'ssp370')
    region = 'EU'
    include_giss = True
    include_ccsm4_cesm1 = False
    x_axis = 'combined'
    baseline = 'hist_1850_1899'
    plot_bg = 'white'

    fig, savepath = make_figure(
        multi_model_dict=multi_model_dict, hosmip_reg_ds_dict=hosmip_reg_ds_dict,
        reg_ds_giss=reg_ds_giss, reg_ds_giss_panel=reg_ds_giss_panel,
        masks=masks, gwl_data=gwl_data,
        season=season, window=window, giss_time_period=giss_time_period,
        ssps=ssps, region=region,
        include_giss=include_giss, include_ccsm4_cesm1=include_ccsm4_cesm1,
        x_axis=x_axis, baseline=baseline, plot_bg=plot_bg,
    )
    print(f'Saved {savepath}')

# %%
