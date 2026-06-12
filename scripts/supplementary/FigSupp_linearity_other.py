########################################
# %%
# LINEARITY DIAGNOSTIC (non-MPI models) — column-paired 2×3 grid
#
# Unified figure for CESM2 (combined-forcing, ssp126+ssp585) and
# GISS-E2-1-G (pooled ssp245 members − composite). Covariance is HAC
# (maxlags=window-1) for both — cluster-robust is degenerate at G=1/2.
#
# Selector: `model='cesm'|'giss'`. GISS adds a `giss_time_period` kwarg
# (default `('2101','2300')`; the other archive convention is
# `('2015','2500')`).
#
# Top row (region-independent maps, viridis, stippled at RESET p<0.05):
#   a) baseline R² of M₀          b) partial R² of curvature    c) ΔR² of total variance
# Bottom row (region-mean diagnostics; default region='EU'):
#   d) pooled regression          e) residuals + augmented      f) M₀ vs augmented prediction
#                                    correction overlay

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
    MODEL = 'giss'                       # 'cesm' or 'giss'
    GISS_TIME_PERIOD = ('2101', '2300')  # used when MODEL='giss'
    SEASON = ''                          # '', 'djf', 'jja' — GISS only for djf/jja

    multi_model_dict, masks = functions.get_full_multi_model_dict()
    if MODEL == 'cesm':
        _, _, boot_data, _ = functions.get_other_studies_data(masks)
        amoc_giss, tas_giss = None, None
        lin_ds = functions.get_linearity_diagnostics_cesm(var='tas')
    elif MODEL == 'giss':
        boot_data = None
        amoc_giss, tas_giss = functions.load_giss_member_amoc_tas(season=SEASON)
        lin_ds = functions.get_linearity_diagnostics_giss(
            time_period=GISS_TIME_PERIOD, var='tas')
    else:
        raise ValueError(f'unknown MODEL {MODEL!r}')

########################################
# %%
# FIGURE FUNCTION


def make_figure(masks, lin_ds, model='cesm', region='EU', season='',
                plot_bg='white', var='tas',
                # CESM-specific
                boot_data=None, multi_model_dict=None,
                # GISS-specific
                amoc_giss=None, tas_giss=None,
                giss_time_period=('2101', '2300')):
    if model not in ('cesm', 'giss'):
        raise ValueError(f'model must be cesm|giss, got {model!r}')
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    fg = 'black' if plot_bg != 'black' else 'white'
    bg = 'white' if plot_bg != 'black' else '#191919'

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.05, 1.0], width_ratios=[1.0, 1.0, 1.0],
        wspace=0.22, hspace=0.15,
    )
    ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=8))
    ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson(central_longitude=8))
    ax_c = fig.add_subplot(gs[0, 2], projection=ccrs.Robinson(central_longitude=8))
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    # ------------------------------------------------------------------
    # TOP ROW: maps of baseline R², partial R², ΔR²_total
    # ------------------------------------------------------------------
    lin = lin_ds.sel(season=season)
    baseline_r2 = lin['baseline_r2'].values
    partial_r2 = lin['partial_r2'].values
    delta_r2 = lin['delta_r2_total'].values
    reset_p = lin['reset_pvalue'].values
    lons = lin['lon'].values
    lats = lin['lat'].values

    stipple = (reset_p < 0.05) & np.isfinite(reset_p)
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

    _draw_map(ax_a, baseline_r2, vmin=0.0, vmax=1.0, n_bins=10,
              label='baseline R² (linear share of total variance)')
    _draw_map(ax_b, partial_r2, vmin=0.0, vmax=0.25, n_bins=5,
              label='partial R² (curvature share of residuals)')
    _draw_map(ax_c, delta_r2, vmin=0.0, vmax=0.05, n_bins=5,
              label='ΔR²_total (curvature share of total variance)')

    season_suffix = f' ({season.upper()})' if season else ''
    model_label = 'CESM2' if model == 'cesm' else 'GISS-E2-1-G'
    ax_a.set_title(f'a) baseline R² ({model_label}){season_suffix}', fontsize=12, loc='left')
    ax_b.set_title(f'b) partial R² ({model_label}){season_suffix}', fontsize=12, loc='left')
    ax_c.set_title(f'c) ΔR²_total ({model_label}){season_suffix}', fontsize=12, loc='left')

    if model == 'cesm':
        caption = ('Stippling: RESET F-test rejects linearity at 5% '
                   '(CESM2 pooled ssp126+ssp585; HAC SE, maxlags=window−1).')
    else:
        caption = ('Stippling: RESET F-test rejects linearity at 5% '
                   f'(GISS-E2-1-G ssp245 pooled members − composite; '
                   f'fit window {giss_time_period[0]}-{giss_time_period[1]}; '
                   f'HAC SE, maxlags=window−1).')
    fig.text(0.5, 0.92, caption, fontsize=10, ha='center',
             style='italic', color=fg)

    # ------------------------------------------------------------------
    # BOTTOM ROW d): canonical pooled regression scatter
    # ------------------------------------------------------------------
    if model == 'cesm':
        functions.cesm_regressions(
            boot_data, multi_model_dict, masks,
            ssp='all', region=region,
            combined_reg=True, add_combined_intercept=True,
            no_plots=False, ext_ax=ax_d, plot_bg=plot_bg,
            compute_linearity=True, season=season, xlim=50, low_ylim=-8,
            equation_y_pos=0.30, equation_y_spacing=0.08,  # fraction-tall simple_eqs
        )
        lc = functions.cesm_regressions.last_call
        x_label = 'Additional AMOC weakening [%]'
        x_extrap_dir = +1   # CESM x grows positive (more weakening); extrapolate right
    else:
        functions.giss_regressions(
            amoc_giss, tas_giss, masks=masks, region=region,
            time_period=giss_time_period, season=season,
            compute_linearity=True, no_plots=False, ext_ax=ax_d,
            plot_bg=plot_bg, add_combined_intercept=True,
        )
        lc = functions.giss_regressions.last_call
        x_label = 'AMOC deviation from composite mean [Sv]'
        # GISS x is bipolar (members above/below composite); extrapolate
        # toward the further-from-zero side of the data.
        x_extrap_dir = 0

    ax_d.set_title(
        f'd) {region}-mean {model_label} pooled regression{season_suffix}',
        fontsize=12, loc='left')

    # ------------------------------------------------------------------
    # Common per-point arrays for panels e and f
    # ------------------------------------------------------------------
    # Detect a failed fit (common for tiny regions on coarse grids, e.g.
    # GISS 'IS' on the 90×144 native grid where the mask has no land cells).
    fit_failed = (lc.get('fit_ok') is False) or (np.size(lc.get('combined_x', [])) == 0)

    if fit_failed:
        for _ax, _label in [(ax_e, 'e'), (ax_f, 'f')]:
            _ax.text(0.5, 0.5,
                     f'no data — region {region!r} has\nno finite cells on the '
                     f'{model_label} native grid',
                     transform=_ax.transAxes, ha='center', va='center',
                     fontsize=11, color=fg, style='italic')
            _ax.set_xticks([]); _ax.set_yticks([])
            _ax.set_title(f'{_label}) (skipped){season_suffix}',
                          fontsize=12, loc='left')
        # Skip the bottom-row scatter drawing; jump to save.
        tp_tag = ''
        if model == 'giss':
            tp_tag = f'_tp-{giss_time_period[0]}-{giss_time_period[1]}'
        savepath = (f'../plots/FigSupp_{model}_linearity{tp_tag}'
                    f'_var-{var}_region-{region}'
                    f"_season-{season or 'annual'}_plotbg-{plot_bg}")
        fig.savefig(savepath + '.png', dpi=200, bbox_inches='tight',
                    transparent=True if plot_bg == 'black' else False)
        fig.savefig(savepath + '.pdf', dpi=400, bbox_inches='tight',
                    transparent=True if plot_bg == 'black' else False)
        return fig, savepath

    x_amoc = np.asarray(lc['combined_x'])
    y_dat = np.asarray(lc['combined_y'])
    resid = np.asarray(lc['combined_resid'])
    scen_names = np.asarray(lc['combined_scenario_name'])

    beta0 = lc['combined_intercept']
    beta1 = lc['combined_coef']
    beta0_aug = lc['combined_intercept_aug']
    beta1_aug = lc['combined_coef_aug']
    g2 = lc['gamma2']
    g3 = lc['gamma3']

    # ------------------------------------------------------------------
    # BOTTOM ROW e): residuals vs x, with augmented-correction overlay.
    # ------------------------------------------------------------------
    for k in range(len(x_amoc)):
        try:
            cl = functions.hosing_colors[scen_names[k]]['ge']
        except KeyError:
            cl = '#888888'
        ax_e.scatter(x_amoc[k], resid[k], marker='o', color=cl, s=18,
                     alpha=0.6, linewidths=0, clip_on=False)
    ax_e.axhline(0, color=fg, lw=1.0, ls='--')

    if np.isfinite(g2) and np.isfinite(g3) and len(x_amoc) > 4:
        # Augmented-minus-linear correction: zero-mean by construction so it
        # overlays the residual cloud. Equals ŷ_aug(x) − ŷ_M₀(x).
        x_curve = np.linspace(np.nanmin(x_amoc), np.nanmax(x_amoc), 200)
        yhat_curve = beta0 + beta1 * x_curve
        yhat_aug_curve = (beta0_aug + beta1_aug * x_curve
                          + g2 * yhat_curve**2 + g3 * yhat_curve**3)
        y_curve = yhat_aug_curve - yhat_curve
        ax_e.plot(x_curve, y_curve, color='#3d8fd4', lw=2.0,
                  label='ŷ_aug(x) − ŷ_M₀(x)  (augmented correction)')
        ax_e.legend(loc='lower left', frameon=False, fontsize=9)

    ax_e.set_xlabel(x_label)
    ax_e.set_ylabel('Residual e₀ = y − ŷ [°C]')
    ax_e.set_xlim(ax_d.get_xlim())
    ax_e.spines['top'].set_visible(False)
    ax_e.spines['right'].set_visible(False)
    ax_e.set_title(f'e) {region}-mean residuals (RESET diagnostic){season_suffix}',
                   fontsize=12, loc='left')

    # ------------------------------------------------------------------
    # BOTTOM ROW f): M₀ line + augmented curve over the data range
    # ------------------------------------------------------------------
    for k in range(len(x_amoc)):
        try:
            cl = functions.hosing_colors[scen_names[k]]['ge']
        except KeyError:
            cl = '#888888'
        ax_f.scatter(x_amoc[k], y_dat[k], marker='o', color=cl, s=18,
                     alpha=0.6, linewidths=0, clip_on=False)

    x_min_data = float(np.nanmin(x_amoc))
    x_max_data = float(np.nanmax(x_amoc))
    if x_extrap_dir > 0:
        x_extrap_right = max(x_max_data + 10.0, ax_d.get_xlim()[1])
        x_full = np.linspace(x_min_data, x_extrap_right, 400)
    else:
        # GISS: extrapolate symmetrically by ~20% beyond data range.
        span = x_max_data - x_min_data
        x_extrap_left = x_min_data - 0.2 * span
        x_extrap_right = x_max_data + 0.2 * span
        x_full = np.linspace(x_extrap_left, x_extrap_right, 400)
    yhat_M0_full = beta0 + beta1 * x_full
    yhat_aug_full = (beta0_aug + beta1_aug * x_full
                     + g2 * yhat_M0_full**2
                     + g3 * yhat_M0_full**3)

    ax_f.plot(x_full, yhat_M0_full, color=fg, lw=1.2, ls='-',
              label='linear: ŷ(x) = β̂₀ + β̂₁·x', zorder=4)
    ax_f.plot(x_full, yhat_aug_full, color='#3d8fd4', lw=2.4, ls='-',
              label='augmented: + γ̂₂·ŷ(x)² + γ̂₃·ŷ(x)³', zorder=5)

    ax_f.axhline(0, color=fg, lw=1.0, ls='--', zorder=2)
    ax_f.axvline(0, color=fg, lw=1.0, ls='--', zorder=2)
    ax_f.set_xlabel(x_label)
    ax_f.set_ylabel(ax_d.get_ylabel())
    ax_f.set_xlim(x_full[0], x_full[-1])
    ax_f.set_ylim(ax_d.get_ylim())
    ax_f.spines['top'].set_visible(False)
    ax_f.spines['right'].set_visible(False)
    ax_f.set_title(f'f) {region}-mean linear vs augmented prediction{season_suffix}',
                   fontsize=12, loc='left')
    ax_f.legend(loc='lower left', frameon=False, fontsize=9)

    sign_str = lambda s: '+' if s > 0 else ('−' if s < 0 else '0')
    annot = (
        f'baseline R² = {lc["baseline_r2"]:.3f}\n'
        f'partial R²  = {lc["partial_r2"]:.3f}\n'
        f'ΔR²_total   = {lc["delta_r2_total"]:.4f}\n'
        f'RESET p     = {lc["reset_pvalue"]:.3f}\n'
        f'(γ̂₂, γ̂₃)   = ({sign_str(lc["beta2_sign"])},{sign_str(lc["beta3_sign"])})'
    )
    ax_f.text(
        0.03, 0.55, annot,
        transform=ax_f.transAxes, fontsize=9,
        va='center', ha='left', family='monospace',
        bbox=dict(facecolor=bg, edgecolor=fg, alpha=0.85,
                  boxstyle='round,pad=0.4'),
    )

    # ------------------------------------------------------------------
    # SAVE — filename encodes model + (giss_time_period if giss)
    # ------------------------------------------------------------------
    tp_tag = ''
    if model == 'giss':
        tp_tag = f'_tp-{giss_time_period[0]}-{giss_time_period[1]}'
    savepath = (f'../plots/FigSupp_{model}_linearity{tp_tag}'
                f'_var-{var}_region-{region}'
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
    if MODEL == 'cesm':
        fig, savepath = make_figure(
            masks, lin_ds, model='cesm', region='EU', season=SEASON,
            plot_bg='white', boot_data=boot_data,
            multi_model_dict=multi_model_dict,
        )
    else:
        fig, savepath = make_figure(
            masks, lin_ds, model='giss', region='EU', season=SEASON,
            plot_bg='white', amoc_giss=amoc_giss, tas_giss=tas_giss,
            giss_time_period=GISS_TIME_PERIOD,
        )
    print(f'Saved {savepath}.(pdf|png)')

# %%
