########################################
# %%
# LINEARITY DIAGNOSTIC — column-paired 2×3 grid
#
# Top row (region-independent maps, all viridis, all stippled at RESET p<0.05):
#   a) baseline R² of M₀          b) partial R² of curvature    c) ΔR² of total variance
# Bottom row (region-mean diagnostics; default region='EU', any region accepted):
#   d) regression M₀ scatter+line e) residuals e₀ vs x with     f) scatter + M₀ line +
#                                    γ₂·ŷ²+γ₃·ŷ³ overlay          augmented curve, with
#                                                                  extrapolation tail shaded
#
# Each column is a metric (top) paired with its EU-data operationalisation
# (bottom). No magnitude threshold anywhere — stippling encodes only formal
# detectability of curvature (RESET F-test), and the magnitude fields show
# how big that curvature actually is.

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
    data_dict = functions.load_mpi_esm_data(eur_only=True)
    lin_ds = functions.get_linearity_diagnostics_mpi(var='tas')

########################################
# %%
# WORST-LINEARITY COUNTRIES (smallest RESET p) — top 10 per season
#
# Ranked by RESET F-test p-value ascending: smallest p = strongest formal
# evidence that a cubic in ŷ adds beyond M₀. Magnitude (partial R²,
# ΔR²_total) is not the sort key; cells can rank high on RESET p while
# the cubic still explains <2% of total variance. Derived from a one-off scan over
# functions.country_codes (var='tas') on 2026-05-28; regenerate with
# /tmp/scan_worst_linearity_countries.py if upstream data, AMOC ref,
# scenario list, mask topology, or regression spec change.
#
# Comments next to each entry: baseR2 = baseline R² of M₀,
# partR2 = partial R² of curvature, dR2t = ΔR² of total variance.
_WORST_LINEARITY_COUNTRIES_BY_SEASON = {
    '': [
        ('PT', 0.0000),  # baseR2=0.890, partR2=0.097, dR2t=0.0107
        ('LU', 0.0013),  # baseR2=0.858, partR2=0.057, dR2t=0.0080
        ('FR', 0.0017),  # baseR2=0.894, partR2=0.068, dR2t=0.0072
        ('ES', 0.0030),  # baseR2=0.904, partR2=0.059, dR2t=0.0057
        ('CH', 0.0072),  # baseR2=0.828, partR2=0.040, dR2t=0.0069
        ('BE', 0.0073),  # baseR2=0.893, partR2=0.062, dR2t=0.0067
        ('NL', 0.0127),  # baseR2=0.879, partR2=0.045, dR2t=0.0054
        ('DE', 0.0129),  # baseR2=0.807, partR2=0.031, dR2t=0.0061
        ('TR', 0.0159),  # baseR2=0.841, partR2=0.029, dR2t=0.0047
        ('IE', 0.0194),  # baseR2=0.949, partR2=0.093, dR2t=0.0048
    ],
    'djf': [
        ('PT', 0.0000),  # baseR2=0.911, partR2=0.170, dR2t=0.0152
        ('ES', 0.0007),  # baseR2=0.887, partR2=0.084, dR2t=0.0095
        ('LU', 0.0227),  # baseR2=0.684, partR2=0.020, dR2t=0.0064
        ('FR', 0.0265),  # baseR2=0.783, partR2=0.027, dR2t=0.0058
        ('XK', 0.0330),  # baseR2=0.478, partR2=0.028, dR2t=0.0148
        ('AL', 0.0353),  # baseR2=0.514, partR2=0.028, dR2t=0.0138
        ('TR', 0.0402),  # baseR2=0.605, partR2=0.035, dR2t=0.0139
        ('BE', 0.0406),  # baseR2=0.743, partR2=0.021, dR2t=0.0055
        ('GR', 0.0597),  # baseR2=0.521, partR2=0.021, dR2t=0.0102
        ('MK', 0.0639),  # baseR2=0.416, partR2=0.019, dR2t=0.0113
    ],
    'jja': [
        ('TR', 0.0004),  # baseR2=0.830, partR2=0.084, dR2t=0.0142
        ('IS', 0.0024),  # baseR2=0.917, partR2=0.081, dR2t=0.0068
        ('BY', 0.0196),  # baseR2=0.572, partR2=0.026, dR2t=0.0111
        ('NL', 0.0258),  # baseR2=0.811, partR2=0.052, dR2t=0.0098
        ('FR', 0.0347),  # baseR2=0.724, partR2=0.048, dR2t=0.0133
        ('LT', 0.0397),  # baseR2=0.638, partR2=0.035, dR2t=0.0126
        ('IE', 0.0409),  # baseR2=0.920, partR2=0.041, dR2t=0.0033
        ('GB', 0.0476),  # baseR2=0.902, partR2=0.041, dR2t=0.0040
        ('PL', 0.0555),  # baseR2=0.538, partR2=0.036, dR2t=0.0166
        ('GR', 0.0561),  # baseR2=0.762, partR2=0.031, dR2t=0.0073
    ],
}

# Alternative ranking #1: partial R² descending (magnitude of curvature,
# agnostic to which polynomial term carries it). Captures countries where
# the cubic explains a large share of M₀ residuals even if the joint
# RESET F is borderline. Same scan run as the dict above.
_WORST_LINEARITY_BY_PARTIAL_R2 = {
    '': [
        ('PT', 0.0974),  # dR2t=0.0107, baseR2=0.890, resetP=0.0000
        ('IE', 0.0933),  # dR2t=0.0048, baseR2=0.949, resetP=0.0194
        ('GB', 0.0735),  # dR2t=0.0040, baseR2=0.945, resetP=0.0357
        ('FR', 0.0682),  # dR2t=0.0072, baseR2=0.894, resetP=0.0017
        ('BE', 0.0624),  # dR2t=0.0067, baseR2=0.893, resetP=0.0073
        ('ES', 0.0594),  # dR2t=0.0057, baseR2=0.904, resetP=0.0030
        ('LU', 0.0567),  # dR2t=0.0080, baseR2=0.858, resetP=0.0013
        ('NO', 0.0452),  # dR2t=0.0069, baseR2=0.847, resetP=0.0547
        ('NL', 0.0446),  # dR2t=0.0054, baseR2=0.879, resetP=0.0127
        ('CH', 0.0400),  # dR2t=0.0069, baseR2=0.828, resetP=0.0072
    ],
    'djf': [
        ('PT', 0.1698),  # dR2t=0.0152, baseR2=0.911, resetP=0.0000
        ('ES', 0.0839),  # dR2t=0.0095, baseR2=0.887, resetP=0.0007
        ('IE', 0.0374),  # dR2t=0.0037, baseR2=0.902, resetP=0.0769
        ('GB', 0.0355),  # dR2t=0.0039, baseR2=0.890, resetP=0.0654
        ('TR', 0.0353),  # dR2t=0.0139, baseR2=0.605, resetP=0.0402
        ('XK', 0.0285),  # dR2t=0.0148, baseR2=0.478, resetP=0.0330
        ('AL', 0.0284),  # dR2t=0.0138, baseR2=0.514, resetP=0.0353
        ('NO', 0.0278),  # dR2t=0.0101, baseR2=0.636, resetP=0.0923
        ('FR', 0.0267),  # dR2t=0.0058, baseR2=0.783, resetP=0.0265
        ('BE', 0.0214),  # dR2t=0.0055, baseR2=0.743, resetP=0.0406
    ],
    'jja': [
        ('TR', 0.0838),  # dR2t=0.0142, baseR2=0.830, resetP=0.0004
        ('IS', 0.0813),  # dR2t=0.0068, baseR2=0.917, resetP=0.0024
        ('NL', 0.0522),  # dR2t=0.0098, baseR2=0.811, resetP=0.0258
        ('FR', 0.0481),  # dR2t=0.0133, baseR2=0.724, resetP=0.0347
        ('DE', 0.0415),  # dR2t=0.0164, baseR2=0.606, resetP=0.0940
        ('GB', 0.0408),  # dR2t=0.0040, baseR2=0.902, resetP=0.0476
        ('IE', 0.0407),  # dR2t=0.0033, baseR2=0.920, resetP=0.0409
        ('DK', 0.0403),  # dR2t=0.0075, baseR2=0.814, resetP=0.0695
        ('CH', 0.0364),  # dR2t=0.0160, baseR2=0.561, resetP=0.0890
        ('PL', 0.0361),  # dR2t=0.0166, baseR2=0.538, resetP=0.0555
    ],
}

# Alternative ranking #2: quadratic-only p ascending. Refit
# y ~ const + x + γ₂·ŷ² with the same cluster-robust covariance
# (exp_id as cluster, use_correction=True) as reset_diagnostics, then
# t-test γ₂=0. Sidesteps the joint-F penalty in RESET that hurts
# cleanly-quadratic countries with γ₃≈0 (notably NO). Sign of γ̂₂ in
# the comment: positive = concave up (accelerating cooling),
# negative = concave down (saturating).
_WORST_LINEARITY_BY_QUADRATIC_P = {
    '': [
        ('BE', 0.0045),  # γ̂₂=+1.55e-01, partR2=0.062, resetP=0.0073
        ('LU', 0.0067),  # γ̂₂=+1.98e-01, partR2=0.057, resetP=0.0013
        ('NO', 0.0107),  # γ̂₂=-1.72e-01, partR2=0.045, resetP=0.0547
        ('CH', 0.0132),  # γ̂₂=+2.28e-01, partR2=0.040, resetP=0.0072
        ('NL', 0.0132),  # γ̂₂=+1.37e-01, partR2=0.045, resetP=0.0127
        ('DE', 0.0134),  # γ̂₂=+1.97e-01, partR2=0.031, resetP=0.0129
        ('FR', 0.0149),  # γ̂₂=+1.57e-01, partR2=0.068, resetP=0.0017
        ('CZ', 0.0238),  # γ̂₂=+2.57e-01, partR2=0.019, resetP=0.0468
        ('PL', 0.0313),  # γ̂₂=+2.41e-01, partR2=0.017, resetP=0.0787
        ('SK', 0.0413),  # γ̂₂=+2.93e-01, partR2=0.014, resetP=0.1383
    ],
    'djf': [
        ('XK', 0.0111),  # γ̂₂=+5.30e-01, partR2=0.028, resetP=0.0330
        ('LU', 0.0155),  # γ̂₂=+1.97e-01, partR2=0.020, resetP=0.0227
        ('AL', 0.0176),  # γ̂₂=+4.92e-01, partR2=0.028, resetP=0.0353
        ('NO', 0.0207),  # γ̂₂=-2.02e-01, partR2=0.028, resetP=0.0923
        ('CH', 0.0223),  # γ̂₂=+2.13e-01, partR2=0.013, resetP=0.0672
        ('MK', 0.0264),  # γ̂₂=+5.25e-01, partR2=0.019, resetP=0.0639
        ('BE', 0.0290),  # γ̂₂=+1.59e-01, partR2=0.021, resetP=0.0406
        ('AT', 0.0367),  # γ̂₂=+2.40e-01, partR2=0.011, resetP=0.1123
        ('GR', 0.0477),  # γ̂₂=+4.59e-01, partR2=0.021, resetP=0.0597
        ('SE', 0.0509),  # γ̂₂=-2.17e-01, partR2=0.016, resetP=0.1664
    ],
    'jja': [
        ('BY', 0.0052),  # γ̂₂=+4.18e-01, partR2=0.026, resetP=0.0196
        ('LT', 0.0071),  # γ̂₂=+3.96e-01, partR2=0.035, resetP=0.0397
        ('NL', 0.0088),  # γ̂₂=+2.21e-01, partR2=0.052, resetP=0.0258
        ('PL', 0.0128),  # γ̂₂=+5.34e-01, partR2=0.036, resetP=0.0555
        ('FR', 0.0143),  # γ̂₂=+3.24e-01, partR2=0.048, resetP=0.0347
        ('GB', 0.0202),  # γ̂₂=+1.03e-01, partR2=0.041, resetP=0.0476
        ('DK', 0.0213),  # γ̂₂=+1.92e-01, partR2=0.040, resetP=0.0695
        ('DE', 0.0268),  # γ̂₂=+4.75e-01, partR2=0.042, resetP=0.0940
        ('LV', 0.0289),  # γ̂₂=+2.81e-01, partR2=0.023, resetP=0.1117
        ('BE', 0.0363),  # γ̂₂=+1.89e-01, partR2=0.032, resetP=0.1165
    ],
}

########################################
# %%
# FIGURE FUNCTION

def make_figure(data_dict, lin_ds, var='tas', region='EU', season='', plot_bg='white'):
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

    # Stippling: formal detection only (RESET p < 0.05). No magnitude factor.
    stipple = (reset_p < 0.05) & np.isfinite(reset_p)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    stipple_lon = lon_grid[stipple]
    stipple_lat = lat_grid[stipple]

    def _draw_map(ax, field, vmin, vmax, n_bins, label):
        # Discrete sequential viridis: sample at bin centres so neither end of
        # the colour ramp collides with set_under/set_over. set_under sampled
        # at viridis(0.0) (darker than first bin), set_over at viridis(1.0)
        # (brighter than last bin) so over/under triangles are visually
        # distinct from the adjacent bin.
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
        functions.add_square(ax, -13.5, 0,  60, 75.5, colour=bg)
        functions.add_square(ax, -30, -18,  68, 75, colour=bg)
        cbar = fig.colorbar(pcm, ax=ax, orientation='horizontal',
                            pad=0.05, aspect=30, extend='max', ticks=bounds)
        cbar.set_label(label, fontsize=10)
        return cbar

    _draw_map(ax_a, baseline_r2, vmin=0.0, vmax=1.0,   n_bins=10,
              label='baseline R² (linear share of total variance)')
    _draw_map(ax_b, partial_r2,  vmin=0.0, vmax=0.25,  n_bins=5,
              label='partial R² (curvature share of residuals)')
    _draw_map(ax_c, delta_r2,    vmin=0.0, vmax=0.025, n_bins=5,
              label='ΔR²_total (curvature share of total variance)')

    season_suffix = f' ({season.upper()})' if season else ''
    ax_a.set_title(f'a) baseline R²{season_suffix}', fontsize=12, loc='left')
    ax_b.set_title(f'b) partial R²{season_suffix}', fontsize=12, loc='left')
    ax_c.set_title(f'c) ΔR²_total{season_suffix}',  fontsize=12, loc='left')

    fig.text(
        0.5, 0.92,
        'Stippling: RESET F-test rejects linearity at 5%.',
        fontsize=10, ha='center', style='italic', color=fg,
    )

    # ------------------------------------------------------------------
    # BOTTOM ROW d): canonical regression scatter + M₀ line (regression_plot)
    # Bottom row mirrors Fig1.py panel c (weakening-% ruler via
    # functions.add_weakening_pct_overlay; data stays in Sv).
    # ------------------------------------------------------------------
    functions.regression_plot(
        data_dict, region=region, var=var, season=season,
        no_plots=False, compute_linearity=True,
        ext_ax=ax_d, plot_bg=plot_bg, ylim=(-2.5, 1.5),
        equation_y_pos=0.25,  # equations in the empty bottom-left, clear of the data
    )
    lc = functions.regression_plot.last_call
    functions.add_weakening_pct_overlay(fig, ax_d, plot_bg=plot_bg, drop_parent_xaxis=True)
    ax_d.set_title(f'd) {region}-mean linear regression{season_suffix}',
                   fontsize=12, loc='left')

    # ------------------------------------------------------------------
    # Common per-point arrays for panels e and f
    # ------------------------------------------------------------------
    x_amoc = np.asarray(lc['combined_x'])
    y_dat = np.asarray(lc['combined_y'])
    resid = np.asarray(lc['combined_resid'])
    ssp_names = np.asarray(lc['combined_ssp_name'])
    hos_names = np.asarray(lc['combined_hos_name'])

    beta0 = lc['combined_intercept']
    beta1 = lc['combined_coef']
    beta0_aug = lc['combined_intercept_aug']
    beta1_aug = lc['combined_coef_aug']
    g2 = lc['gamma2']
    g3 = lc['gamma3']

    # ------------------------------------------------------------------
    # BOTTOM ROW e): residuals vs x, with γ₂·ŷ²+γ₃·ŷ³ overlay
    # Per-hosing markers + per-(ssp,hosing) colours, matching regression_plot.
    # ------------------------------------------------------------------
    for k in range(len(x_amoc)):
        try:
            mk = functions.hosing_markers[hos_names[k]]
            cl = functions.hosing_colors[ssp_names[k]][hos_names[k]]
        except KeyError:
            mk, cl = 'o', '#888888'
        ax_e.scatter(x_amoc[k], resid[k], marker=mk, color=cl, s=25,
                     alpha=0.75, linewidths=0, clip_on=False)
    ax_e.axhline(0, color=fg, lw=1.0, ls='--')

    if np.isfinite(g2) and np.isfinite(g3) and len(x_amoc) > 4:
        x_curve = np.linspace(np.nanmin(x_amoc), np.nanmax(x_amoc), 200)
        yhat_curve = beta0 + beta1 * x_curve
        y_curve = g2 * yhat_curve**2 + g3 * yhat_curve**3
        ax_e.plot(x_curve, y_curve, color='#3d8fd4', lw=2.0,
                  label='γ₂·ŷ(x)² + γ₃·ŷ(x)³  (augmented OLS)')
        ax_e.legend(loc='lower left', frameon=False, fontsize=9)

    ax_e.set_ylabel('Residual e₀ = y − ŷ [°C]')
    ax_e.set_xlim(ax_d.get_xlim())  # Sv, inverted to match d; %-ruler added below
    ax_e.spines['top'].set_visible(False)
    ax_e.spines['right'].set_visible(False)
    functions.add_weakening_pct_overlay(fig, ax_e, plot_bg=plot_bg, drop_parent_xaxis=True)
    ax_e.set_title(f'e) {region}-mean residuals (RESET diagnostic){season_suffix}',
                   fontsize=12, loc='left')

    # ------------------------------------------------------------------
    # BOTTOM ROW f): M₀ line + augmented curve, with extrapolation tail shaded
    # ------------------------------------------------------------------
    for k in range(len(x_amoc)):
        try:
            mk = functions.hosing_markers[hos_names[k]]
            cl = functions.hosing_colors[ssp_names[k]][hos_names[k]]
        except KeyError:
            mk, cl = 'o', '#888888'
        ax_f.scatter(x_amoc[k], y_dat[k], marker=mk, color=cl, s=25,
                     alpha=0.75, linewidths=0, clip_on=False)

    x_min_data = float(np.nanmin(x_amoc))
    x_max_data = float(np.nanmax(x_amoc))
    x_extrap_left = min(-12.0, x_min_data - 1.0)
    x_full = np.linspace(x_extrap_left, x_max_data, 400)
    yhat_M0_full = beta0 + beta1 * x_full
    yhat_M0_for_curve = beta0 + beta1 * x_full
    yhat_aug_full = (beta0_aug + beta1_aug * x_full
                     + g2 * yhat_M0_for_curve**2
                     + g3 * yhat_M0_for_curve**3)

    # M₀ across the full extrapolated range
    ax_f.plot(x_full, yhat_M0_full, color=fg, lw=1.2, ls='-',
              label='linear: ŷ(x) = β̂₀ + β̂₁·x', zorder=4)
    # Augmented across the full extrapolated range
    ax_f.plot(x_full, yhat_aug_full, color='#3d8fd4', lw=2.4, ls='-',
              label='augmented: + γ̂₂·ŷ(x)² + γ̂₃·ŷ(x)³', zorder=5)

    # Shade the extrapolation tail (x outside the data x-range)
    # ax_f.axvspan(x_extrap_left, x_min_data, color=fg, alpha=0.08, zorder=1)
    # ax_f.text(
    #     x_extrap_left + 0.15, ax_d.get_ylim()[1] - 0.15,
    #     'polynomial extrapolation\n(not a statistical statement)',
    #     fontsize=8, color=fg, ha='left', va='top', alpha=0.7, style='italic',
    # )

    ax_f.axhline(0, color=fg, lw=1.0, ls='--', zorder=2)
    ax_f.axvline(0, color=fg, lw=1.0, ls='--', zorder=2)
    ax_f.set_ylabel(ax_d.get_ylabel())
    # Match d's inverted Sv orientation: least-weakening (+6 Sv) left, extrapolation
    # tail (x_extrap_left) right. %-ruler (extended to +60 for the tail) added below.
    ax_f.set_xlim(ax_d.get_xlim()[0], x_extrap_left)
    ax_f.set_ylim(ax_d.get_ylim())
    ax_f.spines['top'].set_visible(False)
    ax_f.spines['right'].set_visible(False)
    functions.add_weakening_pct_overlay(fig, ax_f, plot_bg=plot_bg, drop_parent_xaxis=True, weak_max_pct=60)
    ax_f.set_title(f'f) {region}-mean linear vs augmented prediction{season_suffix}',
                   fontsize=12, loc='left')
    ax_f.legend(loc='lower right', frameon=False, fontsize=9)

    # Annotation block in panel f — baseline R² first as the headline.
    sign_str = lambda s: '+' if s > 0 else ('−' if s < 0 else '0')
    annot = (
        f'baseline R² = {lc["baseline_r2"]:.3f}\n'
        f'partial R²  = {lc["partial_r2"]:.3f}\n'
        f'ΔR²_total   = {lc["delta_r2_total"]:.4f}\n'
        f'RESET p     = {lc["reset_pvalue"]:.3f}\n'
        f'(γ̂₂, γ̂₃)   = ({sign_str(lc["beta2_sign"])},{sign_str(lc["beta3_sign"])})'
    )
    ax_f.text(
        0.97, 0.930, annot,
        transform=ax_f.transAxes, fontsize=9,
        va='top', ha='right', family='monospace',
        bbox=dict(facecolor=bg, edgecolor=fg, alpha=0.85,
                  boxstyle='round,pad=0.4'),
    )

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    savepath = (f"../plots/FigSupp_mpi_linearity_var-{var}_region-{region}"
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
    fig, savepath = make_figure(data_dict, lin_ds, var='tas', region='EU',
                                season='', plot_bg='white')
    print(f'Saved {savepath}.(pdf|png)')

# %%
