"""
GISS-E2-1-G ssp245 Fig1-style analysis (1850–2500).

Two-panel time-series (AMOC, EU tas) for all 10 r1–r10 i1p1f2 members,
with a composite "stable AMOC" reference line (mean of r3, r4, r10).
Right panel: cluster-robust OLS regression of (T_member − T_composite)
on (AMOC_member − AMOC_composite) for the 7 non-composite members,
2015–2500. Anomalies are relative to the GISS-specific 1850–1899
pre-industrial baseline.

Inputs:
- Historical yearly caches (legacy, audited clean, read-only):
  /work/uo1075/m300817/teu_amoc/data/CMIP6/historical/giss-e2-1-g/
- ssp245 to-2500 caches:
  /work/bu1431/T_EU_AMOC/CMIP6/giss-e2-1-g/processed/
- Constants, PI tas climatology, region masks:
  /work/bu1431/T_EU_AMOC/CMIP6/giss-e2-1-g/processed/constants.json
  /work/bu1431/T_EU_AMOC/CMIP6/giss-e2-1-g/processed/giss-e2-1-g_tas_pi_climatology.nc
  /work/bu1431/T_EU_AMOC/CMIP6/giss-e2-1-g/processed/giss-e2-1-g_masks.pkl

Run interactively cell-by-cell, or as a standalone script.
"""

########################################
# %%
# LOAD PACKAGES AND CONFIG

import importlib
import json
import os
import pickle

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.pyplot import cm

import sys, pathlib
if "__file__" in globals():  # standalone run: ensure scripts/ is importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import functions
importlib.reload(functions)

PROC_DIR = '/work/bu1431/T_EU_AMOC/CMIP6/giss-e2-1-g/processed'

MEMBERS = [f'r{i}i1p1f2' for i in range(1, 11)]
COMPOSITE = ['r3i1p1f2', 'r4i1p1f2', 'r10i1p1f2']
NON_COMPOSITE = [m for m in MEMBERS if m not in COMPOSITE]
PI_WINDOW = ('1850', '1899')
SSP_START = '2015'
WINDOW = 10                   # centred rolling mean (years)
REG_TIME = (SSP_START, '2500')

# Season selector. '' = annual; 'djf' / 'jja' route tas through the
# seasonal historical + ssp245-to-2500 files (AMOC stays annual — no
# seasonal sibling at 26°N). Default
# annual; override interactively or via `SEASON=djf python supplementary/FigSupp_giss_sim_regression.py`.
SEASON = os.environ.get('SEASON', '')

# 10 visually distinct marker shapes — one per ensemble member.
# Composite members (r3, r4, r10) still get markers so the time-series
# lines they draw in panels a/b can be identified in the legend.
MEMBER_MARKERS = dict(zip(MEMBERS,
    ['o', 's', '^', 'v', 'D', 'P', 'X', '*', 'p', 'h']))

PLOTS_DIR = '../plots'  # relative to cwd=scripts/, like every other figure script


def load_inputs(season=''):
    """Load GISS member AMOC/tas and derive the smoothed anomaly series that
    make_figure consumes (anomalies, EU-mean tas, composite baseline, centred
    rolling means). Shared entry point for produce_all_paper_figures; the
    interactive __main__ below mirrors these steps cell-by-cell."""
    global AMOC_PI, amoc_comp, tas_eu_comp  # make_figure reads these as module globals (set here or in __main__)
    with open(f'{PROC_DIR}/constants.json') as f:
        const = json.load(f)
    AMOC_PI = const['AMOC_pi_GISS_Sv']
    tas_pi = xr.open_dataarray(
        f'{PROC_DIR}/giss-e2-1-g_tas_pi'
        f"{'_' + season if season else ''}_climatology.nc")
    with open(f'{PROC_DIR}/giss-e2-1-g_masks.pkl', 'rb') as f:
        masks = pickle.load(f)
    amoc, tas = functions.load_giss_member_amoc_tas(season=season)
    amoc_anom = amoc - AMOC_PI
    tas_anom = tas - tas_pi
    tas_eu = functions.weighted_area_lat(
        tas_anom.where(masks['EU']).where(masks['LAND'] == 0)
    ).mean(('lat', 'lon'))
    amoc_comp = amoc_anom.sel(realiz=COMPOSITE).mean('realiz')
    tas_eu_comp = tas_eu.sel(realiz=COMPOSITE).mean('realiz')
    return dict(
        amoc_smooth=amoc_anom.rolling(time=WINDOW, center=True).mean(),
        tas_eu_smooth=tas_eu.rolling(time=WINDOW, center=True).mean(),
        amoc_comp_smooth=amoc_comp.rolling(time=WINDOW, center=True).mean(),
        tas_eu_comp_smooth=tas_eu_comp.rolling(time=WINDOW, center=True).mean(),
        amoc_anom=amoc_anom, tas_eu=tas_eu)

########################################
# %%
# LOAD DATA: AMOC, tas, constants, masks

if __name__ == '__main__':

    with open(f'{PROC_DIR}/constants.json') as f:
        const = json.load(f)
    AMOC_PI = const['AMOC_pi_GISS_Sv']

    tas_pi = xr.open_dataarray(
        f'{PROC_DIR}/giss-e2-1-g_tas_pi'
        f"{'_' + SEASON if SEASON else ''}_climatology.nc")

    with open(f'{PROC_DIR}/giss-e2-1-g_masks.pkl', 'rb') as f:
        masks = pickle.load(f)

    # Canonical loader (single source of truth, shared with get_giss_reg_ds /
    # get_giss_panel_data): historical + ssp245-to-2500, all 10 members, native
    # 90×144 grid; AMOC always annual, tas routed via the requested season.
    amoc, tas = functions.load_giss_member_amoc_tas(season=SEASON)

    print(f'amoc: shape={amoc.shape}, time {amoc.time.values[0]}..{amoc.time.values[-1]}')
    print(f'tas:  shape={tas.shape}')
    print(f'AMOC_pi_GISS = {AMOC_PI:.4f} Sv')


########################################
# %%
# DERIVED FIELDS: anomalies, EU mean, composite baseline


if __name__ == '__main__':

    amoc_anom = amoc - AMOC_PI
    tas_anom = tas - tas_pi

    tas_eu = functions.weighted_area_lat(
        tas_anom.where(masks['EU']).where(masks['LAND'] == 0)
    ).mean(('lat', 'lon'))

    amoc_comp = amoc_anom.sel(realiz=COMPOSITE).mean('realiz')
    tas_eu_comp = tas_eu.sel(realiz=COMPOSITE).mean('realiz')

    amoc_smooth = amoc_anom.rolling(time=WINDOW, center=True).mean()
    tas_eu_smooth = tas_eu.rolling(time=WINDOW, center=True).mean()
    amoc_comp_smooth = amoc_comp.rolling(time=WINDOW, center=True).mean()
    tas_eu_comp_smooth = tas_eu_comp.rolling(time=WINDOW, center=True).mean()

    print(f'tas_eu shape: {tas_eu.shape}')
    print(f'composite (r3+r4+r10) at 2015: '
          f'AMOC_anom={float(amoc_comp.sel(time=SSP_START)):.2f} Sv, '
          f'tas_EU_anom={float(tas_eu_comp.sel(time=SSP_START)):.2f} K')


########################################
# %%
# REGRESSION: HAC-OLS on member-minus-composite, 2015–2500
#
# fit_member_minus_composite was promoted to functions.py on
# 2026-05-23. The covariance was switched from cluster-robust (which
# is inappropriate for a same-forcing ensemble — every ssp245 member
# shares the same external forcing, so within-member dependence is
# autocorrelation, not a treatment shock) to HAC with
# maxlags=window-1, matching how Boot et al. (2024) is fitted in
# hosmip_regression_plot.

fit_member_minus_composite = functions.fit_member_minus_composite


########################################
# %%
# FIGURE FUNCTION


def _clean_axes(ax, hide_bottom=False):
    """Apply Fig1 spine convention: top+right off, left offset by -0.02,
    optional bottom off (top panel of a shared-x stack). Spines are
    then truncated to the first/last visible tick so the axis 'stops at
    ticks'."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('axes', -0.02))
    if hide_bottom:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', which='both', bottom=False, top=False,
                       labelbottom=False)
    # Truncate spines at outermost visible ticks.
    ymin, ymax = ax.get_ylim()
    yticks = [t for t in ax.get_yticks() if ymin <= t <= ymax]
    if yticks:
        ax.spines['left'].set_bounds(yticks[0], yticks[-1])
    if not hide_bottom:
        xmin, xmax = ax.get_xlim()
        xticks = [t for t in ax.get_xticks() if xmin <= t <= xmax]
        if xticks:
            ax.spines['bottom'].set_bounds(xticks[0], xticks[-1])


def make_figure(amoc_smooth, tas_eu_smooth, amoc_comp_smooth, tas_eu_comp_smooth,
                amoc_anom, tas_eu, plot_bg='white', season=''):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10})
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'
    fg = 'black' if plot_bg != 'black' else 'white'

    fig = plt.figure(figsize=(16, 8))
    # gs[0,1] is intentionally left empty — the legend and regression-
    # equation block live in that vertical slot above ax_reg, matching
    # Fig1's layout.
    gs = GridSpec(2, 2, width_ratios=[0.5, 0.5], height_ratios=[1, 1],
                  wspace=0.25, hspace=0.2)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_reg = fig.add_subplot(gs[1, 1])

    # Shared per-member viridis palette + per-member marker. Together
    # they uniquely identify a member across all three panels.
    viridis_levels = cm.viridis(np.linspace(0.0, 1.0, len(MEMBERS)))
    member_color = dict(zip(MEMBERS, viridis_levels))
    member_marker = MEMBER_MARKERS
    composite_color = functions.hosing_colors.get('ssp245', {}).get('ge', '#9e1b32')

    # ----- panel a: AMOC anomaly -----
    years = amoc_smooth.time.dt.year.values
    # Index of the last finite value of a centred window=WINDOW rolling
    # mean (matches simulations_plot convention).
    end_idx = -((WINDOW + 1) // 2)
    end_x_offset = 5  # years; small horizontal offset so marker sits clear of line
    for r in MEMBERS:
        ax_top.plot(years, amoc_smooth.sel(realiz=r).values,
                    color=member_color[r], lw=0.9, alpha=0.85)
        ax_top.plot(years[end_idx] + end_x_offset,
                    amoc_smooth.sel(realiz=r).values[end_idx],
                    marker=member_marker[r], color=member_color[r],
                    markersize=3, linestyle='None', clip_on=False)
    ax_top.plot(years, amoc_comp_smooth.values,
                color=composite_color, lw=2.2)
    # ax_top.plot(years[end_idx] + end_x_offset,
    #             amoc_comp_smooth.values[end_idx],
    #             marker='o', color=composite_color, markersize=8,
    #             linestyle='None', clip_on=False)
    ax_top.axhline(0, color=fg, lw=1., ls='--', alpha=0.5)
    ax_top.axvline(2014.5, color=fg, lw=0.6, ls=':', alpha=0.6)
    ax_top.text(2014.5, ax_top.get_ylim()[1], ' SSP2-4.5', color=fg,
                fontsize=8, va='top', ha='left', alpha=0.7)
    ax_top.set_ylabel(r'$\Delta\mathrm{AMOC}$ @ 26.5°N [Sv]'
                      '\nw.r.t. 1850–1899')
    ax_top.set_xticks(np.arange(1900, 2501, 100))

    # Secondary % weakening axis on the right of ax_top, capped at [0, 80]%.
    # 0% weakening → primary anomaly = 0 Sv; 80% weakening → -0.8·AMOC_PI Sv.
    # The secondary axis only spans the corresponding sub-range of the
    # primary y-axis, mirroring Fig1's sub-range secondary axis pattern.
    SEC_PCT_RANGE = (0, 80)
    prim_y_at_max_pct = -SEC_PCT_RANGE[1] / 100.0 * AMOC_PI   # e.g. -19.47 Sv
    prim_y_at_min_pct = -SEC_PCT_RANGE[0] / 100.0 * AMOC_PI   # 0 Sv
    pri_ylim = ax_top.get_ylim()
    pri_pos = ax_top.get_position()
    frac_low = (max(prim_y_at_max_pct, pri_ylim[0]) - pri_ylim[0]) / \
               (pri_ylim[1] - pri_ylim[0])
    frac_high = (min(prim_y_at_min_pct, pri_ylim[1]) - pri_ylim[0]) / \
                (pri_ylim[1] - pri_ylim[0])
    new_y0 = pri_pos.y0 + frac_low * pri_pos.height
    new_height = (frac_high - frac_low) * pri_pos.height
    secax = fig.add_axes([pri_pos.x1 + 0.002, new_y0, 0.012, new_height])
    secax.set_ylim(SEC_PCT_RANGE[1], SEC_PCT_RANGE[0])   # inverted: max % at bottom
    secax.set_yticks(np.arange(SEC_PCT_RANGE[0], SEC_PCT_RANGE[1] + 1, 20))
    secax.set_yticklabels([f'{t:.0f}' for t in
                           np.arange(SEC_PCT_RANGE[0], SEC_PCT_RANGE[1] + 1, 20)])
    secax.yaxis.tick_right(); secax.yaxis.set_label_position('right')
    secax.set_ylabel('AMOC weakening w.r.t. 1850–1899 [%]', labelpad=5)
    for s in ('top', 'left', 'bottom'):
        secax.spines[s].set_visible(False)
    secax.tick_params(axis='x', which='both', bottom=False, top=False,
                       labelbottom=False)
    secax.set_facecolor('none')

    # ----- panel b: EU-mean tas anomaly -----
    for r in MEMBERS:
        ax_bot.plot(years, tas_eu_smooth.sel(realiz=r).values,
                    color=member_color[r], lw=0.9, alpha=0.85)
        ax_bot.plot(years[end_idx] + end_x_offset,
                    tas_eu_smooth.sel(realiz=r).values[end_idx],
                    marker=member_marker[r], color=member_color[r],
                    markersize=3, linestyle='None', clip_on=False)
    ax_bot.plot(years, tas_eu_comp_smooth.values,
                color=composite_color, lw=2.2)
    # ax_bot.plot(years[end_idx] + end_x_offset,
    #             tas_eu_comp_smooth.values[end_idx],
    #             marker='o', color=composite_color, markersize=8,
    #             linestyle='None', clip_on=False)
    ax_bot.axhline(0, color=fg, lw=1., ls='--', alpha=0.5)
    ax_bot.axvline(2014.5, color=fg, lw=0.6, ls=':', alpha=0.6)
    season_tag_y = '' if season == '' else f' ({season.upper()})'
    ax_bot.set_ylabel(rf'$\Delta\mathrm{{T}}_\mathrm{{EU}}${season_tag_y} [°C]'
                      '\nw.r.t. 1850–1899')
    ax_bot.set_xlabel('')
    ax_bot.set_xticks(np.arange(1900, 2501, 100))
    ax_bot.set_xlim(years[0], years[-1])

    # ----- panel c: regression -----
    amoc_dev = amoc_anom - amoc_comp
    tas_dev = tas_eu - tas_eu_comp

    # Apply rolling mean before fitting (matches regression_plot convention)
    amoc_dev_s = amoc_dev.rolling(time=WINDOW, center=True).mean()
    tas_dev_s = tas_dev.rolling(time=WINDOW, center=True).mean()

    model, x_fit, y_fit, _ = fit_member_minus_composite(
        amoc_dev_s, tas_dev_s, members=NON_COMPOSITE, time_slice=REG_TIME)

    CENTURIES = [('2015', '2100'), ('2101', '2200'), ('2201', '2300'),
                 ('2301', '2400'), ('2401', '2500'), ('2101', '2300')]
    century_colors = cm.plasma(np.linspace(0.1, 0.85, len(CENTURIES)))
    century_fits = []
    for win, ccol in zip(CENTURIES, century_colors):
        m, xc, yc, _ = fit_member_minus_composite(
            amoc_dev_s, tas_dev_s, members=NON_COMPOSITE, time_slice=win)
        century_fits.append((win, ccol, m, xc, yc))

    # Scatter: 7 non-composite members, each with its own marker AND
    # viridis colour so identity is consistent with the time-series
    # panels above. Composite members (r3, r4, r10) contribute ~0 by
    # construction and are omitted.
    for r in NON_COMPOSITE:
        x = amoc_dev_s.sel(realiz=r, time=slice(*REG_TIME)).values
        y = tas_dev_s.sel(realiz=r, time=slice(*REG_TIME)).values
        ax_reg.scatter(x, y, s=15, color=member_color[r],
                       marker=member_marker[r], alpha=0.8,
                       edgecolors='none', clip_on=False)

    # Century-specific regression lines (thin, plasma-coloured).
    for win, ccol, m, xc, yc in century_fits:
        xx = np.linspace(np.nanmin(xc), np.nanmax(xc), 100)
        yy = m.params[0] + m.params[1] * xx
        ax_reg.plot(xx, yy, color=ccol, lw=1.2)

    # Main fit line (full REG_TIME, thick).
    xx = np.linspace(np.nanmin(x_fit), np.nanmax(x_fit), 100)
    yy = model.params[0] + model.params[1] * xx
    ax_reg.plot(xx, yy, color=fg, lw=2.2)

    ax_reg.axhline(0, color=fg, lw=1., ls='--', alpha=0.6)
    ax_reg.axvline(0, color=fg, lw=1., ls='--', alpha=0.6)
    ax_reg.set_xlabel(r'$\Delta\mathrm{AMOC}$ = member − composite [Sv]')
    ax_reg.set_ylabel(rf'$\Delta\mathrm{{T}}_\mathrm{{EU}}${season_tag_y} = member − composite [°C]')

    # ----- Apply Fig1 spine convention to all three panels -----
    _clean_axes(ax_top, hide_bottom=True)
    _clean_axes(ax_bot)
    _clean_axes(ax_reg)

    # ----- Member-identity legend in the top-right slot -----
    # Marker handles for the 10 ensemble members, plus a line handle for
    # the composite mean (used in panels a and b).
    member_handles = [Line2D([0], [0], color=member_color[r],
                             marker=member_marker[r], linestyle='None',
                             markersize=7,
                             label=r + ('  (composite)' if r in COMPOSITE else ''))
                      for r in MEMBERS]
    composite_handle = Line2D([0], [0], color=composite_color, lw=2.2,
                              label=f'composite mean (mean of r3, r4, r10)')
    legend = ax_reg.legend(handles=member_handles + [composite_handle],
                           ncol=2, frameon=False,
                           bbox_to_anchor=(-0.04, 2.15), loc='upper left',
                           fontsize=9, handletextpad=1.0, columnspacing=2)

    # ----- Regression-equation text block (separate from the legend) -----
    # Six rows: full 2015–2500 (bold, fg colour) + 5 century fits
    # (coloured to match the lines drawn in panel c). Sits between the
    # legend and the top of panel c, mirroring Fig1's `equation_y_pos`
    # convention.
    equation_y_pos, equation_y_spacing = 1.50, 0.075
    ax_reg.text(0.0, equation_y_pos + equation_y_spacing,
                'Regression results:',
                transform=ax_reg.transAxes, fontsize=10, va='top',
                color=fg, fontweight='bold')
    ax_reg.text(0.0, equation_y_pos,
                (f'{REG_TIME[0]}–{REG_TIME[1]}:  '
                 f'y = {model.params[1]:+.3f}·x {model.params[0]:+.3f},  '
                 f'R² = {model.rsquared:.2f}  '
                 f'(SE {model.bse[1]:.3f})'),
                transform=ax_reg.transAxes, fontsize=9, va='top',
                color=fg, fontweight='bold')
    for i, (win, ccol, m, _, _) in enumerate(century_fits, start=1):
        ax_reg.text(0.0, equation_y_pos - i * equation_y_spacing,
                    (f'{win[0]}–{win[1]}:  '
                     f'y = {m.params[1]:+.3f}·x {m.params[0]:+.3f},  '
                     f'R² = {m.rsquared:.2f}'),
                    transform=ax_reg.transAxes, fontsize=9, va='top',
                    color=ccol)

    # ----- Subplot labels -----
    for i, ax in enumerate([ax_top, ax_bot, ax_reg]):
        bb = ax.get_position()
        fig.text(bb.x0 - 0.03, bb.y1, 'abc'[i], fontsize=14,
                 ha='right', va='center', fontweight='bold', color=fg)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = (f'{PLOTS_DIR}/FigSupp_giss_sim_regression_plotbg-{plot_bg}'
           f'_window-{WINDOW}_season-{season or "annual"}')
    fig.savefig(out + '.png', dpi=200, bbox_inches='tight',
                transparent=(plot_bg == 'black'))
    fig.savefig(out + '.pdf', dpi=400, bbox_inches='tight',
                transparent=(plot_bg == 'black'))
    print(f'wrote {out}.png/.pdf')
    return fig, out


########################################
# %%
# RUN

def dump_regression_results(amoc_dev_s, tas_dev_s, members):
    """Emit every regression value shown in the figure to (a) stdout in a
    human-readable table, and (b) a sidecar JSON next to the figure.

    Rationale: the figure should never be the only source of any numerical
    result. Quoting values from a rendered PNG legend is unreliable.
    Always cite this dump instead.
    """
    windows = [('2015', '2100'), ('2101', '2200'), ('2201', '2300'),
               ('2301', '2400'), ('2401', '2500'), ('2101', '2300'), REG_TIME]
    rows = []
    for win in windows:
        m, x, y, cl = fit_member_minus_composite(
            amoc_dev_s, tas_dev_s, members=members, time_slice=win)
        rows.append({
            'window':              f'{win[0]}-{win[1]}',
            'slope_K_per_Sv':      float(m.params[1]),
            'intercept_K':         float(m.params[0]),
            'slope_se_hac':        float(m.bse[1]),
            'rsquared':            float(m.rsquared),
            'n_obs':               int(m.nobs),
            'n_members':           int(len(set(cl.tolist()))),
        })
    print('\nRegression results (non-composite members, '
          f'{WINDOW}-yr rolling, HAC SE maxlags={WINDOW - 1}):')
    print(f'  {"window":>11}  {"slope":>8}  {"SE":>7}  {"int":>8}  '
          f'{"R²":>6}  {"n":>5}')
    for r in rows:
        print(f'  {r["window"]:>11}  {r["slope_K_per_Sv"]:+.4f}  '
              f'{r["slope_se_hac"]:.4f}  {r["intercept_K"]:+.4f}  '
              f'{r["rsquared"]:.3f}  {r["n_obs"]:>5}')
    out = f'{PLOTS_DIR}/FigSupp_giss_sim_regression_results_window-{WINDOW}.json'
    with open(out, 'w') as f:
        json.dump({'rows': rows, 'composite': COMPOSITE,
                   'non_composite': list(members),
                   'amoc_pi_Sv': AMOC_PI}, f, indent=2)
    print(f'wrote sidecar {out}')
    return rows


########################################
# %%
# RUN — driver block

if __name__ == '__main__':
    # ----- Fig1-style figure (1-D EU regression) -----
    fig, model = make_figure(
        amoc_smooth, tas_eu_smooth, amoc_comp_smooth, tas_eu_comp_smooth,
        amoc_anom, tas_eu, plot_bg='white', season=SEASON,
        )
    # Re-derive the deviation arrays once outside make_figure() and dump.
    # Identical formulae to those inside make_figure(); divergence = bug.
    amoc_dev_s = (amoc_anom - amoc_comp).rolling(time=WINDOW, center=True).mean()
    tas_dev_s = (tas_eu - tas_eu_comp).rolling(time=WINDOW, center=True).mean()
    dump_regression_results(amoc_dev_s, tas_dev_s, members=NON_COMPOSITE)


# %%
