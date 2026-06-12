"""GISS-E2-1-G analogue of scripts/Fig2.py — multi-panel scaling-factor and
net-cooling-point maps for the Romanou et al. 2023 long-run ensemble.

Layout (4 rows × 3 cols):
  row 0: scaling factors. Annual cell populated from the full-grid GISS
         regression cache (functions.giss_load_or_compute_gridded), rescaled
         via rescale_giss_coef_for_plotter so plot_regression_coefficients'
         internal AMOC_pi_MPI/10 multiplier yields °C per 10% GISS weakening.
         Seasonal cells (JJA, DJF) left empty — no DJF/JJA GISS regression yet.
  row 1: spacer.
  row 2: net cooling points T_ref='pd' (ssp126, ssp245, ssp370).
  row 3: net cooling points T_ref='pi' (ssp126, ssp245, ssp370).

Configurable:
  time_period : '2015-2500' or '2101-2300'. Default '2101-2300'.
"""
########################################
# %%
# LOAD PACKAGES

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle
from matplotlib import ticker
plt.rcParams.update({'font.size': 12})
import cartopy.crs as ccrs
import colorsys

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
    reg_ds_giss = functions.get_giss_reg_ds(recompute=False, masks=masks)

########################################
# %%
# FIGURE FUNCTION
# Mirrors Fig2.py with two key adaptations:
#   - top-row scaling-factor map driven by the full-grid GISS cache, rescaled
#     so the MPI plotter's AMOC_pi_MPI/10 multiplier yields GISS weakening
#     percent (see rescale_giss_coef_for_plotter below);
#   - bottom rows consume the cropped reg_ds_giss directly.

def rescale_giss_coef_for_plotter(reg_ds_giss, amoc_pi_giss,
                                  amoc_pi_mpi=functions.AMOC_pi_MPI,
                                  plot_v_max=None):
    """Rescale a GISS gridded regression dataset into a view that
    ``functions.plot_regression_coefficients`` can render.

    The plotter multiplies ``coef_ensmean`` by ``AMOC_pi_MPI/10`` internally; we
    pre-scale the GISS slope by ``AMOC_pi_GISS / AMOC_pi_MPI`` so that
    multiplication yields °C per 10% *GISS* weakening. If ``plot_v_max`` is
    given, the rescaled coefficient is clipped just below the colorbar ceiling
    so the contourf (``extend='neither'``) does not leave cells uncoloured.
    """
    def _wrap_lon_centred(ds):
        """0–360 lon → [-180, 180]-centred so cartopy contourf does not leave a
        gap at the Greenwich (0°) seam (the GISS cache stores lon ∈ [1.25,
        358.75], whose endpoints are not array-adjacent → a white line at 0°E)."""
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        return ds.sortby('lon')

    scale = amoc_pi_giss / amoc_pi_mpi
    coef = reg_ds_giss.slope_K_per_Sv * scale
    ste = reg_ds_giss.ste_K_per_Sv * scale
    if plot_v_max is not None:
        # Clip a half-bin below the ceiling: clipping exactly at the contourf
        # level boundary leaves degenerate (all-equal-neighbour) cells undrawn
        # — observed over Iceland for late-window fits.
        ceiling = plot_v_max * 10.0 / amoc_pi_mpi
        bin_width = (plot_v_max / (int(10 * plot_v_max))) * 10.0 / amoc_pi_mpi
        coef = coef.clip(max=ceiling - 0.5 * bin_width)
    out = reg_ds_giss.assign(coef_ensmean=coef, ste_ensmean=ste)
    return _wrap_lon_centred(out)


def make_figure(reg_ds_giss, masks, only_pi=False, plot_bg='white',
                weakening_primary=True, time_period='2101-2300',
                colorbar_v_max=1.5, show_cmip_range_contours=True,
                season='', weakening_only=True, future_window=None):
    # Sibling of scripts/Fig2.py: the weakening colorbar and its cbar_x
    # positioning mirror Fig2's weakening_only handling — when Fig2's colorbar /
    # weakening_only logic changes, mirror it here.
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})
    # CMIP6 weakening envelope for the chosen future window (canonical global
    # when future_window is None); shared by the maps and the colorbar bars.
    _ext = functions.get_amoc_extent(future_window)
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    custom_cmap = functions.get_custom_cmap()

    nrows = 3 if only_pi else 4
    ncols = 3

    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f']
    if not only_pi:
        panel_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    reg_plot_labels = ['Annual mean', 'Summer (JJA)', 'Winter (DJF)']

    gridspec_kw = {'height_ratios': [1, 0.05, 1]} if only_pi else {'height_ratios': [1, 0.1, 1, 1]}
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(12, 3 * (nrows - 1)),
        subplot_kw={'projection': ccrs.Robinson(central_longitude=8)},
        constrained_layout=False, gridspec_kw=gridspec_kw)
    axes = axes.flatten()

    # Hide the spacer row.
    for i in range(ncols):
        ax = axes[3 + i]
        ax.set_visible(False)
        ax.remove()

    plt.subplots_adjust(hspace=0.3, wspace=-0.4, top=0.95, bottom=0.05,
                        left=0.05, right=0.95)

    # ---- Top row: scaling factor (annual + DJF + JJA).
    # Load the full-grid bu1431 cache for the chosen time_period per season
    # and rescale so plot_regression_coefficients renders °C per 10% GISS
    # weakening. Seasonal caches written by get_giss_reg_ds(recompute=True)
    # since the 2026-05-26 (g) seasonal pipeline rollout.
    start, end = time_period.split('-')

    trunc_cmap = functions.truncate_colormap(custom_cmap,
                                             maxval=(colorbar_v_max / 1.5))
    # Restore the parent cmap's set_over so the extend triangle is distinct
    # from the last bin (truncate_colormap drops the parent's set_over).
    trunc_cmap.set_over((0.95, 0.3, 0.95, 1.0))

    reg_mappables = []
    panel_axes = []
    for i, _top_season in enumerate(['', 'jja', 'djf']):
        # giss_load_or_compute_gridded raises on cache miss without arrays;
        # the seasonal caches were built by get_giss_reg_ds(recompute=True).
        grid_full_season = functions.giss_load_or_compute_gridded(
            amoc_anom=None, tas_anom=None,
            time_slice=(start, end), season=_top_season)
        reg_view_season = rescale_giss_coef_for_plotter(
            grid_full_season, functions.AMOC_pi_GISS,
            plot_v_max=colorbar_v_max)
        # reg_view_season carries the requested season coord; let
        # plot_regression_coefficients .sel(season=...) on it directly.
        mappable = functions.plot_regression_coefficients(
            reg_view_season, season=_top_season, ext_ax=axes[i], plot_bg=plot_bg,
            custom_cmap=custom_cmap, colorbar_v_max=colorbar_v_max)
        reg_mappables.append(mappable)
        panel_axes.append(axes[i])
        axes[i].text(0.5, -0.03, reg_plot_labels[i],
                     transform=axes[i].transAxes, fontsize=12, ha='center',
                     va='top', fontweight='bold',
                     color='black' if not plot_bg == 'black' else 'white')

    # ---- Bottom rows: net cooling point maps (3 SSPs × pd, pi).
    # Select the chosen time_period from reg_ds_giss before each map call.
    reg_ds_giss_sel = reg_ds_giss.sel(time_period=time_period)

    mappables = []
    j = 2
    for T_ref in ['pd', 'pi']:
        if only_pi and T_ref == 'pd':
            continue
        for i, ssp in enumerate(functions.ssps):
            mappable = functions.net_cooling_point_map(
                reg_ds_giss_sel, ssp, season=season, T_ref=T_ref,
                ext_ax=axes[3 * j + i], plot_bg=plot_bg, title=False,
                custom_cmap=None, weakening_primary=weakening_primary,
                show_cmip_range_contours=show_cmip_range_contours, amoc_extent=_ext)
            mappables.append(mappable)
            panel_axes.append(axes[3 * j + i])
            axes[3 * j + i].text(
                0.5, -0.03, functions.ssp_labels[ssp],
                transform=axes[3 * j + i].transAxes,
                fontsize=12, ha='center', va='top',
                color=functions.hosing_colors[ssp]['ge'], fontweight='bold')
        j += 1

    season_tag = '' if season == '' else f' [{season.upper()}]'
    row_labels = [
        ("Scaling factors: local cooling as a function of additional AMOC "
         f"weakening (GISS-E2-1-G, {time_period})"),
        "",
        f"Net-cooling AMOC weakening w.r.t. {'annual mean ' if season == '' else season.upper() + ' '}present-day temperatures ({functions.PD_LABEL}){season_tag}",
        f"Net-cooling AMOC weakening w.r.t. {'annual mean ' if season == '' else season.upper() + ' '}preindustrial temperatures (1850–1899){season_tag}",
    ]

    left = axes[0].get_position().x0
    for i in [0, 2, 3]:
        if only_pi and i == 2:
            continue
        if only_pi and i == 3:
            row_top = axes[(i - 1) * ncols].get_position().y1
            fig.text(left, row_top + 0.01, row_labels[i], va='center',
                     ha='left', fontsize=12, fontweight='bold',
                     color='black' if not plot_bg == 'black' else 'white')
            continue
        row_top = axes[i * ncols].get_position().y1
        fig.text(left, row_top + 0.01, row_labels[i], va='center', ha='left',
                 fontsize=12, fontweight='bold',
                 color='black' if not plot_bg == 'black' else 'white')

    for ax, label in zip(panel_axes, panel_labels):
        left = ax.get_position().x0
        top = ax.get_position().y1
        fig.text(left + 0.015, top - 0.015, label, transform=fig.transFigure,
                 fontsize=14, ha='right', va='center', fontweight='bold',
                 color='black' if not plot_bg == 'black' else 'white')

    # ---- Colorbars (mirror Fig2.py geometry).
    right = axes[-1].get_position().x1
    cbar_x = right + (0.02 if weakening_only else 0.07)
    cbar_width = 0.02
    top = axes[1].get_position().y0
    cbar1_y = top
    cbar1_height = 0.4 if only_pi else 0.23
    bottom = axes[-1].get_position().y0
    cbar2_y = bottom
    cbar2_height = 0.45 if only_pi else 0.55

    cax1 = fig.add_axes([cbar_x, cbar1_y, cbar_width, cbar1_height])
    cax2 = fig.add_axes([cbar_x, cbar2_y, cbar_width, cbar2_height])

    # ----- Upper cbar: regression / scaling factor.
    # Built from a fresh ScalarMappable matching the BoundaryNorm + truncated
    # cmap used inside plot_regression_coefficients, so extend='max' lands and
    # the next-step set_over magenta is preserved.
    from matplotlib.cm import ScalarMappable
    levels1 = np.linspace(0.0, colorbar_v_max, int(10 * colorbar_v_max) + 1)
    norm1 = BoundaryNorm(levels1, ncolors=trunc_cmap.N, clip=False)
    sm1 = ScalarMappable(norm=norm1, cmap=trunc_cmap)
    sm1.set_array([])
    cbar1 = fig.colorbar(sm1, cax=cax1, orientation='vertical', extend='max')
    cbar1.set_ticks(np.arange(0, colorbar_v_max + 1e-6, 0.3))
    cbar1.ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{x:.1f}°C"))
    cbar1.set_label('Cooling per 10% additional\nAMOC weakening (GISS)',
                    fontsize=12, labelpad=5, loc='center',
                    color='black' if not plot_bg == 'black' else 'white')

    # ----- Lower cbar: AMOC weakening / strength.
    # Net cooling maps use GISS-specific AMOC_pi, so the strength axis here
    # is GISS-anchored. The convert_*_to_* helpers in functions.py hardcode
    # AMOC_pi_MPI; for the GISS panels we mirror Fig2's geometry but anchor
    # the strength conversion to AMOC_pi_GISS via local lambdas.
    cbar2 = fig.colorbar(mappables[-1], cax=cax2, orientation='vertical')
    text_color = 'black' if plot_bg != 'black' else 'white'
    api = functions.AMOC_pi_GISS

    def _w_to_s(w):
        return -w / 100 * api + api

    def _s_to_w(s):
        return (api - s) / api * 100

    if weakening_primary:
        cbar2.ax.invert_yaxis()
        cbar2.set_ticks(list(range(0, 101, 10)))
        cbar2.ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"{x:.0f}%"))
        cbar2.ax.yaxis.set_ticks_position('right')
        cbar2.ax.yaxis.set_label_position('right')
        cbar2.set_label('AMOC weakening', fontsize=12, labelpad=5,
                        loc='bottom', color=text_color)
        if not weakening_only:
            secax = cbar2.ax.secondary_yaxis('left', functions=(_w_to_s, _s_to_w))
            secax.set_ylabel('AMOC strength (GISS)', fontsize=12, labelpad=0,
                             loc='bottom', color=text_color)
            # AMOC_pi_GISS ≈ 24.3 Sv → step every 4 Sv up to 24
            secax.set_yticks(list(range(0, 25, 4)))
            secax.set_yticklabels([f'{s} Sv' for s in range(0, 25, 4)])
    else:
        cbar2.set_label('AMOC strength (GISS)', fontsize=12, labelpad=0,
                        loc='bottom', color=text_color)
        cbar2.ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"{x:.0f} Sv"))
        cbar2.ax.yaxis.set_label_position('left')
        cbar2.ax.yaxis.set_ticks_position('left')
        if not weakening_only:
            secax = cbar2.ax.secondary_yaxis('right',
                                             functions=(_s_to_w, _w_to_s))
            secax.set_ylabel('AMOC weakening', fontsize=12, labelpad=5,
                             loc='bottom', color=text_color)
            secax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            secax.set_yticklabels([f'{x}%' for x in range(10, 101, 10)])

    upper_bound = 18
    range_positions = [2.9, 4.1, 5.3]
    for i, ssp in enumerate(functions.ssps):
        if weakening_primary:
            rect_y = 1 - _ext[ssp][0] / 100
            rect_height = (_ext[ssp][0]
                           - _ext[ssp][1]) / 100
        else:
            rect_y = ((1 - _ext[ssp][0] / 100)
                      / (1 - functions.convert_strength_to_weakening(upper_bound) / 100))
            rect_height = (((_ext[ssp][0]
                             - _ext[ssp][1]) / 100)
                           / (1 - functions.convert_strength_to_weakening(upper_bound) / 100))

        rect = Rectangle(
            (range_positions[i], rect_y),
            1.,
            rect_height,
            transform=cbar2.ax.transAxes,
            color=functions.hosing_colors[ssp]['ge'],
            alpha=1.0, clip_on=False)
        cbar2.ax.add_patch(rect)

        # GISS-projected AMOC strength as horizontal tick.
        giss_amoc_strength = float(
            reg_ds_giss_sel.AMOC_future.sel(scenar=ssp))
        giss_weakening = _s_to_w(giss_amoc_strength)
        if weakening_primary:
            mpi_y = 1 - giss_weakening / 100
        else:
            mpi_y = ((1 - giss_weakening / 100)
                     / (1 - functions.convert_strength_to_weakening(upper_bound) / 100))
        base_rgb = plt.matplotlib.colors.to_rgb(functions.hosing_colors[ssp]['ge'])
        h, l, s = colorsys.rgb_to_hls(*base_rgb)
        line_color = colorsys.hls_to_rgb(h, max(0, l * 0.7), min(1, s * 1.3))
        cbar2.ax.plot(
            [range_positions[i] - 0.08, range_positions[i] + 1.08],
            [mpi_y, mpi_y],
            color=line_color, linewidth=2, transform=cbar2.ax.transAxes,
            clip_on=False, zorder=11)

        bar_text = (f"{functions.ssp_labels[ssp]}" if only_pi
                    else f"{functions.ssp_labels[ssp]} 2100 CMIP6 range")
        if weakening_primary:
            text_y = 1 - (_ext[ssp][0]
                          + _ext[ssp][1]) / 200
        else:
            text_y = ((1 - (_ext[ssp][0]
                            + _ext[ssp][1]) / 200)
                      / (1 - functions.convert_strength_to_weakening(upper_bound) / 100))
        cbar2.ax.text(
            range_positions[i] + 0.6, text_y, bar_text,
            color='white', fontsize=9, rotation=90, va='center', ha='center',
            transform=cbar2.ax.transAxes, zorder=12)

    # Edge cleanup (mirror Fig2.py).
    border_width = 0.001
    border_color = 'white' if plot_bg == 'white' else '#191919'
    for ax in fig.get_axes():
        if hasattr(ax, 'projection'):
            bbox = ax.get_position()
            for x0, y0, w, h in [
                (bbox.x0, bbox.y1 - border_width, bbox.width, border_width),
                (bbox.x0, bbox.y0, bbox.width, border_width),
                (bbox.x0, bbox.y0, border_width, bbox.height),
                (bbox.x1 - border_width, bbox.y0, border_width, bbox.height),
            ]:
                fig.add_artist(plt.Rectangle(
                    (x0, y0), w, h, transform=fig.transFigure,
                    color=border_color, zorder=1000, clip_on=False))

    savepath = (f'../plots/FigSupp_giss_net_cooling_maps_onlypi-{only_pi}'
                f'_plotbg-{plot_bg}_period-{time_period}'
                f'_weakprim-{weakening_primary}'
                f'_cmiprange-{show_cmip_range_contours}'
                f'_weakonly-{weakening_only}'
                f"_season-{season or 'annual'}"
                f'{functions.fw_suffix(future_window)}')
    fig.savefig(savepath + '.png', bbox_inches='tight', dpi=200,
                transparent=(plot_bg == 'black'))
    fig.savefig(savepath + '.pdf', bbox_inches='tight', dpi=400,
                transparent=(plot_bg == 'black'))
    print(f'wrote {savepath}.png/.pdf')

    return fig, savepath


########################################
# %%
# RUN

if __name__ == '__main__':
    only_pi = False
    plot_bg = 'white'
    weakening_primary = True
    time_period = '2101-2300'
    show_cmip_range_contours = True  # overlay CMIP6 AMOC-weakening range edges as black-with-white-halo contours on the scenario maps
    season = ''
    weakening_only = True  # True (canonical, mirrors Fig2): drop the Sv secondary axis from the AMOC-weakening colorbar
    future_window = None  # None -> canonical 2091-2100; e.g. ('2081','2090') for a sensitivity variant

    if future_window is not None:
        reg_ds_giss = functions.get_giss_reg_ds(recompute=False, masks=masks, future_window=future_window)
    fig, savepath = make_figure(
        reg_ds_giss, masks, only_pi=only_pi, plot_bg=plot_bg,
        weakening_primary=weakening_primary, time_period=time_period,
        show_cmip_range_contours=show_cmip_range_contours, season=season,
        weakening_only=weakening_only, future_window=future_window)

# %%
