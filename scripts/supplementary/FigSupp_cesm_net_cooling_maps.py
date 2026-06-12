########################################
# %%
# LOAD PACKAGES

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
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
    bm_data, vwb_data, boot_data, liu_data = functions.get_other_studies_data(masks)
    reg_ds_cesm = functions.get_cesm_reg_ds(recompute=False)

########################################
# %%
# FIGURE FUNCTION
# MAKE FIG. S15: CESM2 REGRESSION AND NET COOLING POINT MAPS

def make_figure(multi_model_dict, masks, boot_data, reg_ds_cesm, plot_bg='white', show_cmip_range_contours=True, weakening_only=True, future_window=None):
    # Sibling of scripts/Fig2.py (CESM analogue). The weakening colorbar and its
    # cbar_x positioning mirror Fig2's weakening_only handling — when Fig2's
    # colorbar / weakening_only logic changes, mirror it here.
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})
    # CMIP6 weakening envelope for the chosen future window (canonical global
    # when future_window is None); shared by the maps and the colorbar bars.
    _ext = functions.get_amoc_extent(future_window)
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    # Create figure with gridspec: top row for regression, bottom two for maps
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.2, 1, 1], hspace=0.35, wspace=-0.3)

    # Top row: CESM regressions plot (spans all 3 columns)
    ax_reg = fig.add_subplot(gs[0, :])

    # Call cesm_regressions with ext_ax (settings from multi_model_analysis.py lines 37-50)
    functions.cesm_regressions(
        boot_data, multi_model_dict, masks,
        ssp='all',
        lat=None, lon=None,
        region='EU',
        window=10,
        regressions=True,
        combined_reg=True,
        add_combined_intercept=True,
        low_ylim=-5,
        xlim=50,
        hosmip_cesm=False,
        plot_bg=plot_bg,
        ext_ax=ax_reg,
        equation_y_pos=0.34,
        equation_y_spacing=0.12  # wider: simple_eqs render °C/10% as a stacked fraction
    )

    ax_reg.text(0.02, 0.94, 'a)', transform=ax_reg.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    # Bottom two rows: Net cooling point maps
    panel_labels = ['b)', 'c)', 'd)', 'e)', 'f)', 'g)']
    ncols = 3

    map_axes = []
    for row in range(2):
        for col in range(ncols):
            ax = fig.add_subplot(gs[row + 1, col], projection=ccrs.Robinson(central_longitude=8))
            map_axes.append(ax)

    mappables = []
    j = 0
    for T_ref in ['pd', 'pi']:
        for i, ssp_scenario in enumerate(functions.ssps):
            ax = map_axes[3*j + i]
            mappable = functions.net_cooling_point_map(reg_ds_cesm, ssp_scenario, season='', T_ref=T_ref, ext_ax=ax, plot_bg=plot_bg, title=False, custom_cmap=None, weakening_primary=True, show_cmip_range_contours=show_cmip_range_contours, amoc_extent=_ext)
            mappables.append(mappable)
            ax.text(0.38, -0.03, panel_labels[3*j+i],
                    transform=ax.transAxes, fontsize=12, ha='right', va='top',
                    color='black' if not plot_bg=='black' else 'white', fontweight='bold')
            ax.text(0.4, -0.03, functions.ssp_labels[ssp_scenario],
                    transform=ax.transAxes, fontsize=12, ha='left', va='top',
                    color=functions.hosing_colors[ssp_scenario]['ge'], fontweight='bold')
        j += 1

    # Adjust top axes to align with maps below
    map_left = map_axes[0].get_position().x0
    map_right = map_axes[2].get_position().x1
    reg_pos = ax_reg.get_position()
    ax_reg.set_position([map_left, reg_pos.y0, map_right - map_left, reg_pos.height])

    # Row labels for maps
    row_labels = [
        f"Net-cooling AMOC weakening w.r.t. annual mean present-day temperatures ({functions.PD_LABEL})",
        "Net-cooling AMOC weakening w.r.t. annual mean preindustrial temperatures (1850-1899)",
    ]

    left = map_axes[0].get_position().x0
    for i in range(2):
        row_top = map_axes[i*ncols].get_position().y1
        fig.text(left, row_top+0.01, row_labels[i], va='center', ha='left', fontsize=12, fontweight='bold',
                 color='black' if not plot_bg=='black' else 'white')

    # Colorbar
    right = map_axes[-1].get_position().x1
    cbar_x = right + (0.03 if weakening_only else 0.07)
    cbar_width = 0.02
    bottom = map_axes[-1].get_position().y0
    cbar_y = bottom
    cbar_height = 0.52

    cax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])

    cbar = fig.colorbar(mappables[0], cax=cax, orientation='vertical')
    text_color = 'black' if plot_bg != 'black' else 'white'

    # CESM-aware conversions: % weakening / Sv labels must use CESM's PI
    # baseline (~17.96 Sv), not MPI's (19.02 Sv hardcoded in
    # functions.convert_strength_to_weakening).
    cesm_amoc_pi = float(reg_ds_cesm.AMOC_pi)
    def cesm_strength_to_weakening(strength):
        return (cesm_amoc_pi - strength) / cesm_amoc_pi * 100
    def cesm_weakening_to_strength(weakening):
        return cesm_amoc_pi * (1 - weakening / 100)

    # Weakening % is the primary axis (governs the discrete colour bins,
    # net_cooling_point_map was called with weakening_primary=True).
    cbar.ax.invert_yaxis()
    cbar.set_ticks(list(range(0, 101, 10)))
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0f}%"))
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.set_label('AMOC weakening', fontsize=12, labelpad=5, loc='bottom', color=text_color)
    if not weakening_only:
        secax = cbar.ax.secondary_yaxis('left', functions=(cesm_weakening_to_strength, cesm_strength_to_weakening))
        secax.set_ylabel('AMOC strength', fontsize=12, labelpad=0, loc='bottom', color=text_color)
        secax.set_yticks(list(range(0, 17, 2)))  # cap at 16 Sv; CESM PI (~17.96 Sv) defines the 0% end
        secax.set_yticklabels([f'{s} Sv' for s in range(0, 17, 2)])

    range_positions = [2.9, 4.1, 5.3]
    for i, ssp_scenario in enumerate(functions.ssps):
        # Weakening-primary axis: positions are direct fractions (axes-y = 1 - w/100
        # after invert_yaxis). Identical convention to Fig2's weakening_primary branch.
        rect_y = 1 - _ext[ssp_scenario][0] / 100
        rect_height = (_ext[ssp_scenario][0] - _ext[ssp_scenario][1]) / 100
        text_y = 1 - (_ext[ssp_scenario][0] + _ext[ssp_scenario][1]) / 200

        rect = Rectangle(
            (range_positions[i], rect_y),
            1.,
            rect_height,
            transform=cbar.ax.transAxes,
            color=functions.hosing_colors[ssp_scenario]['ge'],
            alpha=1.0,
            clip_on=False
        )
        cbar.ax.add_patch(rect)

        # CESM2 projected 2090-2100 weakening as horizontal tick across the bar
        # (lighter sibling of the bar colour, à la Fig3).
        cesm_strength = float(reg_ds_cesm.AMOC_future.sel(scenar=ssp_scenario))
        cesm_weakening = cesm_strength_to_weakening(cesm_strength)
        cesm_y = 1 - cesm_weakening / 100
        base_rgb = plt.matplotlib.colors.to_rgb(functions.hosing_colors[ssp_scenario]['ge'])
        h, l, s = colorsys.rgb_to_hls(*base_rgb)
        line_color = colorsys.hls_to_rgb(h, min(1, l + 0.30), min(1, s * 1.3))
        cbar.ax.plot(
            [range_positions[i] - 0.08, range_positions[i] + 1.08], [cesm_y, cesm_y],
            color=line_color,
            linewidth=2, transform=cbar.ax.transAxes, clip_on=False, zorder=10
        )

        bar_text = f"{functions.ssp_labels[ssp_scenario]} 2100 CMIP6 range"
        cbar.ax.text(
            range_positions[i]+0.6,
            text_y,
            bar_text,
            color='white',
            fontsize=9,
            rotation=90,
            va='center',
            ha='center',
            transform=cbar.ax.transAxes,
            zorder=12
        )

    # Add white borders on top of each map subplot to cover colored edge artifacts in PDF
    border_width = 0.0005
    border_color = 'white' if plot_bg == 'white' else '#191919'
    for i, ax in enumerate(map_axes):
        if hasattr(ax, 'projection'):
            bbox = ax.get_position()
            # Top and bottom borders
            fig.add_artist(plt.Rectangle((bbox.x0, bbox.y1 - border_width), bbox.width, border_width,
                                         transform=fig.transFigure, color=border_color, zorder=1000, clip_on=False))
            fig.add_artist(plt.Rectangle((bbox.x0, bbox.y0), bbox.width, border_width,
                                         transform=fig.transFigure, color=border_color, zorder=1000, clip_on=False))
            # Left border on all
            fig.add_artist(plt.Rectangle((bbox.x0, bbox.y0), border_width, bbox.height,
                                         transform=fig.transFigure, color=border_color, zorder=1000, clip_on=False))
            # Right border only on rightmost column
            if i in [2, 5]:
                fig.add_artist(plt.Rectangle((bbox.x1 - border_width, bbox.y0), border_width, bbox.height,
                                             transform=fig.transFigure, color=border_color, zorder=1000, clip_on=False))

    savepath = f'../plots/FigSupp_cesm_net_cooling_maps_plotbg-{plot_bg}_cmiprange-{show_cmip_range_contours}_weakonly-{weakening_only}{functions.fw_suffix(future_window)}'
    fig.savefig(savepath + '.png', bbox_inches='tight', transparent=True if plot_bg=='black' else False, dpi=200)
    fig.savefig(savepath + '.pdf', bbox_inches='tight', dpi=400, transparent=True if plot_bg=='black' else False)

    return fig, savepath

########################################
# %%
# RUN

if __name__ == '__main__':
    plot_bg = 'white'  # 'white' or 'black'
    show_cmip_range_contours = True  # overlay CMIP6 AMOC-weakening range edges as black-with-white-halo contours on the scenario maps
    weakening_only = True  # True (canonical, mirrors Fig2): drop the Sv secondary axis from the AMOC-weakening colorbar
    future_window = None  # None -> canonical 2091-2100; e.g. ('2081','2090') for a sensitivity variant

    if future_window is not None:
        reg_ds_cesm = functions.get_cesm_reg_ds(recompute=False, future_window=future_window)
    fig, savepath = make_figure(multi_model_dict, masks, boot_data, reg_ds_cesm, plot_bg=plot_bg, show_cmip_range_contours=show_cmip_range_contours, weakening_only=weakening_only, future_window=future_window)

# %%
