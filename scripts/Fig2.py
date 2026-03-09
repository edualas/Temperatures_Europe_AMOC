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
import functions
importlib.reload(functions)

########################################
#%%
# LOAD REGRESSION DATASETS

if __name__ == '__main__':
    reg_ds_mpi = functions.load_regression_ds_mpi()

########################################
#%%
# FIGURE FUNCTION
# MAKE FIG. 2: MULTI-PANEL MAPS
# PRODUCES FIG. S7 FOR WINTER (DJF) and FIG. S8 FOR SUMMER (JJA)

def make_figure(reg_ds_mpi, only_pi=False, plot_season='', plot_bg='white', weakening_primary=True):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3*(nrows-1)), subplot_kw={'projection': ccrs.Robinson(central_longitude=8)}, constrained_layout=False, gridspec_kw=gridspec_kw)
    axes = axes.flatten()

    for i in range(ncols):
        ax = axes[3 + i]
        ax.set_visible(False)
        ax.remove()

    plt.subplots_adjust(hspace=0.3, wspace=-0.4, top=0.95, bottom=0.05, left=0.05, right=0.95)

    # truncated colormap and values
    v_max = 0.7
    trunc_cmap = functions.truncate_colormap(custom_cmap, maxval=(v_max/1.5))
    levels = np.linspace(0.0, v_max, int(10*v_max)+1)
    norm = BoundaryNorm(levels, ncolors=trunc_cmap.N, clip=False)

    reg_mappables = []
    panel_axes = []
    for i, season in enumerate(['', 'jja', 'djf']):
        mappable = functions.plot_regression_coefficients(reg_ds_mpi, season=season, ext_ax=axes[i], plot_bg=plot_bg, custom_cmap=custom_cmap, colorbar_v_max=v_max)
        reg_mappables.append(mappable)
        panel_axes.append(axes[i])
        axes[i].text(0.5, -0.03, reg_plot_labels[i], transform=axes[i].transAxes, fontsize=12, ha='center', va='top', fontweight='bold', color='black' if not plot_bg=='black' else 'white')

    mappables = []
    j = 2
    for T_ref in ['pd', 'pi']:
        if only_pi and T_ref == 'pd':
            continue
        for i, ssp in enumerate(functions.ssps):
            mappable = functions.net_cooling_point_map(reg_ds_mpi, ssp, season=plot_season, T_ref=T_ref, ext_ax=axes[3*j+i], plot_bg=plot_bg, title=False, custom_cmap=None, weakening_primary=weakening_primary)
            mappables.append(mappable)
            panel_axes.append(axes[3*j+i])
            axes[3*j+i].text(
                0.5, -0.03, functions.ssp_labels[ssp],
                transform=axes[3*j+i].transAxes,
                fontsize=12, ha='center', va='top', color=functions.hosing_colors[ssp]['ge'], fontweight='bold'
            )
        j+=1

    row_labels = ["Scaling factors: local cooling as a function of additional AMOC weakening",
                  "",
                  "Net cooling points w.r.t. annual mean present-day temperatures (2000-2029)" if plot_season==''else f"Net cooling points w.r.t. {plot_season.upper()} mean present-day temperatures (2000-2029)",
                  "Net cooling points w.r.t. annual mean preindustrial temperatures (1850-1899)" if plot_season==''else f"Net cooling points w.r.t. {plot_season.upper()} mean preindustrial temperatures (1850-1899)",
                  ]

    left = axes[0].get_position().x0
    for i in [0, 2, 3]:
        if only_pi and i == 2:
            continue
        if only_pi and i == 3:
            row_top = axes[(i-1)*ncols].get_position().y1
            fig.text(left, row_top+0.01, row_labels[i], va='center', ha='left', fontsize=12, fontweight='bold', color='black' if not plot_bg=='black' else 'white')
            continue
        row_top = axes[i*ncols].get_position().y1
        fig.text(left, row_top+0.01, row_labels[i], va='center', ha='left', fontsize=12, fontweight='bold', color='black' if not plot_bg=='black' else 'white')

    for ax, label in zip(panel_axes, panel_labels):
        left = ax.get_position().x0
        top = ax.get_position().y1
        fig.text(left + 0.015, top-0.015, label, transform=fig.transFigure, fontsize=14, ha='right', va='center', fontweight='bold', color='black' if not plot_bg=='black' else 'white')

    right = axes[-1].get_position().x1
    cbar_x = right + 0.07  # x-position in figure coordinates
    cbar_width = 0.02
    top = axes[1].get_position().y0
    cbar1_y = top  # y-position for upper colorbar
    cbar1_height = 0.4 if only_pi else 0.23
    bottom = axes[-1].get_position().y0
    cbar2_y = bottom  # y-position for lower colorbar
    cbar2_height = 0.45 if only_pi else 0.55

    cax1 = fig.add_axes([cbar_x, cbar1_y, cbar_width, cbar1_height])
    cax2 = fig.add_axes([cbar_x, cbar2_y, cbar_width, cbar2_height])

    cbar1 = fig.colorbar(reg_mappables[2], cax=cax1, orientation='vertical')

    cbar1.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.1f}°C"))
    cbar1.set_label('Cooling per 10% additional\nAMOC weakening', fontsize=12, labelpad=5, loc='center', color='black' if not plot_bg=='black' else 'white')

    cbar2 = fig.colorbar(mappables[2], cax=cax2, orientation='vertical')
    text_color = 'black' if plot_bg != 'black' else 'white'

    if weakening_primary:
        cbar2.ax.invert_yaxis()
        cbar2.set_ticks(list(range(0, 101, 10)))
        cbar2.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0f}%"))
        cbar2.ax.yaxis.set_ticks_position('right')
        cbar2.ax.yaxis.set_label_position('right')
        cbar2.set_label('AMOC weakening', fontsize=12, labelpad=5, loc='bottom', color=text_color)
        secax = cbar2.ax.secondary_yaxis('left', functions=(functions.convert_weakening_to_strength, functions.convert_strength_to_weakening))
        secax.set_ylabel('AMOC strength', fontsize=12, labelpad=0, loc='bottom', color=text_color)
        secax.set_yticks(list(range(0, 19, 2)))
        secax.set_yticklabels([f'{s} Sv' for s in range(0, 19, 2)])
    else:
        cbar2.set_label('AMOC strength', fontsize=12, labelpad=0, loc='bottom', color=text_color)
        cbar2.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0f} Sv"))
        cbar2.ax.yaxis.set_label_position('left')
        cbar2.ax.yaxis.set_ticks_position('left')
        secax = cbar2.ax.secondary_yaxis('right', functions=(functions.convert_strength_to_weakening, functions.convert_weakening_to_strength))
        secax.set_ylabel('AMOC weakening', fontsize=12, labelpad=5, loc='bottom', color=text_color)
        secax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        secax.set_yticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])

    upper_bound = 18
    range_positions = [2.9, 4.1, 5.3]
    for i, ssp in enumerate(functions.ssps):
        if weakening_primary:
            rect_y = 1 - functions.AMOC_extent[ssp][0] / 100
            rect_height = (functions.AMOC_extent[ssp][0] - functions.AMOC_extent[ssp][1]) / 100
        else:
            rect_y = (1 - functions.AMOC_extent[ssp][0]/100) / (1 - functions.convert_strength_to_weakening(upper_bound)/100)
            rect_height = ((functions.AMOC_extent[ssp][0]-functions.AMOC_extent[ssp][1])/100) / (1 - functions.convert_strength_to_weakening(upper_bound)/100)

        rect = Rectangle(
            (range_positions[i], rect_y),
            1.,                   # width of the patch
            rect_height, # height in normalized coords
            transform=cbar2.ax.transAxes,
            color=functions.hosing_colors[ssp]['ge'],
            alpha=1.0,
            clip_on=False
        )
        cbar2.ax.add_patch(rect)

        # MPI-ESM projected AMOC strength as horizontal line
        mpi_amoc_strength = float(reg_ds_mpi.AMOC_2090_2100.isel(season=0).sel(scenar=ssp))
        mpi_weakening = functions.convert_strength_to_weakening(mpi_amoc_strength)
        if weakening_primary:
            mpi_y = 1 - mpi_weakening / 100
        else:
            mpi_y = (1 - mpi_weakening/100) / (1 - functions.convert_strength_to_weakening(upper_bound)/100)
        base_rgb = plt.matplotlib.colors.to_rgb(functions.hosing_colors[ssp]['ge'])
        h, l, s = colorsys.rgb_to_hls(*base_rgb)
        line_color = colorsys.hls_to_rgb(h, max(0, l * 0.7), min(1, s * 1.3))
        cbar2.ax.plot(
            [range_positions[i] - 0.08, range_positions[i] + 1.08], [mpi_y, mpi_y],
            color=line_color,
            linewidth=2, transform=cbar2.ax.transAxes, clip_on=False, zorder=11
        )

        bar_text = f"{functions.ssp_labels[ssp]}" if only_pi else f"{functions.ssp_labels[ssp]} 2100 CMIP6 range"
        if weakening_primary:
            text_y = 1 - (functions.AMOC_extent[ssp][0] + functions.AMOC_extent[ssp][1]) / 200
        else:
            text_y = (1 - (functions.AMOC_extent[ssp][0] + functions.AMOC_extent[ssp][1]) / 200) / (1 - functions.convert_strength_to_weakening(upper_bound)/100)
        cbar2.ax.text(
            range_positions[i]+0.6,  # x position
            text_y,
            bar_text,
            color='white',
            fontsize=9,
            rotation=90,
            va='center',
            ha='center',
            transform=cbar2.ax.transAxes,
            zorder=12
        )

    # Add white borders on top of each subplot to cover colored edge artifacts in PDF
    border_width = 0.001  # Width of border in figure coordinates
    border_color = 'white' if plot_bg == 'white' else '#191919'
    for ax in fig.get_axes():
        if hasattr(ax, 'projection'):  # Only for map axes with projections
            bbox = ax.get_position()
            # Add thin rectangles at each edge with very high zorder
            # Top edge
            fig.add_artist(plt.Rectangle((bbox.x0, bbox.y1 - border_width), bbox.width, border_width,
                                         transform=fig.transFigure, color=border_color, zorder=1000, clip_on=False))
            # Bottom edge
            fig.add_artist(plt.Rectangle((bbox.x0, bbox.y0), bbox.width, border_width,
                                         transform=fig.transFigure, color=border_color, zorder=1000, clip_on=False))
            # Left edge
            fig.add_artist(plt.Rectangle((bbox.x0, bbox.y0), border_width, bbox.height,
                                         transform=fig.transFigure, color=border_color, zorder=1000, clip_on=False))
            # Right edge
            fig.add_artist(plt.Rectangle((bbox.x1 - border_width, bbox.y0), border_width, bbox.height,
                                         transform=fig.transFigure, color=border_color, zorder=1000, clip_on=False))

    savepath = f'../plots/Fig2_only_pi_{only_pi}_plotbg-{plot_bg}_season-{plot_season}_weakprim-{weakening_primary}'
    fig.savefig(savepath + '.png', bbox_inches='tight', transparent=True if plot_bg=='black' else False, dpi=200)
    fig.savefig(savepath + '.pdf', bbox_inches='tight', dpi=400, transparent=True if plot_bg=='black' else False)

    return fig, savepath

########################################
#%%
# RUN

if __name__ == '__main__':
    only_pi = False
    plot_bg = 'white'  # 'white' or 'black'
    plot_season = ''
    weakening_primary = True  # True: right axis shows 0–100% weakening as primary ticks

    fig, savepath = make_figure(reg_ds_mpi, only_pi=only_pi, plot_season=plot_season, plot_bg=plot_bg, weakening_primary=weakening_primary)

#%%
