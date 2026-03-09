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

import importlib
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

def make_figure(multi_model_dict, masks, boot_data, reg_ds_cesm, plot_bg='white'):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})
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
        equation_y_pos=0.3,
        equation_y_spacing=0.08
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
            mappable = functions.net_cooling_point_map(reg_ds_cesm, ssp_scenario, season='', T_ref=T_ref, ext_ax=ax, plot_bg=plot_bg, title=False, custom_cmap=None)
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
        "Net cooling points w.r.t. annual mean present-day temperatures (2000-2029)",
        "Net cooling points w.r.t. annual mean preindustrial temperatures (1850-1899)",
    ]

    left = map_axes[0].get_position().x0
    for i in range(2):
        row_top = map_axes[i*ncols].get_position().y1
        fig.text(left, row_top+0.01, row_labels[i], va='center', ha='left', fontsize=12, fontweight='bold',
                 color='black' if not plot_bg=='black' else 'white')

    # Colorbar
    right = map_axes[-1].get_position().x1
    cbar_x = right + 0.07
    cbar_width = 0.02
    bottom = map_axes[-1].get_position().y0
    cbar_y = bottom
    cbar_height = 0.52

    cax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])

    cbar = fig.colorbar(mappables[0], cax=cax, orientation='vertical')

    cbar.set_label('AMOC strength', fontsize=12, labelpad=0, loc='bottom', color='black' if not plot_bg=='black' else 'white')
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0f} Sv"))
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.set_ticks_position('left')

    secax = cbar.ax.secondary_yaxis('right', functions=(functions.convert_strength_to_weakening, functions.convert_weakening_to_strength))
    secax.set_ylabel('AMOC weakening', fontsize=12, labelpad=5, loc='bottom', color='black' if not plot_bg=='black' else 'white')
    secax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    secax.set_yticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])

    upper_bound = 18
    range_positions = [2.9, 4.1, 5.3]
    for i, ssp_scenario in enumerate(functions.ssps):
        rect = Rectangle(
            (range_positions[i], (1 - functions.AMOC_extent[ssp_scenario][0]/100) / (1 - functions.convert_strength_to_weakening(upper_bound)/100)),
            1.,
            (((functions.AMOC_extent[ssp_scenario][0]-functions.AMOC_extent[ssp_scenario][1])/100) / (1 - functions.convert_strength_to_weakening(upper_bound)/100)),
            transform=cbar.ax.transAxes,
            color=functions.hosing_colors[ssp_scenario]['ge'],
            alpha=1.0,
            clip_on=False
        )
        cbar.ax.add_patch(rect)

        bar_text = f"{functions.ssp_labels[ssp_scenario]} 2100 CMIP6 range"
        cbar.ax.text(
            range_positions[i]+0.6,
            (1 - (functions.AMOC_extent[ssp_scenario][0] + functions.AMOC_extent[ssp_scenario][1]) / 200) / (1 - functions.convert_strength_to_weakening(upper_bound)/100),
            bar_text,
            color='white',
            fontsize=9,
            rotation=90,
            va='center',
            ha='center',
            transform=cbar.ax.transAxes,
            zorder=10
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

    savepath = f'../plots/FigS15_cesm_plotbg-{plot_bg}'
    fig.savefig(savepath + '.png', bbox_inches='tight', transparent=True if plot_bg=='black' else False, dpi=200)
    fig.savefig(savepath + '.pdf', bbox_inches='tight', dpi=400, transparent=True if plot_bg=='black' else False)

    return fig, savepath

########################################
# %%
# RUN

if __name__ == '__main__':
    plot_bg = 'white'  # 'white' or 'black'

    fig, savepath = make_figure(multi_model_dict, masks, boot_data, reg_ds_cesm, plot_bg=plot_bg)

# %%
