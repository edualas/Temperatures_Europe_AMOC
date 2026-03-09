########################################
# %%
# LOAD PACKAGES

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
plt.rcParams.update({'font.size': 12})
import cartopy
import cartopy.crs as ccrs
import numpy as np

import importlib
import functions
importlib.reload(functions)

########################################
# %%
# LOAD HOSMIP DATA AND OTHER STUDIES

if __name__ == '__main__':
    multi_model_dict, masks = functions.get_full_multi_model_dict()

########################################
# %%
# LOAD MPI-ESM, CESM and HosMIP REGRESSION DATA

if __name__ == '__main__':
    reg_ds_mpi = functions.load_regression_ds_mpi()
    reg_ds_cesm = functions.get_cesm_reg_ds(recompute=False)
    hosmip_reg_ds_dict = functions.get_hosmip_reg_ds(recompute=False)

########################################
#%%
# FIGURE FUNCTION
# MAKE FIG.3: HOSMIP REGRESSIONS AND NET COOLING RANGES
# PRODUCES FIG. S13 for present-day net cooling points
# PRODUCES FIG. S14 FOR WINTER (DJF) and FIG. S15 FOR SUMMER (JJA)

def make_figure(multi_model_dict, masks, reg_ds_mpi, reg_ds_cesm, hosmip_reg_ds_dict,
                season='', window=10, T_ref='pi', hosmip_markers=False, plot_bg='white'):
    plt.style.use('default')
    # plt.rcParams.update({'font.size': 12})
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 2, width_ratios=[0.35, 0.65], height_ratios=[1, 1, 1, 1], wspace=0.4, hspace=0.08)

    ax_left_1 = fig.add_subplot(gs[0, 0])
    ax_left_2 = fig.add_subplot(gs[1, 0], sharex=ax_left_1)
    ax_left_3 = fig.add_subplot(gs[2, 0], sharex=ax_left_1)
    ax_left_4 = fig.add_subplot(gs[3, 0], sharex=ax_left_1)
    axes_left = [ax_left_1, ax_left_2, ax_left_3, ax_left_4]

    # Right column: one large subplot spanning both rows
    ax_right = fig.add_subplot(gs[:, 1])


    ax_left_2.tick_params(labelbottom=False)
    ax_left_3.tick_params(labelbottom=False)

    plot_regions = ['EU', 'NEU', 'WCE', 'MED']
    region_label_shift = np.array([0, 1, 0, -1]) * 0.025
    subplot_labels = ['a', 'b', 'c', 'd', 'e']
    for i, ax in enumerate(axes_left):
        functions.hosmip_regression_plot(multi_model_dict, region=plot_regions[i], season=season, window=window, plot_bg=plot_bg, quantile_reg=False, no_plots=False, low_ylim=-14, linear_95_reg=True, linear_5_reg=False, central_reg=None, central_reg_intercept=False, hosmip_ref_pi=True, cap_x_range=100, bm_data=True, liu_data=True, vwb_data=True, boot_data=True, boot_regression=True, add_combined_results=True, combined_results_label=False, add_hosmip_mpi_regression=True, ax=ax, markersize=10)
        if i != 3:
            ax.tick_params(labelbottom=False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xlabel("")
            ax.tick_params(axis='x', which='both', bottom=False, top=False)
            ax.tick_params(axis='y', labelsize=10)
        bottom = ax.get_position().y0
        top = ax.get_position().y1
        left = ax.get_position().x0
        tiny_ax = fig.add_axes([0.13, bottom+0.01, 0.1, 0.1], projection=ccrs.PlateCarree())
        functions.get_mask_plot(masks, plot_regions[i], ext_ax=tiny_ax, alpha=0.4, plot_bg=plot_bg)
        fig.text(0.175, bottom+0.05+region_label_shift[i], plot_regions[i], transform=fig.transFigure, fontsize=10, ha='center', va='bottom', fontweight='bold', color='black' if not plot_bg=='black' else 'black',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.5, edgecolor='none'))
        fig.text(left-0.03, top, subplot_labels[i], transform=fig.transFigure, fontsize=14, ha='right', va='center', fontweight='bold', color='black' if not plot_bg=='black' else 'white')

    fig.text(0.465, ax_right.get_position().y1, subplot_labels[4], transform=fig.transFigure, fontsize=14, ha='right', va='center', fontweight='bold', color='black' if not plot_bg=='black' else 'white')

    ax_left_4.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax_left_4.tick_params(axis='both', labelsize=10)

    handles, labels = ax_left_4.get_legend_handles_labels()
    # Split into HosMIP models (first 8) and additional studies (next entries, skip empty)
    label_renames = {'HadGEM3-GC3-1MM': 'HadGEM3-GC3.1-MM', 'HadGEM3-GC3-1LL': 'HadGEM3-GC3.1-LL',
                     'MPI-ESM1-2-HR': 'MPI-ESM1.2-HR', 'MPI-ESM1-2-LR': 'MPI-ESM1.2-LR'}
    h_models = handles[:8]
    l_models = [label_renames.get(l, l) for l in labels[:8]]
    h_studies = [h for h, l in zip(handles[8:], labels[8:]) if l]
    l_studies = [l for l in labels[8:] if l]

    # Position legends horizontally centered with annotations (x=1.25 in ax_left_2 axes fraction)
    ax2_pos = ax_left_2.get_position()
    fig_x = ax2_pos.x0 + 1.25 * ax2_pos.width

    # Studies legend: near top of ax_left_1
    fig_y_top = ax_left_1.get_position().y1
    fig.legend(h_studies, l_studies, frameon=False, ncols=1, fontsize=8,
               loc='upper center', bbox_to_anchor=(fig_x, fig_y_top))

    # Models legend: near bottom of ax_left_4
    fig_y_bot = ax_left_3.get_position().y0 + 0.7 * ax_left_4.get_position().height
    fig.legend(h_models, l_models, frameon=False, ncols=1, fontsize=8,
               loc='upper center', bbox_to_anchor=(fig_x, fig_y_bot))

    ax_left_2.annotate('Combined forcing\nMPI-ESM1.2-LR\nregression\n(this study)', xy=(1.02, 0.835), xytext=(1.25, 1.15), xycoords='axes fraction', ha='center', va='center', fontsize=10, color='brown', fontweight='bold', arrowprops={'arrowstyle': '-|>', 'connectionstyle': 'angle3,angleA=90,angleB=-10', 'color':'brown'})
    ax_left_2.annotate('Preindustrial hosing\nMPI-ESM1.2-LR\nregression\n(NAHosMIP)', xy=(1.02, 0.805 if season!='djf' else 0.775), xytext=(1.25, 0.49), xycoords='axes fraction', ha='center', va='center', fontsize=10, color=functions.hosmip_colors['MPI-ESM1-2-LR'], fontweight='bold', arrowprops={'arrowstyle': '-|>', 'connectionstyle': 'angle3,angleA=90,angleB=-10', 'color':functions.hosmip_colors['MPI-ESM1-2-LR']})
    if season == '':
        ax_left_2.annotate('Combined forcing\nCESM2\nregression\n(Boot et al. 2024)', xy=(0.64, 0.59), xytext=(1.25, -0.1), xycoords='axes fraction', ha='center', va='center', fontsize=10, color='dodgerblue', fontweight='bold', arrowprops={'arrowstyle': '-|>', 'connectionstyle': 'angle3,angleA=200,angleB=150', 'color':'dodgerblue'})

    functions.plot_net_cooling_ranges_mpi_cesm(reg_ds_mpi, reg_ds_cesm, masks, hosmip_reg_ds_dict=hosmip_reg_ds_dict, season=season, hosmip_markers=hosmip_markers, T_ref=T_ref, plot_bg=plot_bg, ext_ax=ax_right, title=False)

    # Align ax_right vertically with the left column (top of ax_left_1, bottom of ax_left_4)
    # The AMOC projection bars/text above the upper x-axis overflow by ~5% of axes height,
    # so shrink the axes top to leave room for that overflow to align with ax_left_1's top.
    pos_left_top = ax_left_1.get_position()
    pos_left_bot = ax_left_4.get_position()
    pos_right = ax_right.get_position()

    extend_down = 0.023
    new_bottom = pos_left_bot.y0 - extend_down
    new_top = pos_left_top.y1
    new_height = new_top - new_bottom
    overflow_frac = 0.05  # fraction of axes height used by AMOC projections above the axis
    new_height_adjusted = new_height / (1 + overflow_frac)
    ax_right.set_position([pos_right.x0, new_bottom, pos_right.width, new_height_adjusted])

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

    savepath = f'../plots/Fig3_plotbg-{plot_bg}_season-{season}_Tref-{T_ref}_markers-{hosmip_markers}'
    fig.savefig(savepath + '.png', dpi=200, bbox_inches='tight', transparent=True if plot_bg=='black' else False)
    fig.savefig(savepath + '.pdf', bbox_inches='tight', dpi=400, transparent=True if plot_bg=='black' else False)

    return fig, savepath

########################################
# %%
# RUN

if __name__ == '__main__':
    plot_bg = 'white'  # 'white' or 'black'
    window = 10
    season = ''
    T_ref = 'pi'  # 'pi' or 'pd'
    hosmip_markers = False

    fig, savepath = make_figure(multi_model_dict, masks, reg_ds_mpi, reg_ds_cesm, hosmip_reg_ds_dict,
                                season=season, window=window, T_ref=T_ref, hosmip_markers=hosmip_markers, plot_bg=plot_bg)
# %%
