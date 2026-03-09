########################################
# %%
# LOAD PACKAGES

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import importlib
import functions
importlib.reload(functions)

########################################
# %%
# LOAD ALL THE MPI-ESM DATA

if __name__ == '__main__':
    data_dict = functions.load_mpi_esm_data(eur_only=True)

########################################
# %%
# FIGURE FUNCTION
# MAKE FIG. 1: SIMULATION AND REGRESSION PLOT
# PRODUCES FIG. S1 FOR WINTER (DJF) and FIG. S2 FOR SUMMER (JJA)
# PRODUCES FIG. S3 FOR 5-yr moving average and FIG. S4 FOR 30-yr moving average
# PRODUCES FIG. S5 for constant hosing only, and FIG. S6 for linearly increasing hosing only

def make_figure(data_dict, season='', window=10, hosing='all', plot_bg='white'):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10})
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 2, width_ratios=[0.5, 0.5], height_ratios=[1, 1], wspace=0.25, hspace=0.2)

    # Left column: two vertical subplots
    ax_left_top = fig.add_subplot(gs[0, 0])
    ax_left_bottom = fig.add_subplot(gs[1, 0], sharex=ax_left_top)

    # Right column: one large subplot spanning both rows
    ax_right = fig.add_subplot(gs[1, 1])

    equation_y_pos, equation_y_spacing = 1.3, 0.07
    functions.simulations_plot(data_dict, season=season, hos_type=hosing, window=window, ext_axs=[ax_left_top, ax_left_bottom], plot_bg=plot_bg)
    functions.regression_plot(data_dict, season=season, hos_type=hosing, window=window, region='EU', ext_ax=ax_right, plot_bg=plot_bg, ylim=(-1.5, 1.0), equation_y_pos=equation_y_pos, equation_y_spacing=equation_y_spacing)

    ax_right.text(0.0, equation_y_pos+equation_y_spacing,
                        "Regression results:",
                        transform=ax_right.transAxes, fontsize=10, verticalalignment='top', color='black' if not plot_bg=='black' else 'white', fontweight='bold')

    ax_left_top.tick_params(labelbottom=False)

    primary_subset_min = 7.606163386737876
    primary_subset_max = 19.01540846684469
    primary_position = ax_left_top.get_position()

    # Calculate the new vertical extent for the secondary axis
    total_height = primary_position.height
    new_y0 = primary_position.y0 + (primary_subset_min - ax_left_top.get_ylim()[0]) / (ax_left_top.get_ylim()[1] - ax_left_top.get_ylim()[0]) * total_height
    new_height = (primary_subset_max - primary_subset_min) / (ax_left_top.get_ylim()[1] - ax_left_top.get_ylim()[0]) * total_height

    secax = fig.add_axes([0.447,  # x-position (0.375 for 0.4/0.6 width ratio)
                        new_y0,  # starting vertical position
                        0.02,  # Width of the secondary axis
                        new_height])  # height of secondary axis

    secax.set_ylabel('AMOC weakening w.r.t. 1850–1899', labelpad=5, fontsize=10)

    secax.spines['top'].set_visible(False)
    secax.spines['left'].set_visible(False)
    secax.spines['bottom'].set_visible(False)
    secax.spines['right'].set_visible(True)

    secax.yaxis.tick_right()
    secax.yaxis.set_label_position('right')
    secax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    secondary_subset_min = functions.convert_strength_to_weakening(primary_subset_max)  # Note: max maps to min
    secondary_subset_max = functions.convert_strength_to_weakening(primary_subset_min)  # Note: min maps to max

    secax.set_ylim(secondary_subset_max, secondary_subset_min)
    secondary_ticks = np.linspace(secondary_subset_max, secondary_subset_min, 7)

    secax.set_yticks(secondary_ticks)
    secax.set_yticklabels([f"{abs(tick):.0f}%" for tick in secondary_ticks])

    secax.set_facecolor('none')

    handles, labels = ax_right.get_legend_handles_labels()
    column_headings = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0']

    # Split handles and labels into three groups
    n = len(handles) // 3
    grouped_handles = [handles[i*n:(i+1)*n] for i in range(3)]
    grouped_labels = [labels[i*n:(i+1)*n] for i in range(3)]

    # Reorder handles/labels based on hosing type
    if hosing == 'all' and n >= 8:
        # Move 'linneg.' from position 7 to position 4
        [grouped_handles[i].insert(4, grouped_handles[i].pop(7)) for i in range(3)]
        [grouped_labels[i].insert(4, grouped_labels[i].pop(7)) for i in range(3)]
    elif hosing == 'linear' and n >= 4:
        # Move 'linneg.' from position 3 (end) to position 0 (start): -0.2, 0.2, 0.6, 1.0
        [grouped_handles[i].insert(0, grouped_handles[i].pop(3)) for i in range(3)]
        [grouped_labels[i].insert(0, grouped_labels[i].pop(3)) for i in range(3)]

    for i in range(3):
        grouped_labels[i] = [l[9:] if len(l) > 10 else l for l in grouped_labels[i]]

    # Create dummy handles for headings (no marker, invisible line)
    heading_handles = [Line2D([], [], color='none', label=heading, linewidth=0) for heading in column_headings]

    # Create MPI-GE handles with correct marker and color for each SSP
    ge_ssp_keys = ['ssp126', 'ssp245', 'ssp370']
    ge_handles = [Line2D([], [], color=functions.hosing_colors[ssp]['ge'],
                         marker=functions.hosing_markers['ge'], linestyle='None',
                         markersize=5, label='MPI-GE') for ssp in ge_ssp_keys]

    # Interleave headings, MPI-GE, and group items for legend
    final_handles = []
    final_labels = []
    for heading, ge, h_group, l_group in zip(heading_handles, ge_handles, grouped_handles, grouped_labels):
        final_handles.append(heading)
        final_labels.append(heading.get_label())
        final_handles.append(ge)
        final_labels.append('MPI-GE')
        final_handles.extend(h_group)
        final_labels.extend(l_group)

    legend = ax_right.legend(final_handles, final_labels, ncol=3, frameon=False, handletextpad=1.5, columnspacing=2, bbox_to_anchor=(-0.04, 2.25), loc='upper left', fontsize=10)

    # Make headings bold and center them
    for text, handle in zip(legend.get_texts(), final_handles):
        if handle in heading_handles:
            text.set_weight('bold')
            text.set_ha('center')
            text.set_color([functions.hosing_colors[ssp]['ge'] for ssp in ['ssp126', 'ssp245', 'ssp370']][heading_handles.index(handle)])

    subplot_labels = ['a', 'b', 'c']
    for i, ax in enumerate([ax_left_top, ax_left_bottom]):
        left = ax.get_position().x0
        top = ax.get_position().y1
        fig.text(left-0.03, top, subplot_labels[i], transform=fig.transFigure, fontsize=14, ha='right', va='center', fontweight='bold', color='black' if not plot_bg=='black' else 'white')
    fig.text(ax_right.get_position().x0-0.04, ax_right.get_position().y1, subplot_labels[2], transform=fig.transFigure, fontsize=14, ha='right', va='center', fontweight='bold', color='black' if not plot_bg=='black' else 'white')

    savepath = f"../plots/Fig1_simulations_regression_plotbg-{plot_bg}_season-{season or 'annual'}_window-{window}_hosing-{hosing}"
    fig.savefig(savepath + '.png', dpi=200, bbox_inches='tight', transparent=True if plot_bg=='black' else False)
    fig.savefig(savepath + '.pdf', bbox_inches='tight', dpi=400, transparent=True if plot_bg=='black' else False)

    return fig, savepath

########################################
# %%
# RUN

if __name__ == '__main__':
    plot_bg = 'white'  # 'white' or 'black'
    season = ''  # '' for annual (Fig. 1), 'djf' for winter (Fig. S1), 'jja' for summer (Fig. S2)
    window = 10  # moving average window in years; 10 for Fig.1, 1 for Fig.S3, 30 for Fig.S4
    hosing = 'all'  # 'all', 'linear' or 'constant'

    fig, savepath = make_figure(data_dict, season=season, window=window, hosing=hosing, plot_bg=plot_bg)

#%%
