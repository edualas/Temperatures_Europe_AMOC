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
import cmip_cooling # type: ignore
importlib.reload(cmip_cooling)

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
    reg_ds_giss = functions.get_giss_reg_ds(recompute=False)
    # Per-season panel data. Each season's panel is cached separately
    # (reg_ds_giss_panel{_djf,_jja}.nc); the make_figure call picks the
    # right slice via the `season` kwarg.
    reg_ds_giss_panel = {
        s: functions.get_giss_panel_data(masks=masks, season=s, recompute=False)
        for s in ['', 'djf', 'jja']
    }
    hosmip_reg_ds_dict = functions.get_hosmip_reg_ds(recompute=False)
    cmip_cooling_ds = cmip_cooling.make_cmip_cooling_ds(recompute=False)

########################################
#%%
# FIGURE FUNCTION
# MAKE FIG.3: HOSMIP REGRESSIONS AND NET COOLING RANGES
# PRODUCES FIG. S13 for present-day net cooling points
# PRODUCES FIG. S14 FOR WINTER (DJF) and FIG. S15 FOR SUMMER (JJA)

def make_figure(multi_model_dict, masks, reg_ds_mpi, reg_ds_cesm, hosmip_reg_ds_dict,
                season='', window=10, T_ref='pi', hosmip_markers=False, plot_bg='white',
                reg_ds_giss=None, giss_time_period='2101-2300',
                giss_panel_data=None, cmip_cooling_ds=None,
                cmip_cooling_show=False, aggregate_first=True, weakening_unit='pct',
                future_window=None):
    plt.style.use('default')
    # plt.rcParams.update({'font.size': 12})
    # Sv-axis extents (only used when weakening_unit='sv'). Panel e extends to the
    # strongest model's PI (rounded up to a multiple of 5); each model's
    # net-cooling point is suppressed beyond its own PI (>100% weakening) inside
    # plot_net_cooling_ranges_mpi_cesm, so only physically achievable points plot.
    # Panels a–d use a data-safe extent (~80% of the strongest PI) so no
    # scatter/overlay clips.
    sv_xmax_e = sv_xmax_ad = None
    if weakening_unit == 'sv':
        pis = [float(ds.AMOC_pi.values) for ds in hosmip_reg_ds_dict.values()]
        pis.append(float(reg_ds_cesm.AMOC_pi.values))
        pis.append(functions.AMOC_pi_MPI)
        if reg_ds_giss is not None:
            pis.append(functions.AMOC_pi_GISS)
        sv_xmax_e = int(np.ceil(max(pis) / 5) * 5)
        sv_xmax_ad = int(np.ceil(0.8 * max(pis) / 5) * 5)
    # CMIP6 weakening envelope for the chosen future window (canonical global
    # when future_window is None); passed into the net-cooling-ranges panel.
    _ext = functions.get_amoc_extent(future_window, unit=weakening_unit)
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
    # giss_panel_data may be a dict keyed by season (preferred) or a flat
    # Dataset (legacy/back-compat). When a dict, pick the season-specific
    # slice so panels a–d show seasonal GISS scatter.
    if isinstance(giss_panel_data, dict):
        giss_panel_for_season = giss_panel_data.get(season)
    else:
        giss_panel_for_season = giss_panel_data
    for i, ax in enumerate(axes_left):
        functions.hosmip_regression_plot(multi_model_dict, region=plot_regions[i], season=season, window=window, plot_bg=plot_bg, quantile_reg=False, no_plots=False, low_ylim=-14, linear_95_reg=True, linear_5_reg=False, central_reg=None, central_reg_intercept=False, hosmip_ref_pi=True, cap_x_range=100, bm_data=True, liu_data=True, vwb_data=True, boot_data=True, boot_regression=True, giss_data=giss_panel_for_season, giss_regression=(giss_panel_for_season is not None), giss_intercept=False, giss_time_period=giss_time_period, add_combined_results=True, combined_results_label=False, add_hosmip_mpi_regression=True, hosmip_reg_ds_dict=hosmip_reg_ds_dict, weakening_unit=weakening_unit, sv_xmax=sv_xmax_ad, ax=ax, markersize=10)
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

    if weakening_unit == 'pct':
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

    ax_left_2.annotate('Combined forcing\nMPI-ESM1.2-LR\nregression\n(this study)', xy=(1.02, 0.84 if season=='' else 0.815 if season=='djf' else 0.86), xytext=(1.25, 1.15), xycoords='axes fraction', ha='center', va='center', fontsize=10, color='brown', fontweight='bold', arrowprops={'arrowstyle': '-|>', 'connectionstyle': 'angle3,angleA=90,angleB=-10', 'color':'brown'})
    ax_left_2.annotate('Preindustrial hosing\nMPI-ESM1.2-LR\nregression\n(NAHosMIP)', xy=(1.02, 0.82 if season=='' else 0.795 if season=='djf' else 0.84), xytext=(1.25, 0.49), xycoords='axes fraction', ha='center', va='center', fontsize=10, color=functions.hosmip_colors['MPI-ESM1-2-LR'], fontweight='bold', arrowprops={'arrowstyle': '-|>', 'connectionstyle': 'angle3,angleA=90,angleB=-10', 'color':functions.hosmip_colors['MPI-ESM1-2-LR']})
    if season == '':
        ax_left_2.annotate('Combined forcing\nCESM2\nregression\n(Boot et al. 2024)', xy=(0.64, 0.59), xytext=(1.25, -0.1), xycoords='axes fraction', ha='center', va='center', fontsize=10, color='dodgerblue', fontweight='bold', arrowprops={'arrowstyle': '-|>', 'connectionstyle': 'angle3,angleA=200,angleB=150', 'color':'dodgerblue'})

    # Pass cmip_cooling_ds only when the overlay should be shown; passing
    # None disables the entire overlay (legend entries + markers). The
    # canonical panel uses cmip_cooling_decade=None ("plot all" mode): every
    # CMIP-cooling point dropped onto the end-of-century panel at its own
    # first-onset-decade weakening.
    cc_ds_for_plot = cmip_cooling_ds if cmip_cooling_show else None
    functions.plot_net_cooling_ranges_mpi_cesm(reg_ds_mpi, reg_ds_cesm, masks, hosmip_reg_ds_dict=hosmip_reg_ds_dict, season=season, hosmip_markers=hosmip_markers, T_ref=T_ref, plot_bg=plot_bg, ext_ax=ax_right, title=False, reg_ds_giss=reg_ds_giss, giss_time_period=giss_time_period, cmip_cooling_ds=cc_ds_for_plot, cmip_cooling_decade=None, aggregate_first=aggregate_first, amoc_extent=_ext, weakening_unit=weakening_unit, sv_xmax=sv_xmax_e)

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

    # giss + gisspanel are coupled in practice (always passed together or not
    # at all), so they fold into a single `giss` facet — mirroring cc_tag.
    giss_tag = f'_giss-{giss_time_period}' if reg_ds_giss is not None else '_giss-off'
    cc_tag = ('_cc-on' if (cmip_cooling_ds is not None and cmip_cooling_show)
              else '_cc-off')
    agg_tag = '_aggfirst-on' if aggregate_first else '_aggfirst-off'
    wunit_tag = f'_wunit-{weakening_unit}'
    savepath = f"../plots/Fig3_plotbg-{plot_bg}_season-{season or 'annual'}_Tref-{T_ref}_markers-{hosmip_markers}{giss_tag}{cc_tag}{agg_tag}{wunit_tag}{functions.fw_suffix(future_window)}"
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
    giss_time_period = '2101-2300'  # '2101-2300' or '2015-2500'
    cmip_cooling_show = False
    aggregate_first = True
    weakening_unit = 'pct'  # 'pct' (% weakening vs PI) or 'sv' (absolute Sv weakening); 'sv' requires aggregate_first=True
    future_window = None  # None -> canonical 2091-2100; e.g. ('2081','2090') for a sensitivity variant

    # A variant future window needs its own (window-keyed) regression caches.
    # reg_ds_giss_panel and cmip_cooling_ds are window-independent, so they
    # are reused as-is.
    if future_window is not None:
        reg_ds_mpi = functions.load_regression_ds_mpi(future_window=future_window)
        reg_ds_cesm = functions.get_cesm_reg_ds(recompute=False, future_window=future_window)
        reg_ds_giss = functions.get_giss_reg_ds(recompute=False, masks=masks, future_window=future_window)
        hosmip_reg_ds_dict = functions.get_hosmip_reg_ds(recompute=False, future_window=future_window)

    # The CMIP-projected cooling overlay defaults off (cmip_cooling_show=False);
    # when on, the canonical panel shows all CMIP-cooling points at their
    # first-onset-decade weakening. Each lands in its own filename via the
    # _cc-on / _cc-off suffix.
    fig, savepath = make_figure(multi_model_dict, masks, reg_ds_mpi, reg_ds_cesm, hosmip_reg_ds_dict,
                                season=season, window=window, T_ref=T_ref, hosmip_markers=hosmip_markers, plot_bg=plot_bg,
                                reg_ds_giss=reg_ds_giss, giss_panel_data=reg_ds_giss_panel, giss_time_period=giss_time_period,
                                cmip_cooling_ds=cmip_cooling_ds,
                                cmip_cooling_show=cmip_cooling_show,
                                aggregate_first=aggregate_first, weakening_unit=weakening_unit,
                                future_window=future_window)
# %%
