########################################
# %%
# LOAD PACKAGES

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.size': 12})

import importlib
import sys, pathlib
if "__file__" in globals():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import functions
importlib.reload(functions)

########################################
# %%
# WARMING-PATTERN UNCERTAINTY (supplementary)
#
# Two-panel side-by-side figure (layout mirrors FigSupp_cmip_cooling.py,
# without the CMIP6 envelope/projection-ticks chrome).
#
#  Left  — warming-pattern decomposition. Per (country, ssp):
#            *  HosMIP MPI-ESM-LR (own T, AMOC, slope)
#            |  bar to HosMIP MPI-ESM-LR with CanESM5 warming pattern (no marker)
#            X  full HosMIP CanESM5
#  Right — full HosMIP/MPI/CESM/GISS net-cooling range with hosmip_markers=True
#          and the top CMIP6-projection apparatus suppressed (top_apparatus=False).

########################################
# %%
# LOAD DATA

if __name__ == '__main__':
    multi_model_dict, masks = functions.get_full_multi_model_dict()
    reg_ds_mpi  = functions.load_regression_ds_mpi()
    reg_ds_cesm = functions.get_cesm_reg_ds(recompute=False)
    reg_ds_giss = functions.get_giss_reg_ds(recompute=False, masks=masks)
    hosmip_reg_ds_dict = functions.get_hosmip_reg_ds(recompute=False)


########################################
# %%
# LEFT-PANEL DRAWING

def _draw_warming_panel(ax, hosmip_reg_ds_dict, masks, *,
                        season='', T_ref='pi', plot_bg='white', title=True):
    bar_height = 0.25
    bar_alpha  = 0.7
    line_alpha = 1.0
    vertical_offsets = {'ssp126': 1., 'ssp245': 0., 'ssp370': -1.}

    regions_cutoff30k = [r for r in functions.regions
                         if r not in ['LU', 'CY', 'XK', 'ME', 'SI', 'MK', 'AL']]
    countries = list(reversed(regions_cutoff30k))
    rows = ['EU'] + countries
    group_centers = np.arange(len(rows))

    def _select(da, model, region):
        da = da.where(masks[model][region])
        if region == 'EU':
            da = da.where(masks[model]['LAND'] == 0)
        return da

    def _amean(da, model, region):
        return float(functions.weighted_area_lat(
            _select(da, model, region)).mean('lat').mean('lon').values)

    def _ncp_pct(amoc_pi, amoc_90, T_90, T_ref_, slope_Sv):
        weakening_sv = amoc_pi - (amoc_90 - (T_90 - T_ref_) / slope_Sv)
        return weakening_sv / amoc_pi * 100.0

    mpi = hosmip_reg_ds_dict['MPI-ESM1-2-LR']
    can = hosmip_reg_ds_dict['CanESM5']

    print(f"{'reg':<4} {'ssp':<7} "
          f"{'MPI':>8} {'hybrid':>8} {'CanESM5':>8} "
          f"{'dir':>4}")

    for i, r in enumerate(rows):
        for ssp_i in functions.ssps:
            amoc_pi_m  = float(mpi.AMOC_pi.values)
            amoc_90_m  = float(mpi.AMOC_future.sel(scenar=ssp_i).values)
            T_base_m   = mpi.T_pi if T_ref == 'pi' else mpi.T_pd_245
            T_pi_m     = _amean(T_base_m.sel(season=season), 'MPI-ESM1-2-LR', r)
            T_90_m     = _amean(mpi.T_future.sel(scenar=ssp_i, season=season), 'MPI-ESM1-2-LR', r)
            slope_m_pct_field = mpi.lin_coef_hosmip.sel(season=season)
            slope_m_pct = _amean(slope_m_pct_field.where(slope_m_pct_field < 0),
                                 'MPI-ESM1-2-LR', r)
            slope_m_Sv  = slope_m_pct * (-100.0) / amoc_pi_m

            amoc_pi_c  = float(can.AMOC_pi.values)
            amoc_90_c  = float(can.AMOC_future.sel(scenar=ssp_i).values)
            T_base_c   = can.T_pi if T_ref == 'pi' else can.T_pd_245
            T_pi_c     = _amean(T_base_c.sel(season=season), 'CanESM5', r)
            T_90_c     = _amean(can.T_future.sel(scenar=ssp_i, season=season), 'CanESM5', r)
            slope_c_pct_field = can.lin_coef_hosmip.sel(season=season)
            slope_c_pct = _amean(slope_c_pct_field.where(slope_c_pct_field < 0),
                                 'CanESM5', r)
            slope_c_Sv  = slope_c_pct * (-100.0) / amoc_pi_c

            ncp_mpi    = _ncp_pct(amoc_pi_m, amoc_90_m, T_90_m, T_pi_m, slope_m_Sv)
            ncp_hybrid = _ncp_pct(amoc_pi_m, amoc_90_m, T_90_c, T_pi_c, slope_m_Sv)
            ncp_can    = _ncp_pct(amoc_pi_c, amoc_90_c, T_90_c, T_pi_c, slope_c_Sv)

            y = group_centers[i] + vertical_offsets[ssp_i] * bar_height
            color = functions.hosing_colors[ssp_i]['ge']

            if np.isfinite(ncp_mpi) and np.isfinite(ncp_hybrid):
                left, width = min(ncp_mpi, ncp_hybrid), abs(ncp_mpi - ncp_hybrid)
                ax.barh(y - 0.01, width, left=left,
                        height=bar_height * 0.8, alpha=0.3,
                        color=color, clip_on=True)

            ax.scatter(ncp_mpi, y, alpha=bar_alpha, color=color,
                       marker='*', s=40, edgecolors='none', zorder=11)
            ax.scatter(ncp_can, y, alpha=bar_alpha, color=color,
                       marker='X', s=22, edgecolors='none', zorder=12)

            if np.isfinite(ncp_mpi) and ncp_mpi < 100:
                _hyb = (np.nan if not np.isfinite(ncp_hybrid) else ncp_hybrid)
                _can = (np.nan if not np.isfinite(ncp_can) else ncp_can)
                if np.isfinite(ncp_hybrid) and np.isfinite(ncp_can):
                    lo, hi = min(ncp_mpi, ncp_can), max(ncp_mpi, ncp_can)
                    direction = 'in' if (lo <= ncp_hybrid <= hi) else 'out'
                else:
                    direction = '?'
                print(f"{r:<4} {ssp_i:<7} {ncp_mpi:8.2f} {_hyb:8.2f} "
                      f"{_can:8.2f} {direction:>4}")

    for i in range(len(rows)):
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, group_centers[i] - 0.5), 120, 1,
                                   color='black', alpha=0.15, zorder=0,
                                   linewidth=0, clip_on=False))

    legend_color = 'white' if plot_bg == 'black' else 'black'
    legend_handles = [
        Line2D([], [], marker='*', color=legend_color, linestyle='None',
               markersize=8, markeredgewidth=0, alpha=bar_alpha),
        Line2D([], [], color=legend_color, alpha=0.3,
               linewidth=5, solid_capstyle='butt'),
        Line2D([], [], marker='X', color=legend_color, linestyle='None',
               markersize=6, markeredgewidth=0, alpha=bar_alpha),
    ]
    legend_labels = [
        'NAHosMIP MPI-ESM1.2-LR',
        'Range until NAHosMIP MPI-ESM1.2-LR with CanESM5 warming (hybrid)',
        'NAHosMIP CanESM5',
    ]
    ax.legend(legend_handles, legend_labels,
              frameon=True, bbox_to_anchor=(0.02, 0.055),
              loc='lower left', fontsize=10,
              title=('Annual' if season=='' else f'Seasonal ({season.upper()})') + ' net-cooling AMOC weakening w.r.t. ' + ('preindustrial' if T_ref=='pi' else 'present-day'),
              title_fontproperties={'weight': 'bold', 'size': 10})

    for x in np.arange(10, 100, 10):
        ax.axvline(x=x, ymin=0.03,
                   color='black' if plot_bg != 'black' else 'white',
                   linestyle=(0, (1, 4)), linewidth=0.8, alpha=1.0, zorder=0)

    ax.set_ylim(-1.7, len(countries) + 0.5)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 110, 10))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.set_xlabel('AMOC weakening')
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_label_position('bottom')
    ax.tick_params(axis='x', which='both', top=True, labeltop=True,
                   bottom=True, labelbottom=True, labelsize=10)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.set_yticks(group_centers)
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_yticklabels(['European mean'] +
                       [functions.country_names[c] for c in countries])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_position(('axes', 0.03))


########################################
# %%
# FIGURE FUNCTION

def make_figure(reg_ds_mpi, reg_ds_cesm, masks, hosmip_reg_ds_dict,
                reg_ds_giss=None, *, season='', T_ref='pi',
                plot_bg='white', giss_time_period='2101-2300'):
    if T_ref not in ('pi', 'pd'):
        raise NotImplementedError("T_ref must be 'pi' or 'pd'.")
    if season != '':
        raise NotImplementedError("Only annual (season='') is implemented.")

    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    fig, axes = plt.subplots(1, 2, figsize=(8.5 * 2, 14))
    text_color = 'black' if plot_bg != 'black' else 'white'

    _draw_warming_panel(axes[0], hosmip_reg_ds_dict, masks,
                        season=season, T_ref=T_ref, plot_bg=plot_bg, title=False)

    functions.plot_net_cooling_ranges_mpi_cesm(
        reg_ds_mpi, reg_ds_cesm, masks,
        hosmip_reg_ds_dict=hosmip_reg_ds_dict,
        hosmip_markers=True, T_ref=T_ref, season=season, plot_bg=plot_bg,
        reg_ds_giss=reg_ds_giss, giss_time_period=giss_time_period,
        ext_ax=axes[1], title=False, aggregate_first=True)

    panel_labels = ['Uncertainty due to projected baseline warming',
                    'Fig. 3b with additional NAHosMIP markers'] # change back if we do not chose Fig3_simple.py as main Fig. 3
    for i, (ax, lbl) in enumerate(zip(axes, panel_labels)):
        ax.text(0.0, 1.07, f'{chr(97 + i)})  {lbl}',
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                color=text_color, ha='left', va='bottom')

    fig.subplots_adjust(wspace=0.3)

    savepath = (f'../plots/FigSupp_warming_uncertainty'
                f'_Tref-{T_ref}_season-{season or "annual"}_plotbg-{plot_bg}')
    fig.savefig(savepath + '.png', dpi=200, bbox_inches='tight',
                transparent=True if plot_bg == 'black' else False)
    fig.savefig(savepath + '.pdf', dpi=400, bbox_inches='tight',
                transparent=True if plot_bg == 'black' else False)
    return fig, savepath


########################################
# %%
# RUN

if __name__ == '__main__':
    plot_bg = 'white'
    T_ref = 'pi'
    fig, savepath = make_figure(reg_ds_mpi, reg_ds_cesm, masks,
                                hosmip_reg_ds_dict,
                                reg_ds_giss=reg_ds_giss,
                                season='', T_ref=T_ref, plot_bg=plot_bg)
    print(f"Saved {savepath}")

# %%
