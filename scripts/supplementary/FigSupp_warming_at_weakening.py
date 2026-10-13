########################################
# %%
# LOAD PACKAGES

import re
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

import importlib
import sys, pathlib
if "__file__" in globals():  # standalone run: ensure scripts/ is importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import functions
importlib.reload(functions)
import plot_warming_at_amoc_weakening as pw # type: ignore
importlib.reload(pw)
import FigSupp_synthetic_scaling_factors as ssf  # type: ignore
importlib.reload(ssf)

########################################
# %%
# WARMING-AT-FIXED-WEAKENING (supplementary)
#
# Two side-by-side panels of pw.make_figure (the flipped net-cooling figure):
# per country, the end-of-century ΔT above 1850-1899 each (model, scenario)
# line predicts at a chosen AMOC-weakening level. Left = 25 %, right = 75 %.
# Since 2026-09-01 the panels carry the Synthetic CMIP6 range of ΔT at each
# level (dT_* vars, range-cache schema v3) in place of the NAHosMIP min-max
# bar; cmip_range_ds=None restores the old bar. Layout mirrors
# FigSupp_cmip_cooling.py; per-panel legend (self-describing via its
# weakening-% title) is intentional.

########################################
# %%
# LOAD DATA

if __name__ == '__main__':
    multi_model_dict, masks = functions.get_full_multi_model_dict()
    reg_ds_mpi  = functions.load_regression_ds_mpi()
    reg_ds_cesm = functions.get_cesm_reg_ds(recompute=False)
    reg_ds_giss = functions.get_giss_reg_ds(recompute=False, masks=masks)
    hosmip_reg_ds_dict = functions.get_hosmip_reg_ds(recompute=False)
    # Fig3-default range spec (pooled, hosing, w); season-matched below.
    cmip_range_ds_by_season = {
        se: ssf.get_cmip_range_ds(recompute=False, state_dep='pooled',
                                  cal_set='hosing', predictors='w', season=se)
        for se in ('', 'djf', 'jja')
    }


########################################
# %%
# FIGURE FUNCTION

def make_figure(reg_ds_mpi, reg_ds_cesm, masks, hosmip_reg_ds_dict,
                reg_ds_giss=None, weakenings=(25.0, 75.0),
                T_ref='pi', season='', giss_time_period='2101-2300',
                cmip_range_ds=None, plot_bg='white'):
    plt.style.use('default')
    plt.rcParams.update({**functions.PRINT_RC, 'font.size': 6})  # 6 pt base, as Fig. 3b: rows are ~8 pt apart
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    n = len(weakenings)
    fig, axes = plt.subplots(1, n, figsize=(183 / 25.4, 183 / 25.4 * 0.7))
    fig.set_size_inches(183 / 25.4, 183 / 25.4 * 0.7, forward=False)  # GUI backends (macosx) round the canvas to whole pixels
    fig.subplots_adjust(top=0.94, bottom=0.05)
    if n == 1:
        axes = [axes]

    for ax, w in zip(axes, weakenings):
        handles, labels = pw.make_figure(
            reg_ds_mpi, reg_ds_cesm, masks,
            hosmip_reg_ds_dict=hosmip_reg_ds_dict,
            amoc_weakening_pct=w, T_ref=T_ref, season=season,
            reg_ds_giss=reg_ds_giss, giss_time_period=giss_time_period,
            cmip_range_ds=cmip_range_ds,
            plot_bg=plot_bg, ext_ax=ax, title=False, savefig=False, legend=False)[1:]

    # One legend for both panels, in panel a's empty ΔT < 0 half (a per-panel legend sat over the bars of the lower rows)
    from matplotlib.legend_handler import HandlerTuple
    axes[0].legend(handles, [re.sub(r', (?=\d{4}–)', ',\n', l.replace(' (', '\n(')) if len(l) > 27 else l for l in labels],
                   frameon=True, loc='lower left', bbox_to_anchor=(0.005, 0.055), fontsize=6, handlelength=1.6, handletextpad=0.6,
                   handler_map={tuple: HandlerTuple(ndivide=1)},
                   title=('Annual' if season == '' else f'Seasonal ({season.upper()})') + ' warming',
                   title_fontproperties={'weight': 'bold', 'size': 6}, alignment='left')

    # Each panel carries its own right-hand country labels: the gutter holds them.
    functions.place_range_panels(fig, axes)
    functions.panel_headers(fig, axes, [f'End-of-century ΔT at {w:.0f}% AMOC weakening' for w in weakenings])

    cr_tag = '_cr-med' if cmip_range_ds is not None else '_cr-off'
    savepath = (f'../plots/FigSupp_warming_at_weakening_'
                f'w-{"-".join(f"{w:.0f}" for w in weakenings)}'
                f'_Tref-{T_ref}_season-{season or "annual"}{cr_tag}_plotbg-{plot_bg}')
    functions.fit_print_width(fig)
    fig.savefig(savepath + '.png', dpi=300, bbox_inches='tight',
                transparent=True if plot_bg == 'black' else False)
    fig.savefig(savepath + '.pdf', dpi=600, bbox_inches='tight',
                transparent=True if plot_bg == 'black' else False)
    return fig, savepath


########################################
# %%
# RUN

if __name__ == '__main__':
    plot_bg = 'white'  # 'white' or 'black'
    season = ''        # '' | 'djf' | 'jja'
    fig, savepath = make_figure(reg_ds_mpi, reg_ds_cesm, masks, hosmip_reg_ds_dict,
                                reg_ds_giss=reg_ds_giss, weakenings=(25.0, 75.0),
                                season=season, T_ref='pi',
                                cmip_range_ds=cmip_range_ds_by_season[season],
                                plot_bg=plot_bg)
    print(f"Saved {savepath}")

# %%
