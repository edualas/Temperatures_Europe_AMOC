########################################
# %%
# LOAD PACKAGES

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

import importlib
import sys, pathlib
if "__file__" in globals():  # standalone run: ensure scripts/ is importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import functions
importlib.reload(functions)
import cmip_cooling # type: ignore
importlib.reload(cmip_cooling)

########################################
# %%
# DECADE-MATCHED CMIP-PROJECTED COOLING (supplementary)
#
# One net-cooling-point-range panel per non-empty first-onset decade, side
# by side. Each panel is built with that decade's window-keyed regression
# caches (so the ranges are computed for the *same* decade) and overlaid
# with only the CMIP-projected cooling points whose first-onset decade is
# that panel's decade. This removes the inconsistency of the canonical Fig 3,
# where end-of-century (2091-2100) ranges carry CMIP-cooling points realised
# anywhere in 2051-2100.

LABEL_TO_WINDOW = {lbl: tuple(lbl.split('-')) for lbl in cmip_cooling.DECADE_LABELS}


def _present_decades(cmip_cooling_ds):
    """Decade labels (chronological) with >=1 first-onset cooling point."""
    onset = cmip_cooling_ds.cmip_cooling_decade_idx.values.ravel()
    present = sorted(set(int(v) for v in onset if v >= 0))
    return [str(cmip_cooling_ds.decade.values[i]) for i in present]


def _load_decade_caches(decade_labels, masks):
    """Window-keyed regression caches + CMIP6 envelope for each decade.

    Returns ``{label: dict(reg_ds_mpi, reg_ds_cesm, reg_ds_giss,
    hosmip_reg_ds_dict, amoc_extent)}``. The canonical 2091-2100 window
    reuses the unsuffixed caches; other decades build (or load) their
    ``_fw{start}-{end}`` siblings on demand.
    """
    reg_by_decade = {}
    for lbl in decade_labels:
        fw = LABEL_TO_WINDOW[lbl]
        print(f"Loading regression caches for {lbl} ...")
        reg_by_decade[lbl] = {
            'reg_ds_mpi': functions.load_regression_ds_mpi(future_window=fw),
            'reg_ds_cesm': functions.get_cesm_reg_ds(recompute=False, future_window=fw),
            'reg_ds_giss': functions.get_giss_reg_ds(recompute=False, masks=masks, future_window=fw),
            'hosmip_reg_ds_dict': functions.get_hosmip_reg_ds(recompute=False, future_window=fw),
            'amoc_extent': functions.get_amoc_extent(fw),
        }
    return reg_by_decade


########################################
# %%
# LOAD DATA

if __name__ == '__main__':
    multi_model_dict, masks = functions.get_full_multi_model_dict()
    cmip_cooling_ds = cmip_cooling.make_cmip_cooling_ds(recompute=False)
    decades = _present_decades(cmip_cooling_ds)
    print(f"Non-empty onset decades: {decades}")
    reg_by_decade = _load_decade_caches(decades, masks)


########################################
# %%
# FIGURE FUNCTION

def make_figure(reg_by_decade, masks, cmip_cooling_ds, decades=None,
                plot_bg='white', giss_time_period='2101-2300',
                aggregate_first=True):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    decades = list(decades) if decades is not None else list(reg_by_decade.keys())
    n = len(decades)
    fig, axes = plt.subplots(1, n, figsize=(8.5 * n, 14))
    if n == 1:
        axes = [axes]

    text_color = 'black' if plot_bg != 'black' else 'white'
    for i, (ax, D) in enumerate(zip(axes, decades)):
        rd = reg_by_decade[D]
        functions.plot_net_cooling_ranges_mpi_cesm(
            rd['reg_ds_mpi'], rd['reg_ds_cesm'], masks,
            hosmip_reg_ds_dict=rd['hosmip_reg_ds_dict'],
            season='', T_ref='pi', plot_bg=plot_bg, ext_ax=ax, title=False,
            reg_ds_giss=rd['reg_ds_giss'], giss_time_period=giss_time_period,
            cmip_cooling_ds=cmip_cooling_ds, cmip_cooling_decade=D,
            aggregate_first=aggregate_first, amoc_extent=rd['amoc_extent'])
        # Decade label above each panel, clear of the net-cooling-range top
        # chrome (CMIP6 bars + "AMOC projections" reach ~1.06 axes frac).
        ax.text(0.0, 1.07, f'{chr(97 + i)})  {D}', transform=ax.transAxes,
                fontsize=15, fontweight='bold', color=text_color,
                ha='left', va='bottom')

    # Each panel carries its own (right-hand) country row labels, so the
    # inter-panel gutter must be wide enough to hold the longest country name
    # of the left panel without overrunning the right panel.
    fig.subplots_adjust(wspace=0.3)

    savepath = (f'../plots/FigSupp_cmip_cooling_over_fig3_plotbg-{plot_bg}'
                f'_decades-{"-".join(decades)}')
    fig.savefig(savepath + '.png', dpi=200, bbox_inches='tight',
                transparent=True if plot_bg == 'black' else False)
    fig.savefig(savepath + '.pdf', dpi=400, bbox_inches='tight',
                transparent=True if plot_bg == 'black' else False)
    return fig, savepath


########################################
# %%
# RUN

if __name__ == '__main__':
    plot_bg = 'white'  # 'white' or 'black'
    aggregate_first = True
    fig, savepath = make_figure(reg_by_decade, masks, cmip_cooling_ds,
                                decades=decades, plot_bg=plot_bg,
                                aggregate_first=aggregate_first)
    print(f"Saved {savepath}")

# %%
