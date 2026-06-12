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
import plot_warming_at_amoc_weakening as pw # type: ignore
importlib.reload(pw)

########################################
# %%
# WARMING-AT-FIXED-WEAKENING (supplementary)
#
# Two side-by-side panels of pw.make_figure (the flipped net-cooling figure):
# per country, the end-of-century ΔT above 1850-1899 each (model, scenario)
# line predicts at a chosen AMOC-weakening level. Left = 25 %, right = 75 %.
# Layout mirrors FigSupp_cmip_cooling.py; per-panel legend (self-describing via
# its weakening-% title) is intentional.

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
# FIGURE FUNCTION

def make_figure(reg_ds_mpi, reg_ds_cesm, masks, hosmip_reg_ds_dict,
                reg_ds_giss=None, weakenings=(25.0, 75.0),
                T_ref='pi', season='', giss_time_period='2101-2300',
                plot_bg='white'):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    n = len(weakenings)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 14))
    if n == 1:
        axes = [axes]

    text_color = 'black' if plot_bg != 'black' else 'white'
    for i, (ax, w) in enumerate(zip(axes, weakenings)):
        pw.make_figure(
            reg_ds_mpi, reg_ds_cesm, masks,
            hosmip_reg_ds_dict=hosmip_reg_ds_dict,
            amoc_weakening_pct=w, T_ref=T_ref, season=season,
            reg_ds_giss=reg_ds_giss, giss_time_period=giss_time_period,
            plot_bg=plot_bg, ext_ax=ax, title=False, savefig=False)
        ax.text(0.0, 1.03, f'{chr(97 + i)}) End-of-century $\Delta \mathrm{{T}}$ at {w:.0f}% AMOC weakening',
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                color=text_color, ha='left', va='bottom')

    # Each panel carries its own right-hand country labels; widen the gutter so
    # the left panel's labels don't overrun the right panel (cf. FigSupp_cmip_cooling).
    fig.subplots_adjust(wspace=0.3)

    savepath = (f'../plots/FigSupp_warming_at_weakening_'
                f'w-{"-".join(f"{w:.0f}" for w in weakenings)}'
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
    plot_bg = 'white'  # 'white' or 'black'
    fig, savepath = make_figure(reg_ds_mpi, reg_ds_cesm, masks, hosmip_reg_ds_dict,
                                reg_ds_giss=reg_ds_giss, weakenings=(25.0, 75.0),
                                season='', T_ref='pi', plot_bg=plot_bg)
    print(f"Saved {savepath}")

# %%
