########################################
# %%
# LOAD PACKAGES

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

import importlib
import sys, pathlib
if "__file__" in globals():  # standalone run: ensure scripts/ is importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import functions
importlib.reload(functions)
import FigSupp_synthetic_scaling_factors as ssf  # type: ignore
importlib.reload(ssf)

########################################
# %%
# STATE-DEPENDENCE / DISPLAY-MODE ROBUSTNESS (supplementary)
#
# Two 2-panel versions of the Synthetic CMIP6 range comparison, each panel
# with the NAHosMIP overlay and GISS markers (split from a single 3-panel
# figure on 2026-09-01 for legibility):
#   version='mode':     (a) no correction, 'medians' display vs
#                       (b) no correction, 'mc' display
#   version='statedep': (a) no correction vs (b) pooled correction,
#                       both at the 'medians' display (paper default)
# cmip_range_mode is a render-time selector only (both 'medians' and 'mc'
# fields live in the same cached dataset), so the 'mode' version reuses one
# loaded cmip_range_ds. Each version is rendered once for the default 'w'
# predictors and once for the 'wt' sibling.

PANEL_LABELS = {
    'none':   'No state-dependence correction',
    'pooled': 'State-dependence correction',
}

########################################
# %%
# LOAD DATA

if __name__ == '__main__':
    multi_model_dict, masks = functions.get_full_multi_model_dict()
    reg_ds_mpi = functions.load_regression_ds_mpi()
    reg_ds_cesm = functions.get_cesm_reg_ds(recompute=False)
    reg_ds_giss = functions.get_giss_reg_ds(recompute=False, masks=masks)
    hosmip_reg_ds_dict = functions.get_hosmip_reg_ds(recompute=False)
    # Keyed by (state_dep, predictors); both w (default) and wt siblings.
    cmip_range_ds = {
        (sd, pred): ssf.get_cmip_range_ds(recompute=False, state_dep=sd,
                                          cal_set='hosing', predictors=pred,
                                          season='')
        for sd in ('none', 'pooled') for pred in ('w', 'wt')
    }


########################################
# %%
# FIGURE FUNCTION

def make_figure(reg_ds_mpi, reg_ds_cesm, masks, hosmip_reg_ds_dict,
                cmip_range_ds_none, cmip_range_ds_pooled, reg_ds_giss=None,
                giss_time_period='2101-2300', plot_bg='white',
                aggregate_first=True, weakening_unit='pct',
                cmip_range_predictors='w', version='statedep'):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    amoc_extent = functions.get_amoc_extent(None, unit=weakening_unit)
    if version == 'statedep':
        panels = [
            ('a', PANEL_LABELS['none'] + ' (medians)',   cmip_range_ds_none,   'medians'),
            ('b', PANEL_LABELS['pooled'] + ' (medians)', cmip_range_ds_pooled, 'medians'),
        ]
    elif version == 'mode':
        panels = [
            ('a', PANEL_LABELS['none'] + ' (medians)',     cmip_range_ds_none, 'medians'),
            ('b', PANEL_LABELS['none'] + ' (Monte Carlo)', cmip_range_ds_none, 'mc'),
        ]
    else:
        raise ValueError(f"version must be 'statedep' or 'mode'; got {version!r}")

    text_color = 'black' if plot_bg != 'black' else 'white'
    fig, axes = plt.subplots(1, len(panels), figsize=(8.5 * len(panels), 14))
    for ax, (letter, desc, cr_ds, mode) in zip(axes, panels):
        functions.plot_net_cooling_ranges_mpi_cesm(
            reg_ds_mpi, reg_ds_cesm, masks,
            hosmip_reg_ds_dict=hosmip_reg_ds_dict,
            season='', T_ref='pi', plot_bg=plot_bg, ext_ax=ax, title=False,
            reg_ds_giss=reg_ds_giss, giss_time_period=giss_time_period,
            aggregate_first=aggregate_first, amoc_extent=amoc_extent,
            weakening_unit=weakening_unit, cmip_range_ds=cr_ds,
            cmip_range_mode=mode, nahosmip_overlay=True)
        # Decade label placement convention from FigSupp_cmip_cooling.py,
        # cleared of the net-cooling-range top chrome (CMIP6 bars +
        # "Projected AMOC weakening" reach ~1.06 axes frac).
        ax.text(0.0, 1.07, f'{letter})  {desc}', transform=ax.transAxes,
                fontsize=15, fontweight='bold', color=text_color,
                ha='left', va='bottom')

    # Each panel carries its own (right-hand) country row labels, so the
    # inter-panel gutter must be wide enough to hold the longest country name
    # of the left panel without overrunning the right panel.
    fig.subplots_adjust(wspace=0.3)

    savepath = (f'../plots/FigSupp_cmip_range_{version}_robustness_plotbg-{plot_bg}'
                f'_cal-hosing_pred-{cmip_range_predictors}')
    functions.check_savepath(savepath)
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
    weakening_unit = 'pct'
    giss_time_period = '2101-2300'
    for _version in ('statedep', 'mode'):
        for _pred in ('w', 'wt'):
            fig, savepath = make_figure(
                reg_ds_mpi, reg_ds_cesm, masks, hosmip_reg_ds_dict,
                cmip_range_ds[('none', _pred)], cmip_range_ds[('pooled', _pred)],
                reg_ds_giss=reg_ds_giss, giss_time_period=giss_time_period,
                plot_bg=plot_bg, aggregate_first=aggregate_first,
                weakening_unit=weakening_unit, cmip_range_predictors=_pred,
                version=_version)
            print(f"Saved {savepath}")

# %%
