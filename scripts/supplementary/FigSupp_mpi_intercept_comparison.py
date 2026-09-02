########################################
# %%
# INTERCEPT-CONVENTION COMPARISON — Fig. 1c side-by-side
#
# Fig. 1c fits OLS with a free intercept, while the paper's NAHosMIP-style
# regressions (Fig. 3) force the fit through the origin. This figure shows the
# MPI-ESM regression panel twice: (a) free intercept exactly as in Fig. 1c,
# (b) intercept forced to 0 (regression_plot's fit_intercept=False), so the
# two conventions can be compared directly. Note the returned R² of the
# through-origin fit is statsmodels' uncentred R² — not comparable to (a)'s.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import importlib
import sys, pathlib
if "__file__" in globals():  # standalone run: ensure scripts/ is importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import functions
importlib.reload(functions)

########################################
# %%
# FIGURE FUNCTION

def make_figure(data_dict, season='', window=10, region='EU', hosing='all', plot_bg='white', ylim=None):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10})
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 2, wspace=0.25, top=0.62, bottom=0.11)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)

    # Fig1.py panel-c clean_corner kwargs (Fig1.py:48-60); a single country spreads
    # wider than the EU mean, so ylim is overridable
    panel_kwargs = dict(season=season, hos_type=hosing, window=window, region=region,
                        plot_bg=plot_bg, xlim=(-8, 6), ylim=ylim or (-1.5, 1.0),
                        eqs_x=0.46, equation_y_pos=1.18, equation_y_spacing=0.085,
                        weakening_xaxis=True, simple_eqs=True)
    functions.regression_plot(data_dict, ext_ax=ax_a, **panel_kwargs)
    functions.regression_plot(data_dict, ext_ax=ax_b, fit_intercept=False, **panel_kwargs)

    fg = 'black' if not plot_bg == 'black' else 'white'
    titles = ['Free intercept (as in Fig. 1c)', 'Intercept forced to 0']
    for ax, title in zip([ax_a, ax_b], titles):
        ax.text(0.46, 1.18 + 0.085, "Regression results:", transform=ax.transAxes,
                fontsize=10, verticalalignment='baseline', color=fg, fontweight='bold')
        # one row above the equation block so the long panel-a title clears the header
        ax.text(0.0, 1.35, title, transform=ax.transAxes,
                fontsize=12, verticalalignment='baseline', color=fg, fontweight='bold')

    # Pooled coefficients to stdout so numbers are citable from text, not the PNG
    res_free = functions.regression_plot(data_dict, season=season, hos_type=hosing, window=window, region=region, no_plots=True)
    res_zero = functions.regression_plot(data_dict, season=season, hos_type=hosing, window=window, region=region, no_plots=True, fit_intercept=False)
    print(f"pooled free-intercept:  intercept={res_free[0]} °C, coef={res_free[1]} K/Sv, ste={res_free[2]}, R2={res_free[3]}")
    print(f"pooled zero-intercept:  coef={res_zero[1]} K/Sv, ste={res_zero[2]}, R2_uncentred={res_zero[3]} (not comparable to free-intercept R2)")

    # Bottom %-weakening ruler per panel — mirror Fig1.py panel c
    for ax in [ax_a, ax_b]:
        functions.add_weakening_pct_overlay(fig, ax, plot_bg=plot_bg, drop_parent_xaxis=True)

    # Shared legend — grouped-by-SSP construction from Fig1.py:117-180 ('all' branch)
    handles, labels = ax_a.get_legend_handles_labels()
    column_headings = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0']
    grouped_handles = [[h for h, l in zip(handles, labels) if l.startswith(head)] for head in column_headings]
    grouped_labels = [[l for l in labels if l.startswith(head)] for head in column_headings]
    n = len(grouped_handles[0])
    if hosing == 'all' and n >= 8:
        # Move 'linneg.' from position 7 to position 4
        [grouped_handles[i].insert(4, grouped_handles[i].pop(7)) for i in range(3)]
        [grouped_labels[i].insert(4, grouped_labels[i].pop(7)) for i in range(3)]
    for i in range(3):
        grouped_labels[i] = [l[9:] if len(l) > 10 else l for l in grouped_labels[i]]

    heading_handles = [Line2D([], [], color='none', label=heading, linewidth=0) for heading in column_headings]
    ge_ssp_keys = ['ssp126', 'ssp245', 'ssp370']
    ge_handles = [Line2D([], [], color=functions.hosing_colors[ssp]['ge'],
                         marker=functions.hosing_markers['ge'], linestyle='None',
                         markersize=5, label='MPI-GE') for ssp in ge_ssp_keys]

    final_handles, final_labels = [], []
    for heading, ge, h_group, l_group in zip(heading_handles, ge_handles, grouped_handles, grouped_labels):
        final_handles.append(heading)
        final_labels.append(heading.get_label())
        final_handles.append(ge)
        final_labels.append('MPI-GE')
        final_handles.extend(h_group)
        final_labels.extend(l_group)

    legend = fig.legend(final_handles, final_labels, ncol=3, frameon=False, handletextpad=1.5,
                        columnspacing=2, loc='lower center', bbox_to_anchor=(0.5, 0.82), fontsize=10)
    for text, handle in zip(legend.get_texts(), final_handles):
        if handle in heading_handles:
            text.set_weight('bold')
            text.set_ha('center')
            text.set_color([functions.hosing_colors[ssp]['ge'] for ssp in ge_ssp_keys][heading_handles.index(handle)])

    for label, ax in zip(['a', 'b'], [ax_a, ax_b]):
        fig.text(ax.get_position().x0 - 0.04, ax.get_position().y1, label, transform=fig.transFigure,
                 fontsize=14, ha='right', va='center', fontweight='bold', color=fg)

    # The manifest facet grammar is key-value with '-' as the separator, so a
    # negative bound must not supply its own '-': "_ylim{-2.0}to{1.0}" parsed
    # as ylim=2.0to1.0, silently dropping the sign. Write the separator
    # explicitly and encode a minus as 'm'.
    ylim_suffix = '' if ylim in (None, (-1.5, 1.0)) else \
        '_ylim-' + 'to'.join(f'{v:g}'.replace('-', 'm') for v in ylim)
    savepath = (f"../plots/FigSupp_mpi_intercept_comparison_region-{region}"
                f"_season-{season or 'annual'}_window-{window}_hosing-{hosing}{ylim_suffix}_plotbg-{plot_bg}")
    fig.savefig(savepath + '.png', dpi=200, bbox_inches='tight', transparent=True if plot_bg == 'black' else False)
    fig.savefig(savepath + '.pdf', dpi=400, bbox_inches='tight', transparent=True if plot_bg == 'black' else False)

    return fig, savepath


########################################
# %%
# LOAD DATA

if __name__ == '__main__':
    season, window, hosing = '', 10, 'all'
    data_dict = functions.load_mpi_esm_data(eur_only=True)

########################################
# %%
# RUN — EU panel, then the country where the two conventions differ most
# (largest |intercept|: the free fit sits furthest from the origin there)

if __name__ == '__main__':
    fig, savepath = make_figure(data_dict, season=season, window=window, hosing=hosing, region='EU')

    country_intercepts = {r: functions.regression_plot(data_dict, region=r, season=season, hos_type=hosing,
                                                       window=window, no_plots=True)[0]
                          for r in functions.regions_cutoff30k}
    country_intercepts = {r: v for r, v in country_intercepts.items() if np.isfinite(v)}
    for r, icpt in sorted(country_intercepts.items(), key=lambda kv: -abs(kv[1])):
        print(f"{r}: {icpt:+.4f} degC")
    top_country = max(country_intercepts, key=lambda r: abs(country_intercepts[r]))
    print(f"largest |intercept|: {top_country} ({country_intercepts[top_country]:+.4f} degC; EU pooled printed above)")

    fig_top, savepath_top = make_figure(data_dict, season=season, window=window, hosing=hosing, region=top_country, ylim=(-2.0, 1.0))

# %%
