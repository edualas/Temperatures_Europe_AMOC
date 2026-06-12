# %%
# Flipped variant of functions.plot_net_cooling_ranges_mpi_cesm:
# at a chosen AMOC weakening level (default 100 %), show the warming
# above 1850-1899 that each (model, scenario, country) line predicts.
# X-axis is delta T [°C] w.r.t. preindustrial (or 1991-2020 with T_ref='pd').
#
# Math (per pixel, per model, per scenario):
#   delta_T(w) = (T_2090 - T_pi) + m * (w - w_2090)
# with m in K per %-weakening. Per-model unit handling lives in
# `_slope_pct(...)` below.

import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import sys, pathlib
if "__file__" in globals():  # standalone run: ensure scripts/ is importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import functions
importlib.reload(functions)


def _slope_pct(coef, model_amoc_pi, slope_units):
    """Return regression slope in K per %-weakening.

    `slope_units='K_per_Sv'`  → multiply by -model_amoc_pi/100.
    `slope_units='K_per_pct'` → already in this unit; return as-is.
    See functions.py:1011 (MPI, K/Sv) vs functions.py:3235 (CESM,
    K per %) for the unit convention.
    """
    if slope_units == 'K_per_Sv':
        return coef * (-model_amoc_pi / 100.0)
    if slope_units == 'K_per_pct':
        return coef
    raise ValueError(f"unknown slope_units: {slope_units!r}")


def _w_2090(amoc_2090, amoc_pi):
    """Projected 2090-2100 AMOC weakening in % of model PI."""
    return (amoc_pi - amoc_2090) / amoc_pi * 100.0


def _delta_T_at_w(delta_T_2090, w_2090, m_pct, w_target):
    """Linear projection: ΔT at AMOC weakening w_target."""
    return delta_T_2090 + m_pct * (w_target - w_2090)


def make_figure(reg_ds_mpi, reg_ds_cesm, masks,
                hosmip_reg_ds_dict=None,
                amoc_weakening_pct=100.0,
                T_ref='pi', season='',
                hosmip_markers=False,
                reg_ds_giss=None, giss_time_period='2101-2300',
                plot_bg='white', ext_ax=None, title=True, savefig=False):
    plt.style.use('default')
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    if ext_ax is None:
        fig, ax = plt.subplots(figsize=(10, 14))
    else:
        ax = ext_ax

    if T_ref not in ('pi', 'pd'):
        raise NotImplementedError("Only T_ref='pi' and T_ref='pd' are implemented.")

    if reg_ds_giss is not None:
        reg_ds_giss_sel = reg_ds_giss.sel(time_period=giss_time_period)

    regions_cutoff30k = [r for r in functions.regions
                         if r not in ['LU', 'CY', 'XK', 'ME', 'SI', 'MK', 'AL']]
    countries = list(reversed(regions_cutoff30k))

    group_centers = np.arange(len(countries) + 1)  # +1 for the EU aggregate
    bar_height = 0.25
    bar_alpha = 0.7
    line_alpha = 1.0
    vertical_offsets = {'ssp126': 1., 'ssp245': 0., 'ssp370': -1.}

    legend_color = 'white' if plot_bg == 'black' else 'black'
    legend_handles = []
    legend_labels = []

    legend_handles.append(Line2D([], [], marker='*', color=legend_color,
                                 linestyle='None', markersize=8))
    legend_labels.append('MPI-ESM1.2-LR (this study)')

    if season == '':
        legend_handles.append(Line2D([], [], marker='d', color=legend_color,
                                     linestyle='None', markersize=5))
        legend_labels.append('CESM2 (Boot et al. 2024)')

        legend_handles.append(Line2D([], [], color=legend_color, alpha=bar_alpha,
                                     linewidth=5, solid_capstyle='butt'))
        legend_labels.append('Combined forcing range (MPI-ESM1.2-LR & CESM2)')

    if reg_ds_giss is not None:
        legend_handles.append(Line2D([], [], marker='o', color=legend_color,
                                     linestyle='None', markersize=5))
        legend_labels.append(
            f'GISS-E2-1-G (Romanou et al. 2023, {giss_time_period} fit)')

    hosmip_bar_h = Line2D([], [], color=legend_color, alpha=0.3,
                          linewidth=5, solid_capstyle='butt')
    legend_handles.append(hosmip_bar_h)
    legend_labels.append('Preindustrial hosing range (NAHosMIP models)')

    if hosmip_markers:
        for mkr, sz, lbl in [
                ('*', 4, 'NAHosMIP MPI-ESM1.2-LR'), ('^', 4, 'NAHosMIP MPI-ESM1.2-HR'),
                ('d', 4, 'NAHosMIP CESM2'),         ('<', 4, 'NAHosMIP HadGEM3-GC3.1-MM'),
                ('>', 4, 'NAHosMIP HadGEM3-GC3.1-LL'), ('v', 4, 'NAHosMIP EC-Earth3'),
                ('X', 4, 'NAHosMIP CanESM5'),       ('P', 4, 'NAHosMIP IPSL-CM6A-LR')]:
            legend_handles.append(Line2D([], [], marker=mkr, color=legend_color,
                                         linestyle='None', markersize=sz, alpha=0.5))
            legend_labels.append(lbl)

    # Same land-aware EU restriction as the original (functions.py:4688).
    def _select(da, model, region):
        da = da.where(masks[model][region])
        if region == 'EU':
            da = da.where(masks[model]['LAND'] == 0)
        return da

    def _strip_season(da):
        return da.sel(season=season) if 'season' in da.dims else da

    # Per-pixel ΔT(w) field for one (model, ssp) pair. Returns a DataArray
    # in K (positive = warming above the chosen reference).
    def _delta_T_field(reg_ds, ssp_i, *, model_kind):
        T_base = reg_ds.T_pi if T_ref == 'pi' else reg_ds.T_pd_245
        T_base = _strip_season(T_base)
        T2090  = _strip_season(reg_ds.T_future.sel(scenar=ssp_i))
        delta_T_2090 = T2090 - T_base

        amoc_2090 = _strip_season(reg_ds.AMOC_future.sel(scenar=ssp_i))

        if model_kind == 'mpi':
            w2090 = functions.convert_strength_to_weakening(amoc_2090)
            m_pct = _slope_pct(_strip_season(reg_ds.coef_ensmean),
                               functions.AMOC_pi_MPI, 'K_per_Sv')
        elif model_kind == 'cesm':
            w2090 = _w_2090(amoc_2090, reg_ds.AMOC_pi)
            m_pct = _slope_pct(_strip_season(reg_ds.coef_ensmean),
                               None, 'K_per_pct')
        elif model_kind == 'hosmip':
            w2090 = _w_2090(amoc_2090, reg_ds.AMOC_pi)
            # Mirror functions.py:3039's sign filter so we project only
            # along cooling-with-weakening fits (consistent contributing
            # pixel set with the original net-cooling figure).
            lin = reg_ds.lin_coef_hosmip.where(reg_ds.lin_coef_hosmip < 0)
            m_pct = _slope_pct(_strip_season(lin), None, 'K_per_pct')
        elif model_kind == 'giss':
            w2090 = _w_2090(amoc_2090, functions.AMOC_pi_GISS)
            m_pct = _slope_pct(_strip_season(reg_ds.coef_ensmean),
                               functions.AMOC_pi_GISS, 'K_per_Sv')
        else:
            raise ValueError(model_kind)

        return _delta_T_at_w(delta_T_2090, w2090, m_pct, amoc_weakening_pct)

    def _country_value(field_da, model, region):
        return float(functions.weighted_area_lat(
            _select(field_da, model, region)).mean('lat').mean('lon'))

    for (i, r) in enumerate(['EU'] + countries):
        for ssp_i in functions.ssps:

            mpi_field  = _delta_T_field(reg_ds_mpi,  ssp_i, model_kind='mpi')
            cesm_field = _delta_T_field(reg_ds_cesm, ssp_i, model_kind='cesm')

            mpi_value  = _country_value(mpi_field,  'MPI-ESM1-2-LR', r)
            cesm_value = _country_value(cesm_field, 'CESM2',         r)

            hosmip_values = []
            for model in functions.hosmip_labels:
                h_field = _delta_T_field(
                    hosmip_reg_ds_dict[model], ssp_i, model_kind='hosmip')
                h_val = _country_value(h_field, model, r)
                hosmip_values.append(h_val)

                if hosmip_markers:
                    mk = {'MPI-ESM1-2-LR':'*','MPI-ESM1-2-HR':'^','CESM2':'d',
                          'HadGEM3-GC3-1MM':'<','HadGEM3-GC3-1LL':'>',
                          'EC-Earth3':'v','CanESM5':'X','IPSL-CM6A-LR':'P'
                          }.get(model, '')
                    ax.scatter(h_val,
                               group_centers[i] + vertical_offsets[ssp_i] * bar_height,
                               alpha=bar_alpha, color=functions.hosing_colors[ssp_i]['ge'],
                               marker=mk, s=10, clip_on=True)

            # NAHosMIP min–max range (thin transparent bar)
            if np.isfinite(np.nanmin(hosmip_values)) and np.isfinite(np.nanmax(hosmip_values)):
                ax.barh(group_centers[i] - 0.01 + vertical_offsets[ssp_i] * bar_height,
                        np.nanmax(hosmip_values) - np.nanmin(hosmip_values),
                        left=np.nanmin(hosmip_values),
                        height=bar_height * 0.8, alpha=0.3,
                        color=functions.hosing_colors[ssp_i]['ge'], clip_on=True)

            # MPI–CESM combined forcing range (thicker, opaque)
            if season == '' and np.isfinite(mpi_value) and np.isfinite(cesm_value):
                ax.barh(group_centers[i] - 0.01 + vertical_offsets[ssp_i] * bar_height,
                        mpi_value - cesm_value,
                        left=cesm_value,
                        height=bar_height * 0.8, alpha=bar_alpha,
                        color=functions.hosing_colors[ssp_i]['ge'])

            ax.scatter(mpi_value,
                       group_centers[i] + vertical_offsets[ssp_i] * bar_height,
                       alpha=line_alpha, color=functions.hosing_colors[ssp_i]['ge'],
                       marker='*', s=50)
            if season == '':
                ax.scatter(cesm_value,
                           group_centers[i] + vertical_offsets[ssp_i] * bar_height,
                           alpha=line_alpha, color=functions.hosing_colors[ssp_i]['ge'],
                           marker='d', s=25)

            if reg_ds_giss is not None:
                giss_field = _delta_T_field(reg_ds_giss_sel, ssp_i, model_kind='giss')
                giss_value = _country_value(giss_field, 'GISS-E2-1-G', r)
                ax.scatter(giss_value,
                           group_centers[i] + vertical_offsets[ssp_i] * bar_height,
                           alpha=line_alpha, color=functions.hosing_colors[ssp_i]['ge'],
                           marker='o', s=30)

    # Reference line at ΔT = 0 (PI / PD baseline).
    # ymin=0.03 stops at the bottom spine position (set below), so the
    # line doesn't extend below the visible x-axis.
    ax.axvline(x=0, ymin=0.03,
               color='black' if plot_bg != 'black' else 'white',
               linewidth=1.0, alpha=0.9, zorder=2)

    # Shade every second row. `clip_on=False` lets the shading extend
    # past the right spine into the country-name area, matching Fig.3.
    x_lo = -10
    x_hi = 10
    ax.set_xlim(x_lo, x_hi)
    # Start at left spine, extend ~20% past the right spine into the
    # country-name strip (mirrors the original `Rectangle((0, ...), 120, ...)`
    # at functions.py:4823 with its xlim of (0, 100)).
    shade_left = x_lo
    shade_width = (x_hi - x_lo) * 1.2
    for i in range(len(countries) + 1):
        if i % 2 == 1:
            ax.add_patch(Rectangle((shade_left, group_centers[i] - 0.5),
                                   shade_width, 1,
                                   color='black', alpha=0.15, zorder=0,
                                   linewidth=0, clip_on=False))

    # Gridlines every 1 °C (skip x=0 — already drawn as the reference line).
    # ymin=0.03 matches the spine position so they stop at the x-axis.
    for x in np.arange(np.ceil(x_lo), x_hi + 0.5, 1.0):
        if abs(x) < 1e-9:
            continue
        ax.axvline(x=x, ymin=0.03,
                   color='black' if plot_bg != 'black' else 'white',
                   linestyle=(0, (1, 4)), linewidth=0.8, alpha=1.0, zorder=0)

    ax.set_ylim(-1.7, len(countries) + 0.5)

    ref_temperature = '1850–1899' if T_ref == 'pi' else functions.PD_LABEL
    season_text = ' in winter' if season == 'djf' else ' in summer' if season == 'jja' else ''

    ax.set_xlabel(rf'$\Delta \mathrm{{T}}$ [°C] w.r.t. {ref_temperature}', labelpad=5)

    if title:
        ax.set_title(
            rf'End-of-century $\Delta \mathrm{{T}}$ at {amoc_weakening_pct:.0f}% AMOC weakening'
            f'{season_text}',
            fontsize=14,
            color='black' if plot_bg != 'black' else 'white')

    # Always-on annotation: keeps the figure self-describing when title=False.
    # ax.text(0.98, 0.985,
    #         f'AMOC weakening: {amoc_weakening_pct:.0f}%',
    #         transform=ax.transAxes, ha='right', va='top', fontsize=11, weight='bold',
    #         color='black' if plot_bg != 'black' else 'white',
    #         bbox=dict(boxstyle='round,pad=0.3',
    #                   fc='white' if plot_bg != 'black' else '#222',
    #                   ec='none', alpha=0.85), zorder=20)

    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_label_position('bottom')
    ax.tick_params(axis='x', which='both', top=True, labeltop=True,
                   bottom=True, labelbottom=True, labelsize=10)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.set_yticks(group_centers)
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_yticklabels(['European mean'] + [functions.country_names[c] for c in countries])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_position(('axes', 0.03))

    ax.legend(legend_handles, legend_labels,
              frameon=True, bbox_to_anchor=(0.02, 0.055),
              loc='lower left', fontsize=10,
              title=('Annual' if season == '' else f'Seasonal ({season.upper()})')
                    + f' warming at {amoc_weakening_pct:.0f}% weakening',
              title_fontproperties={'weight': 'bold', 'size': 10})

    if savefig and ext_ax is None:
        fname = (f'../plots/warming_at_amoc_weakening_w-{amoc_weakening_pct:.0f}'
                 f'_Tref-{T_ref}_season-{season}_plotbg-{plot_bg}'
                 f'_hosmip_markers-{hosmip_markers}')
        fig.savefig(fname + '.png', dpi=400, bbox_inches='tight',
                    transparent=True if plot_bg == 'black' else False)
        fig.savefig(fname + '.pdf', bbox_inches='tight',
                    transparent=True if plot_bg == 'black' else False)
        print(f'wrote {fname}.png and .pdf')

    return ax


# %%
if __name__ == '__main__':
    multi_model_dict, masks = functions.get_full_multi_model_dict()
    reg_ds_mpi  = functions.load_regression_ds_mpi()
    reg_ds_cesm = functions.get_cesm_reg_ds(recompute=False)
    reg_ds_giss = functions.get_giss_reg_ds(recompute=False)
    hosmip_reg_ds_dict = functions.get_hosmip_reg_ds(recompute=False)

    # Default: full AMOC collapse (w = 100 %).
    make_figure(reg_ds_mpi, reg_ds_cesm, masks,
                hosmip_reg_ds_dict=hosmip_reg_ds_dict,
                amoc_weakening_pct=30.0,
                T_ref='pi', season='',
                reg_ds_giss=reg_ds_giss,
                plot_bg='white', title=True, savefig=True)

# %%
