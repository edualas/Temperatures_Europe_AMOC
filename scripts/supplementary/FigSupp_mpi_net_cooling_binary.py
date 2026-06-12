########################################
# %%
# LOAD PACKAGES

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle
from matplotlib import ticker
plt.rcParams.update({'font.size': 12})
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import colorsys

import importlib
import sys, pathlib
if "__file__" in globals():  # standalone run: ensure scripts/ is importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import functions
importlib.reload(functions)

########################################
# %%
# LOAD REGRESSION DATASETS

if __name__ == '__main__':
    reg_ds_mpi = functions.load_regression_ds_mpi()

########################################
# %%
# IN-SCRIPT BINARY HELPER
# Mirrors functions.net_cooling_point_map but fills only the region where
# req_weakening_<T_ref> < threshold (i.e. pixels that net-cool once AMOC has
# weakened by `threshold` percent). Geographic chrome copied verbatim so the
# binary maps register pixel-for-pixel with Fig. 2.

# Fill colour = second-darkest bin of Fig2's YlGnBu-derived discrete cmap
# (the 10-20% weakening bin after `cmap.reversed()`). Hardcoded here for
# standalone use; keep in sync with functions.net_cooling_point_map cmap
# construction if the palette is ever retuned.
_base_cmap = cm.YlGnBu
_trunc_cmap = functions.truncate_colormap(_base_cmap, minval=0.17, maxval=1.0, darken=1.0)
_binary_fill = _trunc_cmap.reversed()(0.15)  # midpoint of bin 1 (10-20%), 2nd from top
_binary_fill_pd = _trunc_cmap.reversed()(0.45)  # midpoint of bin 4 (40-50%), 5th from top


def binary_net_cooling_map(reg_ds, ssp, threshold, T_ref='pi', season='',
                           ext_ax=None, plot_bg='white', fill_color=None,
                           title=False):
    """Binary fill: pixels where req_weakening_<T_ref> < threshold (%)."""
    mid_gray = '#b0b0b0'
    if fill_color is None:
        fill_color = _binary_fill

    plt.style.use('default')
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    if ext_ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.Robinson(central_longitude=8)})
    else:
        ax = ext_ax
    ax.spines['geo'].set_visible(False)
    ax.set_rasterized(True)
    ax.coastlines(color='black' if not plot_bg == 'black' else 'white', linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS, edgecolor='black' if not plot_bg == 'black' else 'white', linewidth=.5)
    ax.add_feature(cfeature.LAND, facecolor=mid_gray)
    ax.add_feature(cfeature.OCEAN, facecolor='white' if not plot_bg == 'black' else '#191919', zorder=2)

    # 'combined': overlay both baselines, pd (lighter) under, pi (smaller) on top.
    if T_ref == 'combined':
        layers = [('pd', _binary_fill_pd), ('pi', _binary_fill)]
    else:
        layers = [(T_ref, fill_color)]

    # contourf binary fill: pixels meeting the threshold map to 1, others to NaN.
    # Casting (<) to float would give 0/1, but 0 would also pick up the fill colour
    # via contourf interpolation. Using NaN for the 'false' class keeps them transparent.
    for tref, fc in layers:
        var = reg_ds[f'req_weakening_{tref}']
        if 'season' in var.dims:
            data = var.sel(scenar=ssp, season=season)
        else:
            data = var.sel(scenar=ssp)
            if season != '':
                print('Seasonal binary net cooling not available, falling back to annual.')
        mask = (data < threshold).where(data < threshold)  # 1 where True, NaN where False
        contour = ax.contourf(
            reg_ds.lon, reg_ds.lat, mask,
            levels=[0.5, 1.5], colors=[fc],
            transform=ccrs.PlateCarree(),
            extend='neither',
        )

    ax.set_extent([-18.5, 38.5, 34.8, 72], crs=ccrs.PlateCarree())
    functions.add_square(ax, -13.5, 0, 60, 75.5, colour='#191919' if plot_bg == 'black' else 'white')
    functions.add_square(ax, -30, -18, 68, 75, colour='#191919' if plot_bg == 'black' else 'white')

    if title:
        T_ref_str = {'pi': 'preindustrial', 'pd': 'present-day',
                     'combined': 'preindustrial & present-day'}[T_ref]
        ax.set_title(f'Net cooling at {threshold}% weakening, {ssp} (vs {T_ref_str})')

    if ext_ax is not None:
        return contour
    return contour


########################################
# %%
# FIGURE FUNCTION
# Top row: scaling factors (annual/JJA/DJF), same as Fig. 2.
# Rows 2-4: binary net-cooling maps at three weakening thresholds, three SSPs.

def make_figure(reg_ds_mpi, plot_season='', plot_bg='white', T_ref='pi',
                thresholds=(30, 50, 80), omit_scaling_row=True):
    assert len(thresholds) == 3, 'thresholds must have length 3 (gridspec is fixed)'

    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    ncols = 3
    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    reg_plot_labels = ['Annual mean', 'Summer (JJA)', 'Winter (DJF)']

    # Layout: with the scaling row, 5 grid rows (scaling + spacer + 3 thresholds);
    # without it, just the 3 threshold rows. row_offset is the first binary grid row.
    if omit_scaling_row:
        nrows, gridspec_kw, row_offset, fig_h = 3, {'height_ratios': [1, 1, 1]}, 0, 9
    else:
        nrows, gridspec_kw, row_offset, fig_h = 5, {'height_ratios': [1, 0.1, 1, 1, 1]}, 2, 12

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(12, fig_h),
        subplot_kw={'projection': ccrs.Robinson(central_longitude=8)},
        constrained_layout=False, gridspec_kw=gridspec_kw,
    )
    axes = axes.flatten()

    # Hide spacer row (grid indices 3, 4, 5) when the scaling row is present.
    if not omit_scaling_row:
        for i in range(ncols):
            axes[3 + i].set_visible(False)
            axes[3 + i].remove()

    plt.subplots_adjust(hspace=0.35, wspace=-0.3, top=0.95, bottom=0.05, left=0.05, right=0.95)

    panel_axes = []
    reg_mappables = []

    # -------- Top row: scaling factors (identical to Fig2) --------
    if not omit_scaling_row:
        v_max = 0.7
        custom_cmap = functions.get_custom_cmap()
        for i, season in enumerate(['', 'jja', 'djf']):
            mappable = functions.plot_regression_coefficients(
                reg_ds_mpi, season=season, ext_ax=axes[i], plot_bg=plot_bg,
                custom_cmap=custom_cmap, colorbar_v_max=v_max,
            )
            reg_mappables.append(mappable)
            panel_axes.append(axes[i])
            axes[i].text(0.5, -0.03, reg_plot_labels[i], transform=axes[i].transAxes,
                         fontsize=12, ha='center', va='top', fontweight='bold',
                         color='black' if not plot_bg == 'black' else 'white')

    # -------- Binary threshold rows --------
    # Row r (0..2) → axes indices 3*(r+2) .. 3*(r+2)+2 (skipping spacer row 1).
    for r, threshold in enumerate(thresholds):
        for i, ssp in enumerate(functions.ssps):
            ax_idx = ncols * (row_offset + r) + i
            binary_net_cooling_map(
                reg_ds_mpi, ssp, threshold, T_ref=T_ref, season=plot_season,
                ext_ax=axes[ax_idx], plot_bg=plot_bg,
            )
            panel_axes.append(axes[ax_idx])
            axes[ax_idx].text(
                0.5, -0.03, functions.ssp_labels[ssp],
                transform=axes[ax_idx].transAxes,
                fontsize=12, ha='center', va='top',
                color=functions.hosing_colors[ssp]['ge'], fontweight='bold',
            )

    # -------- Row labels --------
    season_str = 'annual' if plot_season == '' else plot_season.upper()
    if T_ref == 'combined':
        baseline_suffix = (f'w.r.t. {season_str} mean preindustrial (1850-1899, dark blue) \n'
                           f'and present-day ({functions.PD_LABEL}, light blue) temperatures')
    else:
        T_ref_str = 'preindustrial' if T_ref == 'pi' else 'present-day'
        baseline_years = '1850-1899' if T_ref == 'pi' else functions.PD_LABEL
        baseline_suffix = f'w.r.t. {season_str} mean {T_ref_str} temperatures ({baseline_years})'

    binary_labels = [
        f'Net cooling region for {thresholds[0]}% AMOC weakening, {baseline_suffix}',
        f'Net cooling region for {thresholds[1]}% AMOC weakening',
        f'Net cooling region for {thresholds[2]}% AMOC weakening',
    ]
    left = axes[0].get_position().x0
    label_kw = dict(va='center', ha='left', fontsize=12, fontweight='bold',
                    color='black' if not plot_bg == 'black' else 'white')
    if not omit_scaling_row:
        fig.text(left, axes[0].get_position().y1 + 0.02,
                 'Scaling factors: local cooling as a function of additional AMOC weakening',
                 **label_kw)
    for r, lbl in enumerate(binary_labels):
        row_top = axes[ncols * (row_offset + r)].get_position().y1
        fig.text(left, row_top + 0.02, lbl, **label_kw)

    # -------- Panel labels --------
    for ax, label in zip(panel_axes, panel_labels):
        ax_left = ax.get_position().x0
        ax_top = ax.get_position().y1
        fig.text(ax_left + 0.015, ax_top - 0.01, label, transform=fig.transFigure,
                 fontsize=14, ha='right', va='center', fontweight='bold',
                 color='black' if not plot_bg == 'black' else 'white')

    # -------- Right-hand sidebar --------
    # Upper: scaling-factor colourbar (unchanged from Fig2).
    # Lower: faux colourbar — plain Axes with inverted 0-100% AMOC weakening
    # y-axis, AMOC-strength secondary axis, SSP CMIP6-range rectangles + MPI
    # projection ticks (verbatim from Fig2), plus dashed horizontal lines at
    # the three threshold values to anchor each binary row.
    right = axes[-1].get_position().x1
    cbar_x = right + 0.07
    cbar_width = 0.02

    cbar2_y = axes[-1].get_position().y0
    cbar2_height = axes[ncols * row_offset].get_position().y1 - cbar2_y  # top of first binary row
    sidebar = fig.add_axes([cbar_x, cbar2_y, cbar_width, cbar2_height])

    # Upper colourbar (scaling factors) — only when the scaling row is shown.
    if not omit_scaling_row:
        cbar1_y = axes[1].get_position().y0
        cbar1_height = axes[1].get_position().y1 - cbar1_y
        cax1 = fig.add_axes([cbar_x, cbar1_y, cbar_width, cbar1_height])
        cbar1 = fig.colorbar(reg_mappables[2], cax=cax1, orientation='vertical')
        cbar1.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.1f}°C"))
        cbar1.set_label('Cooling per 10% additional\nAMOC weakening', fontsize=12,
                        labelpad=5, loc='center',
                        color='black' if not plot_bg == 'black' else 'white')

    # Sidebar: plain Axes, AMOC weakening 0-100% inverted. Ticks on the left only.
    text_color = 'black' if plot_bg != 'black' else 'white'
    sidebar.set_xlim(0, 1)
    sidebar.set_ylim(0, 100)
    sidebar.invert_yaxis()
    sidebar.set_xticks([])
    sidebar.set_yticks(list(range(0, 101, 10)))
    sidebar.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0f}%"))
    sidebar.yaxis.set_ticks_position('left')
    sidebar.yaxis.set_label_position('left')
    sidebar.set_ylabel('AMOC weakening', fontsize=12, labelpad=0.5, color=text_color)
    for s in ['top', 'bottom', 'right']:
        sidebar.spines[s].set_visible(False)
    sidebar.spines['left'].set_color(text_color)

    # SSP CMIP6 range rectangles + MPI projected line (mirror Fig2 weakening_primary=True).
    range_positions = [0.9, 2.1, 3.3]
    for i, ssp in enumerate(functions.ssps):
        rect_y = 1 - functions.AMOC_extent[ssp][0] / 100
        rect_height = (functions.AMOC_extent[ssp][0] - functions.AMOC_extent[ssp][1]) / 100
        rect = Rectangle(
            (range_positions[i], rect_y), 1.0, rect_height,
            transform=sidebar.transAxes,
            color=functions.hosing_colors[ssp]['ge'],
            alpha=1.0, clip_on=False,
        )
        sidebar.add_patch(rect)

        mpi_amoc_strength = float(reg_ds_mpi.AMOC_future.isel(season=0).sel(scenar=ssp))
        mpi_weakening = functions.convert_strength_to_weakening(mpi_amoc_strength)
        mpi_y = 1 - mpi_weakening / 100
        base_rgb = plt.matplotlib.colors.to_rgb(functions.hosing_colors[ssp]['ge'])
        h, l, s = colorsys.rgb_to_hls(*base_rgb)
        line_color = colorsys.hls_to_rgb(h, max(0, l * 0.7), min(1, s * 1.3))
        sidebar.plot(
            [range_positions[i] - 0.08, range_positions[i] + 1.08], [mpi_y, mpi_y],
            color=line_color, linewidth=2,
            transform=sidebar.transAxes, clip_on=False, zorder=11,
        )

        text_y = 1 - (functions.AMOC_extent[ssp][0] + functions.AMOC_extent[ssp][1]) / 200
        sidebar.text(
            range_positions[i] + 0.6, text_y,
            f"{functions.ssp_labels[ssp]} 2100 CMIP6 range",
            color='white', fontsize=9, rotation=90, va='center', ha='center',
            transform=sidebar.transAxes, zorder=12,
        )

    # -------- Edge borders (mirror Fig2) --------
    border_width = 0.001
    border_color = 'white' if plot_bg == 'white' else '#191919'
    for ax in fig.get_axes():
        if hasattr(ax, 'projection'):
            bbox = ax.get_position()
            fig.add_artist(plt.Rectangle((bbox.x0, bbox.y1 - border_width), bbox.width, border_width,
                                         transform=fig.transFigure, color=border_color, zorder=1000, clip_on=False))
            fig.add_artist(plt.Rectangle((bbox.x0, bbox.y0), bbox.width, border_width,
                                         transform=fig.transFigure, color=border_color, zorder=1000, clip_on=False))
            fig.add_artist(plt.Rectangle((bbox.x0, bbox.y0), border_width, bbox.height,
                                         transform=fig.transFigure, color=border_color, zorder=1000, clip_on=False))
            fig.add_artist(plt.Rectangle((bbox.x1 - border_width, bbox.y0), border_width, bbox.height,
                                         transform=fig.transFigure, color=border_color, zorder=1000, clip_on=False))

    savepath = (
        f"../plots/FigSupp_mpi_net_cooling_binary_plotbg-{plot_bg}_season-{plot_season or 'annual'}"
        f'_Tref-{T_ref}'
        f'_thresh-{"-".join(str(t) for t in thresholds)}'
        f'_omitscale-{omit_scaling_row}'
    )
    fig.savefig(savepath + '.png', bbox_inches='tight',
                transparent=True if plot_bg == 'black' else False, dpi=200)
    fig.savefig(savepath + '.pdf', bbox_inches='tight', dpi=400,
                transparent=True if plot_bg == 'black' else False)
    print(f'wrote {savepath}.png/.pdf')

    return fig, savepath


########################################
# %%
# RUN

if __name__ == '__main__':
    plot_bg = 'white'         # 'white' or 'black'
    plot_season = ''          # '', 'djf', 'jja'
    T_ref = 'combined'              # 'pi', 'pd', or 'combined'
    thresholds = (30, 50, 80)  # AMOC weakening percentages, length 3
    omit_scaling_row = True    # True: drop the top scaling-factor row entirely

    fig, savepath = make_figure(
        reg_ds_mpi, plot_season=plot_season, plot_bg=plot_bg,
        T_ref=T_ref, thresholds=thresholds, omit_scaling_row=omit_scaling_row,
    )

# %%
