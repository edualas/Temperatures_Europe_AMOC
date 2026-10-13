########################################
# %%
# LOAD PACKAGES

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker
import matplotlib.cm as cm
plt.rcParams.update({'font.size': 12})
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import importlib
import sys, pathlib
if "__file__" in globals():  # standalone run: ensure scripts/ is importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import functions
importlib.reload(functions)

########################################
# %%
# LOAD MPI-ESM DATA

if __name__ == '__main__':
    data_dict = functions.load_mpi_esm_data(eur_only=True)

########################################
# %%
# FIGURE FUNCTION
# MAKE FIG. S10: 4-PANEL TAS/TMN MAPS (2100 AND 2300 VS PREINDUSTRIAL)

def make_figure(data_dict, ssp='ssp245', hosing='1', plot_bg='white', future_window=None):
    plt.style.use('default')
    plt.rcParams.update(functions.PRINT_RC)
    fw = functions.FUTURE_WINDOW if future_window is None else tuple(future_window)
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    nrows = 2
    ncols = 2

    sc = 0.744  # print scale: old crop 245.9 mm → 183 mm (fit_print_width sets the exact canvas)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * sc, 8 * sc),
                             subplot_kw={'projection': ccrs.Robinson(central_longitude=8)},
                             constrained_layout=False)
    fig.set_size_inches(10 * sc, 8 * sc, forward=False)  # GUI backends (macosx) round the canvas to whole pixels
    fg = 'black' if plot_bg != 'black' else 'white'
    axes = axes.flatten()

    plt.subplots_adjust(hspace=0.04, wspace=0.05, top=0.92, bottom=0.08, left=0.05, right=0.85)  # hspace 0.15 → 0.04: the panel titles below the maps became column headers above

    # Get preindustrial reference (1850-1899) from historical data
    his_tas = data_dict['']['tas']['his']
    his_tmn = data_dict['']['tmn']['his']

    tas_pi_ref = his_tas.tas.mean(dim='realiz').sel(time=slice(*functions.PI_WINDOW)).mean(dim='time')
    tmn_pi_ref = his_tmn.tmn.mean(dim='realiz').sel(time=slice(*functions.PI_WINDOW)).mean(dim='time')

    # Get ssphos data for TAS and TMN
    ssphos_tas = data_dict['']['tas']['ssphos']
    ssphos_tmn = data_dict['']['tmn']['ssphos']

    # Time periods for end-of-century and 2300 (the 2300 window is unrelated to
    # FUTURE_WINDOW and stays fixed; data goes to 2298).
    time_2100 = slice(*fw)
    time_2300 = slice('2289', '2298')

    # Calculate anomalies vs preindustrial
    # TAS 2100
    tas_2100 = ssphos_tas.sel(scenar=ssp, hosing=hosing).mean(dim='realiz').sel(time=time_2100).mean(dim='time').tas
    tas_anom_2100 = tas_2100 - tas_pi_ref

    # TMN 2100
    tmn_2100 = ssphos_tmn.sel(scenar=ssp, hosing=hosing).sel(time=time_2100).mean(dim='time').tmn
    tmn_anom_2100 = tmn_2100 - tmn_pi_ref

    # TAS 2300
    tas_2300 = ssphos_tas.sel(scenar=ssp, hosing=hosing).mean(dim='realiz').sel(time=time_2300).mean(dim='time').tas
    tas_anom_2300 = tas_2300 - tas_pi_ref

    # TMN 2300
    tmn_2300 = ssphos_tmn.sel(scenar=ssp, hosing=hosing).sel(time=time_2300).mean(dim='time').tmn
    tmn_anom_2300 = tmn_2300 - tmn_pi_ref

    # Colormap setup
    original_cmap = cm.coolwarm
    vmin, vmax = -5, 5
    levels = np.linspace(vmin, vmax, 11)

    # Truncate colormap to center on zero
    if abs(vmax) < abs(vmin):
        trunc_cmap = functions.truncate_colormap(original_cmap, minval=0.0, maxval=(1+abs(vmax)/abs(vmin))*0.5)
    elif abs(vmax) > abs(vmin):
        trunc_cmap = functions.truncate_colormap(original_cmap, minval=(1 - abs(vmin)/abs(vmax))*0.5, maxval=1.0)
    else:
        trunc_cmap = original_cmap

    n_intervals = len(levels) - 1
    sampled_colors = trunc_cmap(np.linspace(0, 1, n_intervals + 2))

    under_color = sampled_colors[0]
    over_color = sampled_colors[-1]
    main_colors = sampled_colors[1:-1]

    main_cmap = LinearSegmentedColormap.from_list('main_cmap', main_colors)
    main_cmap.set_under(under_color)
    main_cmap.set_over(over_color)

    norm = BoundaryNorm(levels, ncolors=main_cmap.N, clip=False)

    # Data to plot: [TAS 2100, TMN 2100, TAS 2300, TMN 2300]
    data_to_plot = [tas_anom_2100, tmn_anom_2100, tas_anom_2300, tmn_anom_2300]
    lon = ssphos_tas.lon
    lat = ssphos_tas.lat

    mappables = []
    for i, ax in enumerate(axes):
        ax.spines['geo'].set_visible(False)
        ax.coastlines(color='black' if plot_bg != 'black' else 'white', linewidth=0.45)
        ax.add_feature(cfeature.BORDERS, edgecolor='black' if plot_bg != 'black' else 'white', linewidth=0.25)
        ax.add_feature(cfeature.LAND, facecolor='white' if plot_bg != 'black' else '#191919')
        ax.add_feature(cfeature.OCEAN, facecolor='white' if plot_bg != 'black' else '#191919', zorder=2)

        contour = ax.contourf(lon, lat, data_to_plot[i],
                              levels=levels, cmap=main_cmap, norm=norm,
                              transform=ccrs.PlateCarree(), extend='both')
        functions.rasterize_map(ax, contour)
        mappables.append(contour)

        ax.set_extent([-18.5, 38.5, 34.8, 72], crs=ccrs.PlateCarree())

        # Add white squares to cover artifacts
        functions.add_square(ax, -13.5, 0, 60, 75.5, colour='#191919' if plot_bg == 'black' else 'white')
        functions.add_square(ax, -30, -18, 68, 75, colour='#191919' if plot_bg == 'black' else 'white')

    # Column headers (variable) above the maps, row headers (window) left of them, letters inside each map's top-left
    # corner; figure-level texts, as the map axes are rasterised.
    fig.draw_without_rendering()
    for ax, title in zip(axes[:2], ['Annual mean surface temperature', 'Annual minimum surface temperature']):
        fig.add_artist(matplotlib.text.Annotation(title, (0.5, 1), xycoords=ax.transAxes, xytext=(0, 3), textcoords='offset points',
                                                  fontsize=7, fontweight='bold', ha='center', va='bottom', color=fg))
    for ax, window in zip(axes[::2], [f'{fw[0]}–{fw[1]}', '2289–2298']):
        fig.add_artist(matplotlib.text.Annotation(window, (0, 0.5), xycoords=ax.transAxes, xytext=(-3, 0), textcoords='offset points',
                                                  fontsize=7, fontweight='bold', ha='right', va='center', rotation=90, color=fg))
    for ax, letter in zip(axes, 'abcd'):
        fig.add_artist(matplotlib.text.Annotation(letter, (0, 1), xycoords=ax.transAxes, xytext=(0, -1), textcoords='offset points',
                                                  fontsize=8, fontweight='bold', ha='left', va='top', color=fg))

    # Colorbar
    right = axes[1].get_position().x1
    cbar_x = right + 0.03
    cbar_width = 0.02
    bottom = axes[3].get_position().y0
    top = axes[1].get_position().y1
    cbar_height = top - bottom

    cax = fig.add_axes([cbar_x, bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(mappables[0], cax=cax, orientation='vertical', extend='both')
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0f} °C".replace('-', '−')))
    cbar.set_label(f'ΔT w.r.t. 1850–1899, {functions.ssp_labels[ssp]} with {hosing} Sv hosing', labelpad=3, color=fg)

    # Add white borders on top of each subplot to cover colored edge artifacts in PDF
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

    savepath = f'../plots/FigSupp_mpi_tas_tmn_anomalies_{ssp}_plotbg-{plot_bg}{functions.fw_suffix(future_window)}'
    functions.fit_print_width(fig)
    fig.savefig(savepath + '.png', bbox_inches='tight', transparent=True if plot_bg == 'black' else False, dpi=300)
    fig.savefig(savepath + '.pdf', bbox_inches='tight', dpi=600, transparent=True if plot_bg == 'black' else False)

    return fig, savepath

########################################
# %%
# RUN

if __name__ == '__main__':
    plot_bg = 'white'  # 'white' or 'black'
    ssp = 'ssp245'  # ssp126 or ssp245
    hosing = '1'  # 1Sv hosing for extended runs
    future_window = None  # None -> canonical 2091-2100; e.g. ('2081','2090') for a sensitivity variant

    fig, savepath = make_figure(data_dict, ssp=ssp, hosing=hosing, plot_bg=plot_bg, future_window=future_window)

# %%
