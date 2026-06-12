########################################
# %%
# LOAD PACKAGES

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm
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
# LOAD DATA

if __name__ == '__main__':
    # Load HosMIP multi-model data (includes other studies data)
    multi_model_dict, masks = functions.get_full_multi_model_dict()

    # Load MPI-ESM, CESM and HosMIP regression datasets
    reg_ds_mpi = functions.load_regression_ds_mpi()
    reg_ds_cesm = functions.get_cesm_reg_ds(recompute=False)
    hosmip_reg_ds_dict = functions.get_hosmip_reg_ds(recompute=False)

########################################
# %%
# FIGURE FUNCTION
# VISUALIZE THE REGRESSION COEFFICIENTS FOR ALL MODELS
# FIG. S12 FOR WINTER AND FIG. S13 FOR SUMMER
# FIG. S14 FOR STANDARD ERRORS (ANNUAL)

def make_figure(multi_model_dict, masks, reg_ds_mpi, reg_ds_cesm, hosmip_reg_ds_dict,
                season='', plot_ste=False, ste_relative=False, plot_bg='white'):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = 'none'
        plt.rcParams['figure.facecolor'] = 'none'

    season_idx = 0 if season == '' else 1 if season == 'djf' else 2
    custom_cmap = functions.get_custom_cmap()

    if season == '' and not plot_ste:
        nrows, ncols = (4, 4)
        figsize = (16, 11)
    elif plot_ste:
        nrows, ncols = (3, 4)  # 8 HosMIP + 4 CESM = 12 subplots
        figsize = (16, 8)
    else:
        nrows, ncols = (2, 4)
        figsize = (16, 5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw={'projection': ccrs.Robinson(central_longitude=8)})
    plt.subplots_adjust(hspace=0.45, wspace=-0.3, bottom=0.08)

    if plot_ste:
        if ste_relative:
            levels = np.linspace(0.0, 20, 11)  # Relative standard errors in %
        else:
            levels = np.linspace(0.0, 0.1, 11)  # Absolute standard errors
        ste_cmap = functions.truncate_colormap(cm.viridis, maxval=0.92)
        ste_cmap.set_over(cm.viridis(1.0))  # Full yellow as distinct over color
        norm = BoundaryNorm(levels, ncolors=ste_cmap.N, clip=False)
    else:
        levels = np.linspace(0.0, 1.5, 16)
        norm = BoundaryNorm(levels, ncolors=custom_cmap.N, clip=False)

    # Plot HosMIP model regression coefficients (rows 1 & 2)
    for i, model in enumerate(functions.hosmip_labels[4:] + functions.hosmip_labels[1:4] + [functions.hosmip_labels[0]]):
        ax = axes.flat[i]
        ax.spines['geo'].set_visible(False)
        ax.set_rasterized(True)
        ax.coastlines(color='black' if not plot_bg=='black' else 'white', linewidth=.5)
        ax.add_feature(cfeature.BORDERS, edgecolor='black' if not plot_bg=='black' else 'white', linewidth=.5)
        ax.add_feature(cfeature.LAND, facecolor='white' if not plot_bg=='black' else '#191919')
        ax.add_feature(cfeature.OCEAN, facecolor='white' if not plot_bg=='black' else '#191919', zorder=2)
        # Use the new hosmip_reg_ds_dict structure: lin_coef_hosmip or ste_hosmip with (lat, lon, season) dims
        if plot_ste:
            if ste_relative:
                plot_data = -hosmip_reg_ds_dict[model].ste_hosmip.isel(season=season_idx) / hosmip_reg_ds_dict[model].lin_coef_hosmip.isel(season=season_idx) * 100
            else:
                plot_data = hosmip_reg_ds_dict[model].ste_hosmip.isel(season=season_idx) * 10  # Scale to per 10% weakening
            plot_cmap = ste_cmap
        else:
            plot_data = -10 * hosmip_reg_ds_dict[model].lin_coef_hosmip.isel(season=season_idx)
            plot_cmap = custom_cmap
        mesh = ax.pcolormesh(hosmip_reg_ds_dict[model].lon, hosmip_reg_ds_dict[model].lat,
                             plot_data, transform=ccrs.PlateCarree(),
                             cmap=plot_cmap, norm=norm, shading='auto')
        ax.set_extent([-18.5, 38.5, 34.8, 72], crs=ccrs.PlateCarree())
        functions.add_square(ax, -13.5, 0, 60, 75.5, colour='#191919' if plot_bg == 'black' else 'white')
        functions.add_square(ax, -30, -18, 68, 75, colour='#191919' if plot_bg == 'black' else 'white')
        # Letter label (a-h) and model name below subplot
        letter = chr(ord('a') + i)
        ax.text(0.5, -0.08, f'{letter}) {model}', transform=ax.transAxes, fontsize=13,
                ha='center', va='top')

    # Prepare data for other studies plots (rows 3 & 4)
    # Liu et al. (2020): AMOC weakens by 23.6% (from control to hosing under RCP8.5)
    liu_tas_diff = (multi_model_dict['CCSM4'].tas.sel(type='hosing', scenar='ghg', season='') -
                    multi_model_dict['CCSM4'].tas.sel(type='control', scenar='ghg', season=''))
    liu_amoc_weakening = (multi_model_dict['CCSM4'].amoc.sel(type='control', scenar='ghg') -
                          multi_model_dict['CCSM4'].amoc.sel(type='hosing', scenar='ghg')) / \
                         multi_model_dict['CCSM4'].amoc.sel(type='control', scenar='pi') * 100

    # Bellomo & Mehling: 4xCO2 experiment with EC-Earth3 (AMOC weakens by 46%)
    bm_tas_4x = multi_model_dict['EC-Earth3'].tas.sel(type='diff', scenar='ghg', season='').isel(time=0)
    bm_amoc_4x_weakening = multi_model_dict['EC-Earth3'].amoc.sel(type='diff', scenar='ghg', season='').isel(time=0) / \
                           multi_model_dict['EC-Earth3'].amoc.sel(type='control', scenar='pi', season='').isel(time=0) * 100

    # van Westen & Baatsen: AMOC weakens by 71% more in 1500 case than in 600 case
    vwb_tas_diff = ((multi_model_dict['CESM1'].tas.sel(type='hosing', scenar='ghg', season='') -
                     multi_model_dict['CESM1'].tas.sel(type='hosing', scenar='pi', season='')) -
                    (multi_model_dict['CESM1'].tas.sel(type='control', scenar='ghg', season='') -
                     multi_model_dict['CESM1'].tas.sel(type='control', scenar='pi', season='')))
    vwb_amoc_diff = ((multi_model_dict['CESM1'].amoc.sel(type='control', scenar='ghg') -
                      multi_model_dict['CESM1'].amoc.sel(type='control', scenar='pi')) -
                     (multi_model_dict['CESM1'].amoc.sel(type='hosing', scenar='ghg') -
                      multi_model_dict['CESM1'].amoc.sel(type='hosing', scenar='pi'))) / \
                    multi_model_dict['CESM1'].amoc.sel(type='control', scenar='pi') * 100

    add_plots = [
        'bellomo4x',
        'vwb',
        'liu',
        'combined',
        'boot_combined',
        'boot_combined_intercept',
        'boot126',
        'boot585',
        ]

    if season != '':
        add_plots = []  # Skip other studies for seasonal plots
    elif plot_ste:
        # For standard errors, only show CESM plots (last 4 in the list)
        add_plots = ['boot_combined', 'boot_combined_intercept', 'boot126', 'boot585']

    # Coordinate data for plotting
    coord_data_plots = {
        'liu': multi_model_dict['CCSM4'],
        'bellomo4x': multi_model_dict['EC-Earth3'],
        'vwb': multi_model_dict['CESM1'],
        'combined': reg_ds_mpi,
        'boot_combined': reg_ds_cesm,
        'boot_combined_intercept': reg_ds_cesm,
        'boot126': reg_ds_cesm,
        'boot585': reg_ds_cesm,
    }

    # Temperature data normalized to cooling per 10% AMOC weakening
    tas_data_plot = {
        'liu': liu_tas_diff / liu_amoc_weakening.values * (10),  # cooling per 10% weakening
        'bellomo4x': -bm_tas_4x / bm_amoc_4x_weakening.values * 10,  # cooling per 10% weakening
        'vwb': vwb_tas_diff / vwb_amoc_diff.values * (-10),  # cooling per 10% weakening
        'combined': reg_ds_mpi.coef_ensmean.sel(season=season) * functions.AMOC_pi_MPI / 10,  # original values are per Sv change
        'boot_combined': -reg_ds_cesm.coef_ensmean * 10,  # original values are per % weakening
        'boot_combined_intercept': -reg_ds_cesm.coef_ensmean_intercept * 10,  # original values are per % weakening
        'boot126': -reg_ds_cesm.coef_ssp.sel(scenar='ssp126') * 10,
        'boot585': -reg_ds_cesm.coef_ssp.sel(scenar='ssp585') * 10,
    }

    # Standard error data for CESM plots
    ste_data_plot = {
        'boot_combined': reg_ds_cesm.ste_ensmean * 10 if not ste_relative else -reg_ds_cesm.ste_ensmean / reg_ds_cesm.coef_ensmean * 100,
        'boot_combined_intercept': reg_ds_cesm.ste_ensmean_intercept * 10 if not ste_relative else -reg_ds_cesm.ste_ensmean_intercept / reg_ds_cesm.coef_ensmean_intercept * 100,
        'boot126': reg_ds_cesm.ste_ssp.sel(scenar='ssp126') * 10 if not ste_relative else -reg_ds_cesm.ste_ssp.sel(scenar='ssp126') / reg_ds_cesm.coef_ssp.sel(scenar='ssp126') * 100,
        'boot585': reg_ds_cesm.ste_ssp.sel(scenar='ssp585') * 10 if not ste_relative else -reg_ds_cesm.ste_ssp.sel(scenar='ssp585') / reg_ds_cesm.coef_ssp.sel(scenar='ssp585') * 100,
    }

    label_plot = {
        'liu': 'Liu et al. (2020)\nRCP8.5 in 2061-2080\nwith CCSM4',
        'bellomo4x': 'Bellomo & Mehling\n'+r'4x$\mathrm{{CO}}_2$ experiment'+'\nwith EC-Earth3',
        'vwb': 'v. Westen & Baatsen\nRCP4.5 in 2400-2500\nwith CESM1',
        'combined': 'This study\nall emissions scenarios\nwith MPI-ESM1.2-LR',
        'boot_combined': 'Boot et al. (2024)\nboth emissions scenarios\nwith CESM2',
        'boot_combined_intercept': 'Boot et al. (2024)\nboth emissions scenarios\nwith CESM2 (non-0 intercept)',
        'boot126': 'Boot et al. (2024)\nSSP1-2.6 with CESM2\n(non-0 intercept)',
        'boot585': 'Boot et al. (2024)\nSSP5-8.5 with CESM2\n(non-0 intercept)',
    }

    for j, plot in enumerate(add_plots):
        ax = axes.flat[8+j]
        ax.spines['geo'].set_visible(False)
        ax.set_rasterized(True)
        ax.coastlines(color='black' if not plot_bg=='black' else 'white', linewidth=.5)
        ax.add_feature(cfeature.BORDERS, edgecolor='black' if not plot_bg=='black' else 'white', linewidth=.5)
        ax.add_feature(cfeature.LAND, facecolor='white' if not plot_bg=='black' else '#191919')
        ax.add_feature(cfeature.OCEAN, facecolor='white' if not plot_bg=='black' else '#191919', zorder=2)
        if plot_ste:
            plot_data_j = ste_data_plot[plot]
            plot_cmap_j = ste_cmap
        else:
            plot_data_j = tas_data_plot[plot]
            plot_cmap_j = custom_cmap
        mesh = ax.pcolormesh(coord_data_plots[plot].lon, coord_data_plots[plot].lat,
                             plot_data_j, transform=ccrs.PlateCarree(),
                             cmap=plot_cmap_j, norm=norm, shading='auto')
        ax.set_extent([-18.5, 38.5, 34.8, 72], crs=ccrs.PlateCarree())
        functions.add_square(ax, -13.5, 0, 60, 75.5, colour='#191919' if plot_bg == 'black' else 'white')
        functions.add_square(ax, -30, -18, 68, 75, colour='#191919' if plot_bg == 'black' else 'white')
        # Letter label (i-l) and plot name below subplot
        letter = chr(ord('a') + 8 + j)
        ax.text(0.5, -0.08, f'{letter}) {label_plot[plot]}', transform=ax.transAxes, fontsize=13,
                ha='center', va='top')

    # Remove unused axes if any
    for j in range(len(functions.hosmip_labels)+len(add_plots), nrows*ncols):
        fig.delaxes(axes.flat[j])

    # Shared colorbar
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation='vertical', pad=0.02, aspect=30, extend='max')
    if plot_ste:
        if ste_relative:
            cbar.set_ticks(np.arange(0, 21, 2))
            cbar.set_label('Relative standard error [%]', fontsize=14)
        else:
            cbar.set_ticks(np.arange(0, 0.11, 0.01))
            cbar.set_label('Standard error per 10% AMOC weakening [°C]', fontsize=14)
    else:
        cbar.set_ticks(np.arange(0, 1.6, 0.3))
        cbar.set_label('Cooling per 10% AMOC weakening [°C]', fontsize=14)

    season_str = 'Winter ' if season == 'djf' else 'Summer ' if season == 'jja' else ''
    ste_str = ('relative ' if ste_relative else '') + 'standard errors' if plot_ste else 'scaling factors'
    if plot_ste:
        fig.suptitle(f'{"Annual" if season == "" else season_str.strip()} {ste_str} for all models', fontsize=16, y=0.95)
    elif season == '':
        fig.suptitle('Annual scaling factors for all models (a-h: preindustrial; i-p: combined forcing)', fontsize=16, y=0.92)
    else:
        fig.suptitle(season_str + 'scaling factors for all NAHosMIP models', fontsize=16, y=0.92)

    ste_val = 'off' if not plot_ste else ('rel' if ste_relative else 'abs')
    savepath = f"../plots/FigSupp_multimodel_regression_maps_plotbg-{plot_bg}_season-{season or 'annual'}_ste-{ste_val}"
    fig.savefig(savepath + '.png', bbox_inches='tight', dpi=200, transparent=True if plot_bg=='black' else False)
    fig.savefig(savepath + '.pdf', bbox_inches='tight', dpi=400, transparent=True if plot_bg=='black' else False)

    return fig, savepath

########################################
# %%
# RUN

if __name__ == '__main__':
    plot_bg = 'white'
    season = ''
    plot_ste = False  # If True, plot standard errors instead of regression coefficients
    ste_relative = False  # If True, plot relative standard errors (ste/coef), else absolute

    fig, savepath = make_figure(multi_model_dict, masks, reg_ds_mpi, reg_ds_cesm, hosmip_reg_ds_dict,
                                season=season, plot_ste=plot_ste, ste_relative=ste_relative, plot_bg=plot_bg)

# %%
