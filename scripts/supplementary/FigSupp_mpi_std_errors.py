########################################
# %%
# LOAD PACKAGES

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
plt.rcParams.update({'font.size': 12})

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
    reg_ds_mpi = functions.load_regression_ds_mpi()

########################################
# %%
# PLOT STANDARD ERRORS FOR MPI-ESM (SINGLE PANEL, EXPLORATORY)
# Toggle between relative and absolute standard errors

if __name__ == '__main__':
    plot_bg = 'white'
    season = ''
    ste_relative = False  # If True, plot relative standard errors (ste/coef * 100), else absolute

    plt.style.use('default')
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = 'none'
        plt.rcParams['figure.facecolor'] = 'none'

    # Set colorbar max based on relative vs absolute
    if ste_relative:
        colorbar_v_max = 8  # Relative standard error in %
    else:
        colorbar_v_max = 0.016  # Absolute standard error in degC per 10% weakening

    functions.plot_regression_coefficients(
        reg_ds_mpi,
        season=season,
        var='tas',
        std_error=True,
        ste_relative=ste_relative,
        plot_bg=plot_bg,
        colorbar_v_max=colorbar_v_max,
        savefig=True
    )

########################################
# %%
# FIGURE FUNCTION: SIDE-BY-SIDE ABSOLUTE AND RELATIVE STANDARD ERRORS

def make_figure(reg_ds_mpi, season='', plot_bg='white'):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = 'none'
        plt.rcParams['figure.facecolor'] = 'none'

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), subplot_kw={'projection': ccrs.Robinson(central_longitude=8)})
    plt.subplots_adjust(wspace=0.1)

    # Absolute standard errors
    contour_abs = functions.plot_regression_coefficients(
        reg_ds_mpi,
        season=season,
        var='tas',
        std_error=True,
        ste_relative=False,
        plot_bg=plot_bg,
        colorbar_v_max=0.016,
        ext_ax=axes[0],
        savefig=False
    )
    axes[0].set_title('a) Absolute standard error', fontsize=14)

    # Relative standard errors
    contour_rel = functions.plot_regression_coefficients(
        reg_ds_mpi,
        season=season,
        var='tas',
        std_error=True,
        ste_relative=True,
        plot_bg=plot_bg,
        colorbar_v_max=8,
        ext_ax=axes[1],
        savefig=False
    )
    axes[1].set_title('b) Relative standard error', fontsize=14)

    # Add colorbars
    cbar_abs = fig.colorbar(contour_abs, ax=axes[0], orientation='vertical', pad=0.02, aspect=30, extend='max')
    cbar_abs.set_label('Standard error per 10% AMOC weakening [°C]', fontsize=12)
    cbar_abs.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{np.round(x, 4)}°C"))

    cbar_rel = fig.colorbar(contour_rel, ax=axes[1], orientation='vertical', pad=0.02, aspect=30, extend='max')
    cbar_rel.set_label('Relative standard error [%]', fontsize=12)
    cbar_rel.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{np.round(x, 1)}%"))

    savepath = f"../plots/FigSupp_mpi_std_errors_plotbg-{plot_bg}_season-{season or 'annual'}"
    fig.savefig(savepath + '.png', dpi=200, bbox_inches='tight', transparent=True if plot_bg=='black' else False)
    fig.savefig(savepath + '.pdf', bbox_inches='tight', dpi=400, transparent=True if plot_bg=='black' else False)

    return fig, savepath

########################################
# %%
# RUN

if __name__ == '__main__':
    fig, savepath = make_figure(reg_ds_mpi, season='', plot_bg='white')

# %%
