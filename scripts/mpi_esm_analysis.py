########################################
# %% 
# LOAD PACKAGES

import functions
import importlib
importlib.reload(functions)

########################################
# %% 
# LOAD ALL THE MPI-ESM DATA

data_dict = functions.load_mpi_esm_data(eur_only=True)

########################################
#%%
# DEFINE COMMON PLOT PARAMETERS

season = ''
region = 'EU'
plot_bg = 'white'
var = 'tas'
ssp = 'all'
hos_type = 'all' # all_plus_1Sv
window = 10

########################################
#%%
# PRODUCE SIMULATIONS PLOT

functions.simulations_plot(data_dict, season=season, region=region, ssp=ssp, hos_type=hos_type, plot_bg=plot_bg, var=var, window=window)

########################################
#%%
# PRODUCE REGRESSIONS PLOT

lat, lon = None, None
regressions = True
combined_reg = True
ylim = (-0.25, 0.15) if var=='pr' else (-2.5, 1.5)

functions.regression_plot(data_dict, season=season, region=region, ssp=ssp, hos_type=hos_type, lat=lat, lon=lon, plot_bg=plot_bg, var=var, window=window, regressions=regressions, combined_reg=combined_reg, ylim=ylim)

########################################
#%% 
# LOAD REGRESSION DATASETS (ALL SEASONS)

reg_ds_mpi = functions.load_regression_ds_mpi(data_dict=data_dict, var=var)

########################################
#%% 
# PLOT REGRESSION DATASETS

vmax_tas = 0.7
vmax_pr = 0.1
ste_error = True
ste_relative = False  # If True, plot relative standard errors (ste/coef * 100), else absolute
colorbar_v_max = vmax_pr if var=='pr' else vmax_tas
if ste_error:
    colorbar_v_max = 4 if ste_relative else 0.004

functions.plot_regression_coefficients(reg_ds_mpi, season=season, var=var, std_error=ste_error, ste_relative=ste_relative, plot_bg=plot_bg, colorbar_v_max=colorbar_v_max)

########################################
#%%
# MAKE AND SAVE CONTOUR NET COOLING MAPS

T_ref = 'pi'
ssp = 'ssp126'
title = False

functions.net_cooling_point_map(reg_ds_mpi, ssp, season=season, T_ref=T_ref, title=title, plot_bg=plot_bg)

#%%