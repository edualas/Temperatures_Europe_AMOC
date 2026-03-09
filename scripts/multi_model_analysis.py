########################################
# %% 
# LOAD PACKAGES

import functions
import importlib
importlib.reload(functions)

########################################
# %% 
# LOAD THE HOSMIP DATA

multi_model_dict, masks = functions.get_full_multi_model_dict()

########################################
# %% 
# LOAD THE DATA FROM OTHER STUDIES

bm_data, vwb_data, boot_data, liu_data = functions.get_other_studies_data(masks)

########################################
#%%
# PLOT HOSMIP REGRESSION RESULTS TOGETHER WITH OTHER STUDIES

functions.hosmip_regression_plot(multi_model_dict, plot_bg='white', low_ylim=-16, quantile_reg=False, season='', region='EU', hosmip_ref_pi=True, bm_data=True, liu_data=True, vwb_data=True, boot_data=True, boot_regression=True, boot_intercept=False, add_combined_results=True, add_hosmip_mpi_regression=True)

########################################
#%% 
# CREATE DATASET OF UNHOSED CMIP6 SIMULATIONS

cmip6_ctrl_data = functions.get_cmip_projections(masks)

########################################
# %% 
# CESM2 REGRESSIONS PLOT

plot_bg = 'white'
window = 10
region = 'EU'
ssp = 'all' # 'ssp126', 'ssp585' or 'all'
lat, lon = None, None # 20, 30
regressions = True
combined_reg = True
add_combined_intercept = True
no_plots = False
low_ylim = -5
xlim = 50
hosmip_cesm = False

functions.cesm_regressions(boot_data, multi_model_dict, masks, ssp=ssp, lat=lat, lon=lat, region=region, window=window, regressions=regressions, combined_reg=combined_reg, add_combined_intercept=add_combined_intercept, no_plots=no_plots, plot_bg=plot_bg, low_ylim=low_ylim, xlim=xlim, hosmip_cesm=hosmip_cesm, savefig=f'../plots/cesm2_reg_intercept-{add_combined_intercept}_{region}.pdf')

########################################
# %% 
# LOAD MPI-ESM, CESM and HosMIP REGRESSION DATA

reg_ds_mpi = functions.load_regression_ds_mpi()
reg_ds_cesm = functions.get_cesm_reg_ds(recompute=False)
hosmip_reg_ds_dict = functions.get_hosmip_reg_ds(recompute=False)

########################################
#%%
# PLOTTING CESM NET COOLING MAPS ALSO WORKS WITH MPI-ESM FUNCTION

ssp = 'ssp245'
functions.net_cooling_point_map(reg_ds_cesm, ssp, plot_bg='white')

########################################
#%%
# MAKE NEW RANGE PLOT FOR MPI-ESM AND CESM2

plot_bg = 'white'
functions.plot_net_cooling_ranges_mpi_cesm(reg_ds_mpi, reg_ds_cesm, masks, hosmip_reg_ds_dict=hosmip_reg_ds_dict, season='', hosmip_markers=False, T_ref='pi', plot_bg=plot_bg, ext_ax=None, title=False)

#%%