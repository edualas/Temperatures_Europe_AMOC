#%%

"""
Produce all paper figures from a single script.

Usage:
    python produce_all_paper_figures.py              # Generate all figures
    python produce_all_paper_figures.py Fig1 Fig3    # Generate only Fig1 and Fig3 variants
    python produce_all_paper_figures.py FigSupp_mpi_std_errors

This script is the single source of truth for which figures exist: the
canonical workflow is to wipe plots/ and regenerate from here. Main figures
keep stable numbers (Fig1/Fig2/Fig3); every standalone supplementary figure
is FigSupp_<kind> (kind leads with the model). Supplementary figures register
one canonical variant each, except where noted (scenario-independence has a
scenario + state variant; linearity_other loops cesm + giss).
"""

import sys
import importlib
import matplotlib.pyplot as plt
import functions  # imported first: its shim puts scripts/ subfolders on sys.path

import Fig1, Fig2, Fig3
import Fig3_simple  # type: ignore
import FigSupp_mpi_std_errors as supp_std_errors # type: ignore
import FigSupp_mpi_tas_tmn_anomalies as supp_tas_anom # type: ignore
import FigSupp_multimodel_regression_maps as supp_reg_maps # type: ignore
import FigSupp_cesm_net_cooling_maps as supp_cesm_maps # type: ignore
import FigSupp_giss_net_cooling_maps as supp_giss_maps # type: ignore
import FigSupp_giss_sim_regression as supp_giss_ts # type: ignore
import FigSupp_scenario_independence as supp_scenindep # type: ignore
import FigSupp_scenario_independence_cesm as supp_scenindep_cesm # type: ignore
import FigSupp_mpi_linearity as supp_linearity_mpi # type: ignore
import FigSupp_linearity_other as supp_linearity_other # type: ignore
import FigSupp_mpi_net_cooling_binary as supp_binary # type: ignore
import cmip_cooling, FigSupp_cmip_cooling as supp_cmip_cooling # type: ignore
import FigSupp_warming_at_weakening as supp_warming_weak # type: ignore
import FigSupp_warming_uncertainty as supp_warm_unc # type: ignore
import FigSupp_scaling_factor_correlations as supp_scaling_corr # type: ignore

for _mod in [functions, Fig1, Fig2, Fig3, Fig3_simple, supp_std_errors, supp_tas_anom,
             supp_reg_maps, supp_cesm_maps, supp_giss_maps, supp_giss_ts,
             supp_scenindep, supp_scenindep_cesm, supp_linearity_mpi,
             supp_linearity_other, supp_binary, cmip_cooling, supp_cmip_cooling,
             supp_warming_weak, supp_warm_unc, supp_scaling_corr]:
    importlib.reload(_mod)

make_fig1 = Fig1.make_figure
make_fig2 = Fig2.make_figure
make_fig3 = Fig3.make_figure
make_fig3_simple = Fig3_simple.make_figure
make_std_errors = supp_std_errors.make_figure
make_tas_anom = supp_tas_anom.make_figure
make_reg_maps = supp_reg_maps.make_figure
make_cesm_maps = supp_cesm_maps.make_figure
make_giss_maps = supp_giss_maps.make_figure
make_giss_ts = supp_giss_ts.make_figure
make_scenindep = supp_scenindep.make_figure
make_scenindep_cesm = supp_scenindep_cesm.make_figure
make_linearity_mpi = supp_linearity_mpi.make_figure
make_linearity_other = supp_linearity_other.make_figure
make_binary = supp_binary.make_figure
make_cmip_cooling = supp_cmip_cooling.make_figure
make_warming_weak = supp_warming_weak.make_figure
make_warm_unc = supp_warm_unc.make_figure
make_scaling_corr = supp_scaling_corr.make_figure

#%%
########################################
# FUTURE-WINDOW SENSITIVITY (opt-in)
# Leave as None for the canonical 2091-2100 run: the loads and FIGURE_CONFIGS
# below are then byte-for-byte the submitted pipeline. Set to a (start, end)
# tuple, e.g. ('2081', '2090'), to ALSO build the five future-window-dependent
# figures off window-keyed parallel caches. Variant caches and figures carry a
# _fw-{start}-{end} suffix; canonical outputs are never overwritten.
FUTURE_WINDOW_OVERRIDE = None

#%%
########################################
# DATA LOADING
# Each dataset is loaded once and shared across all figures that need it.

print("Loading MPI-ESM data...")
data_dict = functions.load_mpi_esm_data(eur_only=True)            # Fig1, tas_anom, linearity, scen-indep

print("Loading MPI-ESM regression dataset...")
reg_ds_mpi = functions.load_regression_ds_mpi()                    # Fig2, Fig3, std_errors, reg_maps, binary

print("Loading CESM2 regression dataset...")
reg_ds_cesm = functions.get_cesm_reg_ds(recompute=False)           # Fig3, reg_maps, cesm_maps

print("Loading HosMIP multi-model data...")
multi_model_dict, masks = functions.get_full_multi_model_dict()    # Fig3, reg_maps, cesm_maps, linearity, scen-indep

print("Loading HosMIP regression datasets...")
hosmip_reg_ds_dict = functions.get_hosmip_reg_ds(recompute=False)  # Fig3, reg_maps, scen-indep (state)

print("Loading GISS-E2-1-G regression and per-season panel data...")
reg_ds_giss = functions.get_giss_reg_ds(recompute=False, masks=masks)  # Fig3 panel e markers, giss_maps
reg_ds_giss_panel = {
    s: functions.get_giss_panel_data(masks=masks, season=s, recompute=False)
    for s in ['', 'djf', 'jja']
}                                                                       # Fig3 panel a-d scatter

print("Loading GWL diagnostic data (CMIP6 hist ens-mean AMOC)...")
gwl_data = functions.get_gwl_diagnostic_data(recompute=False)           # scaling_factor_correlations (hist baseline)

print("Loading other studies data...")
_, _, boot_data, _ = functions.get_other_studies_data(masks)        # cesm_maps, linearity_other, scen-indep_cesm

print("Loading MPI-ESM scenario-independence diagnostics...")
si_ds = functions.get_scenario_independence_diagnostics_mpi(var='tas')  # scen-indep (scenario variant)

# ── Supplementary-figure-specific diagnostics (cached reads) ──────────────
print("Loading linearity diagnostics (MPI/CESM/GISS)...")
lin_ds_mpi = functions.get_linearity_diagnostics_mpi(var='tas')          # linearity (mpi)
lin_ds_cesm = functions.get_linearity_diagnostics_cesm(var='tas')        # linearity_other (cesm)
lin_ds_giss = functions.get_linearity_diagnostics_giss(var='tas')        # linearity_other (giss)

print("Loading CESM scenario-independence diagnostics...")
si_ds_cesm = functions.get_scenario_independence_diagnostics_cesm(var='tas')  # scen-indep_cesm

print("Loading MPI state-independence diagnostics...")
si_ds_state = supp_scenindep.load_scenario_independence_diagnostics_mpi_state(
    data_dict=data_dict, multi_model_dict=multi_model_dict,
    hosmip_reg_ds_dict=hosmip_reg_ds_dict, var='tas', recompute=False)   # scen-indep (state variant)

print("Loading GISS member series for the GISS time-series figure...")
amoc_giss, tas_giss = functions.load_giss_member_amoc_tas(season='')     # linearity_other (giss)
giss_ts_inputs = supp_giss_ts.load_inputs(season='')                     # giss_sim_regression (FigGiss1)

print("Loading CMIP-projected cooling dataset...")
cmip_cooling_ds = cmip_cooling.make_cmip_cooling_ds(recompute=False)

print("All data loaded.\n")

# Variant future-window caches (only when a sensitivity window is requested).
if FUTURE_WINDOW_OVERRIDE is not None:
    _fw = FUTURE_WINDOW_OVERRIDE
    print(f"Loading future-window-variant regression datasets for {_fw}...")
    reg_ds_mpi_fw = functions.load_regression_ds_mpi(future_window=_fw)
    reg_ds_cesm_fw = functions.get_cesm_reg_ds(recompute=False, future_window=_fw)
    hosmip_reg_ds_dict_fw = functions.get_hosmip_reg_ds(recompute=False, future_window=_fw)
    reg_ds_giss_fw = functions.get_giss_reg_ds(recompute=False, masks=masks, future_window=_fw)
    print("Variant data loaded.\n")

#%%
########################################
# FIGURE CONFIGURATIONS

FIGURE_CONFIGS = {
    'Fig1': {
        'function': make_fig1,
        'data_args': {'data_dict': data_dict},
        'variants': [
            {'label': 'Fig1',            'kwargs': {'season': '', 'window': 10, 'hosing': 'all'}},
            {'label': 'FigS1 (DJF)',     'kwargs': {'season': 'djf', 'window': 10, 'hosing': 'all'}},
            {'label': 'FigS2 (JJA)',     'kwargs': {'season': 'jja', 'window': 10, 'hosing': 'all'}},
            {'label': 'FigS3 (1yr)',     'kwargs': {'season': '', 'window': 1, 'hosing': 'all'}},
            {'label': 'FigS4 (30yr)',    'kwargs': {'season': '', 'window': 30, 'hosing': 'all'}},
            {'label': 'FigS5 (const)',   'kwargs': {'season': '', 'window': 10, 'hosing': 'constant'}},
            {'label': 'FigS6 (linear)',  'kwargs': {'season': '', 'window': 10, 'hosing': 'linear'}},
            {'label': 'Fig1 (+1Sv)',     'kwargs': {'season': '', 'window': 10, 'hosing': 'all1Sv'}},
        ],
    },
    'Fig2': {
        'function': make_fig2,
        'data_args': {'reg_ds_mpi': reg_ds_mpi},
        'variants': [
            {'label': 'Fig2',                  'kwargs': {'only_pi': False, 'plot_season': ''}},
            {'label': 'Fig2',                  'kwargs': {'only_pi': True, 'plot_season': ''}},
            {'label': 'Fig2 (cbar1_left)',      'kwargs': {'only_pi': True, 'plot_season': '', 'cbar1_left': True}},
            {'label': 'Fig2',                  'kwargs': {'only_pi': True, 'plot_season': 'djf'}},
            {'label': 'Fig2',                  'kwargs': {'only_pi': True, 'plot_season': 'jja'}},
            {'label': 'FigS7 (DJF)',           'kwargs': {'only_pi': False, 'plot_season': 'djf'}},
            {'label': 'FigS8 (JJA)',           'kwargs': {'only_pi': False, 'plot_season': 'jja'}},
        ],
    },
    'Fig3': {
        'function': make_fig3,
        'data_args': {
            'multi_model_dict': multi_model_dict, 'masks': masks,
            'reg_ds_mpi': reg_ds_mpi, 'reg_ds_cesm': reg_ds_cesm,
            'hosmip_reg_ds_dict': hosmip_reg_ds_dict,
            'reg_ds_giss': reg_ds_giss,
            'giss_panel_data': reg_ds_giss_panel,
            'cmip_cooling_ds': cmip_cooling_ds,
        },
        'variants': [
            {'label': 'Fig3',                'kwargs': {'season': '', 'T_ref': 'pi'}},
            {'label': 'FigS (PD ref)',       'kwargs': {'season': '', 'T_ref': 'pd'}},
            {'label': 'FigS (DJF)',          'kwargs': {'season': 'djf', 'T_ref': 'pi'}},
            {'label': 'FigS (JJA)',          'kwargs': {'season': 'jja', 'T_ref': 'pi'}},
            {'label': 'FigS (PD ref, DJF)',  'kwargs': {'season': 'djf', 'T_ref': 'pd'}},
            {'label': 'FigS (PD ref, JJA)',  'kwargs': {'season': 'jja', 'T_ref': 'pd'}},
            {'label': 'Fig3 (markers)',      'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': True}},
            {'label': 'Fig3 (CMIP cooling)', 'kwargs': {'season': '', 'T_ref': 'pi', 'cmip_cooling_show': True}},
        ],
    },
    'Fig3_simple': {
        'function': make_fig3_simple,
        'data_args': {
            'multi_model_dict': multi_model_dict, 'masks': masks,
            'reg_ds_mpi': reg_ds_mpi, 'reg_ds_cesm': reg_ds_cesm,
            'hosmip_reg_ds_dict': hosmip_reg_ds_dict,
            'reg_ds_giss': reg_ds_giss,
            'giss_panel_data': reg_ds_giss_panel,
            'cmip_cooling_ds': cmip_cooling_ds,
        },
        'variants': [
            {'label': 'Fig3_simple',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct'}},
            {'label': 'Fig3_simple',
             'kwargs': {'season': '', 'T_ref': 'pd', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct'}},
            {'label': 'Fig3_simple',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': True, 'aggregate_first': True, 'weakening_unit': 'pct'}},
            {'label': 'Fig3_simple',
             'kwargs': {'season': 'djf', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct'}},
            {'label': 'Fig3_simple',
             'kwargs': {'season': 'jja', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct'}},
            {'label': 'Fig3_simple',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct'}},
        ],
    },
    'FigSupp_mpi_std_errors': {
        'function': make_std_errors,
        'data_args': {'reg_ds_mpi': reg_ds_mpi},
        'variants': [
            {'label': 'FigSupp_mpi_std_errors', 'kwargs': {'season': ''}},
        ],
    },
    'FigSupp_mpi_tas_tmn_anomalies': {
        'function': make_tas_anom,
        'data_args': {'data_dict': data_dict},
        'variants': [
            {'label': 'FigSupp_mpi_tas_tmn_anomalies', 'kwargs': {'ssp': 'ssp245'}},
        ],
    },
    'FigSupp_multimodel_regression_maps': {
        'function': make_reg_maps,
        'data_args': {
            'multi_model_dict': multi_model_dict, 'masks': masks,
            'reg_ds_mpi': reg_ds_mpi, 'reg_ds_cesm': reg_ds_cesm,
            'hosmip_reg_ds_dict': hosmip_reg_ds_dict,
        },
        'variants': [
            {'label': 'FigSupp_multimodel_regression_maps (annual)',  'kwargs': {'season': '', 'plot_ste': False}},
            {'label': 'FigSupp_multimodel_regression_maps (DJF)',     'kwargs': {'season': 'djf', 'plot_ste': False}},
            {'label': 'FigSupp_multimodel_regression_maps (JJA)',     'kwargs': {'season': 'jja', 'plot_ste': False}},
            {'label': 'FigSupp_multimodel_regression_maps (std err)', 'kwargs': {'season': '', 'plot_ste': True, 'ste_relative': False}},
            {'label': 'FigSupp_multimodel_regression_maps (rel ste)', 'kwargs': {'season': '', 'plot_ste': True, 'ste_relative': True}},
        ],
    },
    'FigSupp_cesm_net_cooling_maps': {
        'function': make_cesm_maps,
        'data_args': {
            'multi_model_dict': multi_model_dict, 'masks': masks,
            'boot_data': boot_data, 'reg_ds_cesm': reg_ds_cesm,
        },
        'variants': [
            {'label': 'FigSupp_cesm_net_cooling_maps', 'kwargs': {}},
        ],
    },
    'FigSupp_giss_net_cooling_maps': {
        'function': make_giss_maps,
        'data_args': {'reg_ds_giss': reg_ds_giss, 'masks': masks},
        'variants': [
            {'label': 'FigSupp_giss_net_cooling_maps',        'kwargs': {'season': ''}},
            {'label': 'FigSupp_giss_net_cooling_maps (DJF)',  'kwargs': {'season': 'djf'}},
            {'label': 'FigSupp_giss_net_cooling_maps (JJA)',  'kwargs': {'season': 'jja'}},
        ],
    },
    'FigSupp_giss_sim_regression': {
        'function': make_giss_ts,
        'data_args': {**giss_ts_inputs},
        'variants': [
            {'label': 'FigSupp_giss_sim_regression', 'kwargs': {'season': ''}},
        ],
    },
    'FigSupp_scenario_independence': {
        'function': make_scenindep,
        'data_args': {'data_dict': data_dict, 'si_ds': si_ds},
        'variants': [
            {'label': 'FigSupp_mpi_scenario_independence', 'kwargs': {'var': 'tas', 'region': 'EU', 'season': ''}},
        ],
    },
    'FigSupp_mpi_state_independence': {
        'function': make_scenindep,
        'data_args': {'data_dict': data_dict, 'si_ds': si_ds_state},
        'variants': [
            {'label': 'FigSupp_mpi_state_independence',
             'kwargs': {'var': 'tas', 'region': 'EU', 'season': '',
                        'state_dependence': True, 'multi_model_dict': multi_model_dict,
                        'hosmip_reg_ds_dict': hosmip_reg_ds_dict}},
        ],
    },
    'FigSupp_scenario_independence_cesm': {
        'function': make_scenindep_cesm,
        'data_args': {'boot_data': boot_data, 'multi_model_dict': multi_model_dict,
                      'masks': masks, 'si_ds': si_ds_cesm},
        'variants': [
            {'label': 'FigSupp_cesm_scenario_independence', 'kwargs': {'var': 'tas', 'region': 'EU', 'season': ''}},
        ],
    },
    'FigSupp_mpi_linearity': {
        'function': make_linearity_mpi,
        'data_args': {'data_dict': data_dict, 'lin_ds': lin_ds_mpi},
        'variants': [
            {'label': 'FigSupp_mpi_linearity', 'kwargs': {'var': 'tas', 'region': 'EU', 'season': ''}},
        ],
    },
    'FigSupp_linearity_other': {
        'function': make_linearity_other,
        'data_args': {'masks': masks},
        'variants': [
            {'label': 'FigSupp_cesm_linearity',
             'kwargs': {'model': 'cesm', 'region': 'EU', 'season': '', 'lin_ds': lin_ds_cesm,
                        'boot_data': boot_data, 'multi_model_dict': multi_model_dict}},
            {'label': 'FigSupp_giss_linearity',
             'kwargs': {'model': 'giss', 'region': 'EU', 'season': '', 'lin_ds': lin_ds_giss,
                        'amoc_giss': amoc_giss, 'tas_giss': tas_giss,
                        'giss_time_period': ('2101', '2300')}},
        ],
    },
    'FigSupp_mpi_net_cooling_binary': {
        'function': make_binary,
        'data_args': {'reg_ds_mpi': reg_ds_mpi},
        'variants': [
            {'label': 'FigSupp_mpi_net_cooling_binary',
             'kwargs': {'plot_season': '', 'T_ref': 'combined',
                        'thresholds': (30, 50, 80), 'omit_scaling_row': True}},
        ],
    },
    'FigSupp_warming_at_weakening': {
        'function': make_warming_weak,
        'data_args': {
            'reg_ds_mpi': reg_ds_mpi, 'reg_ds_cesm': reg_ds_cesm, 'masks': masks,
            'hosmip_reg_ds_dict': hosmip_reg_ds_dict, 'reg_ds_giss': reg_ds_giss,
        },
        'variants': [
            {'label': 'FigSupp_warming_at_weakening',
             'kwargs': {'weakenings': (25.0, 75.0), 'season': '', 'T_ref': 'pi'}},
        ],
    },
    'FigSupp_warming_uncertainty': {
        'function': make_warm_unc,
        'data_args': {
            'reg_ds_mpi': reg_ds_mpi, 'reg_ds_cesm': reg_ds_cesm, 'masks': masks,
            'hosmip_reg_ds_dict': hosmip_reg_ds_dict, 'reg_ds_giss': reg_ds_giss,
        },
        'variants': [
            {'label': 'FigSupp_warming_uncertainty (T_ref=pi)',
             'kwargs': {'season': '', 'T_ref': 'pi'}},
            {'label': 'FigSupp_warming_uncertainty (T_ref=pd)',
             'kwargs': {'season': '', 'T_ref': 'pd'}},
        ],
    },
    'FigSupp_scaling_factor_correlations': {
        'function': make_scaling_corr,
        'data_args': {
            'multi_model_dict': multi_model_dict, 'masks': masks,
            'hosmip_reg_ds_dict': hosmip_reg_ds_dict,
            'reg_ds_giss': reg_ds_giss, 'reg_ds_giss_panel': reg_ds_giss_panel,
            'gwl_data': gwl_data,
        },
        'variants': [
            {'label': 'FigSupp_scaling_factor_correlations',
             'kwargs': {'x_axis': 'combined', 'baseline': 'hist_1850_1899'}},
        ],
    },
}

#%%
########################################
# FUTURE-WINDOW VARIANT CONFIGURATIONS (only when a window is requested)
# One variant per in-scope figure, off the window-keyed caches, with
# future_window threaded into kwargs (drives the _fw- suffix on every savepath
# and the window-keyed AMOC_extent inside each make_figure).

if FUTURE_WINDOW_OVERRIDE is not None:
    FIGURE_CONFIGS_FW = {
        'Fig2_fw': {
            'function': make_fig2,
            'data_args': {'reg_ds_mpi': reg_ds_mpi_fw},
            'variants': [
                {'label': f'Fig2 [fw {FUTURE_WINDOW_OVERRIDE}]', 'kwargs': {'only_pi': False, 'plot_season': '', 'future_window': FUTURE_WINDOW_OVERRIDE}},
            ],
        },
        'Fig3_fw': {
            'function': make_fig3,
            'data_args': {
                'multi_model_dict': multi_model_dict, 'masks': masks,
                'reg_ds_mpi': reg_ds_mpi_fw, 'reg_ds_cesm': reg_ds_cesm_fw,
                'hosmip_reg_ds_dict': hosmip_reg_ds_dict_fw,
                'reg_ds_giss': reg_ds_giss_fw,
                'giss_panel_data': reg_ds_giss_panel,
            },
            'variants': [
                {'label': f'Fig3 [fw {FUTURE_WINDOW_OVERRIDE}]', 'kwargs': {'season': '', 'T_ref': 'pi', 'future_window': FUTURE_WINDOW_OVERRIDE}},
            ],
        },
        'FigSupp_mpi_tas_tmn_anomalies_fw': {
            'function': make_tas_anom,
            'data_args': {'data_dict': data_dict},
            'variants': [
                {'label': f'tas_anom [fw {FUTURE_WINDOW_OVERRIDE}]', 'kwargs': {'ssp': 'ssp245', 'future_window': FUTURE_WINDOW_OVERRIDE}},
            ],
        },
        'FigSupp_cesm_net_cooling_maps_fw': {
            'function': make_cesm_maps,
            'data_args': {
                'multi_model_dict': multi_model_dict, 'masks': masks,
                'boot_data': boot_data, 'reg_ds_cesm': reg_ds_cesm_fw,
            },
            'variants': [
                {'label': f'cesm_maps [fw {FUTURE_WINDOW_OVERRIDE}]', 'kwargs': {'future_window': FUTURE_WINDOW_OVERRIDE}},
            ],
        },
        'FigSupp_giss_net_cooling_maps_fw': {
            'function': make_giss_maps,
            'data_args': {'reg_ds_giss': reg_ds_giss_fw, 'masks': masks},
            'variants': [
                {'label': f'giss_maps [fw {FUTURE_WINDOW_OVERRIDE}]', 'kwargs': {'season': '', 'future_window': FUTURE_WINDOW_OVERRIDE}},
            ],
        },
    }
    FIGURE_CONFIGS.update(FIGURE_CONFIGS_FW)

#%%
########################################
# EXECUTION

# Filter by CLI arguments if provided (skip in interactive/Jupyter environments)
def _is_interactive():
    try:
        get_ipython() # type: ignore
        return True
    except NameError:
        return False

if _is_interactive():
    requested = list(FIGURE_CONFIGS.keys()) + ['FigSupp_cmip_cooling']  # generate all when running interactively
else:
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(FIGURE_CONFIGS.keys()) + ['FigSupp_cmip_cooling']

# Dedicated loader branch for the decade-matched CMIP-projected cooling figure:
# its per-decade window-keyed regression caches are built on demand (heavy),
# so register it only when actually requested. Decades are auto-pruned to those
# with >=1 first-onset cooling point (mirrors FigSupp_cmip_cooling.__main__).
if 'FigSupp_cmip_cooling' in requested and 'FigSupp_cmip_cooling' not in FIGURE_CONFIGS:
    print("Loading CMIP-projected cooling dataset + per-decade caches...")
    cmip_cooling_ds = cmip_cooling.make_cmip_cooling_ds(recompute=False)
    _cc_decades = supp_cmip_cooling._present_decades(cmip_cooling_ds)
    _cc_reg = supp_cmip_cooling._load_decade_caches(_cc_decades, masks)
    FIGURE_CONFIGS['FigSupp_cmip_cooling'] = {
        'function': make_cmip_cooling,
        'data_args': {'reg_by_decade': _cc_reg, 'masks': masks,
                      'cmip_cooling_ds': cmip_cooling_ds},
        'variants': [
            {'label': 'FigSupp_cmip_cooling', 'kwargs': {'decades': _cc_decades}},
        ],
    }

# Validate requested figures
for name in list(requested):
    if name not in FIGURE_CONFIGS:
        print(f"Warning: '{name}' not found in FIGURE_CONFIGS. Available: {list(FIGURE_CONFIGS.keys())}")
        requested.remove(name)

for fig_name in requested:
    config = FIGURE_CONFIGS[fig_name]
    func = config['function']
    data_args = config['data_args']

    for variant in config['variants']:
        label = variant['label']
        kwargs = variant['kwargs']
        print(f"Generating {label}...")
        try:
            fig, savepath = func(**data_args, **kwargs)
            plt.close(fig)
            print(f"  Saved: {savepath}")
        except Exception as e:
            print(f"  ERROR generating {label}: {e}")

print("\nDone.")

# %%
