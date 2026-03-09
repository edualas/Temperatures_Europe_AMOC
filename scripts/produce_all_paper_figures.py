#%%

"""
Produce all paper figures from a single script.

Usage:
    python produce_all_paper_figures.py              # Generate all figures
    python produce_all_paper_figures.py Fig1 Fig3    # Generate only Fig1 and Fig3 variants
    python produce_all_paper_figures.py FigS9        # Generate only FigS9
"""

import sys
import importlib
import matplotlib.pyplot as plt
import functions

import Fig1, Fig2, Fig3, FigS9, FigS10, FigS11, FigS15
for _mod in [functions, Fig1, Fig2, Fig3, FigS9, FigS10, FigS11, FigS15]:
    importlib.reload(_mod)

from Fig1 import make_figure as make_fig1
from Fig2 import make_figure as make_fig2
from Fig3 import make_figure as make_fig3
from FigS9 import make_figure as make_figs9
from FigS10 import make_figure as make_figs10
from FigS11 import make_figure as make_figs11
from FigS15 import make_figure as make_figs15

#%%
########################################
# DATA LOADING
# Each dataset is loaded once and shared across all figures that need it.

print("Loading MPI-ESM data...")
data_dict = functions.load_mpi_esm_data(eur_only=True)            # Fig1, FigS10

print("Loading MPI-ESM regression dataset...")
reg_ds_mpi = functions.load_regression_ds_mpi()                    # Fig2, Fig3, FigS9, FigS11

print("Loading CESM2 regression dataset...")
reg_ds_cesm = functions.get_cesm_reg_ds(recompute=False)           # Fig3, FigS11, FigS15

print("Loading HosMIP multi-model data...")
multi_model_dict, masks = functions.get_full_multi_model_dict()    # Fig3, FigS11, FigS15

print("Loading HosMIP regression datasets...")
hosmip_reg_ds_dict = functions.get_hosmip_reg_ds(recompute=False)  # Fig3, FigS11

print("Loading other studies data...")
_, _, boot_data, _ = functions.get_other_studies_data(masks)        # FigS15

print("All data loaded.\n")

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
        ],
    },
    'Fig2': {
        'function': make_fig2,
        'data_args': {'reg_ds_mpi': reg_ds_mpi},
        'variants': [
            {'label': 'Fig2',            'kwargs': {'only_pi': False, 'plot_season': ''}},
            {'label': 'FigS7 (DJF)',     'kwargs': {'only_pi': False, 'plot_season': 'djf'}},
            {'label': 'FigS8 (JJA)',     'kwargs': {'only_pi': False, 'plot_season': 'jja'}},
        ],
    },
    'Fig3': {
        'function': make_fig3,
        'data_args': {
            'multi_model_dict': multi_model_dict, 'masks': masks,
            'reg_ds_mpi': reg_ds_mpi, 'reg_ds_cesm': reg_ds_cesm,
            'hosmip_reg_ds_dict': hosmip_reg_ds_dict,
        },
        'variants': [
            {'label': 'Fig3',            'kwargs': {'season': '', 'T_ref': 'pi'}},
            {'label': 'FigS (PD ref)',   'kwargs': {'season': '', 'T_ref': 'pd'}},
            {'label': 'FigS (DJF)',      'kwargs': {'season': 'djf', 'T_ref': 'pi'}},
            {'label': 'FigS (JJA)',      'kwargs': {'season': 'jja', 'T_ref': 'pi'}},
            {'label': 'Fig3 (markers)',   'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': True}},
        ],
    },
    'FigS9': {
        'function': make_figs9,
        'data_args': {'reg_ds_mpi': reg_ds_mpi},
        'variants': [
            {'label': 'FigS9',           'kwargs': {'season': ''}},
        ],
    },
    'FigS10': {
        'function': make_figs10,
        'data_args': {'data_dict': data_dict},
        'variants': [
            {'label': 'FigS10 (ssp245)', 'kwargs': {'ssp': 'ssp245'}},
        ],
    },
    'FigS11': {
        'function': make_figs11,
        'data_args': {
            'multi_model_dict': multi_model_dict, 'masks': masks,
            'reg_ds_mpi': reg_ds_mpi, 'reg_ds_cesm': reg_ds_cesm,
            'hosmip_reg_ds_dict': hosmip_reg_ds_dict,
        },
        'variants': [
            {'label': 'FigS11 (annual)',       'kwargs': {'season': '', 'plot_ste': False}},
            {'label': 'FigS11 (DJF)',          'kwargs': {'season': 'djf', 'plot_ste': False}},
            {'label': 'FigS11 (JJA)',          'kwargs': {'season': 'jja', 'plot_ste': False}},
            {'label': 'FigS11 (std errors)',   'kwargs': {'season': '', 'plot_ste': True, 'ste_relative': False}},
            {'label': 'FigS11 (rel std err)',  'kwargs': {'season': '', 'plot_ste': True, 'ste_relative': True}},
        ],
    },
    'FigS15': {
        'function': make_figs15,
        'data_args': {
            'multi_model_dict': multi_model_dict, 'masks': masks,
            'boot_data': boot_data, 'reg_ds_cesm': reg_ds_cesm,
        },
        'variants': [
            {'label': 'FigS15',           'kwargs': {}},
        ],
    },
}

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
    requested = list(FIGURE_CONFIGS.keys())  # generate all when running interactively
else:
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(FIGURE_CONFIGS.keys())

# Validate requested figures
for name in requested:
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
