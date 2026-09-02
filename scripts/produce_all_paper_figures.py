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
import FigSupp_synthetic_scaling_factors as supp_synth_sf # type: ignore
import FigSupp_mpi_intercept_comparison as supp_icpt_comp # type: ignore
import FigSupp_cmip_range_statedep_robustness as supp_range_robustness # type: ignore

for _mod in [functions, Fig1, Fig2, Fig3, Fig3_simple, supp_std_errors, supp_tas_anom,
             supp_reg_maps, supp_cesm_maps, supp_giss_maps, supp_giss_ts,
             supp_scenindep, supp_scenindep_cesm, supp_linearity_mpi,
             supp_linearity_other, supp_binary, cmip_cooling, supp_cmip_cooling,
             supp_warming_weak, supp_warm_unc, supp_scaling_corr, supp_synth_sf,
             supp_icpt_comp, supp_range_robustness]:
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
make_synth_sf = supp_synth_sf.make_figure
make_icpt_comp = supp_icpt_comp.make_figure
make_range_robustness = supp_range_robustness.make_figure

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
try:
    data_dict = functions.load_mpi_esm_data(eur_only=True)        # Fig1, tas_anom, linearity, scen-indep
except FileNotFoundError as e:
    # off-Levante cache mirror: raw MPI-GE/ssphos fields not mirrored; the
    # figures needing data_dict (Fig1 family) fail loudly at dispatch instead
    print(f"WARNING: raw MPI-ESM data unavailable ({e}); data_dict = None")
    data_dict = None

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

print("Loading CMIP6 target (w, t) for synthetic scaling factors...")
wt_ds = supp_synth_sf.get_target_wt(recompute=False)

# Per-country Synthetic CMIP6 range caches for Fig3/Fig3_simple panel e.
# Three state-dependence treatments x four calibration configurations
# (exclusions compose with the hosing default since 2026-08-31, so each
# variant differs from the default by one knob; predictors default to 'w'
# since 2026-09-01).
# The cache is cache-only here (build via the FigSupp script's BUILD RANGE
# CACHE cell), and a missing one raises rather than silently rendering cr-off.
print("Loading Synthetic CMIP6 range caches...")
CMIP_RANGE_CALS = ('hosing', 'full', 'nocesm2', 'consistentssp')
cmip_range_ds = {
    (sd, cs, ''): supp_synth_sf.get_cmip_range_ds(
        recompute=False, state_dep=sd, cal_set=cs, verbose=False)
    for sd in ('none', 'pooled', 'interact') for cs in CMIP_RANGE_CALS}
# Seasonal siblings: the PI-calibrated base and the paper default only, at the
# default calibration set. The seasonal state-dependence rests on the MPI,
# CESM2 and GISS warm rows (the EC-Earth3 pair's BM diff fields are annual),
# and 'loglin' has no seasonal form.
cmip_range_ds.update({
    (sd, 'hosing', se): supp_synth_sf.get_cmip_range_ds(
        recompute=False, state_dep=sd, cal_set='hosing', season=se, verbose=False)
    for sd in ('none', 'pooled') for se in ('djf', 'jja')})
# wt-conditioning reference for the pooled default (pred-wt sibling; the
# mirror of the pre-2026-09-01 pred-w sibling).
cmip_range_ds_pred_wt = supp_synth_sf.get_cmip_range_ds(
    recompute=False, state_dep='pooled', cal_set='hosing', predictors='wt',
    verbose=False)
# wt sibling of the 'none' treatment, for the state-dependence robustness
# figure's second (wt) render.
cmip_range_ds_none_pred_wt = supp_synth_sf.get_cmip_range_ds(
    recompute=False, state_dep='none', cal_set='hosing', predictors='wt',
    verbose=False)

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
            # Fig3 is supplementary now (Fig3_simple is the paper figure), so it
            # follows every behaviour change but carries a minimal variant set.
            # vwb is on by default; one vwb-off render is kept for verification.
            {'label': 'Fig3 (no vwb, verification)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'vwb_data': False}},
            # Range variants mirroring the paper figure's default (pooled,
            # hosing, w): annual plus the two seasonal siblings consumed by
            # the manuscript's fig3-old-djf/jja. The 3x4 sweep lives on
            # Fig3_simple.
            {'label': 'Fig3 (range pooled)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'cmip_range_show': True,
                        'cmip_range_ds': cmip_range_ds[('pooled', 'hosing', '')],
                        'cmip_range_statedep': 'pooled', 'cmip_range_calset': 'hosing'}},
            {'label': 'Fig3 (range pooled, DJF)',
             'kwargs': {'season': 'djf', 'T_ref': 'pi', 'cmip_range_show': True,
                        'cmip_range_ds': cmip_range_ds[('pooled', 'hosing', 'djf')],
                        'cmip_range_statedep': 'pooled', 'cmip_range_calset': 'hosing'}},
            {'label': 'Fig3 (range pooled, JJA)',
             'kwargs': {'season': 'jja', 'T_ref': 'pi', 'cmip_range_show': True,
                        'cmip_range_ds': cmip_range_ds[('pooled', 'hosing', 'jja')],
                        'cmip_range_statedep': 'pooled', 'cmip_range_calset': 'hosing'}},
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
            # The paper figure: the Synthetic CMIP6 range is on by default
            # (pooled correction, hosing calibration, w predictors). T_ref='pd'
            # variants cannot carry the range (the renderer defines it for the
            # PI-referenced panel only) and are explicit cr-off.
            {'label': 'Fig3_simple',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct',
                        'cmip_range_show': True, 'cmip_range_ds': cmip_range_ds[('pooled', 'hosing', '')],
                        'cmip_range_statedep': 'pooled', 'cmip_range_calset': 'hosing'}},
            {'label': 'Fig3_simple',
             'kwargs': {'season': '', 'T_ref': 'pd', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_range_show': False}},
            {'label': 'Fig3_simple',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': True, 'aggregate_first': True, 'weakening_unit': 'pct',
                        'cmip_range_show': True, 'cmip_range_ds': cmip_range_ds[('pooled', 'hosing', '')],
                        'cmip_range_statedep': 'pooled', 'cmip_range_calset': 'hosing'}},
            {'label': 'Fig3_simple',
             'kwargs': {'season': 'djf', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct',
                        'cmip_range_show': True, 'cmip_range_ds': cmip_range_ds[('pooled', 'hosing', 'djf')],
                        'cmip_range_statedep': 'pooled', 'cmip_range_calset': 'hosing'}},
            {'label': 'Fig3_simple',
             'kwargs': {'season': 'jja', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct',
                        'cmip_range_show': True, 'cmip_range_ds': cmip_range_ds[('pooled', 'hosing', 'jja')],
                        'cmip_range_statedep': 'pooled', 'cmip_range_calset': 'hosing'}},
            {'label': 'Fig3_simple',
             'kwargs': {'season': 'djf', 'T_ref': 'pd', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_range_show': False}},
            {'label': 'Fig3_simple',
             'kwargs': {'season': 'jja', 'T_ref': 'pd', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_range_show': False}},
            # Explicit range-off render: regression anchor (nothing outside the
            # range branch may move) and the clean panel-b look.
            {'label': 'Fig3_simple (range off)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_range_show': False}},
            # Appendix/internal: original NAHosMIP min-max hatched on top.
            {'label': 'Fig3_simple (NAHosMIP overlay)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct',
                        'cmip_range_show': True, 'cmip_range_ds': cmip_range_ds[('pooled', 'hosing', '')],
                        'cmip_range_statedep': 'pooled', 'cmip_range_calset': 'hosing',
                        'nahosmip_overlay': True}},
            # wt-conditioning reference: the pooled default on the pred-wt
            # sibling cache, with the NAHosMIP overlay (appendix/internal).
            {'label': 'Fig3_simple (wt, NAHosMIP overlay)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct',
                        'cmip_range_show': True, 'cmip_range_ds': cmip_range_ds_pred_wt,
                        'cmip_range_statedep': 'pooled', 'cmip_range_calset': 'hosing',
                        'cmip_range_predictors': 'wt', 'nahosmip_overlay': True}},
            # vwb is on by default; one vwb-off render is kept for verification
            # (range off to keep the anchor minimal).
            {'label': 'Fig3_simple (no vwb, verification)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'vwb_data': False, 'cmip_range_show': False}},
            {'label': 'Fig3_simple (CMIP cooling)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_cooling_show': True, 'cmip_range_show': False}},
            {'label': 'Fig3_simple (aggregate last)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': False, 'weakening_unit': 'pct', 'cmip_range_show': False}},
            # Synthetic CMIP6 range sweep: 3 state-dependence treatments x 4
            # calibration configurations, minus the default (pooled, hosing)
            # which is the first variant above. 'none' is the PI-calibrated
            # base, 'pooled' the proportional restriction, 'interact' the
            # unrestricted warm line (its PI side is the PI-only OLS).
            # Exclusion sets compose with the hosing default (2026-08-31), so
            # each variant differs from the default by exactly one knob.
            {'label': 'Fig3_simple (range none)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_range_show': True,
                        'cmip_range_ds': cmip_range_ds[('none', 'full', '')],
                        'cmip_range_statedep': 'none', 'cmip_range_calset': 'full'}},
            {'label': 'Fig3_simple (range none, nocesm2)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_range_show': True,
                        'cmip_range_ds': cmip_range_ds[('none', 'nocesm2', '')],
                        'cmip_range_statedep': 'none', 'cmip_range_calset': 'nocesm2'}},
            {'label': 'Fig3_simple (range none, hosing)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_range_show': True,
                        'cmip_range_ds': cmip_range_ds[('none', 'hosing', '')],
                        'cmip_range_statedep': 'none', 'cmip_range_calset': 'hosing'}},
            {'label': 'Fig3_simple (range none, consistentssp)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_range_show': True,
                        'cmip_range_ds': cmip_range_ds[('none', 'consistentssp', '')],
                        'cmip_range_statedep': 'none', 'cmip_range_calset': 'consistentssp'}},
            {'label': 'Fig3_simple (range pooled)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_range_show': True,
                        'cmip_range_ds': cmip_range_ds[('pooled', 'full', '')],
                        'cmip_range_statedep': 'pooled', 'cmip_range_calset': 'full'}},
            {'label': 'Fig3_simple (range pooled, nocesm2)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_range_show': True,
                        'cmip_range_ds': cmip_range_ds[('pooled', 'nocesm2', '')],
                        'cmip_range_statedep': 'pooled', 'cmip_range_calset': 'nocesm2'}},
            {'label': 'Fig3_simple (range pooled, consistentssp)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_range_show': True,
                        'cmip_range_ds': cmip_range_ds[('pooled', 'consistentssp', '')],
                        'cmip_range_statedep': 'pooled', 'cmip_range_calset': 'consistentssp'}},
            {'label': 'Fig3_simple (range interact)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_range_show': True,
                        'cmip_range_ds': cmip_range_ds[('interact', 'full', '')],
                        'cmip_range_statedep': 'interact', 'cmip_range_calset': 'full'}},
            {'label': 'Fig3_simple (range interact, nocesm2)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_range_show': True,
                        'cmip_range_ds': cmip_range_ds[('interact', 'nocesm2', '')],
                        'cmip_range_statedep': 'interact', 'cmip_range_calset': 'nocesm2'}},
            {'label': 'Fig3_simple (range interact, hosing)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_range_show': True,
                        'cmip_range_ds': cmip_range_ds[('interact', 'hosing', '')],
                        'cmip_range_statedep': 'interact', 'cmip_range_calset': 'hosing'}},
            {'label': 'Fig3_simple (range interact, consistentssp)',
             'kwargs': {'season': '', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_range_show': True,
                        'cmip_range_ds': cmip_range_ds[('interact', 'consistentssp', '')],
                        'cmip_range_statedep': 'interact', 'cmip_range_calset': 'consistentssp'}},
            # Seasonal PI-calibrated base ranges at the default calibration
            # set (the seasonal paper defaults with the pooled range are the
            # plain seasonal variants above). Warming side is seasonal
            # throughout; the weakening side stays annual (no seasonal AMOC
            # at 26N).
            {'label': 'Fig3_simple (range none, DJF)',
             'kwargs': {'season': 'djf', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_range_show': True,
                        'cmip_range_ds': cmip_range_ds[('none', 'hosing', 'djf')],
                        'cmip_range_statedep': 'none', 'cmip_range_calset': 'hosing'}},
            {'label': 'Fig3_simple (range none, JJA)',
             'kwargs': {'season': 'jja', 'T_ref': 'pi', 'hosmip_markers': False, 'aggregate_first': True, 'weakening_unit': 'pct', 'cmip_range_show': True,
                        'cmip_range_ds': cmip_range_ds[('none', 'hosing', 'jja')],
                        'cmip_range_statedep': 'none', 'cmip_range_calset': 'hosing'}},
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
            'reg_ds_giss': reg_ds_giss,
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
        # Since 2026-09-01 the panels carry the Synthetic CMIP6 range of dT
        # at each weakening level (schema-v3 dT_* vars, Fig3-default pooled/
        # hosing/w spec, season-matched) instead of the NAHosMIP min-max bar.
        'variants': [
            {'label': 'FigSupp_warming_at_weakening',
             'kwargs': {'weakenings': (25.0, 75.0), 'season': '', 'T_ref': 'pi',
                        'cmip_range_ds': cmip_range_ds[('pooled', 'hosing', '')]}},
            {'label': 'FigSupp_warming_at_weakening (DJF)',
             'kwargs': {'weakenings': (25.0, 75.0), 'season': 'djf', 'T_ref': 'pi',
                        'cmip_range_ds': cmip_range_ds[('pooled', 'hosing', 'djf')]}},
            {'label': 'FigSupp_warming_at_weakening (JJA)',
             'kwargs': {'weakenings': (25.0, 75.0), 'season': 'jja', 'T_ref': 'pi',
                        'cmip_range_ds': cmip_range_ds[('pooled', 'hosing', 'jja')]}},
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
    'FigSupp_synthetic_scaling_factors': {
        'function': make_synth_sf,
        'data_args': {
            'wt_ds': wt_ds,
            'multi_model_dict': multi_model_dict, 'masks': masks,
            'hosmip_reg_ds_dict': hosmip_reg_ds_dict,
            'reg_ds_giss': reg_ds_giss, 'reg_ds_giss_panel': reg_ds_giss_panel,
            'gwl_data': gwl_data,
            # Needed by the sdep-pooled variants below, and supplied to every
            # other variant so the sidecar's state-dependence block is populated
            # in canonical renders too (state_dep defaults to 'none' there, so
            # those figures are unaffected).
            'reg_ds_mpi': reg_ds_mpi, 'reg_ds_cesm': reg_ds_cesm,
        },
        'variants': [
            # Mirrors the Fig3 range sweep's calibration sets, each with both
            # predictor specs: the w default (since 2026-09-01) and the wt
            # sibling. cal_set='hosing' default since 2026-08-31; exclusion
            # sets compose with the hosing default, and 'full' adds GISS back.
            {'label': 'FigSupp_synthetic_scaling_factors',
             'kwargs': {}},
            {'label': 'FigSupp_synthetic_scaling_factors (wt)',
             'kwargs': {'predictors': 'wt'}},
            {'label': 'FigSupp_synthetic_scaling_factors (full)',
             'kwargs': {'cal_set': 'full'}},
            {'label': 'FigSupp_synthetic_scaling_factors (full, wt)',
             'kwargs': {'cal_set': 'full', 'predictors': 'wt'}},
            {'label': 'FigSupp_synthetic_scaling_factors (no CESM2)',
             'kwargs': {'cal_set': 'nocesm2'}},
            {'label': 'FigSupp_synthetic_scaling_factors (no CESM2, wt)',
             'kwargs': {'cal_set': 'nocesm2', 'predictors': 'wt'}},
            {'label': 'FigSupp_synthetic_scaling_factors (consistent SSPs)',
             'kwargs': {'cal_set': 'consistentssp'}},
            {'label': 'FigSupp_synthetic_scaling_factors (consistent SSPs, wt)',
             'kwargs': {'cal_set': 'consistentssp', 'predictors': 'wt'}},
            # Seasonal siblings at the default calibration set (2026-08-23;
            # re-pointed to the hosing default 2026-08-31). The cal_set
            # sensitivities stay annual; season and calibration set are
            # independent axes and the seasonal question is about the fit, not
            # about which models are in it.
            {'label': 'FigSupp_synthetic_scaling_factors (DJF)',
             'kwargs': {'season': 'djf'}},
            {'label': 'FigSupp_synthetic_scaling_factors (DJF, wt)',
             'kwargs': {'predictors': 'wt', 'season': 'djf'}},
            {'label': 'FigSupp_synthetic_scaling_factors (JJA)',
             'kwargs': {'season': 'jja'}},
            {'label': 'FigSupp_synthetic_scaling_factors (JJA, wt)',
             'kwargs': {'predictors': 'wt', 'season': 'jja'}},
            # State-dependence: the targets are predicted on the fitted warm
            # line, so the Boot (CESM2) and BM (EC-Earth3) warm slopes reach the
            # plotted panels rather than only the sidecar. 'pooled' at the
            # default calibration set is the treatment Fig3 defaults to, so the
            # annual variant below is the counterpart of the paper figure; the
            # seasonal fits have three warm observations (MPI, CESM2, GISS)
            # against four annually, EC-Earth3's BM diff fields being
            # annual-only.
            {'label': 'FigSupp_synthetic_scaling_factors (pooled)',
             'kwargs': {'state_dep': 'pooled'}},
            {'label': 'FigSupp_synthetic_scaling_factors (pooled, wt)',
             'kwargs': {'predictors': 'wt', 'state_dep': 'pooled'}},
            {'label': 'FigSupp_synthetic_scaling_factors (DJF, pooled)',
             'kwargs': {'season': 'djf', 'state_dep': 'pooled'}},
            {'label': 'FigSupp_synthetic_scaling_factors (JJA, pooled)',
             'kwargs': {'season': 'jja', 'state_dep': 'pooled'}},
        ],
    },
    # Stacked three-version comparison (2026-08-31): hosing/w, nocesm2/w,
    # hosing/wt, all state_dep='none'; full a+b row per version. EU + IE per
    # the standing between-version comparison convention.
    'FigSupp_synthetic_scaling_factors_stack': {
        'function': supp_synth_sf.make_stacked_figure,
        'data_args': {
            'wt_ds': wt_ds,
            'multi_model_dict': multi_model_dict, 'masks': masks,
            'hosmip_reg_ds_dict': hosmip_reg_ds_dict,
            'reg_ds_giss': reg_ds_giss, 'reg_ds_giss_panel': reg_ds_giss_panel,
            'gwl_data': gwl_data,
            'reg_ds_mpi': reg_ds_mpi, 'reg_ds_cesm': reg_ds_cesm,
        },
        'variants': [
            {'label': 'FigSupp_synthetic_scaling_factors_stack',
             'kwargs': {}},
            {'label': 'FigSupp_synthetic_scaling_factors_stack (IE)',
             'kwargs': {'region': 'IE'}},
            # Same 3-row stack, state-dependence-corrected instead of 'none'
            # (EU + IE per the standing between-version comparison convention).
            {'label': 'FigSupp_synthetic_scaling_factors_stack (pooled)',
             'kwargs': {'state_dep': 'pooled'}},
            {'label': 'FigSupp_synthetic_scaling_factors_stack (pooled, IE)',
             'kwargs': {'state_dep': 'pooled', 'region': 'IE'}},
        ],
    },
    'FigSupp_mpi_intercept_comparison': {
        'function': make_icpt_comp,
        'data_args': {'data_dict': data_dict},
        'variants': [
            {'label': 'FigSupp_mpi_intercept_comparison', 'kwargs': {'season': '', 'window': 10, 'hosing': 'all'}},
            # PT has the largest free-intercept of the 34 countries >= ~30k km2
            # (-0.147 degC vs -0.036 for EU), so the two conventions differ most there;
            # its spread needs the wider ylim.
            {'label': 'FigSupp_mpi_intercept_comparison (PT)',
             'kwargs': {'season': '', 'window': 10, 'hosing': 'all', 'region': 'PT', 'ylim': (-2.0, 1.0)}},
        ],
    },
    # Two 2-panel robustness checks for the Synthetic CMIP6 range (split from
    # one 3-panel figure 2026-09-01 for legibility): version='statedep' is
    # no correction vs pooled correction (both 'medians'); version='mode' is
    # 'medians' vs 'mc' display without correction. cmip_range_mode is a
    # render-time selector only, so the 'mode' variants reuse one loaded
    # cmip_range_ds (see FigSupp_cmip_range_statedep_robustness.py's LOAD
    # DATA cell / docstring). Each version renders for w and the wt sibling.
    'FigSupp_cmip_range_statedep_robustness': {
        'function': make_range_robustness,
        'data_args': {
            'reg_ds_mpi': reg_ds_mpi, 'reg_ds_cesm': reg_ds_cesm, 'masks': masks,
            'hosmip_reg_ds_dict': hosmip_reg_ds_dict, 'reg_ds_giss': reg_ds_giss,
        },
        'variants': [
            {'label': 'FigSupp_cmip_range_statedep_robustness',
             'kwargs': {'cmip_range_ds_none': cmip_range_ds[('none', 'hosing', '')],
                        'cmip_range_ds_pooled': cmip_range_ds[('pooled', 'hosing', '')],
                        'cmip_range_predictors': 'w', 'version': 'statedep'}},
            {'label': 'FigSupp_cmip_range_statedep_robustness (wt)',
             'kwargs': {'cmip_range_ds_none': cmip_range_ds_none_pred_wt,
                        'cmip_range_ds_pooled': cmip_range_ds_pred_wt,
                        'cmip_range_predictors': 'wt', 'version': 'statedep'}},
            {'label': 'FigSupp_cmip_range_mode_robustness',
             'kwargs': {'cmip_range_ds_none': cmip_range_ds[('none', 'hosing', '')],
                        'cmip_range_ds_pooled': cmip_range_ds[('pooled', 'hosing', '')],
                        'cmip_range_predictors': 'w', 'version': 'mode'}},
            {'label': 'FigSupp_cmip_range_mode_robustness (wt)',
             'kwargs': {'cmip_range_ds_none': cmip_range_ds_none_pred_wt,
                        'cmip_range_ds_pooled': cmip_range_ds_pred_wt,
                        'cmip_range_predictors': 'wt', 'version': 'mode'}},
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
