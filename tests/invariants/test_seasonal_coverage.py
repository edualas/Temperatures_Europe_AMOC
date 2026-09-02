"""Tier 1 — invariants on seasonal tas coverage.

The seasonal ensembles are derived from the annual ones by
``scripts/processing/process_seasonal_tas.py``, so seasonal coverage can
never exceed annual coverage, and for the models the seasonal figures
consume it should equal it. Both loaders concatenate whatever members they
find into a single ``realiz`` dimension, so a silent shortfall changes the
ensemble mean without changing anything visible.

Guards against:
- a seasonal file appearing with no annual counterpart (wrong member id,
  stale staging, a file written under the wrong model directory),
- the consumed ensembles quietly falling behind the annual ones again,
  which is what happened between 2026-05-24 and 2026-08-17.

ssp585 is excluded: it is opt-in, absent from the submitted figures, and
seasonally empty by design.
"""

from __future__ import annotations

import pytest


# Models drawn in the seasonal figure variants: the HosMIP set
# (``functions.hosmip_labels``) plus GISS-E2-1-G, which supplies the Fig3
# panel-e markers. MPI-ESM1-2-LR is absent on purpose — it is loaded from
# MPI-GE, not the CMIP6 archive, so it never appears in the inventory scan.
CONSUMED = [
    'CanESM5', 'EC-Earth3', 'CESM2', 'IPSL-CM6A-LR',
    'HadGEM3-GC31-MM', 'HadGEM3-GC31-LL', 'MPI-ESM1-2-HR', 'GISS-E2-1-G',
]

SCENARIOS = ('historical', 'ssp126', 'ssp245', 'ssp370')

# Members deliberately not staged seasonally, with the reason. CESM2 ssp370
# r5 and r6 are byte-identical upstream; staging both would double-count one
# realisation, so neither is staged until the annual duplicate is resolved.
EXPECTED_GAPS = {
    ('ssp370', 'CESM2'): {'r5i1p1f1', 'r6i1p1f1'},
}


@pytest.fixture(scope='module')
def inv(scripts_dir):
    import sys
    sys.path.insert(0, str(scripts_dir))
    import cmip6_inventory
    return cmip6_inventory


@pytest.mark.parametrize('scenario', SCENARIOS)
@pytest.mark.parametrize('season', ('djf', 'jja'))
def test_seasonal_is_subset_of_annual(inv, scenario, season):
    """No seasonal file may exist without its annual counterpart."""
    annual = inv.available_realisations(scenario, 'tas')
    seasonal = inv.available_seasonal_realisations(scenario, season)
    orphans = []
    for model, reas in seasonal.items():
        extra = set(reas) - set(annual.get(model, []))
        orphans += [f'{model}/{r}' for r in sorted(extra)]
    assert not orphans, (
        f'{scenario} {season}: {len(orphans)} seasonal file(s) with no annual '
        f'counterpart: {orphans[:10]}')


@pytest.mark.parametrize('scenario', SCENARIOS)
@pytest.mark.parametrize('season', ('djf', 'jja'))
def test_consumed_ensembles_are_complete(inv, scenario, season):
    """For the models the seasonal figures draw, every annual member must
    have a seasonal counterpart apart from the documented refusals.
    """
    missing = inv.missing_seasonal(scenario, season, models=CONSUMED)
    unexpected = {}
    for model, reas in missing.items():
        gap = set(reas) - EXPECTED_GAPS.get((scenario, model), set())
        if gap:
            unexpected[model] = sorted(gap)
    assert not unexpected, (
        f'{scenario} {season}: consumed models missing seasonal members '
        f'{unexpected}; stage them or add them to EXPECTED_GAPS with a reason')


def test_expected_gaps_are_still_gaps(inv):
    """A refusal that has quietly been filled should be removed from
    EXPECTED_GAPS rather than left as a stale exemption.
    """
    stale = []
    for (scenario, model), reas in EXPECTED_GAPS.items():
        for season in ('djf', 'jja'):
            on_disk = set(inv.available_seasonal_realisations(
                scenario, season, models=[model])[model])
            stale += [f'{scenario}/{season}/{model}/{r}'
                      for r in sorted(reas & on_disk)]
    assert not stale, (
        f'EXPECTED_GAPS lists members that are now staged: {stale}')


def test_both_seasons_cover_the_same_members(inv):
    """DJF and JJA are written from the same monthly source in one pass, so
    a member present in one season and not the other means a half-finished
    staging run.
    """
    lopsided = []
    for scenario in SCENARIOS:
        djf = inv.available_seasonal_realisations(scenario, 'djf')
        jja = inv.available_seasonal_realisations(scenario, 'jja')
        for model in set(djf) | set(jja):
            d, j = set(djf.get(model, [])), set(jja.get(model, []))
            lopsided += [f'{scenario}/{model}/{r} (djf only)'
                         for r in sorted(d - j)]
            lopsided += [f'{scenario}/{model}/{r} (jja only)'
                         for r in sorted(j - d)]
    assert not lopsided, (
        f'{len(lopsided)} member(s) staged in one season only: {lopsided[:10]}')


# ── Target-side (w, t) cache: the seasonal CMIP range inputs ────────────────
# The synthetic-scaling targets carry a season dim on the warming side only.
# AMOC at 26N has no seasonal sibling, and the member set is decided on the
# annual availability, so w / baseline_amoc / n_real must be bit-identical
# across seasons. A season-dependent w would mean the builder had silently
# re-derived membership per season.

WT_COUNTRIES = 'cmip_synth_wt_countries.nc'


@pytest.fixture(scope='module')
def wt_ds(cached_data_path):
    import xarray as xr
    p = cached_data_path / WT_COUNTRIES
    if not p.exists():
        pytest.skip(f'{WT_COUNTRIES} not cached')
    ds = xr.open_dataset(p)
    if 'season' not in ds.dims:
        pytest.skip(f'{WT_COUNTRIES} predates the season dim')
    return ds


@pytest.mark.parametrize('var', ('w', 'baseline_amoc', 'n_real'))
def test_wt_amoc_side_has_no_season_dim(wt_ds, var):
    assert 'season' not in wt_ds[var].dims, (
        f'{var} gained a season dim; AMOC at 26N has no seasonal sibling')


def test_wt_warming_side_varies_with_season(wt_ds):
    """A season dim whose values are identical everywhere means the seasonal
    tas never reached the builder (the failure the DJF/JJA panels cannot show).
    """
    import numpy as np
    spread = float(np.nanmax(wt_ds.t.std('season').values))
    assert spread > 0.05, (
        f'target warming is season-invariant (max std {spread:.4f} K); the '
        f'seasonal tas is not reaching target_wt_model_multi')


def test_wt_targets_are_seasonally_complete(inv):
    """Every member the target (w, t) build consumes must exist in both
    seasons, or the seasonal ensemble mean silently averages a different set.
    """
    import sys
    sys.path.insert(0, str(inv.__file__.rsplit('/', 1)[0] + '/supplementary'))
    import FigSupp_synthetic_scaling_factors as ssf
    gaps = []
    for scenario in ('historical',) + ssf.MAIN_SSPS:
        tas = inv.available_realisations(scenario, 'tas',
                                         models=ssf.TARGET_MODELS, esgf_names=False)
        amoc = inv.available_realisations(scenario, 'amoc',
                                          models=ssf.TARGET_MODELS, esgf_names=False)
        for season in ('djf', 'jja'):
            seas = inv.available_seasonal_realisations(
                scenario, season, models=ssf.TARGET_MODELS, esgf_names=False)
            for model in ssf.TARGET_MODELS:
                consumed = set(tas.get(model, [])) & set(amoc.get(model, []))
                gaps += [f'{scenario}/{season}/{model}/{r}'
                         for r in sorted(consumed - set(seas.get(model, [])))]
    assert not gaps, (
        f'{len(gaps)} target member(s) consumed annually but absent '
        f'seasonally: {gaps[:10]}')
