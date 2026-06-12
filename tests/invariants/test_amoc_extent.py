"""Tier 2 — invariants on the recomputed CMIP6 AMOC weakening envelope.

These guard the JSON sidecar at ``data/cmip6_amoc_extent.json`` produced by
``functions.compute_amoc_extent(recompute=True)``. The cache feeds the
right-hand colorbar bars on Fig 2, FigGiss2, FigS15. If a methodology
change (PI window, future window, retracted-rea filter) silently slips in,
or if newly mirrored data accidentally shrinks the per-SSP rea count, this
catches it.

Skipped cleanly when the sidecar is absent (fresh checkout before the
first recompute), in line with the existing tier-2 pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


CACHE = Path('/home/m/m300940/teu_amoc/data/cmip6_amoc_extent.json')
BOOTSTRAP = Path('/home/m/m300940/teu_amoc/data/cmip6_inventory_snapshots/'
                 '2026-05-25_pre-rewire.json')


@pytest.fixture(scope='module')
def cache():
    if not CACHE.exists():
        pytest.skip(
            f'AMOC_extent cache missing at {CACHE}; run '
            'functions.compute_amoc_extent(recompute=True) to populate.')
    with open(CACHE, 'r') as f:
        return json.load(f)


@pytest.fixture(scope='module')
def bootstrap():
    if not BOOTSTRAP.exists():
        pytest.skip(f'bootstrap snapshot missing at {BOOTSTRAP}')
    with open(BOOTSTRAP, 'r') as f:
        return json.load(f)


def test_pi_and_future_windows_match_canonical(cache):
    """Cache must use the canonical PI / future windows (1850–1899 and
    2091–2100). If a methodology change updates the windows, this test
    fires — update the test together with the intended change.
    """
    assert cache['pi_window'] == ['1850', '1899']
    assert cache['future_window'] == ['2091', '2100']


def test_paper_ssps_present(cache):
    """The three paper SSPs must each have an envelope."""
    assert set(cache['ssps'].keys()) >= {'ssp126', 'ssp245', 'ssp370'}


@pytest.mark.parametrize('ssp', ['ssp126', 'ssp245', 'ssp370'])
def test_bounds_monotonic(cache, ssp):
    """Max weakening must exceed min weakening (sanity)."""
    lo, hi = cache['ssps'][ssp]['bounds_pct']
    assert lo > hi, f'{ssp} bounds {lo,hi} are not max>min'
    assert -200 < hi < 200, f'{ssp} min {hi}% is out of plausible range'
    assert -200 < lo < 200, f'{ssp} max {lo}% is out of plausible range'


@pytest.mark.parametrize('ssp', ['ssp126', 'ssp245', 'ssp370'])
def test_count_at_least_bootstrap(cache, bootstrap, ssp):
    """Per-SSP ``per_realisation`` count must not be lower than the
    bootstrap-snapshot count for that SSP. New data only widens the
    ensemble; never narrows it.
    """
    bootstrap_pairs = bootstrap['realisations']['amoc26'][ssp]
    bootstrap_count = sum(len(reas) for model, reas in bootstrap_pairs.items()
                          if model != 'MPI-ESM1-2-LR')
    cache_count = cache['ssps'][ssp]['n_realisations']
    assert cache_count >= bootstrap_count, (
        f'{ssp}: cache has {cache_count} reas but bootstrap had '
        f'{bootstrap_count}; new data should never shrink the envelope')


def test_cesm2_r4_present_in_ssp370(cache):
    """Spot check: a well-known rea (CESM2 r4 ssp370) appears in the cache."""
    pairs = cache['ssps']['ssp370']['per_realisation']
    rea_keys = {(r['model'], r['variant_id']) for r in pairs}
    assert ('CESM2', 'r4i1p1f1') in rea_keys


@pytest.mark.parametrize('ssp', ['ssp126', 'ssp245', 'ssp370'])
def test_both_aggregations_present(cache, ssp):
    """Sidecar must carry both per-realisation and per-model bounds for
    every SSP, so consumers can pick the aggregation matching their
    convention without recomputing.
    """
    entry = cache['ssps'][ssp]
    assert 'bounds_pct' in entry
    assert 'bounds_pct_model' in entry
    # Per-model envelope is narrower than per-realisation (means lie
    # strictly inside the realisation cloud, except at the single-rea
    # edge where they coincide).
    rea_lo, rea_hi = entry['bounds_pct']
    mod_lo, mod_hi = entry['bounds_pct_model']
    assert rea_lo >= mod_lo - 1e-9, (
        f'{ssp}: per-realisation max {rea_lo:.3f}% should be ≥ per-model '
        f'max {mod_lo:.3f}%')
    assert rea_hi <= mod_hi + 1e-9, (
        f'{ssp}: per-realisation min {rea_hi:.3f}% should be ≤ per-model '
        f'min {mod_hi:.3f}%')


@pytest.mark.parametrize('ssp', ['ssp126', 'ssp245', 'ssp370'])
def test_per_model_uses_matched_rea_set(cache, ssp):
    """For every per-model summary entry, the listed ``variant_ids`` must
    be exactly the reas that contributed to the per_realisation list
    (i.e. the hist∩scen intersection guaranteed by available_pairs).
    """
    rea_by_model = {}
    for r in cache['ssps'][ssp]['per_realisation']:
        rea_by_model.setdefault(r['model'], set()).add(r['variant_id'])
    for m in cache['ssps'][ssp]['per_model']:
        assert set(m['variant_ids']) == rea_by_model[m['model']], (
            f"{ssp} {m['model']}: per_model variant_ids "
            f"{m['variant_ids']} disagree with per_realisation "
            f"{sorted(rea_by_model[m['model']])}")


def test_retracted_absent_from_cache(cache):
    """No CESM2 r1-r3 i1p1f1 entries in any SSP envelope."""
    for ssp_entry in cache['ssps'].values():
        for r in ssp_entry['per_realisation']:
            assert not (r['model'] == 'CESM2'
                        and r['variant_id'] in {
                            'r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1'})
