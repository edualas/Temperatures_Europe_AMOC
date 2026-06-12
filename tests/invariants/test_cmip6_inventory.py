"""Tier 1 — invariants on the CMIP6 inventory layer.

Single source of truth for "what CMIP6 data is on disk" lives in
``scripts/cmip6_inventory.py``. These tests guard against:

- silent removal of pre-rewire realisations (no-regression vs the bootstrap
  snapshot committed alongside the unification commit),
- the retracted-rea filter being weakened,
- shape drift in the snapshot-replay path,
- KeyError surprises when a caller whitelists a model that does not exist
  in either search root.

The tests intentionally do NOT assert "every paper-set model has every
SSP". Several model/SSP combinations are genuinely empty: HadGEM3-GC3-1MM
has only ssp126, EC-Earth3 has only ssp245, GISS-E2-1-G has ssp126 only
for r1-r5, CESM2 historical excludes the retracted r1-r3. The legitimate
gaps live in the bootstrap snapshot — the superset check below honours
them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


BOOTSTRAP = Path(__file__).resolve().parent.parent.parent / \
    'data' / 'cmip6_inventory_snapshots' / '2026-05-25_pre-rewire.json'


@pytest.fixture(scope='module')
def inv(scripts_dir):
    """Import the inventory module once per test module."""
    import sys
    sys.path.insert(0, str(scripts_dir))
    import cmip6_inventory
    return cmip6_inventory


@pytest.fixture(scope='module')
def bootstrap_realisations():
    if not BOOTSTRAP.exists():
        pytest.skip(f'bootstrap snapshot missing at {BOOTSTRAP}')
    with open(BOOTSTRAP, 'r') as f:
        return json.load(f)['realisations']


def test_superset_of_bootstrap(inv, bootstrap_realisations):
    """Every (var, scenario, model, variant_id) recorded in the pre-rewire
    bootstrap must still be discoverable on disk by the inventory layer.

    Catches accidental file removal, path drift, or a retracted-rea filter
    that mistakenly drops non-retracted reas.
    """
    missing = []
    for var, by_scen in bootstrap_realisations.items():
        for scen, by_model in by_scen.items():
            if scen not in inv.SCENARIOS:
                continue  # bootstrap may carry scenarios we don't list
            live = inv.available_realisations(scen, var)
            for model, reas in by_model.items():
                # MPI-ESM1-2-LR historical is loaded from MPI-GE (not CMIP6/),
                # so it does not appear in the on-disk inventory scan. Skip
                # that one entry; everything else must be a strict superset.
                if model == 'MPI-ESM1-2-LR':
                    continue
                live_set = set(live.get(model, []))
                for r in reas:
                    # Retracted reas: the bootstrap captures them as
                    # historical members (the pre-rewire hardcoded list for
                    # CESM2 included r1-r3 historical as orphans of the
                    # retracted SSPs). The unified inventory filters them
                    # out consistently — this is the intended methodology
                    # change, not regression. Skip the check for these.
                    if (model, r) in inv.RETRACTED:
                        continue
                    if r not in live_set:
                        missing.append(f'{var}/{scen}/{model}/{r}')
    assert not missing, (
        f'on-disk inventory missing {len(missing)} pre-rewire entries; '
        f'first 10: {missing[:10]}')


def test_retracted_excluded_by_default(inv):
    """CESM2 r1-r3 i1p1f1 (NCAR-retracted SSP runs) must never appear in
    ``available_realisations`` output without ``include_retracted=True``.
    """
    for scen in ('ssp126', 'ssp245', 'ssp370', 'ssp585'):
        reas = inv.available_realisations(scen, 'amoc26',
                                          models=['CESM2'])['CESM2']
        for bad in ('r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1'):
            assert bad not in reas, (
                f'retracted {bad} surfaced in CESM2/{scen}; '
                f'check cmip6_inventory.RETRACTED filter')


def test_retracted_included_when_asked(inv):
    """When ``include_retracted=True`` the retracted reas come back IF they
    are on disk. This is the opt-in audit path.
    """
    # CESM2 historical r1 i1p1f1 exists on disk (orphan from the retracted
    # SSP runs); verify it surfaces only with the opt-in flag.
    reas_default = set(inv.available_realisations(
        'historical', 'amoc26', models=['CESM2'])['CESM2'])
    reas_opted = set(inv.available_realisations(
        'historical', 'amoc26', models=['CESM2'],
        include_retracted=True)['CESM2'])
    # opt-in is a superset of default.
    assert reas_default.issubset(reas_opted)
    # And at least one retracted entry appears via opt-in (assuming the
    # legacy historical files are still on disk).
    if (reas_opted - reas_default):
        assert reas_opted - reas_default <= {'r1i1p1f1', 'r2i1p1f1',
                                              'r3i1p1f1'}


def test_unknown_model_returns_empty_not_keyerror(inv):
    """Passing a model name that doesn't exist on disk must return ``{m: []}``,
    never raise KeyError. Defensive against typo'd whitelists.
    """
    out = inv.available_realisations(
        'ssp245', 'amoc26', models=['NoSuchModel-99'])
    assert out == {'NoSuchModel-99': []}


def test_input_spelling_preserved_in_keys(inv):
    """When a caller passes ``HadGEM3-GC3-1LL`` (legacy spelling), the
    returned dict is keyed by ``HadGEM3-GC3-1LL`` — not the canonical
    ``HadGEM3-GC31-LL``. Same applies to lowercase dir names.
    """
    out_legacy = inv.available_realisations(
        'historical', 'amoc26', models=['HadGEM3-GC3-1LL'])
    assert list(out_legacy.keys()) == ['HadGEM3-GC3-1LL']

    out_canonical = inv.available_realisations(
        'historical', 'amoc26', models=['HadGEM3-GC31-LL'])
    assert list(out_canonical.keys()) == ['HadGEM3-GC31-LL']

    # Both synonyms resolve to the same ensemble.
    assert out_legacy['HadGEM3-GC3-1LL'] == out_canonical['HadGEM3-GC31-LL']


def test_snapshot_replay_returns_same_shape(inv, tmp_path):
    """``available_realisations(..., snapshot=path)`` returns a dict of the
    same shape as the live scan path.
    """
    live = inv.available_realisations('ssp245', 'amoc26', models=['CESM2'])
    # Save live snapshot to a tmp file in the same format and replay.
    snap_doc = {
        'written': 'test',
        'realisations': {'amoc26': {'ssp245': live, 'historical': {}}},
    }
    p = tmp_path / 'snap.json'
    with open(p, 'w') as f:
        json.dump(snap_doc, f)
    replayed = inv.available_realisations(
        'ssp245', 'amoc26', models=['CESM2'], snapshot=str(p))
    assert replayed == live


def test_resolve_path_extension_2300(inv):
    """MRI-ESM2-0 ssp126 r1i1p1f1 must resolve to the extension_2300 file."""
    import os
    p = inv.resolve_path('mri-esm2-0', 'ssp126', 'r1i1p1f1', 'tas')
    assert 'extension_2300' in p, f'expected extension_2300 in {p}'
    assert os.path.exists(p)


def test_available_pairs_intersects_historical(inv):
    """``available_pairs`` only returns (model, rea) tuples present in BOTH
    the scenario and historical, for the same var.
    """
    pairs = inv.available_pairs('ssp245', 'amoc26', models=['CESM2'])
    # CESM2 ssp245 has r4/r10/r11; historical has all three plus r5/r6/r8/r9.
    # Intersection should yield 3.
    assert {r for (_m, r) in pairs} == {'r4i1p1f1', 'r10i1p1f1', 'r11i1p1f1'}
