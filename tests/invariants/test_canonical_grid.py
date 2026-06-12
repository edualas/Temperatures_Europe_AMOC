"""Tier 2 — canonical-grid invariants for ``cmip6_inventory.open_canonical``.

Guards the project's "one (lat, lon) per model" convention. Three
classes of assertion:

1. ``open_canonical`` actually snaps loaded files to the canonical grid
   (bit-identical lat/lon coords).
2. The drift-detection assert fires when a synthetic file genuinely
   doesn't match the canonical grid (different resolution).
3. Across all reas on disk for a given (model, scenario, var), every
   opened file passes the snap — i.e. no rea has lat/lon further than
   the configured tolerance from canonical. Catches the original
   EC-Earth3 Group A vs Group B problem at the test level.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


@pytest.fixture(scope='module')
def inv(scripts_dir):
    sys.path.insert(0, str(scripts_dir))
    import cmip6_inventory
    return cmip6_inventory


def test_canonical_grid_loads_for_every_paper_model(inv):
    """Every entry in CANONICAL_GRID_SOURCE must produce a (lat, lon)
    pair without raising. Confirms the chosen source files exist on disk."""
    for model in inv.CANONICAL_GRID_SOURCE:
        lat, lon = inv.canonical_grid(model)
        assert lat.ndim == 1 and lat.size > 1, f'{model}: bad lat'
        assert lon.ndim == 1 and lon.size > 1, f'{model}: bad lon'


def test_open_canonical_snaps_to_grid(inv):
    """A representative SSP file, opened via open_canonical, has lat/lon
    bit-identical to the canonical grid (regardless of whether the file's
    own coords would have been slightly drifted)."""
    # EC-Earth3 ssp245 r2 is the canonical test case — its native file
    # uses Group B encoding (-89.46282), canonical is Group A
    # (-89.46282156...) from r1 historical. The snap should re-stamp.
    path = ('/work/uo1075/m300817/teu_amoc/data/CMIP6/ssp245/ec-earth3/'
            'ec-earth3_r2i1p1f1_tas_yr.nc')
    if not Path(path).exists():
        pytest.skip(f'fixture file missing: {path}')
    can_lat, can_lon = inv.canonical_grid('EC-Earth3')
    da = inv.open_canonical(path, 'EC-Earth3')
    assert np.array_equal(da.lat.values, can_lat)
    assert np.array_equal(da.lon.values, can_lon)


def test_open_canonical_raises_on_real_regrid(inv, tmp_path):
    """A synthetic netCDF on a clearly-different grid (half the canonical
    lat resolution) must raise — the drift-detection threshold (1e-3°)
    is well below any real grid spacing."""
    can_lat, can_lon = inv.canonical_grid('EC-Earth3')
    # Half-resolution grid: keep every other lat/lon
    halved_lat = can_lat[::2].astype('float64')
    halved_lon = can_lon[::2].astype('float64')
    da = xr.DataArray(
        np.zeros((halved_lat.size, halved_lon.size)),
        dims=('lat', 'lon'),
        coords={'lat': halved_lat, 'lon': halved_lon},
        name='tas',
    )
    p = tmp_path / 'fake_half_res.nc'
    da.to_netcdf(p)
    with pytest.raises(ValueError, match='[Gg]rid.*mismatch'):
        inv.open_canonical(str(p), 'EC-Earth3')


@pytest.mark.parametrize('model,scenario,var', [
    ('EC-Earth3', 'historical', 'amoc26'),  # the Group A/B problem case
    ('EC-Earth3', 'ssp245', 'tas'),
    ('CESM2',     'historical', 'tas'),
    ('CanESM5',   'ssp370', 'tas'),
    ('GISS-E2-1-G', 'historical', 'tas'),
])
def test_all_reas_within_canonical_tolerance(inv, model, scenario, var):
    """For every rea on disk for (model, scenario, var), opening via
    open_canonical must succeed. Catches the kind of cross-rea encoding
    drift that surfaced with EC-Earth3 historical — without the snap,
    xr.concat over realiz unions the lat dim."""
    reas = inv.available_realisations(scenario, var, models=[model])[model]
    if not reas:
        pytest.skip(f'no reas on disk for {model}/{scenario}/{var}')
    can_lat, can_lon = inv.canonical_grid(model)
    failures = []
    for rea in reas:
        try:
            path = inv.resolve_path(model, scenario, rea, var)
            da = inv.open_canonical(path, model)
            # AMOC files (1D time series at 26.5N) skip the lat/lon snap;
            # only check 2D-gridded files.
            if 'lat' in da.dims and 'lon' in da.dims:
                assert np.array_equal(da.lat.values, can_lat), (
                    f'{model}/{scenario}/{rea}: lat not snapped')
                assert np.array_equal(da.lon.values, can_lon), (
                    f'{model}/{scenario}/{rea}: lon not snapped')
        except Exception as e:
            failures.append(f'{rea}: {e}')
    assert not failures, (
        f'{model}/{scenario}/{var}: {len(failures)} reas failed canonical '
        f'snap; first 3: {failures[:3]}')
