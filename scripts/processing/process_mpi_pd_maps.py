"""Build the shareable NetCDF behind FigSupp_mpi_pd_maps.

Ingredients only (dT_future, w_future, slope, slope_se, cmip6_w_range,
land_mask) from data/reg_ds_mpi_tas.nc; the derived fields are computed in
FigSupp_mpi_pd_maps.make_figure, which needs nothing but this file. Rerun
whenever reg_ds_mpi_tas.nc changes.

Run from scripts/, interactively or as `python processing/process_mpi_pd_maps.py`.
"""

# %% imports
import os
import sys
import pathlib
import numpy as np
import xarray as xr

if '__file__' in globals():  # standalone run: ensure scripts/ is importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import functions

NC_PATH = '../data/mpi_esm_europe_amoc_pd_maps.nc'


# %% build
def build_dataset(reg_ds_mpi, masks, path=NC_PATH):
    """Write the ingredients for the maps from reg_ds_mpi_tas.nc to `path`."""
    r = reg_ds_mpi.rename(scenar='scenario')
    amoc_pi = functions.AMOC_pi_MPI
    k = -amoc_pi / 100.0  # K/Sv -> K per % weakening (weakening = -100·ΔAMOC/AMOC_pi)
    ext = functions.get_amoc_extent()
    land = (masks['MPI-ESM1-2-LR']['LAND'].sel(lat=r.lat, lon=r.lon) == 0).astype('int8')
    ds = xr.Dataset(
        dict(
            dT_future=(r.T_future - r.T_pd_245).transpose('season', 'scenario', 'lat', 'lon'),
            w_future=functions.convert_strength_to_weakening(r.AMOC_future.sel(season='', drop=True)),
            slope=(r.coef_ensmean * k),
            slope_se=(r.ste_ensmean * abs(k)),
            cmip6_w_range=(('scenario', 'bound'), np.array([sorted(ext[s]) for s in r.scenario.values])),
            land_mask=land,
        ),
        coords=dict(bound=['lo', 'hi']))
    ds['season'] = ['annual', 'djf', 'jja']
    for v, attrs in dict(
        dT_future=dict(units='K', long_name='Temperature change 2091-2100 minus 1991-2020 (ensemble mean)',
                       comment='positive = warmer than 1991-2020'),
        w_future=dict(units='%', long_name='Projected AMOC weakening 2091-2100 at 26.5N, % of the 1850-1899 AMOC strength'),
        slope=dict(units='K per %', long_name='Temperature change per percentage point of additional AMOC weakening',
                   comment='negative = cooling as the AMOC weakens; from hosing-experiment regression'),
        slope_se=dict(units='K per %', long_name='Standard error of slope'),
        cmip6_w_range=dict(units='%', long_name='CMIP6 multi-model range of AMOC weakening 2091-2100, % of 1850-1899'),
        land_mask=dict(long_name='Land mask', comment='1 = land, 0 = ocean'),
    ).items():
        ds[v].attrs = attrs
    ds.lat.attrs, ds.lon.attrs = dict(units='degrees_north'), dict(units='degrees_east')
    ds.attrs = dict(
        title='European temperature change under global warming and AMOC weakening, MPI-ESM1.2-LR, w.r.t. 1991-2020',
        model='MPI-ESM1.2-LR (native grid, European subset)',
        future_window='2091-2100',
        present_day_reference='1991-2020 (historical 1991-2014 + SSP2-4.5 2015-2020)',
        amoc_reference=f'1850-1899 historical ensemble mean, {amoc_pi:.3f} Sv',
        formula_dT_at_w='dT(w) = dT_future + slope * (w - w_future)   [w in % weakening of the 1850-1899 AMOC]',
        formula_w_net_cooling='w_net_cooling = w_future - dT_future / slope, defined only where slope < 0 (NaN elsewhere): '
                              'the AMOC weakening at which end-of-century temperature returns to its 1991-2020 level',
        usage='Any weakening level w can be computed from dT_future, slope and w_future with formula_dT_at_w.',
        scenario_labels='ssp126: SSP1-2.6, ssp245: SSP2-4.5, ssp370: SSP3-7.0',
        references='Alastrué de Asenjo & Schaumann (2026). AMOC weakening unlikely to cool large parts of Europe in a warming climate.',
        contact='fschaumann@ethz.ch',
    )
    ds.to_netcdf(path)
    return ds


# %% run
if __name__ == '__main__':
    _, masks = functions.get_full_multi_model_dict()
    ds = build_dataset(functions.load_regression_ds_mpi(), masks)
    print(ds)
