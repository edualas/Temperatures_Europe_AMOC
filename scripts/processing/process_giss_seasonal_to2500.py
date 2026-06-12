"""Stage seasonal (DJF + JJA) yearly tas for GISS-E2-1-G ssp245 to 2500.

Sibling of ``process_seasonal_tas.py`` (which handled HosMIP + GISS *historical*
seasonal tas). Source is the per-member monthly chunks at
``/work/bu1431/T_EU_AMOC/CMIP6/giss-e2-1-g/ssp245/{member}/tas_Amon_...nc``
(downloaded by the archived ``_archive/process_giss_long_runs.py``).

Outputs land at
``/work/bu1431/T_EU_AMOC/CMIP6/giss-e2-1-g/processed/
giss-e2-1-g_{member}_tas_{djf,jja}_to2500.nc``
next to the existing annual ``_tas_yr_to2500.nc`` files. These feed the
seasonal branch of ``functions.giss_load_or_compute_gridded`` /
``functions.get_giss_reg_ds`` once the GISS pipeline is made season-aware.

Run interactively cell-by-cell, or as a standalone script.
"""

# %% imports + config ---------------------------------------------------------
import os
from glob import glob

import xarray as xr

import process_seasonal_tas as pst

MEMBERS = [f'r{i}i1p1f2' for i in range(1, 11)]
SSP_ROOT = '/work/bu1431/T_EU_AMOC/CMIP6/giss-e2-1-g/ssp245'
OUT_ROOT = '/work/bu1431/T_EU_AMOC/CMIP6/giss-e2-1-g/processed'
SSP_START_YEAR = 2015


def _open_monthly(member):
    """Open the 10 monthly chunks (2015-2500) for one GISS r{member} ssp245.

    Files: tas_Amon_GISS-E2-1-G_ssp245_{member}_gn_{YYYYMM-YYYYMM}.nc
    parallel=False: with uneven chunk sizes, open_mfdataset(parallel=True)
    can silently drop the last chunk.
    """
    pattern = f'{SSP_ROOT}/{member}/tas_Amon_GISS-E2-1-G_ssp245_{member}_gn_*.nc'
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f'No monthly tas chunks for {member} at {pattern}')
    return xr.open_mfdataset(
        files, parallel=False, use_cftime=True,
        data_vars='minimal', coords='minimal', compat='override')


def write_seasonal_to2500(member, season, overwrite=False):
    """Write giss-e2-1-g_{member}_tas_{season}_to2500.nc using pst.seasonalize."""
    out_path = f'{OUT_ROOT}/giss-e2-1-g_{member}_tas_{season}_to2500.nc'
    if os.path.exists(out_path) and not overwrite:
        print(f'  skip (exists): {out_path}')
        return
    monthly = _open_monthly(member)
    seasonal = pst.seasonalize(monthly['tas'], season,
                               start_year=SSP_START_YEAR)
    expected = monthly.time.dt.year.max().item() - SSP_START_YEAR + 1
    # `seasonalize` keeps a leading partial-year row for DJF (Jan-Feb of the
    # start year, from the QS-DEC chunk centred on the prior Dec) and drops
    # the trailing partial-year row (Dec of end_year only). Net: DJF size is
    # `expected` rows (485 full DJFs + 1 leading partial = 486 rows for
    # 2015-2500). JJA has no partial-year rows → exactly `expected`. Match
    # the convention the existing historical seasonal files already use.
    assert seasonal.sizes['time'] == expected, (
        f'{member} {season}: got {seasonal.sizes["time"]}, expected {expected}')
    seasonal = seasonal.to_dataset(name='tas').load()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seasonal.to_netcdf(out_path)
    print(f'  wrote: {out_path}  (time={seasonal.sizes["time"]})')


# %% driver -------------------------------------------------------------------
if __name__ == '__main__':
    for member in MEMBERS:
        print(f'\n=== {member} ===')
        for season in ('djf', 'jja'):
            write_seasonal_to2500(member, season, overwrite=False)
    print('\nAll members done.')

# %%
