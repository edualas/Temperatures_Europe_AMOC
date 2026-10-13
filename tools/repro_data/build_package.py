"""Build the reproducibility data package (data/package/) from the Levante sources.

Run on Levante from scripts/ (needs the raw MPI-ESM and GISS fields and the cached inputs):
    python ../tools/repro_data/build_package.py [mpi nahosmip giss cesm2_boot other_studies cmip]
No argument builds every source. Each source is one netCDF file with groups. Fields are cut to the
European window the pipeline already uses for that model (ocean included) and stored as float32;
derived caches keep their dtype and carry the pipeline filename they are unpacked to in package mode
(see unpack_package in src/teu_functions.py).
"""
import os, sys, json, pickle, glob, subprocess
sys.path.insert(0, os.getcwd())
import numpy as np
import xarray as xr
import functions

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'package')) + '/'
os.makedirs(OUT, exist_ok=True)
DATA = functions.local_path
try:
    COMMIT = subprocess.run(['/sw/spack-levante/git-2.43.7-2ofazl/bin/git', 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
except FileNotFoundError:
    COMMIT = 'unknown'
SEASON_GROUPS = {'annual': '', 'djf': 'djf', 'jja': 'jja'}
PROJ_VARS = ['AMOC_pi', 'AMOC_future', 'T_pi', 'T_pd_245', 'T_future']


def put(path, group, obj, f32=True, **attrs):
    """Write one Dataset/DataArray as a netCDF group (zlib; floats to float32 unless f32=False)."""
    ds = obj.to_dataset(name=obj.name or 'data') if isinstance(obj, xr.DataArray) else obj.copy()
    for v in list(ds.data_vars):
        if ds[v].dtype == bool:
            ds[v] = ds[v].astype('int8').assign_attrs(bool_mask=1)
        elif f32 and ds[v].dtype == np.float64:
            ds[v] = ds[v].astype('float32')
    ds.attrs.update(attrs)
    for a in [ds.attrs] + [ds[v].attrs for v in ds.variables]:   # netCDF attrs: no bool, None or dict
        for k, val in list(a.items()):
            a[k] = int(val) if isinstance(val, (bool, np.bool_)) else val if isinstance(val, (str, int, float, np.number, list, np.ndarray)) else str(val)
    enc = {v: {'zlib': True, 'complevel': 4} for v in ds.data_vars if ds[v].dtype != object}
    ds.to_netcdf(path, group=group, mode='a' if os.path.exists(path) else 'w', encoding=enc)


def pack_nc(path, group, filename, ds=None, f32=False):
    put(path, group, xr.open_dataset(DATA + filename).load() if ds is None else ds, f32=f32, filename=filename, kind='nc')


def pack_json(path, group, filename):
    with open(DATA + filename) as fh:
        put(path, group, xr.Dataset(), filename=filename, kind='json', text=fh.read())


def pack_pkl(path, group, filename, obj=None, f32=False):
    """A pickled dict of Datasets (or of dicts of DataArrays) as one group with a subgroup per key."""
    if obj is None:
        with open(DATA + filename, 'rb') as fh:
            obj = pickle.load(fh)
    put(path, group, xr.Dataset(), filename=filename, kind='pkl')
    for i, (key, val) in enumerate(obj.items()):
        if isinstance(val, dict):
            put(path, f'{group}/k{i}', xr.Dataset({r: strip(m) for r, m in val.items()}), f32=f32, key=key, container='vars')
        else:
            put(path, f'{group}/k{i}', val, f32=f32, key=key)


def strip(da):
    """DataArray without non-dimension coords (scalar labels like season= that would clash in one Dataset)."""
    return da.drop_vars([c for c in da.coords if c not in da.dims])


def root_attrs(path, **attrs):
    xr.Dataset(attrs={'built_by': 'tools/repro_data/build_package.py', 'git_commit': COMMIT, **attrs}).to_netcdf(path, mode='a')


def fresh(name):
    if os.path.exists(OUT + name):
        os.remove(OUT + name)
    return OUT + name


def region_series(tas, mask, regions):
    """Area-weighted land mean per region (mask: Dataset/DataArray with a region dim incl. 'LAND'), as in the plots."""
    land = mask.sel(region='LAND')
    return xr.concat([functions.weighted_area_lat(tas.where(mask.sel(region=r)).where(land == 0)).mean(('lat', 'lon'))
                      for r in regions], dim=xr.DataArray(regions, dims='region', name='region'))


# MPI-ESM1.2-LR: grand ensemble (ensemble mean + member regional-mean series) and hosing runs
def build_mpi():
    data_dict = functions.load_mpi_esm_data(eur_only=True)
    masks = data_dict['masks']
    like = data_dict['']['tas']['his']
    land = masks['LAND']
    regions = [r for r in masks if r != 'LAND']

    path = fresh('mpi_ge.nc')
    for grp, s in SEASON_GROUPS.items():
        for exp in ('his', 'ssp'):
            ds = data_dict[s]['tas'][exp]
            if 'scenar' in ds.dims:
                ds = ds.sel(scenar=functions.ssps)
            put(path, f'{grp}/tas_{exp}', ds.mean('realiz'))
            # member spread of the regional land mean (Fig. 1 bands); same reduction as simulations_plot
            rm = xr.concat([functions.weighted_area_lat(ds.where(masks[r]).where(land == 0)).mean(('lat', 'lon'))
                            for r in regions], dim=xr.DataArray(regions, dims='region', name='region'))
            put(path, f'{grp}/regmean_{exp}', rm, f32=False)
    for exp in ('his', 'ssp'):
        a = data_dict['']['amoc'][exp]
        put(path, f'amoc_{exp}', a.sel(scenar=functions.ssps) if 'scenar' in a.dims else a, f32=False)
    put(path, 'masks', xr.Dataset({r: strip(m).sel(lat=like.lat, lon=like.lon) for r, m in masks.items()}))
    pack_nc(path, 'pd_maps', 'mpi_esm_europe_amoc_pd_maps.nc')
    root_attrs(path, title='MPI-ESM1.2-LR grand ensemble, EU window',
               content='tas ensemble means (historical 1850-2014; SSP1-2.6, SSP2-4.5, SSP3-7.0 2015-2100; annual, DJF, JJA); '
                       'regmean_*: regional land-mean tas per member; amoc_*: AMOC at 26N per member (Sv); region masks; '
                       'pd_maps: present-day maps of FigSupp_mpi_pd_maps')

    path = fresh('mpi_hosing.nc')
    keep = [h for h in data_dict['']['amoc']['ssphos'].hosing.values if h != '1']   # +1 Sv run (SSP2-4.5 only) not shipped
    cut = lambda ds: ds.sel(hosing=keep, time=slice(None, '2115'))   # to 2115: Fig. 1's centred rolling means (up to 30 yr) reach past 2100
    for grp, s in SEASON_GROUPS.items():
        put(path, f'{grp}/tas', cut(data_dict[s]['tas']['ssphos']))
    put(path, 'amoc', cut(data_dict['']['amoc']['ssphos']), f32=False)
    root_attrs(path, title='MPI-ESM1.2-LR SSP hosing experiments, EU window',
               content='tas (annual, DJF, JJA) and AMOC at 26N per member, 2015-2115, the hosing experiments of Fig. 1')


# NAHosMIP: preindustrial control/hosing fields per model, region masks, CMIP6 projection side of the regressions
def build_nahosmip():
    mmd, masks = functions.get_full_multi_model_dict()
    with open(DATA + 'hosmip_reg_ds_dict.pkl', 'rb') as fh:
        hosmip_reg = pickle.load(fh)
    path = fresh('nahosmip.nc')
    nah, nah_masks = {}, {}
    for m in functions.hosmip_labels:
        ds = mmd[m]
        if m == 'CESM2':    # Boot et al. (2024) SSP runs are merged in from cesm2_boot.nc
            ds = ds.sel(scenar=['pi'])
        if m == 'EC-Earth3':  # Bellomo & Mehling (2024) difference fields are merged in from other_studies.nc
            assert int(np.isfinite(ds.tas.sel(scenar='ghg', type=['control', 'hosing'])).sum()) == 0
            ds = ds.sel(scenar=['pi'], type=['control', 'hosing'])
        nah[m] = ds
        nah_masks[m] = {r: strip(a).sel(lat=ds.lat, lon=ds.lon) for r, a in masks[m].items()}
        put(path, f'projection/k{len(nah) - 1}', hosmip_reg[m][PROJ_VARS], f32=False, key=m)
    pack_pkl(path, 'multi_model_dict', 'multi_model_dict.pkl', nah, f32=True)
    pack_pkl(path, 'masks', 'hosmip_masks.pkl', nah_masks)
    root_attrs(path, title='NAHosMIP preindustrial hosing experiments, EU window',
               content='multi_model_dict: control and hosing tas fields and AMOC per model (Jackson et al. 2023); masks: region '
                       'masks per model grid; projection/: CMIP6 ensemble-mean AMOC_pi, AMOC_future, T_pi, T_pd_245, T_future '
                       'per model (1850-1899, 1991-2020, 2091-2100)')


# GISS-E2-1-G: 10 members historical + SSP2-4.5 to 2500
def build_giss():
    reg = xr.open_dataset(DATA + 'reg_ds_giss.nc').load()
    box = lambda da: da.sel(lat=slice(float(reg.lat.min()), float(reg.lat.max())))\
        .where(lambda x: ((x.lon + 180) % 360 - 180 >= float(reg.lon.min())) & ((x.lon + 180) % 360 - 180 <= float(reg.lon.max())), drop=True)
    proc = functions.GISS_PROCESSED_PATH
    path = fresh('giss.nc')
    for grp, s in SEASON_GROUPS.items():
        amoc, tas = functions.load_giss_member_amoc_tas(season=s)
        if s == '':
            put(path, 'amoc', amoc.rename('amoc'), f32=False)
        put(path, f'{grp}/tas', box(tas).rename('tas'))
        name = f"giss-e2-1-g_tas_pi{'_' + s if s else ''}_climatology.nc"
        pack_nc(path, f'tas_pi_{grp}', 'giss_processed/' + name, box(xr.open_dataarray(proc + name).load()).to_dataset())
    with open(proc + 'giss-e2-1-g_masks.pkl', 'rb') as fh:
        gm = pickle.load(fh)
    put(path, 'masks', xr.Dataset({r: box(strip(a)) for r, a in gm.items()}), filename='giss_processed/giss-e2-1-g_masks.pkl', kind='pkl_vars')
    with open(proc + 'constants.json') as fh:
        put(path, 'constants', xr.Dataset(), filename='giss_processed/constants.json', kind='json', text=fh.read())
    put(path, 'projection', reg[PROJ_VARS], f32=False)
    root_attrs(path, title='GISS-E2-1-G SSP2-4.5 to 2500 (Romanou et al. 2023), EU window',
               content='tas per member (annual, DJF, JJA; historical 1850-2014 + SSP2-4.5 2015-2500, native longitudes), AMOC at '
                       '26N per member, preindustrial climatologies, region masks, constants; projection: CMIP6 ensemble-mean '
                       'AMOC_pi, AMOC_future, T_pi, T_pd_245, T_future')


# Boot et al. (2024) CESM2: regional land-mean series only (fields not public); CESM2 regression caches
def build_cesm2_boot():
    mmd, _ = functions.get_full_multi_model_dict()
    cesm = mmd['CESM2'].sel(scenar=['ssp126', 'ssp585'])
    cesm = cesm.sel(time=cesm.amoc.notnull().any(['type', 'season', 'scenar']))   # Boot years only (the merge padded them to the NAHosMIP axis)
    regions = [r for r in cesm.region.values if r != 'LAND']
    path = fresh('cesm2_boot.nc')
    boot = xr.Dataset({'tas_reg': region_series(cesm.tas, cesm.mask, regions), 'amoc': cesm.amoc})
    put(path, 'boot', boot, f32=False)
    for f in ['reg_ds_cesm.nc', 'linearity_diagnostics_cesm_tas.nc', 'scenario_independence_diagnostics_cesm_tas.nc'] \
            + sorted(os.path.basename(p) for p in glob.glob(DATA + 'reg_ds_cesm_fw-*.nc')):
        pack_nc(path, f.replace('.nc', ''), f)
    root_attrs(path, title='CESM2 SSP hosing experiments (Boot et al. 2024), derived',
               content='boot: area-weighted regional land-mean tas (tas_reg; annual, DJF, JJA) and AMOC at 26N for the control '
                       'and hosing runs under SSP1-2.6 and SSP5-8.5; CESM2 regression and diagnostic caches')


# Other studies: differences the figures use, no copies of the source files
def build_other_studies():
    _, masks = functions.get_full_multi_model_dict()
    bm, vwb, _, liu = functions.get_other_studies_data(masks)
    path = fresh('other_studies.nc')
    put(path, 'bellomo', bm, f32=False, source='Bellomo & Mehling (2024), plotting data of their Fig. 4')
    sel = lambda ds, t, sc: ds.sel(type=t, scenar=sc)
    dd = xr.Dataset({f'{v}_dd': (sel(vwb[v], 'hosing', 'ghg') - sel(vwb[v], 'hosing', 'pi'))
                                - (sel(vwb[v], 'control', 'ghg') - sel(vwb[v], 'control', 'pi')) for v in ('tas', 'tas_lo', 'tas_hi') if v in vwb})
    put(path, 'vanwesten', xr.merge([dd, vwb[['amoc', 'mask']]]), f32=False,
        source='van Westen & Baatsen (2025): hosing-minus-control difference of the RCP4.5 minus preindustrial difference, '
               'per RCP4.5 window (vwb_window)')
    diff = (liu.tas.sel(type='control', scenar='ghg') - liu.tas.sel(type='hosing', scenar='ghg')).rename('tas_diff')
    put(path, 'liu', xr.merge([diff, liu[['amoc', 'mask']]]), f32=False,
        source='Liu et al. (2020) CCSM4: control minus AMOC-stabilised RCP8.5 climatology')
    bm2 = xr.open_dataset(functions.local_path + 'Bellomo2024_plotting_data/data_figs_2_S3.nc').load()
    w = (bm2.tas_4x - bm2.tas_pi).rename('warming')
    w = w.where((w.lat >= 25) & (w.lat <= 80) & (((w.lon + 180) % 360 - 180 >= -50) & ((w.lon + 180) % 360 - 180 <= 60)), drop=True)
    pack_nc(path, 'bellomo_warming', 'Bellomo2024_plotting_data/data_figs_2_S3_warming.nc', w.to_dataset())
    root_attrs(path, title='Results of other studies used in the figures (derived, EU window)',
               content='bellomo: EC-Earth3 hosing and 4xCO2 tas differences and AMOC changes; vanwesten: double-difference tas '
                       '(central, low, high) and AMOC; liu: tas difference and AMOC; bellomo_warming: EC-Earth3 4xCO2 warming')


# CMIP6 projection ensemble: caches only
def build_cmip():
    path = fresh('cmip.nc')
    files = sorted(os.path.basename(p) for p in glob.glob(DATA + 'cmip_range_fig3*.nc') + glob.glob(DATA + 'cmip_synth_wt_*.nc')) \
        + ['cmip_cooling.nc'] + sorted(os.path.basename(p) for p in glob.glob(DATA + 'reg_ds_mpi_tas_fw-*.nc') + glob.glob(DATA + 'reg_ds_giss_fw-*.nc'))
    for i, f in enumerate(files):
        pack_nc(path, f'nc{i}', f)
    for i, f in enumerate(sorted(os.path.basename(p) for p in glob.glob(DATA + 'cmip_range_fig3*_fit_stats.json') + glob.glob(DATA + 'cmip6_amoc_extent*.json'))):
        pack_json(path, f'json{i}', f)
    pack_pkl(path, 'gwl', 'gwl_diagnostic.pkl')
    for i, f in enumerate(sorted(os.path.basename(p) for p in glob.glob(DATA + 'hosmip_reg_ds_dict_fw-*.pkl'))):
        pack_pkl(path, f'hosmip_fw{i}', f)
    root_attrs(path, title='CMIP6 projection ensemble, derived caches',
               content='synthetic CMIP6 range variants and fit statistics, (w, t) targets, CMIP-projected cooling, AMOC extent, '
                       'global-warming-level diagnostic, decade-matched regression caches of FigSupp_cmip_cooling')


BUILDERS = {'mpi': build_mpi, 'nahosmip': build_nahosmip, 'giss': build_giss, 'cesm2_boot': build_cesm2_boot,
            'other_studies': build_other_studies, 'cmip': build_cmip}

if __name__ == '__main__':
    for name in sys.argv[1:] or list(BUILDERS):
        print(f'=== building {name}', flush=True)
        BUILDERS[name]()
    for f in sorted(os.listdir(OUT)):
        print(f'{f}: {os.path.getsize(OUT + f) / 1e6:.1f} MB')
