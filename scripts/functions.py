# region PACKAGES

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle
from matplotlib import ticker
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
plt.rcParams.update({'font.size': 12})
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.geometry import LineString, Polygon
from shapely.ops import split, unary_union
import xarray as xr
import regionmask
import statsmodels.api as sm
from scipy import stats as scipy_stats
import pickle
import colorsys

# endregion

########################################################################################################################
# GLOBAL DEFINITIONS
########################################################################################################################

# region GLOBAL DEFINITIONS

global_plot_bg = 'white'

data_path = '/work/uo1075/m300817/teu_amoc/data/'
local_path = '/home/m/m300940/teu_amoc/data/'

# define CMIP ranges of AMOC weakening in % (first value: model realisation with highest weakening; second value: model realisation with lowest weakening)
# these values refer to the models that are documented in the .json file of the respective /work/uo1075/m300817/teu_amoc/data/CMIP6/{ssp}/ folder
# which are the ones that are being loaded in the CMIP6 section of analysis_teu_amoc.ipynb.
    # end_weakenings = {}
    # for i, sce in enumerate(["ssp126", "ssp245", "ssp370"]):
    #     countmod, countsim = 0,0 # not include MPI-ESM, set to 1,50 otherwise
    #     end_weakenings[sce] = []

    #     for model in sce_amoc_cmip6[sce]:
    #         if model in historical_amoc_cmip6.keys():
    #             countmod = countmod+1
    #             model_sim_weakening = []
    #             for rea in sce_amoc_cmip6[sce][model]:
    #                 if rea in historical_amoc_cmip6[model].keys():
    #                     initial = historical_amoc_cmip6[model][rea].amoc.isel(time=slice(0,50)).mean("time")
    #                     weakening = 100*(sce_amoc_cmip6[sce][model][rea].isel(time=slice(-10, None)).mean(dim='time').amoc-initial)/initial
    #                     model_sim_weakening.append(weakening.values)
    #                     end_weakenings[sce].append(weakening.values)
    #             # print(f"{sce} - {model}: {np.mean(np.array(model_sim_weakening))}")
    #     print(f'{countmod} models for {sce}: min/max weakening at end of century: {np.min(-np.array(end_weakenings[sce]))}/{np.max(-np.array(end_weakenings[sce]))} %')
AMOC_extent = {}
AMOC_extent['ssp126'] = (49.0, 0.8) # (49.0, 10.1) for only HosMIP models  # (48.6, 0.8) for only json models
AMOC_extent['ssp245'] = (51.3, 13.6) # (49.1, 18.7) for only HosMIP models  # (51.3, 13.6) for only json models
AMOC_extent['ssp370'] = (53.7, 17.9) # (48.1, 24.2) for only HosMIP models  # (53.7, 17.9) for only json models
# hosmip values arrived at through:
# end_weakenings = {}
# for ssp in ssps:
#     end_weakenings[ssp] = []
#     for model in functions.hosmip_labels:
#         strength = float(cmip6_ctrl_data[model].isel(time=slice(-10, None)).mean(dim='time').amoc.sel(scenar=ssp).values)
#         pi_strength = float(multi_model_dict[model].amoc.sel(season='', scenar='pi', type='control').isel(time=0).values)
#         end_weakenings[ssp].append((pi_strength-strength)/pi_strength*100)


AMOC_pi_MPI = 19.0154

ssps = ['ssp126', 'ssp245', 'ssp370']

# regions = ['RU', 'NO', 'FR', 'SE', 'BY', 'UA', 'PL', 'AT', 'HU', 'MD', 'RO', 'LT', 'LV', 'EE', 'DE', 'BG', 'GR', 'TR', 'HR', 'CH', 'BE', 'NL', 'PT', 'ES', 'IE', 'IT', 'DK', 'GB', 'SI', 'FI', 'SK', 'CZ', 'MK', 'RS', 'XK', 'IS']

# these are sorted by effect size in range plot for both MPI-ESM and CESM2 data
regions = ['IE', 'GB', 'IS', 'NO', 'NL', 'DK', 'SE', 'BE', 'PT', 'FI', 'EE', 'FR', 'LT', 'LV', 'DE', 'LU', 'RU', 'PL', 'BY', 'ES', 'CZ', 'UA', 'MD', 'SK', 'CH', 'AT', 'RO', 'GR', 'BG', 'HU', 'AL', 'TR', 'HR', 'ME', 'RS', 'IT', 'XK', 'SI', 'BA', 'MK', 'CY']

ssp_labels = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5"
}

hosing_colors = {
    'ssp126': {
        # Exact base RGBA value:
        'ge': np.array([23/255, 60/255, 102/255]) * 2.5 if global_plot_bg == 'black' else (23/255, 60/255, 102/255),
        # Slightly darker blues for negative hosing:
        'neg01': 'midnightblue',
        # Slightly lighter / more saturated blues for positive hosing:
        '01': 'steelblue',
        '03': 'cornflowerblue',
        '05': 'deepskyblue',
        '1': 'green',
        # For linear hosing, shift a bit toward greenish tones:
        'linneg02': 'teal',
        'lin02': 'cadetblue',
        'lin06': 'mediumaquamarine',
        'lin10': 'aquamarine'
    },
    'ssp245': {
        'ge': (247/255, 148/255, 32/255),
        'neg01': 'darkorange',
        '01': 'gold',
        '03': 'yellow',
        '05': 'navajowhite',
        '1': 'orange',
        'linneg02': 'olive',
        'lin02': 'darkkhaki',
        'lin06': 'khaki',
        'lin10': 'palegoldenrod'
    },
    'ssp370': {
        'ge': (231/255, 29/255, 37/255),
        'neg01': 'darkred',
        '01': 'orangered',
        '03': 'salmon',
        '05': 'lightsalmon',
        '1': 'green',
        'linneg02': 'mediumorchid',
        'lin02': 'hotpink',
        'lin06': 'plum',
        'lin10': 'pink'
    },
    'ssp585': {
        'ge': (149/255, 27/255, 30/255),
    }
}

hosing_names = dict(zip(['ssp126', 'ssp245', 'ssp370'], 
                        [dict(zip(['ge', 'neg01', '01', '03', '05', '1', 'linneg02', 'lin02', 'lin06', 'lin10'], 
                                  ['MPI-ESM GE SSP1-2.6', 'SSP1-2.6  -0.1Sv', 'SSP1-2.6 +0.1Sv', 'SSP1-2.6 +0.3Sv', 'SSP1-2.6 +0.5Sv', 'SSP1-2.6 +1.0Sv', 'SSP1-2.6 lin. -0.2Sv', 'SSP1-2.6 lin.+0.2Sv', 'SSP1-2.6 lin.+0.6Sv', 'SSP1-2.6 lin.+1.0Sv'])), 
                         dict(zip(['ge', 'neg01', '01', '03', '05', '1', 'linneg02', 'lin02', 'lin06', 'lin10'], 
                                  ['MPI-ESM GE SSP2-4.5', 'SSP2-4.5  -0.1Sv', 'SSP2-4.5 +0.1Sv', 'SSP2-4.5 +0.3Sv', 'SSP2-4.5 +0.5Sv', 'SSP2-4.5 +1.0Sv', 'SSP2-4.5 lin. -0.2Sv', 'SSP2-4.5 lin.+0.2Sv', 'SSP2-4.5 lin.+0.6Sv', 'SSP2-4.5 lin.+1.0Sv'])),
                         dict(zip(['ge', 'neg01', '01', '03', '05', 'linneg02', 'lin02', 'lin06', 'lin10'], 
                                  ['MPI-ESM GE SSP3-7.0', 'SSP3-7.0  -0.1Sv', 'SSP3-7.0 +0.1Sv', 'SSP3-7.0 +0.3Sv', 'SSP3-7.0 +0.5Sv', 'SSP3-7.0 lin. -0.2Sv', 'SSP3-7.0 lin.+0.2Sv', 'SSP3-7.0 lin.+0.6Sv', 'SSP3-7.0 lin.+1.0Sv'])),
                         ]))

hosing_names['ssp585'] = {}
hosing_names['ssp585']['ge'] = 'MPI-ESM GE SSP5-8.5'

hosing_symbols = {
    'ge': '●',
    '01': '▲',
    '03': '✕',
    '05': '★',
    '1': '◆',
    'neg01': '▼',
    'lin02': '▶',
    'lin06': '⬟',
    'lin10': '■',
    'linneg02': '◀',
}

hosing_markers = {
    'ge': 'o',
    '01': '^',
    '03': 'x',
    '05': '*',
    '1': 'D',
    'neg01': 'v',
    'lin02': '>',
    'lin06': 'p',
    'lin10': 's',
    'linneg02': '<',
}

hosmip_labels = ['CanESM5', 'EC-Earth3', 'CESM2', 'IPSL-CM6A-LR', 'HadGEM3-GC3-1MM', 'HadGEM3-GC3-1LL', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR']

hosmip_colors = {
    'CanESM5': 'mediumseagreen', #'mediumturquoise',
    'EC-Earth3': 'darkturquoise', #'cadetblue',
    'CESM2': 'steelblue',
    'IPSL-CM6A-LR': 'darkslateblue',
    'HadGEM3-GC3-1MM': 'mediumorchid',
    'HadGEM3-GC3-1LL': 'plum',
    'MPI-ESM1-2-HR': 'mediumvioletred',
    'MPI-ESM1-2-LR': 'brown' #'firebrick' #'crimson' 'goldenrod
}

# this is for loading seasonal CMIP data, where dimensions and variables differ between models
drop_stuff = {}
drop_stuff['MPI-ESM1-2-HR'] = {}
drop_stuff['MPI-ESM1-2-HR']['dim'] = ['bnds']
drop_stuff['MPI-ESM1-2-HR']['var'] = ['height']

drop_stuff['MPI-ESM1-2-LR'] = {}
drop_stuff['MPI-ESM1-2-LR']['dim'] = ['bnds']
drop_stuff['MPI-ESM1-2-LR']['var'] = ['height']

drop_stuff['CanESM5'] = {}
drop_stuff['CanESM5']['dim'] = ['bnds']
drop_stuff['CanESM5']['var'] = ['height']

drop_stuff['CESM2'] = {}
drop_stuff['CESM2']['dim'] = 'nbnd'

drop_stuff['IPSL-CM6A-LR'] = {}
drop_stuff['IPSL-CM6A-LR']['var'] = ['height']

drop_stuff['HadGEM3-GC3-1LL'] = {}
drop_stuff['HadGEM3-GC3-1LL']['dim'] = ['bnds']
drop_stuff['HadGEM3-GC3-1LL']['var'] = ['height']

drop_stuff['HadGEM3-GC3-1MM'] = {}
drop_stuff['HadGEM3-GC3-1MM']['dim'] = ['bnds']
drop_stuff['HadGEM3-GC3-1MM']['var'] = ['height']

drop_stuff['EC-Earth3'] = {}
drop_stuff['EC-Earth3']['dim'] = ['bnds']
drop_stuff['EC-Earth3']['var'] = ['height']

def get_hosmip_colors(plot_bg='white'):
    if plot_bg == 'black':
        # Lighter colors for dark background - blue to green spectrum
        return {
            'CanESM5': '#4A90E2',      # Bright blue
            'EC-Earth3': '#5BA3D4',    # Blue-cyan
            'CESM2': '#6BB6C6',        # Cyan
            'IPSL-CM6A-LR': '#7BC9B8', # Blue-green
            'HadGEM3-GC3-1MM': '#8BDCAA', # Green-cyan
            'HadGEM3-GC3-1LL': '#9BEF9C', # Light green
            'MPI-ESM1-2-HR': '#6DA9D4', # Medium blue-cyan
            'MPI-ESM1-2-LR': '#A8A8A8'  # Gray for contrast
        }
    else:  # white background
        # Darker colors for light background - blue to green spectrum
        return {
            'CanESM5': '#1E3A8A',      # Dark blue
            'EC-Earth3': 'darkturquoise',
            'CESM2': 'steelblue',
            'IPSL-CM6A-LR': '#1E9A6E', # Dark blue-green
            'HadGEM3-GC3-1MM': '#808080', #  gray for contrast '#1EBA5A', # Dark green-cyan
            'HadGEM3-GC3-1LL': '#2EDA46', # Dark green
            'MPI-ESM1-2-HR': '#1E7A82', # Dark cyan
            'MPI-ESM1-2-LR': 'blueviolet' #'brown' 
        }

# Update your existing code:
hosmip_colors = get_hosmip_colors()

country_names = {
    'RU': 'Russia',
    'NO': 'Norway', 
    'FR': 'France',
    'SE': 'Sweden',
    'BY': 'Belarus',
    'UA': 'Ukraine',
    'PL': 'Poland',
    'AT': 'Austria',
    'HU': 'Hungary',
    'MD': 'Moldova',
    'RO': 'Romania',
    'LT': 'Lithuania',
    'LV': 'Latvia',
    'EE': 'Estonia',
    'DE': 'Germany',
    'BG': 'Bulgaria',
    'GR': 'Greece',
    'TR': 'Turkey',
    'HR': 'Croatia',
    'CH': 'Switzerland',
    'BE': 'Belgium',
    'NL': 'Netherlands',
    'PT': 'Portugal',
    'ES': 'Spain',
    'IE': 'Ireland',
    'IT': 'Italy',
    'DK': 'Denmark',
    'GB': 'United Kingdom',
    'SI': 'Slovenia',
    'FI': 'Finland',
    'SK': 'Slovakia',
    'CZ': 'Czechia',
    'MK': 'North Macedonia',
    'RS': 'Serbia',
    'XK': 'Kosovo',
    'IS': 'Iceland',
    'BA': 'Bosnia and Herz.',
    'AL': 'Albania',
    'ME': 'Montenegro',
    'CY': 'Cyprus',
    'LU': 'Luxembourg',
}

country_codes = {
    'RU': 18,  # Russia
    'NO': 21,  # Norway
    'FR': 43,  # France
    'SE': 110, # Sweden
    'BY': 111, # Belarus
    'UA': 112, # Ukraine
    'PL': 113, # Poland
    'AT': 114, # Austria
    'HU': 115, # Hungary
    'MD': 116, # Moldova
    'RO': 117, # Romania
    'LT': 118, # Lithuania
    'LV': 119, # Latvia
    'EE': 120, # Estonia
    'DE': 121, # Germany
    'BG': 122, # Bulgaria
    'GR': 123, # Greece
    'TR': 124, # Turkey
    'HR': 126, # Croatia
    'CH': 127, # Switzerland
    'BE': 129, # Belgium
    'NL': 130, # Netherlands
    'PT': 131, # Portugal
    'ES': 132, # Spain
    'IE': 133, # Ireland
    'IT': 141, # Italy
    'DK': 142, # Denmark
    'GB': 143, # United Kingdom
    'SI': 150, # Slovenia
    'FI': 151, # Finland
    'SK': 152, # Slovakia
    'CZ': 153, # Czechia
    'RS': 172, # Serbia
    'XK': 174, # Kosovo
    'IS': 144, # Iceland
    'MK': 171, # North Macedonia
    # 'CN': 160, # N. Cyprus
    'ME': 173, # Montenegro
    'CY': 161, # Cyprus
    'BA': 171, # Bosnia and Herz.
    'AL': 125, # Albania
    'LU': 128, # Luxembourg
}

# endregion

########################################################################################################################
# HELPER FUNCTIONS
########################################################################################################################

def weighted_area_lat(ds):
    """
    Calculate the area-weighted temperature over its domain. 
    In a regular latitude/ longitude grid the grid cell area decreases towards the pole.
    We can use the cosine of the latitude as proxy for the grid cell area.
    IMPORTANT: after applying function, do .mean('lat') before 'lon'
    Taken from https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html
    
    Parameters:
    ds (xr.Dataset): Dataset containing latitude data.
    
    Returns:
    xr.Dataset: Dataset with weighted area.
    """
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"
    return ds.weighted(weights)

def define_region_masks(ds, min_overlap=0.0):
    """
    Define IPCC region masks for different parts of Europe. If no centroid falls into a country,
    fallback to an overlap-based mask.

    Parameters:
    ds (xr.Dataset): Dataset with grid information.
    min_overlap (float): Minimum overlap fraction (0.0 to 1.0) for a grid cell to be included in the fallback mask.

    Returns:
    dict: Dictionary containing masks for different regions of Europe.
    """
    from shapely.geometry import Polygon, Point
    from shapely.prepared import prep

    # Create the initial masks using centroids
    mask_eur = regionmask.defined_regions.ar6.land[16, 17, 18, 19].mask(ds)
    base_mask = ((mask_eur == 16) | (mask_eur == 17) | (mask_eur == 19)).copy()
    shifted_west = base_mask.roll(lon=-15, roll_coords=False)
    shifted_east = base_mask.roll(lon=5, roll_coords=False)
    mask_eu_buffer = base_mask | shifted_west | shifted_east

    masks_eur_dict = {
        'NEU': mask_eur == 16,
        'WCE': mask_eur == 17,
        'EEU': mask_eur == 18,
        'MED': mask_eur == 19,
        'EU': (mask_eur == 16) | (mask_eur == 17) | (mask_eur == 19),
        'EU_EEU': (mask_eur == 16) | (mask_eur == 17) | (mask_eur == 18) | (mask_eur == 19),
        'EU_buffer': mask_eu_buffer,
    }

    shpfilename = shpreader.natural_earth(resolution='10m',
                            category='cultural',
                            name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    countries = list(reader.records())

    # Compute grid cell boundaries as midpoints between consecutive coordinates
    lon_vals = ds.lon.values
    lat_vals = ds.lat.values
    lon_bounds = np.concatenate([[1.5 * lon_vals[0] - 0.5 * lon_vals[1]],
                                 (lon_vals[:-1] + lon_vals[1:]) / 2,
                                 [1.5 * lon_vals[-1] - 0.5 * lon_vals[-2]]])
    lat_bounds = np.concatenate([[1.5 * lat_vals[0] - 0.5 * lat_vals[1]],
                                 (lat_vals[:-1] + lat_vals[1:]) / 2,
                                 [1.5 * lat_vals[-1] - 0.5 * lat_vals[-2]]])

    # Add masks for individual countries
    country_masks = {}
    for country, code in country_codes.items():
        # Create the initial mask using centroids
        country_mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(ds) == code

        # Check if the mask is entirely False (no centroids in country)
        if not np.any(country_mask):
            print(f"    No centroids found for {country}. Falling back to overlap-based mask.")

            # Get the polygon for the country
            for c in countries:
                if c.attributes['ISO_A2_EH'] == country:
                    country_polygon = c.geometry

            prepared_polygon = prep(country_polygon)

            # Create a fallback mask based on overlap
            fallback_mask = np.zeros_like(country_mask, dtype=bool)
            for i, lon in enumerate(ds.lon.values):
                for j, lat in enumerate(ds.lat.values):
                    # Create a grid cell polygon using actual grid boundaries
                    grid_cell = Polygon([
                        (lon_bounds[i], lat_bounds[j]),
                        (lon_bounds[i+1], lat_bounds[j]),
                        (lon_bounds[i+1], lat_bounds[j+1]),
                        (lon_bounds[i], lat_bounds[j+1]),
                    ])
                    # Check if the overlap meets the minimum threshold
                    if prepared_polygon.intersects(grid_cell):
                        overlap_area = country_polygon.intersection(grid_cell).area  # Use the original geometry here
                        grid_cell_area = grid_cell.area
                        if overlap_area / grid_cell_area >= min_overlap:
                            fallback_mask[j, i] = True

            # Convert the fallback mask to an xarray.DataArray
            country_mask = xr.DataArray(
                fallback_mask,
                dims=masks_eur_dict['EU'].dims,
                coords=masks_eur_dict['EU'].coords,
            )

        country_masks[country] = country_mask

        if not np.any(country_mask):
            print(f"    Warning: Fallback mask for {country} is still empty.")

    # Add country masks to the dictionary
    masks_eur_dict.update(country_masks)

    return masks_eur_dict

def make_legend(fig, plot_ssp, hos_type, hos_strength='all', incl_ge=True, line_spacing=0.045, pos=(1.01, 1.0), col_width=0.15, plot_bg='white', markers=True, black_text=False):
    """
    Create a legend for the plot.
    
    Parameters:
    fig (matplotlib.Figure): Figure object to add the legend to.
    plot_ssp (str): SSP scenario to plot.
    hos_type (str): Hosing type: all, constant, linear.
    incl_ge (bool): Whether to include GE historical data in the legend (default is True).
    """
    i, left_out = -2, 0

    if plot_ssp != 'all':
        left_out = 0 # originally 2 but for simulation plot only 0 worked

    for (k, ssp) in enumerate(['ssp370', 'ssp245', 'ssp126', 'hist']):
        if ssp == 'hist':
            if incl_ge:
                fig.text(pos[0], pos[1], 'MPI-ESM GE historical', fontsize=10, color='k' if not plot_bg=='black' else 'white', weight='bold', va='top', ha='left', clip_on=False)
                i+=1
            else:
                i -= 1
        else:
            if plot_ssp != 'all' and ssp != plot_ssp:
                left_out += 1
                continue
            for exp in ['ge', 'neg01', '01', '03', '05', 'linneg02', 'lin02', 'lin06', 'lin10']:
                if exp == 'ge' and not incl_ge:
                    # print only scenario names
                    fig.text(pos[0], 
                            pos[1] - line_spacing*i + 4*(k-left_out-1)*line_spacing - (k-left_out)*line_spacing,
                            hosing_names[ssp][exp][11:], fontsize=10, color=hosing_colors[ssp][exp], weight='bold', va='top', ha='left', clip_on=False)
                    i += 1
                    continue
                if hos_type != 'all':
                    if hos_type == 'linear':
                        if 'lin' not in exp and exp != 'ge':
                            i += 1
                            continue
                    elif hos_type == 'constant':
                        if 'lin' in exp:
                            i += 1
                            continue
                    elif hos_type == None: # this might work only if ssp != 'all'
                        # print('k:',k, 'exp:', exp, 'i:', i, 'left_out:', left_out)
                        if exp != 'ge':
                            continue
                    else:
                        print('Unknown hosing type in legend.')
                        continue
                    if hos_strength != 'all' and exp != hos_strength:
                        continue
                col_offset = col_width if 'lin' in exp else 0.
                col_offset = 0 if hos_type == 'linear' else col_offset
                vert_shift = 4*(k-left_out)*line_spacing if 'lin' in exp else 4 * (k-left_out-1) * line_spacing
                fig.text(pos[0] + col_offset,
                        pos[1] - line_spacing*i + vert_shift - (k-left_out)*line_spacing,
                        (hosing_symbols[exp]+' '+hosing_names[ssp][exp][9:] if markers else hosing_names[ssp][exp][9:]) if exp!='ge' else (hosing_symbols[exp]+' '+hosing_names[ssp][exp] if markers else hosing_names[ssp][exp]), fontsize=10, color='k' if black_text else hosing_colors[ssp][exp], weight='bold', va='top', ha='left', clip_on=False)
                i += 1

def add_square(ax, lon_min, lon_max, lat_min, lat_max, colour='white'):
    rectangle = Polygon([(lon_min, lat_min), (lon_min, lat_max), (lon_max, lat_max), (lon_max, lat_min)])
    ax.add_geometries([rectangle], crs=ccrs.PlateCarree(), facecolor=colour, edgecolor=colour, zorder=4)

def get_custom_cmap():

    orig_cmap = plt.get_cmap('ocean_r')
    n_colors = 256

    replace_frac = 0.4
    colors = orig_cmap(np.linspace(0.04, (1-replace_frac), n_colors - 100))

    end_color = np.array([[0.8, 0.2, 0.8, 1.0]])  # pink for strong cooling

    transition = np.linspace(colors[-1], end_color.flatten(), 100)
    colors = np.vstack([colors, transition])

    custom_cmap = LinearSegmentedColormap.from_list('ocean_r_purple', colors)
    custom_cmap.set_over((0.95, 0.3, 0.95, 1.0))  # Brighter magenta for extend triangle

    return custom_cmap

########################################################################################################################
# MPI-ESM ANALYSIS
########################################################################################################################

# HELPER FUNCTIONS FOR MPI-ESM ANALYSIS

def split_ds_exper_dims(ds):
    if np.shape(ds.exper) == ():
        scenar = str(ds.exper.values).split('_')[0]
        hosing = str(ds.exper.values).split('_')[1][3:-2]
        # For 0-dim: drop exper and assign scalar coords directly
        return ds.drop_vars('exper').assign_coords(scenar=scenar, hosing=hosing)
    else:
        scenar = [val.split('_')[0] for val in ds.exper.values]
        hosing = [val.split('_')[1][3:-2] for val in ds.exper.values]
        ds = ds.assign_coords(
            scenar=('exper', scenar),
            hosing=('exper', hosing)
        )
        return ds.set_index(exper=['scenar', 'hosing']).unstack('exper')

def convert_strength_to_weakening(strength):
    return (AMOC_pi_MPI - strength) / AMOC_pi_MPI * 100

def convert_weakening_to_strength(weakening):
    return - weakening/100 * AMOC_pi_MPI + AMOC_pi_MPI

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256, darken=1.0):
    
    new_colors = cmap(np.linspace(minval, maxval, n))
    
    # Darken: multiply RGB channels by darken factor (keep alpha unchanged)
    new_colors[:, :3] = new_colors[:, :3] * darken
    new_colors = np.clip(new_colors, 0, 1)
    
    truncated_cmap = LinearSegmentedColormap.from_list('trunc({},{:.2f},{:.2f})'.format(cmap.name, minval, maxval), new_colors)
    
    return truncated_cmap

# LOADING MPI-ESM DATA

def load_mpi_esm_data(eur_only=True, load_daily=False):

    print("Loading MPI-ESM data...")

    his_amoc_yr = xr.open_dataarray(data_path+"MPI-GE/his_amoc26_yr.nc", use_cftime=True).to_dataset(name='AMOC_strength').drop_vars('lev')
    his_tas_yr = xr.open_dataarray(data_path+"MPI-GE/his_tas_yr.nc", use_cftime=True).to_dataset(name='tas') - 273.15
    his_tas_djf = xr.open_dataset(data_path+"MPI-GE/his_tas_djf.nc", use_cftime=True).drop_vars(['height', 'lat_bnds', 'lon_bnds']) - 273.15
    his_tas_jja = xr.open_dataset(data_path+"MPI-GE/his_tas_jja.nc", use_cftime=True).drop_vars(['height', 'lat_bnds', 'lon_bnds']) - 273.15

    ssp_amoc_yr = xr.open_dataarray(data_path+"MPI-GE/ssp_amoc26_yr.nc", use_cftime=True).to_dataset(name='AMOC_strength').drop_vars('lev')
    ssp_tas_yr = xr.open_dataarray(data_path+"MPI-GE/ssp_tas_yr.nc", use_cftime=True).to_dataset(name='tas') - 273.15
    ssp_tas_djf = xr.open_dataset(data_path+"MPI-GE/ssp_tas_djf.nc", use_cftime=True).drop_vars(['height', 'lat_bnds', 'lon_bnds']) - 273.15
    ssp_tas_jja = xr.open_dataset(data_path+"MPI-GE/ssp_tas_jja.nc", use_cftime=True).drop_vars(['height', 'lat_bnds', 'lon_bnds']) - 273.15

    ssphos_amoc_yr = xr.open_dataarray(data_path+"ssphos/ssphos_amoc_yr.nc", use_cftime=True).to_dataset(name='AMOC_strength')
    ssphos_tas_yr = xr.open_dataarray(data_path+"ssphos/ssphos_tas_yr.nc", use_cftime=True).to_dataset(name='tas') - 273.15
    ssphos_tas_djf = xr.open_dataarray(data_path+"ssphos/ssphos_tas_djf.nc", use_cftime=True).to_dataset(name='tas') - 273.15
    ssphos_tas_jja = xr.open_dataarray(data_path+"ssphos/ssphos_tas_jja.nc", use_cftime=True).to_dataset(name='tas') - 273.15

    his_pr_yr = xr.open_dataarray(data_path+"MPI-GE/his_pr_yr.nc", use_cftime=True).to_dataset(name='pr') * 3600 * 24  # convert from kg m-2 s-1 to mm d-1
    ssp_pr_yr = xr.open_dataarray(data_path+"MPI-GE/ssp_pr_yr.nc", use_cftime=True).to_dataset(name='pr') * 3600 * 24  # convert from kg m-2 s-1 to mm d-1
    ssphos_pr_yr = xr.open_dataarray(data_path+"ssphos/ssphos_pr_yr.nc", use_cftime=True).to_dataset(name='pr') * 3600 * 24  # convert from kg m-2 s-1 to mm d-1

    his_tmn_yr = xr.open_dataarray(data_path+"MPI-GE/his_tmn_year.nc", use_cftime=True).to_dataset(name='tmn') - 273.15
    ssp_tmn_yr = xr.open_dataarray(data_path+"MPI-GE/ssp_tmn_year.nc", use_cftime=True).to_dataset(name='tmn') - 273.15
    ssphos_tmn_yr_ssp245 = xr.open_dataarray(data_path+"ssphos/tmn_ssp245_grc1Sv_year.nc", use_cftime=True).to_dataset(name='tmn') - 273.15
    ssphos_tmn_yr_ssp126 = xr.open_dataarray(data_path+"ssphos/tmn_ssp126_grc1Sv_year.nc", use_cftime=True).to_dataset(name='tmn') - 273.15
    ssphos_tmn_yr = xr.concat([ssphos_tmn_yr_ssp126, ssphos_tmn_yr_ssp245], dim='exper')

    # Monthly tmn (eager load - reasonable size)
    ssphos_tmn_mon = xr.open_dataarray(data_path+"ssphos/tmn_ssp245_grc1Sv_mon.nc", use_cftime=True).to_dataset(name='tmn') - 273.15

    # Daily tas (lazy load via dask - large file)
    if load_daily:
        ssphos_tas_day = xr.open_dataarray(data_path+"ssphos/ssphos_tas_day.nc", chunks={'time': 365}, use_cftime=True).to_dataset(name='tas') - 273.15

    # split hosing datasets into scenarios and hosing types
    ssphos_amoc_yr = split_ds_exper_dims(ssphos_amoc_yr)
    ssphos_tas_yr = split_ds_exper_dims(ssphos_tas_yr)
    ssphos_tas_djf = split_ds_exper_dims(ssphos_tas_djf)
    ssphos_tas_jja = split_ds_exper_dims(ssphos_tas_jja)
    ssphos_pr_yr = split_ds_exper_dims(ssphos_pr_yr)
    ssphos_tmn_yr = split_ds_exper_dims(ssphos_tmn_yr)
    ssphos_tmn_mon = split_ds_exper_dims(ssphos_tmn_mon)
    if load_daily:
        ssphos_tas_day = split_ds_exper_dims(ssphos_tas_day)

    # make grids align
    ssp_tas_yr.coords['lon'] = (ssp_tas_yr.coords['lon'] + 180) % 360 - 180
    ssp_tas_yr = ssp_tas_yr.sortby(ssp_tas_yr.lon)
    ssp_tas_djf.coords['lon'] = (ssp_tas_djf.coords['lon'] + 180) % 360 - 180
    ssp_tas_djf = ssp_tas_djf.sortby(ssp_tas_djf.lon)
    ssp_tas_jja.coords['lon'] = (ssp_tas_jja.coords['lon'] + 180) % 360 - 180
    ssp_tas_jja = ssp_tas_jja.sortby(ssp_tas_jja.lon)

    his_tas_yr.coords['lon'] = (his_tas_yr.coords['lon'] + 180) % 360 - 180
    his_tas_yr = his_tas_yr.sortby(his_tas_yr.lon)
    his_tas_djf.coords['lon'] = (his_tas_djf.coords['lon'] + 180) % 360 - 180
    his_tas_djf = his_tas_djf.sortby(his_tas_djf.lon)
    his_tas_jja.coords['lon'] = (his_tas_jja.coords['lon'] + 180) % 360 - 180
    his_tas_jja = his_tas_jja.sortby(his_tas_jja.lon)

    his_pr_yr.coords['lon'] = (his_pr_yr.coords['lon'] + 180) % 360 - 180
    his_pr_yr = his_pr_yr.sortby(his_pr_yr.lon)
    ssp_pr_yr.coords['lon'] = (ssp_pr_yr.coords['lon'] + 180) % 360 - 180
    ssp_pr_yr = ssp_pr_yr.sortby(ssp_pr_yr.lon)
    ssphos_pr_yr = ssphos_pr_yr.sortby(ssphos_pr_yr.lat)

    ssphos_tas_yr = ssphos_tas_yr.sortby(ssphos_tas_yr.lat)
    ssphos_tas_djf = ssphos_tas_djf.sortby(ssphos_tas_djf.lat)
    ssphos_tas_jja = ssphos_tas_jja.sortby(ssphos_tas_jja.lat)
    
    his_tmn_yr.coords['lon'] = (his_tmn_yr.coords['lon'] + 180) % 360 - 180
    his_tmn_yr = his_tmn_yr.sortby(his_tmn_yr.lat)
    ssp_tmn_yr.coords['lon'] = (ssp_tmn_yr.coords['lon'] + 180) % 360 - 180
    ssp_tmn_yr = ssp_tmn_yr.sortby(ssp_tmn_yr.lat)

    ssphos_tmn_yr.coords['lon'] = (ssphos_tmn_yr.coords['lon'] + 180) % 360 - 180
    ssphos_tmn_yr = ssphos_tmn_yr.sortby(ssphos_tmn_yr.lat)

    ssphos_tmn_mon.coords['lon'] = (ssphos_tmn_mon.coords['lon'] + 180) % 360 - 180
    ssphos_tmn_mon = ssphos_tmn_mon.sortby(ssphos_tmn_mon.lat)

    if load_daily:
        ssphos_tas_day.coords['lon'] = (ssphos_tas_day.coords['lon'] + 180) % 360 - 180
        ssphos_tas_day = ssphos_tas_day.sortby(ssphos_tas_day.lat)

    masks_dict = define_region_masks(ssp_tas_yr)
    mask_land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(ssphos_tas_yr)
    masks_dict['LAND'] = mask_land

    data_dict = {}
    data_dict[''] = {}
    data_dict['djf'] = {}
    data_dict['jja'] = {}
    data_dict['masks'] = masks_dict

    data_dict['']['amoc'] = {}
    data_dict['']['amoc']['his'] = his_amoc_yr
    data_dict['']['amoc']['ssp'] = ssp_amoc_yr
    data_dict['']['amoc']['ssphos'] = ssphos_amoc_yr

    data_dict['']['tas'] = {}
    data_dict['djf']['tas'] = {}
    data_dict['jja']['tas'] = {}

    data_dict['']['tas']['his'] = his_tas_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else his_tas_yr
    data_dict['']['tas']['ssp'] = ssp_tas_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssp_tas_yr
    data_dict['']['tas']['ssphos'] = ssphos_tas_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssphos_tas_yr

    data_dict['djf']['tas']['his'] = his_tas_djf.where(masks_dict['EU_buffer'], drop=True) if eur_only else his_tas_djf
    data_dict['djf']['tas']['ssp'] = ssp_tas_djf.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssp_tas_djf
    data_dict['djf']['tas']['ssphos'] = ssphos_tas_djf.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssphos_tas_djf

    data_dict['jja']['tas']['his'] = his_tas_jja.where(masks_dict['EU_buffer'], drop=True) if eur_only else his_tas_jja
    data_dict['jja']['tas']['ssp'] = ssp_tas_jja.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssp_tas_jja
    data_dict['jja']['tas']['ssphos'] = ssphos_tas_jja.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssphos_tas_jja

    data_dict['']['pr'] = {}
    data_dict['']['pr']['his'] = his_pr_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else his_pr_yr
    data_dict['']['pr']['ssp'] = ssp_pr_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssp_pr_yr
    data_dict['']['pr']['ssphos'] = ssphos_pr_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssphos_pr_yr

    data_dict['']['tmn'] = {}
    data_dict['']['tmn']['his'] = his_tmn_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else his_tmn_yr
    data_dict['']['tmn']['ssp'] = ssp_tmn_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssp_tmn_yr
    data_dict['']['tmn']['ssphos'] = ssphos_tmn_yr.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssphos_tmn_yr    #.expand_dims(['hosing'])

    data_dict['mon'] = {}
    if load_daily:
        data_dict['day'] = {}

    data_dict['mon']['tmn'] = {}
    data_dict['mon']['tmn']['ssphos'] = ssphos_tmn_mon.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssphos_tmn_mon

    if load_daily:
        data_dict['day']['tas'] = {}
        data_dict['day']['tas']['ssphos'] = ssphos_tas_day.where(masks_dict['EU_buffer'], drop=True) if eur_only else ssphos_tas_day

    return data_dict

# TRANSFORMING MPI-ESM DATA

def get_regression_ds(data_dict, season='', var='tas'):

    print(f"Calculating {'annual' if season=='' else season} MPI-ESM regression coefficients...")

    static_ds = data_dict[season][var]['ssphos'].isel(time=0, drop=True).copy(deep=True).drop_vars(var)

    reg_ds = static_ds.assign(
        coef_ensmean=(['lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, scenar=0, hosing=0, drop=True).sizes.values()), np.nan)),
        ste_ensmean=(['lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, scenar=0, hosing=0, drop=True).sizes.values()), np.nan)), # standard error of regression coefficient
        rsq_ensmean=(['lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, scenar=0, hosing=0, drop=True).sizes.values()), np.nan)), # R^2 of regression
        AMOC_2090_2100=(['scenar'], np.full(tuple(static_ds.isel(lon=0, lat=0, realiz=0, hosing=0, drop=True).sizes.values()), np.nan)),
        AMOC_pi=([], np.full(tuple(static_ds.isel(lon=0, lat=0, realiz=0, hosing=0, scenar=0, drop=True).sizes.values()), np.nan)),
        T_2090_2100=(['scenar', 'lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, hosing=0, drop=True).sizes.values()), np.nan)),
        T_pi=(['lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, scenar=0, hosing=0, drop=True).sizes.values()), np.nan)),
        T_pd_245=(['lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, scenar=0, hosing=0, drop=True).sizes.values()), np.nan)),
        req_strength_pi=(['scenar', 'lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, hosing=0, drop=True).sizes.values()), np.nan)),
        req_strength_pd=(['scenar', 'lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, hosing=0, drop=True).sizes.values()), np.nan)),
        req_weakening_pi=(['scenar', 'lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, hosing=0, drop=True).sizes.values()), np.nan)),
        req_weakening_pd=(['scenar', 'lat', 'lon'], np.full(tuple(static_ds.isel(realiz=0, hosing=0, drop=True).sizes.values()), np.nan)),
    )

    reg_ds.AMOC_pi.values = AMOC_pi_MPI
    reg_ds.T_pi.values[:, :] = data_dict[season][var]['his'].mean(dim='realiz').sel(time=slice('1850', '1899')).mean(dim='time')[var]

    i=0
    for lat in np.arange(len(data_dict[season][var]['ssphos'].lat)):
        for lon in np.arange(len(data_dict[season][var]['ssphos'].lon)):
            reg_quadruple = regression_plot(data_dict, season=season, var=var, lat=lat, lon=lon, no_plots=True)
            reg_ds.coef_ensmean.values[lat, lon] = reg_quadruple[1]
            reg_ds.ste_ensmean.values[lat, lon] = reg_quadruple[2]
            reg_ds.rsq_ensmean.values[lat, lon] = reg_quadruple[3]
            i += 1
            if i % 100 == 0:
                print(f"{i}/{len(data_dict[season][var]['ssphos'].lat)*len(data_dict[season][var]['ssphos'].lon)} regression coefficients calculated.")

    for ssp_i in reg_ds.scenar.values:
        reg_ds.AMOC_2090_2100.loc[ssp_i] = data_dict['']['amoc']['ssp'].mean(dim='realiz').sel(time=slice('2091', '2100')).mean(dim='time').sel(scenar=ssp_i).AMOC_strength
        ssp_index = list(reg_ds.scenar.values).index(ssp_i)
        reg_ds.T_2090_2100.values[ssp_index, :, :] = data_dict[season][var]['ssp'].mean(dim='realiz').sel(time=slice('2091', '2100')).mean(dim='time').sel(scenar=ssp_i)[var]
        reg_ds.T_pd_245.values[:, :] = xr.concat([data_dict[season][var]['his'].mean(dim='realiz').sel(time=slice('2000', '2014')), data_dict[season][var]['ssp'].mean(dim='realiz').sel(time=slice('2015', '2029'), scenar='ssp245')], dim='time').mean(dim='time')[var]
        reg_ds.req_strength_pi.loc[ssp_i].values[:, :] = reg_ds.AMOC_2090_2100.loc[ssp_i] - (reg_ds.T_2090_2100.loc[ssp_i] - reg_ds.T_pi) / reg_ds.coef_ensmean.values
        reg_ds.req_strength_pd.loc[ssp_i].values[:, :] = reg_ds.AMOC_2090_2100.loc[ssp_i] - (reg_ds.T_2090_2100.loc[ssp_i] - reg_ds.T_pd_245) / reg_ds.coef_ensmean.values
        reg_ds.req_weakening_pi.loc[ssp_i].values[:, :] = convert_strength_to_weakening(reg_ds.req_strength_pi.loc[ssp_i].values)
        reg_ds.req_weakening_pd.loc[ssp_i].values[:, :] = convert_strength_to_weakening(reg_ds.req_strength_pd.loc[ssp_i].values)
    
    return reg_ds

def load_regression_ds_mpi(data_dict=None, var='tas', ens_mean=True):
    if os.path.exists(local_path+f'reg_ds_mpi_{var}.nc'):
        print("Loading precomputed MPI-ESM regression dataset...")
        reg_ds_mpi = xr.open_dataset(local_path+f'reg_ds_mpi_{var}.nc')
    else:
        print("Calculating MPI-ESM regression dataset...")
        if data_dict is None:
            data_dict = load_mpi_esm_data(eur_only=True)
        if var == 'tas':
            reg_ds_seasonal = [get_regression_ds(data_dict, season=s, var=var) for s in ['', 'djf', 'jja']]
            reg_ds_mpi = xr.concat([ds.assign_coords(season=s) for ds, s in zip(reg_ds_seasonal, ['', 'djf', 'jja'])], dim='season')
        else:
            reg_ds_mpi = get_regression_ds(data_dict, season='', var=var)
        reg_ds_mpi.to_netcdf(local_path+f'reg_ds_mpi_{var}.nc')
        print(f"Saved MPI-ESM regression dataset to {local_path}/reg_ds_mpi_{var}.nc.")
    return reg_ds_mpi.mean(dim='realiz') if ens_mean else reg_ds_mpi

# PLOTTING MPI-ESM DATA

def simulations_plot(data_dict, ssp='all', hos_type='all', season='', var='tas', region='EU', window=10, plot_bg='white', ext_axs=None):

    # always compare against all-year AMOC strength
    his_amoc_yr = data_dict['']['amoc']['his']
    ssp_amoc_yr = data_dict['']['amoc']['ssp']
    ssphos_amoc_yr = data_dict['']['amoc']['ssphos']
    # potentially seasonal variable
    his_var = data_dict[season][var]['his']
    ssp_var = data_dict[season][var]['ssp']
    ssphos_var = data_dict[season][var]['ssphos']
    # season-invariant masks
    masks_eur_dict = data_dict['masks']
    mask_land = data_dict['masks']['LAND']

    his_years = his_amoc_yr.time.dt.year
    ssp_years = ssp_amoc_yr.time.dt.year

    plt.style.use('default')
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    if ext_axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
        fig.subplots_adjust(hspace=0.2)
    else:
        axs = ext_axs
        plt.rcParams.update({'font.size': 10})
    axs[1].sharex(axs[0])


    amoc = xr.concat([his_amoc_yr, ssp_amoc_yr.sel(scenar='ssp245')], dim='time').rolling(time=window, center=True).mean('time').AMOC_strength
    axs[0].plot(his_years,
                amoc.mean(dim='realiz')[:len(his_years)], c='k' if not plot_bg=='black' else 'white', lw=2.5, clip_on=True)
    axs[0].fill_between(his_years,
                        amoc.mean(dim='realiz')[:len(his_years)] - amoc.std(dim='realiz')[:len(his_years)],
                        amoc.mean(dim='realiz')[:len(his_years)] + amoc.std(dim='realiz')[:len(his_years)],
                        alpha=0.4, linewidth=0, color='k' if not plot_bg=='black' else 'white', clip_on=True)

    tas_eur =  weighted_area_lat(xr.concat([his_var, ssp_var.sel(scenar='ssp245')], dim='time').where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon').rolling(time=window, center=True).mean("time")[var]
    his_tas_eur_1850_1899 = weighted_area_lat(his_var.where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon')[var].mean(dim='realiz').sel(time=slice('1850', '1899')).mean(dim='time')
    axs[1].plot(his_years,
                tas_eur.mean(dim='realiz')[:len(his_years)] - his_tas_eur_1850_1899, 
                c='k' if not plot_bg=='black' else 'white', lw=2.5, clip_on=True)
    axs[1].fill_between(his_years,
                        tas_eur.mean(dim='realiz')[:len(his_years)] - his_tas_eur_1850_1899 - tas_eur.std(dim='realiz')[:len(his_years)],
                        tas_eur.mean(dim='realiz')[:len(his_years)] - his_tas_eur_1850_1899 + tas_eur.std(dim='realiz')[:len(his_years)],
                        alpha=0.4, linewidth=0, color='k' if not plot_bg=='black' else 'white', clip_on=True)

    for ssp_i in ssphos_amoc_yr.scenar.values:
        if ssp != 'all' and ssp_i != ssp:
            continue
        for hos_i in ssphos_amoc_yr.hosing.values:
            if hos_type != 'all_plus_1Sv':
                if hos_type == 'all':
                    if hos_i == '1':
                        continue
                elif hos_type == 'linear':
                    if 'lin' not in hos_i:
                        continue
                elif hos_type == 'constant':
                    if 'lin' in hos_i:
                        continue
                    if hos_i == '1':
                        continue
                elif hos_type == '1':
                    if hos_i != '1':
                        continue
                elif hos_type == None:
                    continue
                else:
                    print('Unknown hosing type.')
                    continue
            if hos_i == '1' and ssp_i not in ['ssp126', 'ssp245']:
                continue
            amoc = xr.concat([his_amoc_yr, ssphos_amoc_yr.sel(scenar=ssp_i, hosing=hos_i)], dim='time').rolling(time=window, center=True).mean('time').isel(time=slice(0, -198)).AMOC_strength  # after including 1.0 Sv, ds goes until 2298
            axs[0].plot(ssp_years,
                    amoc.mean(dim='realiz')[-len(ssp_years):], color=hosing_colors[ssp_i][hos_i], lw=1.5, alpha=0.98, clip_on=False)
            axs[0].plot(ssp_years[-((window+1)//2)]+1.5,
                amoc.mean(dim='realiz')[-((window+1)//2)],
                marker=hosing_markers[hos_i], color=hosing_colors[ssp_i][hos_i], markersize=6, clip_on=False)

            if hos_i != '1' or var != 'tmn':
                tas_eur = weighted_area_lat(xr.concat([his_var.mean(dim='realiz'), ssphos_var.sel(scenar=ssp_i, hosing=hos_i).mean(dim='realiz')], dim='time').where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon').rolling(time=window, center=True).mean('time').isel(time=slice(0, -198))[var]   # after including 1.0 Sv, ds goes until 2298
            else: # separate treatment for 1.0 Sv hosing tmn because there is no ensemble, just one realisation
                tas_eur = weighted_area_lat(xr.concat([his_var.mean(dim='realiz'), ssphos_var.sel(scenar=ssp_i, hosing=hos_i)], dim='time').where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon').rolling(time=window, center=True).mean('time').isel(time=slice(0, -198))[var]   # after including 1.0 Sv, ds goes until 2298
            his_tas_eur_1850_1899 = weighted_area_lat(his_var.where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon')[var].mean(dim='realiz').sel(time=slice('1850', '1899')).mean(dim='time')
            axs[1].plot(ssp_years,
                    tas_eur[-len(ssp_years):] - his_tas_eur_1850_1899, color=hosing_colors[ssp_i][hos_i], lw=1.5, alpha=0.98, clip_on=False)
            axs[1].plot(ssp_years[-((window+1)//2)]+1.5,
                tas_eur[-((window+1)//2)] - his_tas_eur_1850_1899,
                marker=hosing_markers[hos_i], color=hosing_colors[ssp_i][hos_i], markersize=6, clip_on=False)
            
        amoc = xr.concat([his_amoc_yr, ssp_amoc_yr.sel(scenar=ssp_i)], dim='time').rolling(time=window, center=True).mean('time').AMOC_strength
        axs[0].plot(ssp_years,
                    amoc.mean(dim='realiz')[-len(ssp_years):], color=hosing_colors[ssp_i]['ge'], lw=2.5, clip_on=False)
        axs[0].fill_between(ssp_years,
                            amoc.mean(dim='realiz')[-len(ssp_years):] - amoc.std(dim='realiz')[-len(ssp_years):],
                            amoc.mean(dim='realiz')[-len(ssp_years):] + amoc.std(dim='realiz')[-len(ssp_years):],
                            alpha=0.4, linewidth=0, color=hosing_colors[ssp_i]['ge'], clip_on=False, zorder=3)
        axs[0].plot(ssp_years[-((window+1)//2)]+1.5,
                amoc.mean(dim='realiz')[-((window+1)//2)],
                marker='o', color=hosing_colors[ssp_i]['ge'], markersize=6, label=f"{ssp_i}", clip_on=False, zorder=3)

        tas_eur = weighted_area_lat(xr.concat([his_var, ssp_var.sel(scenar=ssp_i)], dim='time').where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon').rolling(time=window, center=True).mean("time")[var]
        his_tas_eur_1850_1899 = weighted_area_lat(his_var.where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon')[var].mean(dim='realiz').sel(time=slice('1850', '1899')).mean(dim='time')
        axs[1].plot(ssp_years,
                    tas_eur.mean(dim='realiz')[-len(ssp_years):] - his_tas_eur_1850_1899,
                    color=hosing_colors[ssp_i]['ge'], lw=2.5, clip_on=False)
        axs[1].fill_between(ssp_years,
                            tas_eur.mean(dim='realiz')[-len(ssp_years):] - his_tas_eur_1850_1899 - tas_eur.std(dim='realiz')[-len(ssp_years):],
                            tas_eur.mean(dim='realiz')[-len(ssp_years):] - his_tas_eur_1850_1899 + tas_eur.std(dim='realiz')[-len(ssp_years):],
                            alpha=0.4, linewidth=0, color=hosing_colors[ssp_i]['ge'], clip_on=False)
        axs[1].plot(ssp_years[-((window+1)//2)]+1.5,
                tas_eur.mean(dim='realiz')[-((window+1)//2)] - his_tas_eur_1850_1899,
                marker='o', color=hosing_colors[ssp_i]['ge'], markersize=5, clip_on=False)


    ############################ formatting

    axs[0].set_ylabel('AMOC strength [Sv]')
    axs[0].set_ylim(8, 22)
    axs[1].set_xlabel('')
    axs[1].set_ylabel((season.upper()+' ' if season else '')+rf'$\Delta \mathrm{{T}}_\mathrm{{{region}}}$ [°C] w.r.t. 1850–1899' if var=='tas' else rf'$\Delta \mathrm{{Pr}}_\mathrm{{{region}}} \left[ \frac{{\mathrm{{mm}}}}{{\mathrm{{d}}}} \right]$ ' if var=='pr' else rf'$\Delta \mathrm{{TMn}}_\mathrm{{{region}}}$ [°C]' if var=='tmn' else 'fill var label', labelpad=5)
    axs[1].set_xlim(2000, 2100)
    if var == 'tas':
        axs[1].set_ylim(1, 6)
    elif var == 'pr':
        axs[1].set_ylim(-0.25, 0.2)

    axs[0].spines['top'].set_visible(False), axs[1].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False)

    # axs[1].yaxis.tick_right(), axs[1].yaxis.set_label_position('right')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines["left"].set_position(("axes", -0.02))
    axs[0].spines["left"].set_position(("axes", -0.02))

    axs[1].set_xticks(np.arange(2000, 2101, 25))

    if ext_axs is not None:
        return

    primary_subset_min = 7.606163386737876
    primary_subset_max = 19.01540846684469
    primary_position = axs[0].get_position()

    # Calculate the new vertical extent for the secondary axis
    total_height = primary_position.height
    new_y0 = primary_position.y0 + (primary_subset_min - axs[0].get_ylim()[0]) / (axs[0].get_ylim()[1] - axs[0].get_ylim()[0]) * total_height
    new_height = (primary_subset_max - primary_subset_min) / (axs[0].get_ylim()[1] - axs[0].get_ylim()[0]) * total_height

    secax = fig.add_axes([0.9,  # x-position
                        new_y0,  # starting vertical position
                        0.02,  # Width of the secondary axis
                        new_height])  # height of secondary axis

    secax.set_ylabel('AMOC weakening [%]', labelpad=5)

    secax.spines['top'].set_visible(False)
    secax.spines['left'].set_visible(False)
    secax.spines['bottom'].set_visible(False)
    secax.spines['right'].set_visible(True)

    secax.yaxis.tick_right()
    secax.yaxis.set_label_position('right')
    secax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    secondary_subset_min = convert_strength_to_weakening(primary_subset_max)  # Note: max maps to min
    secondary_subset_max = convert_strength_to_weakening(primary_subset_min)  # Note: min maps to max

    secax.set_ylim(secondary_subset_max, secondary_subset_min)
    secondary_ticks = np.linspace(secondary_subset_max, secondary_subset_min, 7)

    secax.set_yticks(secondary_ticks)
    secax.set_yticklabels([f"{tick:.0f}" for tick in secondary_ticks])

    secax.set_facecolor('none')

    make_legend(fig, ssp, hos_type, line_spacing=0.015, pos=(0.18, 0.62), col_width=0.12, plot_bg=plot_bg)

    plt.savefig(f'../plots/simulations_ssp-{ssp}_hos-{hos_type}_window-{window}_{region}_bg-{plot_bg}.png', dpi=400, bbox_inches='tight', transparent=True if plot_bg=='black' else False)

def regression_plot(data_dict, lat=None, lon=None, region=None, var='tas', ssp='all', hos_type='all', hos_strength='all', window=10, regressions=True, combined_reg=True, no_plots=False, plot_bg='white', season='', ext_ax=None, xlim=(-8, 6), ylim=(-2.5, 1.5), equation_y_pos=0.95, equation_y_spacing=0.05):

    # always compare against all-year AMOC strength
    ssp_amoc_yr = data_dict['']['amoc']['ssp']
    ssphos_amoc_yr = data_dict['']['amoc']['ssphos'].isel(time=slice(0, -198))
    # potentially seasonal variable
    ssp_var = data_dict[season][var]['ssp']
    ssphos_var = data_dict[season][var]['ssphos'].isel(time=slice(0, -198))
    # season-invariant masks
    masks_eur_dict = data_dict['masks']
    mask_land = data_dict['masks']['LAND']

    if not no_plots:
        plt.style.use('default')
        if plot_bg == 'black':
            plt.style.use('dark_background')
            plt.rcParams['axes.facecolor'] = '#191919'
            plt.rcParams['figure.facecolor'] = '#191919'
        if ext_ax is None:
            fig, ax = plt.subplots(figsize=(9, 6))
        else:
            ax = ext_ax

    is_latlon, is_region, is_ocean = False, False, False
    if lat != None and lon != None:
        is_latlon = True
        if np.sum(~np.isnan(ssphos_var.isel(lat=lat, lon=lon)[var].values))==0:
            is_ocean = True
    elif region != None:
        is_region = True
    else:
        print('No lat/lon or region provided.')
        return

    combined_x = []
    combined_y = []
    combined_exp_id = []

    # for ssp_i in ssp_amoc_yr.scenar.values: # need to drop ssp585
    for ssp_id, ssp_i in enumerate(['ssp126', 'ssp245', 'ssp370']):
        ssp_x = []
        ssp_y = []
        ssp_exp_id = []
        
        if ssp != 'all' and ssp_i != ssp:
            continue

        for exp_id, exp in enumerate(np.concat([[ssphos_amoc_yr.hosing.values[-1]], ssphos_amoc_yr.hosing.values[:-1]])):
            if hos_type != 'all_plus_1Sv':
                if hos_type == 'all':
                    if exp == '1':
                        continue                    
                elif hos_type == 'linear':
                    if 'lin' not in exp:
                        continue
                elif hos_type == 'constant':
                    if 'lin' in exp:
                        continue
                    if exp == '1':
                        continue
                elif hos_type == '1':
                    if exp != '1':
                        continue
                else:
                    print('Unknown hosing type.')
                    continue
                if hos_strength != 'all' and exp != hos_strength:
                    continue
            if is_ocean:
                continue

            if exp == '1' and ssp_i not in  ['ssp126', 'ssp245']:
                continue
            
            x = ssphos_amoc_yr.sel(scenar=ssp_i, hosing=exp).rolling(time=window, center=True).mean("time").mean(dim='realiz').AMOC_strength.values - ssp_amoc_yr.sel(scenar=ssp_i).rolling(time=window, center=True).mean('time').mean(dim='realiz').AMOC_strength.values
            if is_latlon:
                if exp == '1' and var=='tmn':
                    y = ssphos_var.sel(scenar=ssp_i, hosing=exp).isel(lat=lat, lon=lon).rolling(time=window, center=True).mean("time")[var].values - ssp_var.sel(scenar=ssp_i).isel(lat=lat, lon=lon).rolling(time=window, center=True).mean("time").mean(dim='realiz')[var].values
                else:
                    y = ssphos_var.sel(scenar=ssp_i, hosing=exp).isel(lat=lat, lon=lon).rolling(time=window, center=True).mean("time").mean(dim='realiz')[var].values - ssp_var.sel(scenar=ssp_i).isel(lat=lat, lon=lon).rolling(time=window, center=True).mean("time").mean(dim='realiz')[var].values
            elif is_region:
                if exp == '1' and var=='tmn':
                    y = weighted_area_lat(ssphos_var.sel(scenar=ssp_i, hosing=exp).where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon').rolling(time=window, center=True).mean("time")[var].values - weighted_area_lat(ssp_var.sel(scenar=ssp_i).where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon').rolling(time=window, center=True).mean("time").mean(dim='realiz')[var].values
                else:
                    y = weighted_area_lat(ssphos_var.sel(scenar=ssp_i, hosing=exp).where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon').rolling(time=window, center=True).mean("time").mean(dim='realiz')[var].values - weighted_area_lat(ssp_var.sel(scenar=ssp_i).where(masks_eur_dict[region]).where(mask_land==0)).mean('lat').mean('lon').rolling(time=window, center=True).mean("time").mean(dim='realiz')[var].values
            else:
                print('No lat/lon or region provided.')

            # Filter out NaN values
            mask = ~np.isnan(x) & ~np.isnan(y)
            x = x[mask]
            y = y[mask]
            
            ssp_x.extend(x)
            ssp_y.extend(y)
            ssp_exp_id.extend([exp_id]*len(x))
            combined_x.extend(x)
            combined_y.extend(y)
            combined_exp_id.extend([9*ssp_id + exp_id]*len(x))
        
            if not no_plots:
                ax.scatter(x, y,
                            label=hosing_names[ssp_i][exp], c=hosing_colors[ssp_i][exp],
                            marker=hosing_markers[exp], s=25, clip_on=False)

        if not is_ocean:
            # Perform linear regression using statsmodels
            x_with_const = sm.add_constant(ssp_x)
            model = sm.OLS(ssp_y, x_with_const).fit(cov_type='cluster', cov_kwds={'groups': ssp_exp_id, 'use_correction': True})

            # Manual df correction: use G-1 degrees of freedom for cluster-robust inference
            G_ssp = len(set(ssp_exp_id))
            df_cluster_ssp = G_ssp - 1
            t_crit_ssp = scipy_stats.t.ppf(0.975, df=df_cluster_ssp)
            t_stats_ssp = model.params / model.bse
            p_values_ssp = 2 * scipy_stats.t.sf(np.abs(t_stats_ssp), df=df_cluster_ssp)

            # DEBUG prints
            # print(f"\n=== {ssp_i} per-SSP regression ===")
            # print(f"  N={len(ssp_y)}, G={G_ssp}, df_cluster={df_cluster_ssp}")
            # print(f"  statsmodels df_resid={model.df_resid} (should be {df_cluster_ssp})")
            # print(f"  t_crit(0.975, df={df_cluster_ssp}) = {t_crit_ssp:.4f}")
            # for i_param, name in enumerate(['const', 'x1']):
            #     print(f"  {name}: coef={model.params[i_param]:.4f}, se={model.bse[i_param]:.4f}, "
            #           f"t={t_stats_ssp[i_param]:.3f}, "
            #           f"p_corrected={p_values_ssp[i_param]:.6f} (was {model.pvalues[i_param]:.6f}), "
            #           f"CI=[{model.params[i_param] - t_crit_ssp*model.bse[i_param]:.4f}, "
            #           f"{model.params[i_param] + t_crit_ssp*model.bse[i_param]:.4f}]")

            x_range = np.linspace(-10, 6, 200)
            x_range_with_const = sm.add_constant(x_range)
            y_pred = model.predict(x_range_with_const)
            if regressions and not no_plots:
                ax.plot(x_range, y_pred,
                        color=hosing_colors[ssp_i]['ge'], lw=1.5, linestyle='-')

            # Calculate confidence intervals manually with corrected df
            pred_se = model.get_prediction(x_range_with_const).se
            lower_bound = y_pred - t_crit_ssp * pred_se
            upper_bound = y_pred + t_crit_ssp * pred_se
            if regressions and not no_plots:
                ax.fill_between(x_range, lower_bound, upper_bound,
                                color=hosing_colors[ssp_i][exp], alpha=0.4, linewidth=0)

            # Plot regression equation and coefficients with corrected p-values
            coef = model.params[1]
            intercept = model.params[0]
            p_value = p_values_ssp[1]
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            else:
                significance = ''
            if is_latlon:
                # equation = fr"$\Delta T_{{lat{lat}\_lon{lon}}}$ = {coef:.2f}{significance} $\Delta AMOC$"+(" +" if intercept>0 else " ") +f"{intercept:.2f}"
                var_label = f'T_{{lat{lat}\_lon{lon}}}' if var=='tas' else f'Pr_{{lat{lat}\_lon{lon}}}' if var=='pr' else 'fill var label'
            else:
                # equation = fr"$\Delta T_{{{region}}}$ = {coef:.2f}{significance} $\Delta AMOC$"+(" +" if intercept>0 else " ") +f"{intercept:.2f}"
                var_label = f'T_{{{region}}}' if var=='tas' else f'Pr_{{{region}}}' if var=='pr' else f'TMn_{{{region}}}' if var == 'tmn' else 'fill var label'
            coef_display = f'{coef:.2f}' if var=='tas' or var=='tmn' else f'{coef:.3f}' if var=='pr' else f'{coef}'
            intercept_display = f'{intercept:.2f}' if var=='tas' or var=='tmn' else f'{intercept:.3f}' if var=='pr' else f'{intercept}'
            equation = fr"$\Delta {var_label}$ = {coef_display}$^{{{{\text{{{significance}}}}}}}$°C/Sv $\Delta AMOC$"+(" +" if intercept>0 else " ") +f"{intercept_display} $^{{\circ}}$C"
            if regressions and not no_plots:
                ax.text(0.0, equation_y_pos,
                        f"{hosing_names[ssp_i]['ge'][11:]}: {equation}",
                        transform=ax.transAxes, fontsize=10, verticalalignment='top', color=hosing_colors[ssp_i]['ge'])
            equation_y_pos -= equation_y_spacing

    if not is_ocean:
        # Combine both datasets for a third regression line
        combined_x = np.array(combined_x)
        combined_y = np.array(combined_y)

        combined_x_with_const = sm.add_constant(combined_x)
        combined_model = sm.OLS(combined_y, combined_x_with_const).fit(cov_type='cluster', cov_kwds={'groups': combined_exp_id, 'use_correction': True})

        # Manual df correction for combined model
        G_combined = len(set(combined_exp_id))
        df_cluster_combined = G_combined - 1
        t_crit_combined = scipy_stats.t.ppf(0.975, df=df_cluster_combined)
        t_stats_combined = combined_model.params / combined_model.bse
        p_values_combined = 2 * scipy_stats.t.sf(np.abs(t_stats_combined), df=df_cluster_combined)

        # DEBUG prints
        # print(f"\n=== Combined regression ===")
        # print(f"  N={len(combined_y)}, G={G_combined}, df_cluster={df_cluster_combined}")
        # print(f"  statsmodels df_resid={combined_model.df_resid} (should be {df_cluster_combined})")
        # print(f"  t_crit(0.975, df={df_cluster_combined}) = {t_crit_combined:.4f}")
        # for i_param, name in enumerate(['const', 'x1']):
        #     print(f"  {name}: coef={combined_model.params[i_param]:.4f}, se={combined_model.bse[i_param]:.4f}, "
        #           f"t={t_stats_combined[i_param]:.3f}, "
        #           f"p_corrected={p_values_combined[i_param]:.6f} (was {combined_model.pvalues[i_param]:.6f}), "
        #           f"CI=[{combined_model.params[i_param] - t_crit_combined*combined_model.bse[i_param]:.4f}, "
        #           f"{combined_model.params[i_param] + t_crit_combined*combined_model.bse[i_param]:.4f}]")

        combined_x_range = np.linspace(-10, 6, 200)
        combined_x_range_with_const = sm.add_constant(combined_x_range)
        combined_y_pred = combined_model.predict(combined_x_range_with_const)
        if (regressions and combined_reg) and not no_plots:
            ax.plot(combined_x_range, combined_y_pred,
                    color='black' if not plot_bg=='black' else 'white', lw=1.5, linestyle='-')

        # Calculate confidence intervals manually with corrected df
        combined_pred_se = combined_model.get_prediction(combined_x_range_with_const).se
        combined_lower_bound = combined_y_pred - t_crit_combined * combined_pred_se
        combined_upper_bound = combined_y_pred + t_crit_combined * combined_pred_se
        if (regressions and combined_reg) and not no_plots:
            ax.fill_between(combined_x_range, combined_lower_bound, combined_upper_bound,
                            color='black' if not plot_bg=='black' else 'white', alpha=0.4, linewidth=0)

        # Plot combined regression equation and coefficients with corrected p-values
        combined_coef = combined_model.params[1]
        combined_intercept = combined_model.params[0]
        combined_p_value = p_values_combined[1]
        combined_ste = combined_model.bse[1]
        combined_rsq = combined_model.rsquared
        if combined_p_value < 0.001:
            combined_significance = '***'
        elif combined_p_value < 0.01:
            combined_significance = '**'
        elif combined_p_value < 0.05:
            combined_significance = '*'
        else:
            combined_significance = ''
        if is_latlon:
            # combined_equation = fr"$\Delta T_{{lat{lat}\_lon{lon}}}$ = {combined_coef:.2f}{combined_significance} $\Delta AMOC$"+(" +" if intercept>0 else " ") +f"{combined_intercept:.2f}"
            var_label = f'T_{{lat{lat}\_lon{lon}}}' if var=='tas' else f'Pr_{{lat{lat}\_lon{lon}}}' if var=='pr' else 'fill var label'
        else: 
            # combined_equation = fr"$\Delta T_{{{region}}}$ = {combined_coef:.2f}{combined_significance} $\Delta AMOC$"+(" +" if intercept>0 else " ") +f"{combined_intercept:.2f}"
            var_label = f'T_{{{region}}}' if var=='tas' else f'Pr_{{{region}}}' if var=='pr' else f'TMn_{{{region}}}' if var == 'tmn' else 'fill var label'
        combined_coef_display = f'{combined_coef:.2f}' if var=='tas' or var=='tmn' else f'{combined_coef:.3f}' if var=='pr' else f'{combined_coef}'
        combined_intercept_display = f'{combined_intercept:.2f}' if var=='tas' or var=='tmn' else f'{combined_intercept:.3f}' if var=='pr' else f'{combined_intercept}'
        combined_equation = fr"$\Delta {var_label}$ = {combined_coef_display}$^{{{{\text{{{combined_significance}}}}}}}$°C/Sv $\Delta AMOC$"+(" +" if intercept>0 else " ") +f"{combined_intercept_display} $^{{\circ}}$C"
        if (regressions and combined_reg) and not no_plots:
            ax.text(0.0, equation_y_pos,
                    f"All SSPs:  {combined_equation}",
                    transform=ax.transAxes, fontsize=10, verticalalignment='top', color='black' if not plot_bg=='black' else 'white')

    if not no_plots:
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("Deviation from MPI-GE AMOC strength [Sv]")
        if is_latlon:
            ax.set_ylabel((season.upper()+' ' if season else '')+f"Deviation from GE ensemble mean lat{lat}_lon{lon} temperature [°C]" if var=='tas' else rf'Deviation from MPI-GE $\Delta \mathrm{{Pr}}_\mathrm{{lat{lat}lon{lon}}} \left[ \frac{{\mathrm{{mm}}}}{{\mathrm{{d}}}} \right]$ ' if var=='pr' else rf'Deviation from MPI-GE $\Delta \mathrm{{TMn}}_\mathrm{{lat{lat}lon{lon}}}$ [°C]' if var=='tmn' else 'fill var label')
        else:
            ax.set_ylabel((season.upper()+' ' if season else '')+rf"Deviation from MPI-GE $\Delta \mathrm{{T}}_\mathrm{{{region}}}$ [°C]" if var=='tas' else rf'Deviation from MPI-GE $\Delta \mathrm{{Pr}}_\mathrm{{{region}}} \left[ \frac{{\mathrm{{mm}}}}{{\mathrm{{d}}}} \right]$ ' if var=='pr' else rf'Deviation from MPI-GE $\Delta \mathrm{{TMn}}_\mathrm{{{region}}}$ [°C]' if var=='tmn' else 'fill var label')
        ax.axhline(0, color='black' if not plot_bg=='black' else 'white', lw=1., ls='--')
        ax.axvline(0, color='black' if not plot_bg=='black' else 'white', lw=1., ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines["left"].set_position(("axes", -0.02))

        if ext_ax is not None:
            return

        if is_latlon:
            plt.title(f"Location: {ssphos_var.lat.isel(lat=lat).values:.2f}°N, {ssphos_var.lon.isel(lon=lon).values:.2f}°E", loc='left', fontsize=12)

        make_legend(fig, ssp, hos_type, hos_strength=hos_strength, line_spacing=0.023, pos=(0.65, 0.58), col_width=0.12, plot_bg=plot_bg, incl_ge=False, markers=True)

        fig.savefig(f'../plots/regression-{regressions}_ssp-{ssp}_hos-{hos_type}_strength-{hos_strength}_window-{window}_region-{region}_lat-{lat}_lon-{lon}_bg-{plot_bg}_{season}.png', dpi=400, bbox_inches='tight', transparent=True if plot_bg=='black' else False)

    if not is_ocean:
        return np.round(combined_intercept, 4), np.round(combined_coef, 4), np.round(combined_ste, 5), np.round(combined_rsq, 4)
    else:
        return np.nan, np.nan, np.nan, np.nan
    
def net_cooling_point_map(reg_ds, ssp, T_ref='pi', season='', title=False, plot_bg='white', ext_ax=None, custom_cmap=None, weakening_primary=False):

    mid_gray = '#b0b0b0' # '#b0b0b0', "#989898"

    plt.style.use('default')
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    if ext_ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.Robinson(central_longitude=8)})
    else:
        ax = ext_ax
    ax.spines['geo'].set_visible(False)
    ax.set_rasterized(True)
    ax.coastlines(color='black' if not plot_bg=='black' else 'white', linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS, edgecolor='black' if not plot_bg=='black' else 'white', linewidth=.5)
    ax.add_feature(cfeature.LAND, facecolor=mid_gray)
    ax.add_feature(cfeature.OCEAN, facecolor='white' if not plot_bg=='black' else '#191919', zorder=2) # standard ocean: #A6CAE0

    base_cmap = cm.YlGnBu # options: PuBu, YlGnBu, cool
    mod_cmap = truncate_colormap(base_cmap, minval=0.17, maxval=1.0, darken=1.) # darkens for darken < 1, otherwise brightens
    cmap = mod_cmap if custom_cmap is None else custom_cmap

    if weakening_primary:
        levels = np.linspace(0, 100, 11)
        cmap = cmap.reversed()
        cmap.set_under(mid_gray)
        cmap.set_over(mid_gray)
    else:
        upper_bound = 18
        levels = np.linspace(0, upper_bound, 10)
        cmap.set_under(mid_gray)
        cmap.set_over(mid_gray)

    norm = BoundaryNorm(levels, ncolors=plt.get_cmap('PuBu').N, clip=False)

    if T_ref == 'pi':
        if 'season' in reg_ds.dims:
            data = reg_ds.req_weakening_pi.sel(scenar=ssp, season=season) if weakening_primary else reg_ds.req_strength_pi.sel(scenar=ssp, season=season)
        else:
            data = reg_ds.req_weakening_pi.sel(scenar=ssp) if weakening_primary else reg_ds.req_strength_pi.sel(scenar=ssp)
            if season != '':
                print('Seasonal net cooling not available!')
    else:
        if 'season' in reg_ds.dims:
            data = reg_ds.req_weakening_pd.sel(scenar=ssp, season=season) if weakening_primary else reg_ds.req_strength_pd.sel(scenar=ssp, season=season)
        else:
            data = reg_ds.req_weakening_pd.sel(scenar=ssp) if weakening_primary else reg_ds.req_strength_pd.sel(scenar=ssp)
            if season != '':
                print('Seasonal net cooling not available!')

    contour = ax.contourf(
        reg_ds.lon, reg_ds.lat, data,
        levels=levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='neither'
    )
    #  w/o Iceland: -9.5
    ax.set_extent([-18.5, 38.5, 34.8, 72], crs=ccrs.PlateCarree())
    
    # add_square(ax, 39.5, 65,  64.5, 75.5, colour='#191919' if plot_bg=='black' else 'white') # cut off North Russia where previously there was no data
    add_square(ax, -13.5, 0,  60, 75.5, colour='#191919' if plot_bg=='black' else 'white') # get rid of Nordic sea small islands
    add_square(ax, -30, -18,  68, 75, colour='#191919' if plot_bg=='black' else 'white') # get rid of Greenland

    if ext_ax is not None:
        return contour

    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.08, aspect=30, extend='neither')

    if weakening_primary:
        cbar.ax.invert_yaxis()
        cbar.set_ticks(list(range(0, 101, 10)))
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0f}%"))
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')
        cbar.set_label('AMOC weakening', fontsize=12, labelpad=2, loc='bottom')
        secax = cbar.ax.secondary_yaxis('left', functions=(convert_weakening_to_strength, convert_strength_to_weakening))
        secax.set_ylabel('AMOC strength [Sv]', fontsize=12.8, labelpad=0, loc='bottom')
        secax.set_yticks(list(range(0, 19, 2)))
        secax.set_yticklabels([f'{s} Sv' for s in range(0, 19, 2)])
    else:
        cbar.set_label('AMOC strength [Sv]', fontsize=12, labelpad=2, loc='bottom')
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.yaxis.set_ticks_position('left')
        secax = cbar.ax.secondary_yaxis('right', functions=(convert_strength_to_weakening, convert_weakening_to_strength))
        secax.set_ylabel('AMOC weakening', fontsize=12.8, labelpad=0, loc='bottom')
        secax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        secax.set_yticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])

    if weakening_primary:
        rect_y = 1 - AMOC_extent[ssp][0] / 100
        rect_height = (AMOC_extent[ssp][0] - AMOC_extent[ssp][1]) / 100
        text_y = 1 - (AMOC_extent[ssp][0] + AMOC_extent[ssp][1]) / 200
    else:
        rect_y = (1 - AMOC_extent[ssp][0]/100) / (1 - convert_strength_to_weakening(upper_bound)/100)
        rect_height = ((AMOC_extent[ssp][0]-AMOC_extent[ssp][1])/100) / (1 - convert_strength_to_weakening(upper_bound)/100)
        text_y = (1 - (AMOC_extent[ssp][0] + AMOC_extent[ssp][1]) / 200) / (1 - convert_strength_to_weakening(upper_bound)/100)

    rect = Rectangle(
        (-3.2, rect_y),
        1.2,                   # width of the patch
        rect_height, # height in normalized coords
        transform=cbar.ax.transAxes,
        color=hosing_colors[ssp]['ge'],
        alpha=1.0,
        clip_on=False
    )
    cbar.ax.add_patch(rect)

    # Add vertical text inside the rectangle
    cbar.ax.text(
        -2.55,  # x position
        text_y,
        "2100 CMIP6 range",
        color='white',
        fontsize=10,
        rotation=90,
        va='center',
        ha='center',
        transform=cbar.ax.transAxes,
        zorder=10
    )

    fig.savefig(f'../plots/net_cooling_maps_{ssp}_{T_ref}_bg-{plot_bg}.png', dpi=400, bbox_inches='tight', transparent=True if plot_bg=='black' else False)

    T_ref_str = 'preindustrial' if T_ref == 'pi' else 'present-day'
    if title:
            plt.title(f'Compensating warming since {T_ref_str} under {ssp}')

def plot_regression_coefficients(reg_ds, season='', var='tas', std_error=False, ste_relative=False, plot_bg='white', custom_cmap=cm.ocean_r, colorbar_v_max=0.7, ext_ax=None, savefig=True):

    if var =='tas':
        reg_ds = reg_ds.sel(season=season)
    else:
        if season != '':
            print('Seasonal regression coefficients only available for tas.')
            return

    plt.style.use('default')
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    if ext_ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.Robinson(central_longitude=8)})
    else:
        ax = ext_ax
    ax.spines['geo'].set_visible(False)
    ax.set_rasterized(True)
    ax.coastlines(color='black' if not plot_bg=='black' else 'white', linewidth=.5, zorder=3)
    ax.add_feature(cfeature.BORDERS, edgecolor='black' if not plot_bg=='black' else 'white', linewidth=.5)
    ax.add_feature(cfeature.LAND, facecolor='white' if not plot_bg=='black' else '#191919')
    ax.add_feature(cfeature.OCEAN, facecolor='white' if not plot_bg=='black' else '#191919', zorder=2)

    # truncated colormap and values
    v_max = colorbar_v_max
    if std_error:
        cmap = cm.viridis
        levels = np.linspace(0.0, v_max, 9)
    else:
        if var == 'tas':
            trunc_cmap = truncate_colormap(custom_cmap, maxval=(v_max/1.5))
            levels = np.linspace(0.0, v_max, int(10*v_max)+1)
            norm = BoundaryNorm(levels, ncolors=trunc_cmap.N, clip=False)
        elif var == 'pr':
            trunc_cmap = truncate_colormap(cm.BrBG_r, minval=0.3)
            levels = np.linspace(-v_max*2/5, v_max, int(100*v_max)+1)
            norm = BoundaryNorm(levels, ncolors=trunc_cmap.N, clip=False)

    if std_error:
        if ste_relative:
            # Relative standard error: ste / coef * 100 (in %)
            plot_data = reg_ds.ste_ensmean / reg_ds.coef_ensmean * 100
        else:
            # Absolute standard error: scale from °C/Sv to °C per 10% weakening
            plot_data = reg_ds.ste_ensmean * AMOC_pi_MPI / 10
        contour = ax.contourf(
            reg_ds.lon, reg_ds.lat, plot_data,
            transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, extend='max' if ste_relative else 'max'
        )
    else:
        contour = ax.contourf(
            reg_ds.lon, reg_ds.lat, AMOC_pi_MPI/10 * reg_ds.coef_ensmean,
            transform=ccrs.PlateCarree(), cmap=trunc_cmap, levels=levels, norm=norm, extend='neither'
        )

    ax.set_extent([-18.5, 38.5, 34.8, 72], crs=ccrs.PlateCarree())

    # add_square(ax, 39.5, 65,  64.5, 75.5, colour='#191919' if plot_bg=='black' else 'white') # cut off North Russia where previously there was no data
    add_square(ax, -13.5, 0,  60, 75.5, colour='#191919' if plot_bg=='black' else 'white') # get rid of Nordic sea small islands
    add_square(ax, -30, -18,  68, 75, colour='#191919' if plot_bg=='black' else 'white') # get rid of Greenland

    if ext_ax is not None:
        return contour

    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.02, aspect=30, extend='max' if std_error else 'neither')
    if var == 'tas':
        if std_error:
            if ste_relative:
                cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{np.round(x, 1)}%"))
            else:
                cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{np.round(x, 4)}°C"))
        else:
            cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{np.round(x, 2)}°C"))
    elif var == 'pr':
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.2f} mm/d"))
    if std_error:
        if ste_relative:
            cbar.set_label('Relative standard error [%]', fontsize=14)
        else:
            cbar.set_label('Standard error per 10% AMOC weakening [°C]', fontsize=14)
    else:
        cbar.set_label('Cooling per 10% AMOC weakening' if var=='tas' else 'Drying per 10% AMOC weakening' if var=='pr' else 'fill in var description', fontsize=14)

    if std_error:
        ste_type = 'relative' if ste_relative else 'absolute'
        plt.title(f'Regression coefficient {ste_type} standard errors for MPI-ESM', fontsize=14)
    else:
        plt.title(f'Regression coefficients for MPI-ESM', fontsize=14)

    if savefig and ext_ax is None:
        if std_error:
            ste_type = 'relative' if ste_relative else 'absolute'
            fig.savefig(f'../plots/regression_std_errors_{ste_type}_EU_bg-{plot_bg}.pdf', dpi=400, bbox_inches='tight', transparent=True if plot_bg=='black' else False)
        else:
            fig.savefig(f'../plots/regression_coefficients_EU_bg-{plot_bg}.pdf', dpi=400, bbox_inches='tight', transparent=True if plot_bg=='black' else False)

########################################################################################################################
# MULTI-MODEL ANALYSIS
########################################################################################################################

# LOADING MULTI-MODEL DATA

def load_hosmip_data(apply_ocean_mask=False):
    """
    apply_ocean_mask only changes things for preindustrial mean tas calculation.
    """

    data_path = '/work/uo1075/m300817/hosing/nahosmip/post/'
    pi_data_path = '/work/uo1075/m300817/teu_amoc/data/CMIP6/picontrol/'
    local_path = '../data/'
    use_local = False

    print('Loading HOSMIP data...')

    print('Loading CanESM5 data...')
    # CanESM5
    canesm_03_amoc = xr.open_dataarray((local_path if use_local else data_path)+"CanESM5_u03-hos_amoc26_yr.nc", use_cftime=True).drop_vars('lev').to_dataset(name='amoc')

    canesm_03_tas = xr.open_dataarray((local_path if use_local else data_path)+"CanESM5_u03-hos_tas_yr.nc", use_cftime=True).to_dataset(name='tas') - 273.15
    canesm_03_tas_djf = xr.open_dataset((local_path if use_local else data_path)+"CanESM5_u03-hos_tas_djf.nc", use_cftime=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15
    canesm_03_tas_jja = xr.open_dataset((local_path if use_local else data_path)+"CanESM5_u03-hos_tas_jja.nc", use_cftime=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15

    canesm_03_tas.coords['lon'] = (canesm_03_tas.coords['lon'] + 180) % 360 - 180
    canesm_03_tas = canesm_03_tas.sortby(canesm_03_tas.lon)

    canesm_03_tas_djf.coords['lon'] = (canesm_03_tas_djf.coords['lon'] + 180) % 360 - 180
    canesm_03_tas_djf = canesm_03_tas_djf.sortby(canesm_03_tas_djf.lon)

    canesm_03_tas_jja.coords['lon'] = (canesm_03_tas_jja.coords['lon'] + 180) % 360 - 180
    canesm_03_tas_jja = canesm_03_tas_jja.sortby(canesm_03_tas_jja.lon)

    canesm_masks = define_region_masks(canesm_03_tas)
    canesm_mask_land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(canesm_03_tas)
    canesm_masks['LAND'] = canesm_mask_land

    canesm_03_tas = canesm_03_tas.where(canesm_masks['EU_buffer'], drop=True)
    canesm_03_tas_djf = canesm_03_tas_djf.where(canesm_masks['EU_buffer'], drop=True)
    canesm_03_tas_jja = canesm_03_tas_jja.where(canesm_masks['EU_buffer'], drop=True)

    print('Loading CESM2 data...')
    # CESM2
    cesm_03_amoc = xr.open_dataarray((local_path if use_local else data_path)+"CESM2_u03-hos_amoc26_yr.nc", use_cftime=True).drop_vars(['moc_z', 'moc_components']).to_dataset(name='amoc')

    cesm_03_tas = xr.open_dataarray((local_path if use_local else data_path)+"CESM2_u03-hos_tas_yr.nc", use_cftime=True).to_dataset(name='tas') - 273.15
    cesm_03_tas_djf = xr.open_dataset((local_path if use_local else data_path)+"CESM2_u03-hos_tas_djf.nc", use_cftime=True).tas.to_dataset(name='tas') - 273.15
    cesm_03_tas_jja = xr.open_dataset((local_path if use_local else data_path)+"CESM2_u03-hos_tas_jja.nc", use_cftime=True).tas.to_dataset(name='tas') - 273.15

    cesm_03_tas.coords['lon'] = (cesm_03_tas.coords['lon'] + 180) % 360 - 180
    cesm_03_tas = cesm_03_tas.sortby(cesm_03_tas.lon)

    cesm_03_tas_djf.coords['lon'] = (cesm_03_tas_djf.coords['lon'] + 180) % 360 - 180
    cesm_03_tas_djf = cesm_03_tas_djf.sortby(cesm_03_tas_djf.lon)

    cesm_03_tas_jja.coords['lon'] = (cesm_03_tas_jja.coords['lon'] + 180) % 360 - 180
    cesm_03_tas_jja = cesm_03_tas_jja.sortby(cesm_03_tas_jja.lon)

    cesm_masks = define_region_masks(cesm_03_tas)
    cesm_mask_land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(cesm_03_tas)
    cesm_masks['LAND'] = cesm_mask_land

    cesm_03_tas = cesm_03_tas.where(cesm_masks['EU_buffer'], drop=True)
    cesm_03_tas_djf = cesm_03_tas_djf.where(cesm_masks['EU_buffer'], drop=True)
    cesm_03_tas_jja = cesm_03_tas_jja.where(cesm_masks['EU_buffer'], drop=True)

    print('Loading EC-Earth3 data...')
    # EC-Earth3
    ecearth_03_amoc = xr.open_dataarray((local_path if use_local else data_path)+"EC-Earth3_u03-hos_amoc26_yr.nc", use_cftime=True).drop_vars(['lev', 'sector']).to_dataset(name='amoc')

    ecearth_03_tas = xr.open_dataarray((local_path if use_local else data_path)+"EC-Earth3_u03-hos_tas_yr.nc", use_cftime=True).to_dataset(name='tas') - 273.15
    ecearth_03_tas_djf = xr.open_dataset((local_path if use_local else data_path)+"EC-Earth3_u03-hos_tas_djf.nc", use_cftime=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15
    ecearth_03_tas_jja = xr.open_dataset((local_path if use_local else data_path)+"EC-Earth3_u03-hos_tas_jja.nc", use_cftime=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15

    ecearth_03_tas.coords['lon'] = (ecearth_03_tas.coords['lon'] + 180) % 360 - 180
    ecearth_03_tas = ecearth_03_tas.sortby(ecearth_03_tas.lon)

    ecearth_03_tas_djf.coords['lon'] = (ecearth_03_tas_djf.coords['lon'] + 180) % 360 - 180
    ecearth_03_tas_djf = ecearth_03_tas_djf.sortby(ecearth_03_tas_djf.lon)

    ecearth_03_tas_jja.coords['lon'] = (ecearth_03_tas_jja.coords['lon'] + 180) % 360 - 180
    ecearth_03_tas_jja = ecearth_03_tas_jja.sortby(ecearth_03_tas_jja.lon)

    ecearth_masks = define_region_masks(ecearth_03_tas)
    ecearth_mask_land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(ecearth_03_tas)
    ecearth_masks['LAND'] = ecearth_mask_land

    ecearth_03_tas = ecearth_03_tas.where(ecearth_masks['EU_buffer'], drop=True)
    ecearth_03_tas_djf = ecearth_03_tas_djf.where(ecearth_masks['EU_buffer'], drop=True)
    ecearth_03_tas_jja = ecearth_03_tas_jja.where(ecearth_masks['EU_buffer'], drop=True)

    print('Loading HadGEM3-GC3-1LL data...')
    # HadGEM3-GC3-1LL
    hadgem1ll_03_amoc = xr.open_dataarray((local_path if use_local else data_path)+"HadGEM3-GC3-1LL_u03-hos_amoc26_yr.nc", use_cftime=True).drop_vars(['nav_lat', 'nav_lon', 'depthw', 'time_centered']).to_dataset(name='amoc')

    hadgem1ll_03_tas = xr.open_dataarray((local_path if use_local else data_path)+"HadGEM3-GC3-1LL_u03-hos_tas_yr.nc", use_cftime=True).to_dataset(name='tas') - 273.15
    hadgem1ll_03_tas_djf = xr.open_dataset((local_path if use_local else data_path)+"HadGEM3-GC3-1LL_u03-hos_tas_djf.nc", use_cftime=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15
    hadgem1ll_03_tas_jja = xr.open_dataset((local_path if use_local else data_path)+"HadGEM3-GC3-1LL_u03-hos_tas_jja.nc", use_cftime=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15

    hadgem1ll_03_tas.coords['lon'] = (hadgem1ll_03_tas.coords['lon'] + 180) % 360 - 180
    hadgem1ll_03_tas = hadgem1ll_03_tas.sortby(hadgem1ll_03_tas.lon)

    hadgem1ll_03_tas_djf.coords['lon'] = (hadgem1ll_03_tas_djf.coords['lon'] + 180) % 360 - 180
    hadgem1ll_03_tas_djf = hadgem1ll_03_tas_djf.sortby(hadgem1ll_03_tas_djf.lon)

    hadgem1ll_03_tas_jja.coords['lon'] = (hadgem1ll_03_tas_jja.coords['lon'] + 180) % 360 - 180
    hadgem1ll_03_tas_jja = hadgem1ll_03_tas_jja.sortby(hadgem1ll_03_tas_jja.lon)

    hadgem1ll_masks = define_region_masks(hadgem1ll_03_tas)
    hadgem1ll_mask_land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(hadgem1ll_03_tas)
    hadgem1ll_masks['LAND'] = hadgem1ll_mask_land

    hadgem1ll_03_tas = hadgem1ll_03_tas.where(hadgem1ll_masks['EU_buffer'], drop=True)
    hadgem1ll_03_tas_djf = hadgem1ll_03_tas_djf.where(hadgem1ll_masks['EU_buffer'], drop=True)
    hadgem1ll_03_tas_jja = hadgem1ll_03_tas_jja.where(hadgem1ll_masks['EU_buffer'], drop=True)

    print('Loading HadGEM3-GC3-1MM data...')
    # HadGEM3-GC3-1MM
    hadgem1mm_03_amoc = xr.open_dataarray((local_path if use_local else data_path)+"HadGEM3-GC3-1MM_u03-hos_amoc26_yr.nc", use_cftime=True).drop_vars(['nav_lat', 'nav_lon', 'depthw', 'time_centered']).to_dataset(name='amoc')

    hadgem1mm_03_tas = xr.open_dataarray((local_path if use_local else data_path)+"HadGEM3-GC3-1MM_u03-hos_tas_yr.nc", use_cftime=True).to_dataset(name='tas') - 273.15
    hadgem1mm_03_tas_djf = xr.open_dataset((local_path if use_local else data_path)+"HadGEM3-GC3-1MM_u03-hos_tas_djf.nc", use_cftime=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15
    hadgem1mm_03_tas_jja = xr.open_dataset((local_path if use_local else data_path)+"HadGEM3-GC3-1MM_u03-hos_tas_jja.nc", use_cftime=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15

    hadgem1mm_03_tas.coords['lon'] = (hadgem1mm_03_tas.coords['lon'] + 180) % 360 - 180
    hadgem1mm_03_tas = hadgem1mm_03_tas.sortby(hadgem1mm_03_tas.lon)

    hadgem1mm_03_tas_djf.coords['lon'] = (hadgem1mm_03_tas_djf.coords['lon'] + 180) % 360 - 180
    hadgem1mm_03_tas_djf = hadgem1mm_03_tas_djf.sortby(hadgem1mm_03_tas_djf.lon)

    hadgem1mm_03_tas_jja.coords['lon'] = (hadgem1mm_03_tas_jja.coords['lon'] + 180) % 360 - 180
    hadgem1mm_03_tas_jja = hadgem1mm_03_tas_jja.sortby(hadgem1mm_03_tas_jja.lon)

    hadgem1mm_masks = define_region_masks(hadgem1mm_03_tas)
    hadgem1mm_mask_land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(hadgem1mm_03_tas)
    hadgem1mm_masks['LAND'] = hadgem1mm_mask_land

    hadgem1mm_03_tas = hadgem1mm_03_tas.where(hadgem1mm_masks['EU_buffer'], drop=True)
    hadgem1mm_03_tas_djf = hadgem1mm_03_tas_djf.where(hadgem1mm_masks['EU_buffer'], drop=True)
    hadgem1mm_03_tas_jja = hadgem1mm_03_tas_jja.where(hadgem1mm_masks['EU_buffer'], drop=True)

    print('Loading IPSL-CM6A-LR data...')
    # IPSL-CM6A-LR
    ipsl_03_amoc = xr.open_dataarray((local_path if use_local else data_path)+"IPSL-CM6A-LR_u03-hos_amoc26_yr.nc", use_cftime=True).drop_vars(['nav_lat', 'nav_lon', 'olevel']).to_dataset(name='amoc')

    ipsl_03_tas = xr.open_dataarray((local_path if use_local else data_path)+"IPSL-CM6A-LR_u03-hos_tas_yr.nc", use_cftime=True).to_dataset(name='tas') - 273.15
    ipsl_03_tas_djf = xr.open_dataset((local_path if use_local else data_path)+"IPSL-CM6A-LR_u03-hos_tas_djf.nc", use_cftime=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15
    ipsl_03_tas_jja = xr.open_dataset((local_path if use_local else data_path)+"IPSL-CM6A-LR_u03-hos_tas_jja.nc", use_cftime=True).tas.to_dataset(name='tas').drop_vars('height') - 273.15

    ipsl_03_tas.coords['lon'] = (ipsl_03_tas.coords['lon'] + 180) % 360 - 180
    ipsl_03_tas = ipsl_03_tas.sortby(ipsl_03_tas.lon)

    ipsl_03_tas_djf.coords['lon'] = (ipsl_03_tas_djf.coords['lon'] + 180) % 360 - 180
    ipsl_03_tas_djf = ipsl_03_tas_djf.sortby(ipsl_03_tas_djf.lon)

    ipsl_03_tas_jja.coords['lon'] = (ipsl_03_tas_jja.coords['lon'] + 180) % 360 - 180
    ipsl_03_tas_jja = ipsl_03_tas_jja.sortby(ipsl_03_tas_jja.lon)

    ipsl_masks = define_region_masks(ipsl_03_tas)
    ipsl_mask_land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(ipsl_03_tas)
    ipsl_masks['LAND'] = ipsl_mask_land

    ipsl_03_tas = ipsl_03_tas.where(ipsl_masks['EU_buffer'], drop=True)
    ipsl_03_tas_djf = ipsl_03_tas_djf.where(ipsl_masks['EU_buffer'], drop=True)
    ipsl_03_tas_jja = ipsl_03_tas_jja.where(ipsl_masks['EU_buffer'], drop=True)

    print('Loading MPI-ESM1-2-HR data...')
    # MPI-ESM1-2-HR
    mpiesmhr_03_amoc = xr.open_dataarray((local_path if use_local else data_path)+"MPI-ESM1-2-HR_u03-hos_amoc26_yr.nc", use_cftime=True).drop_vars(['depth_2']).to_dataset(name='amoc')

    mpiesmhr_03_tas = xr.open_dataarray((local_path if use_local else data_path)+"MPI-ESM1-2-HR_u03-hos_tas_yr.nc", use_cftime=True).to_dataset(name='tas') - 273.15
    mpiesmhr_03_tas_djf = xr.open_dataset((local_path if use_local else data_path)+"MPI-ESM1-2-HR_u03-hos_tas_djf.nc", use_cftime=True).tas.to_dataset(name='tas') - 273.15
    mpiesmhr_03_tas_jja = xr.open_dataset((local_path if use_local else data_path)+"MPI-ESM1-2-HR_u03-hos_tas_jja.nc", use_cftime=True).tas.to_dataset(name='tas') - 273.15

    mpiesmhr_03_tas.coords['lon'] = (mpiesmhr_03_tas.coords['lon'] + 180) % 360 - 180
    mpiesmhr_03_tas = mpiesmhr_03_tas.sortby(mpiesmhr_03_tas.lon)

    mpiesmhr_03_tas_djf.coords['lon'] = (mpiesmhr_03_tas_djf.coords['lon'] + 180) % 360 - 180
    mpiesmhr_03_tas_djf = mpiesmhr_03_tas_djf.sortby(mpiesmhr_03_tas_djf.lon)

    mpiesmhr_03_tas_jja.coords['lon'] = (mpiesmhr_03_tas_jja.coords['lon'] + 180) % 360 - 180
    mpiesmhr_03_tas_jja = mpiesmhr_03_tas_jja.sortby(mpiesmhr_03_tas_jja.lon)

    mpiesmhr_masks = define_region_masks(mpiesmhr_03_tas)
    mpiesmhr_mask_land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(mpiesmhr_03_tas)
    mpiesmhr_masks['LAND'] = mpiesmhr_mask_land

    mpiesmhr_03_tas = mpiesmhr_03_tas.where(mpiesmhr_masks['EU_buffer'], drop=True)
    mpiesmhr_03_tas_djf = mpiesmhr_03_tas_djf.where(mpiesmhr_masks['EU_buffer'], drop=True)
    mpiesmhr_03_tas_jja = mpiesmhr_03_tas_jja.where(mpiesmhr_masks['EU_buffer'], drop=True)

    print('Loading MPI-ESM1-2-LR data...')
    # MPI-ESM1-2-LR
    mpiesmlr_03_amoc = xr.open_dataarray((local_path if use_local else data_path)+"MPI-ESM1-2-LR_u03-hos_amoc26_yr.nc", use_cftime=True).drop_vars(['depth_2']).to_dataset(name='amoc')

    mpiesmlr_03_tas = xr.open_dataarray((local_path if use_local else data_path)+"MPI-ESM1-2-LR_u03-hos_tas_yr.nc", use_cftime=True).to_dataset(name='tas') - 273.15
    mpiesmlr_03_tas_djf = xr.open_dataset((local_path if use_local else data_path)+"MPI-ESM1-2-LR_u03-hos_tas_djf.nc", use_cftime=True).tas.to_dataset(name='tas') - 273.15
    mpiesmlr_03_tas_jja = xr.open_dataset((local_path if use_local else data_path)+"MPI-ESM1-2-LR_u03-hos_tas_jja.nc", use_cftime=True).tas.to_dataset(name='tas') - 273.15

    mpiesmlr_03_tas.coords['lon'] = (mpiesmlr_03_tas.coords['lon'] + 180) % 360 - 180
    mpiesmlr_03_tas = mpiesmlr_03_tas.sortby(mpiesmlr_03_tas.lon)

    mpiesmlr_03_tas_djf.coords['lon'] = (mpiesmlr_03_tas_djf.coords['lon'] + 180) % 360 - 180
    mpiesmlr_03_tas_djf = mpiesmlr_03_tas_djf.sortby(mpiesmlr_03_tas_djf.lon)

    mpiesmlr_03_tas_jja.coords['lon'] = (mpiesmlr_03_tas_jja.coords['lon'] + 180) % 360 - 180
    mpiesmlr_03_tas_jja = mpiesmlr_03_tas_jja.sortby(mpiesmlr_03_tas_jja.lon)

    mpiesmlr_masks = define_region_masks(mpiesmlr_03_tas)
    mpiesmlr_mask_land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(mpiesmlr_03_tas)
    mpiesmlr_masks['LAND'] = mpiesmlr_mask_land

    mpiesmlr_03_tas = mpiesmlr_03_tas.where(mpiesmlr_masks['EU_buffer'], drop=True)
    mpiesmlr_03_tas_djf = mpiesmlr_03_tas_djf.where(mpiesmlr_masks['EU_buffer'], drop=True)
    mpiesmlr_03_tas_jja = mpiesmlr_03_tas_jja.where(mpiesmlr_masks['EU_buffer'], drop=True)

    amoc_data= {
        'CanESM5': canesm_03_amoc,
        'EC-Earth3': ecearth_03_amoc,
        'CESM2': cesm_03_amoc,
        'MPI-ESM1-2-LR': mpiesmlr_03_amoc,
        'HadGEM3-GC3-1MM': hadgem1mm_03_amoc,
        'HadGEM3-GC3-1LL': hadgem1ll_03_amoc,
        'MPI-ESM1-2-HR': mpiesmhr_03_amoc,
        'IPSL-CM6A-LR': ipsl_03_amoc
    }

    tas_data = {}
    tas_data[''] = {
        'CanESM5': canesm_03_tas,
        'EC-Earth3': ecearth_03_tas,
        'CESM2': cesm_03_tas,
        'MPI-ESM1-2-LR': mpiesmlr_03_tas,
        'HadGEM3-GC3-1MM': hadgem1mm_03_tas,
        'HadGEM3-GC3-1LL': hadgem1ll_03_tas,
        'MPI-ESM1-2-HR': mpiesmhr_03_tas,
        'IPSL-CM6A-LR': ipsl_03_tas
    }
    tas_data['djf'] = {
        'CanESM5': canesm_03_tas_djf,
        'EC-Earth3': ecearth_03_tas_djf,
        'CESM2': cesm_03_tas_djf,
        'MPI-ESM1-2-LR': mpiesmlr_03_tas_djf,
        'HadGEM3-GC3-1MM': hadgem1mm_03_tas_djf,
        'HadGEM3-GC3-1LL': hadgem1ll_03_tas_djf,
        'MPI-ESM1-2-HR': mpiesmhr_03_tas_djf,
        'IPSL-CM6A-LR': ipsl_03_tas_djf
    }
    tas_data['jja'] = {
        'CanESM5': canesm_03_tas_jja,
        'EC-Earth3': ecearth_03_tas_jja,
        'CESM2': cesm_03_tas_jja,
        'MPI-ESM1-2-LR': mpiesmlr_03_tas_jja,
        'HadGEM3-GC3-1MM': hadgem1mm_03_tas_jja,
        'HadGEM3-GC3-1LL': hadgem1ll_03_tas_jja,
        'MPI-ESM1-2-HR': mpiesmhr_03_tas_jja,
        'IPSL-CM6A-LR': ipsl_03_tas_jja
    }

    masks = {
        'CanESM5': canesm_masks,
        'EC-Earth3': ecearth_masks,
        'CESM2': cesm_masks,
        'MPI-ESM1-2-LR': mpiesmlr_masks,
        'HadGEM3-GC3-1MM': hadgem1mm_masks,
        'HadGEM3-GC3-1LL': hadgem1ll_masks,
        'MPI-ESM1-2-HR': mpiesmhr_masks,
        'IPSL-CM6A-LR': ipsl_masks
    }

    data_cutoffs = {}
    data_cutoffs[''] = {
        'CanESM5': [0, 0],
        'EC-Earth3': [0, 0],
        'CESM2': [0, 1],
        'MPI-ESM1-2-LR': [0, 0],
        'HadGEM3-GC3-1MM': [0, 1],
        'HadGEM3-GC3-1LL': [45, 0],
        'MPI-ESM1-2-HR': [0, 0],
        'IPSL-CM6A-LR': [0, 50]
    }
    data_cutoffs['jja'] = {
        'CanESM5': [0, 0],
        'EC-Earth3': [0, 0],
        'CESM2': [0, 28],
        'MPI-ESM1-2-LR': [0, 0],
        'HadGEM3-GC3-1MM': [0, 1],
        'HadGEM3-GC3-1LL': [45, 0],
        'MPI-ESM1-2-HR': [0, 0],
        'IPSL-CM6A-LR': [0, 50]
    }
    data_cutoffs['djf'] = {
        'CanESM5': [0, 0],
        'EC-Earth3': [0, 0],
        'CESM2': [0, 28],
        'MPI-ESM1-2-LR': [0, 0],
        'HadGEM3-GC3-1MM': [0, 1],
        'HadGEM3-GC3-1LL': [45, 0],
        'MPI-ESM1-2-HR': [0, 0],
        'IPSL-CM6A-LR': [0, 50]
    }

    pi_data_path = '/work/uo1075/m300817/teu_amoc/data/CMIP6/picontrol/'
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

    hosmip_realiz_dict = {
        'MPI-ESM1-2-HR': 'r1i1p1f1', 
        'MPI-ESM1-2-LR': 'r1i1p1f1',
        'CanESM5': 'r1i1p1f1',
        'CESM2': 'r1i1p1f1',
        'HadGEM3-GC31-LL': 'r1i1p1f1',
        'HadGEM3-GC31-MM': 'r1i1p1f1',
        'IPSL-CM6A-LR': 'r1i2p1f1',
        'EC-Earth3': 'r1i1p1f1'
        }

    hosmip_amoc_pi_data = {}
    hosmip_tas_pi_data = {}
    for model in hosmip_realiz_dict.keys():
        print('Loading preindustrial data from '+model)
        
        hosmip_amoc_pi_data[model] = xr.open_dataarray(pi_data_path+f'{model.lower()}/{model.lower()}_{hosmip_realiz_dict[model]}_amoc26_yr.nc', decode_times=time_coder).drop_vars('lev' if model != 'IPSL-CM6A-LR' else 'olevel').to_dataset(name='amoc')

        for season in ['', 'djf', 'jja']:
            if season == '':
                raw_tas_data = xr.open_dataarray(pi_data_path+f'{model.lower()}/{model.lower()}_{hosmip_realiz_dict[model]}_tas_yr.nc', decode_times=time_coder).to_dataset(name='tas') - 273.15

                if model in ['CESM2', 'CanESM5', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM']:
                    new_time = xr.cftime_range(start=str(raw_tas_data.time.values[0]), freq='YS', periods=raw_tas_data.sizes['time'], calendar='proleptic_gregorian')
                    raw_tas_data = raw_tas_data.assign_coords(time=new_time)

                raw_tas_data.coords['lon'] = (raw_tas_data.coords['lon'] + 180) % 360 - 180
                raw_tas_data = raw_tas_data.sortby(raw_tas_data.lon)

                raw_tas_data.coords['season'] = season
                raw_tas_data = raw_tas_data.expand_dims('season', axis=-1)

                all_season_tas_data = raw_tas_data
                
                mask_land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(raw_tas_data)
            else:
                raw_tas_data = xr.open_dataset(pi_data_path+f'{model.lower()}/tas_{season}/{model.lower()}_{hosmip_realiz_dict[model]}_tas_{season}.nc', decode_times=time_coder) - 273.15

                model_key = 'HadGEM3-GC3-1LL' if model == 'HadGEM3-GC31-LL' else ('HadGEM3-GC3-1MM' if model == 'HadGEM3-GC31-MM' else model)
                if 'dim' in drop_stuff[model_key].keys():
                    raw_tas_data = raw_tas_data.drop_dims(drop_stuff[model_key]['dim'])
                if 'var' in drop_stuff[model_key].keys():
                    raw_tas_data = raw_tas_data.drop_vars(drop_stuff[model_key]['var'])

                raw_tas_data.coords['lon'] = (raw_tas_data.coords['lon'] + 180) % 360 - 180
                raw_tas_data = raw_tas_data.sortby(raw_tas_data.lon)

                raw_tas_data.coords['season'] = season
                raw_tas_data = raw_tas_data.expand_dims('season', axis=-1)

                all_season_tas_data = xr.concat([all_season_tas_data, raw_tas_data], dim='season')

            hosmip_tas_pi_data[model] = all_season_tas_data.where(mask_land==0, drop=True) if apply_ocean_mask else all_season_tas_data

    # rename HadGEM model keys
    hosmip_amoc_pi_data['HadGEM3-GC3-1LL'] = hosmip_amoc_pi_data.pop('HadGEM3-GC31-LL')
    hosmip_amoc_pi_data['HadGEM3-GC3-1MM'] = hosmip_amoc_pi_data.pop('HadGEM3-GC31-MM')
    hosmip_tas_pi_data['HadGEM3-GC3-1LL'] = hosmip_tas_pi_data.pop('HadGEM3-GC31-LL')
    hosmip_tas_pi_data['HadGEM3-GC3-1MM'] = hosmip_tas_pi_data.pop('HadGEM3-GC31-MM')

    return amoc_data, hosmip_amoc_pi_data, tas_data, hosmip_tas_pi_data, masks, data_cutoffs

def build_hosmip_dataset(amoc_data, hosmip_amoc_pi_data, tas_data, hosmip_tas_pi_data, masks, data_cutoffs):

    multi_model_dict = {}

    print('Building combined HosMIP dataset...')

    for model in hosmip_labels:
        print('Processing model: ', model)

        for season in ['', 'djf', 'jja']:

            amoc = amoc_data[model] if data_cutoffs[season][model][0] == 0 else amoc_data[model].isel(time=slice(0, -data_cutoffs[season][model][0]))
            tas = tas_data[season][model] if data_cutoffs[season][model][1] == 0 else tas_data[season][model].isel(time=slice(0, -data_cutoffs[season][model][1]))

            masks_ds = xr.concat([masks[model][k] for k in masks[model].keys()], 
                                dim=pd.Index(list(masks[model].keys()), name='region'))
            masks_ds = masks_ds.where(masks[model]['EU_buffer'], drop=True)

            amoc_tas = xr.merge([amoc, tas])
            amoc_tas.coords['season'] = season
            amoc_tas.coords['scenar'] = 'pi'
            amoc_tas.coords['type'] = 'hosing'

            amoc_tas = amoc_tas.expand_dims(['scenar', 'season', 'type'], axis=[-1, -2, -3])

            if season == '':
                amoc_tas_all = amoc_tas
            else:
                amoc_tas_all = xr.merge([amoc_tas_all, amoc_tas])

        ctrl_amoc = hosmip_amoc_pi_data[model].mean(dim='time')
        ctrl_tas = hosmip_tas_pi_data[model].where(masks[model]['EU_buffer'], drop=True).mean(dim='time')

        model_ctrl = xr.merge([ctrl_amoc, ctrl_tas])

        model_ctrl.coords['scenar'] = 'pi'
        model_ctrl.coords['type'] = 'control'
        model_ctrl = model_ctrl.expand_dims(['scenar', 'type'], axis=[-1, -2])

        amoc_tas_all = xr.merge([amoc_tas_all, model_ctrl]) 

        amoc_tas_masks = xr.merge([amoc_tas_all, masks_ds])

        multi_model_dict[model] = amoc_tas_masks
    
    return multi_model_dict

def get_other_studies_data(masks):

    print('Loading Bellomo & Mehling (2024) data...')

    data_f4_tas = xr.open_dataset(local_path + 'Bellomo2024_plotting_data/data_tas_fig4.nc')

    data_f4_tas.coords['lon'] = (data_f4_tas.coords['lon'] + 180) % 360 - 180
    data_f4_tas = data_f4_tas.sortby(data_f4_tas.lon)

    tas_4x = data_f4_tas.tas_4x.where(masks['EC-Earth3']['EU'], drop=True).drop_vars('height')
    tas_hosing = data_f4_tas.tas_hosing.where(masks['EC-Earth3']['EU'], drop=True).drop_vars('height')

    tas_4x = tas_4x.assign_coords(scenar='ghg', type='diff')
    tas_hosing = tas_hosing.assign_coords(scenar='pi', type='diff')

    tas_diff = xr.concat([tas_4x, tas_hosing], dim='scenar')
    tas_diff = tas_diff.expand_dims('type', axis=-1)

    tas_diff = tas_diff.transpose('lat', 'lon', 'scenar', 'type')
    tas_diff = tas_diff.assign_coords(season='')
    tas_diff = tas_diff.expand_dims('season', axis=-3)
    tas_diff = tas_diff.transpose('lat', 'lon', 'season', 'scenar', 'type')

    bm_data = tas_diff.to_dataset(name='tas')

    amoc_4x = xr.DataArray(8.2)
    amoc_hosing = xr.DataArray(10.2)

    amoc_4x = amoc_4x.assign_coords(scenar='ghg', type='diff')
    amoc_hosing = amoc_hosing.assign_coords(scenar='pi', type='diff')

    amoc_diff = xr.concat([amoc_4x, amoc_hosing], dim='scenar')
    amoc_diff = amoc_diff.expand_dims('type', axis=-1)

    amoc_diff = amoc_diff.assign_coords(season='')
    amoc_diff = amoc_diff.expand_dims('season', axis=-3)
    amoc_diff = amoc_diff.transpose('season', 'scenar', 'type')

    bm_data['amoc'] = amoc_diff
    
    ########################################

    print('Loading van Westen & Baatsten (2025) data...')

    tas_vwb_600_pi = xr.open_dataset(local_path + 'RenevanWesten-AMOC-TEMP-Extremes-458bcf4/Data/CESM_0600_PI/Atmosphere/TEMP_2m_Europe_month_1-1.nc')
    amoc_vwb_600_pi = xr.open_dataset(local_path + 'RenevanWesten-AMOC-TEMP-Extremes-458bcf4/Data/CESM_0600_PI/Ocean/AMOC_transport_depth_0-1000m.nc')
    tas_vwb_1500_pi = xr.open_dataset(local_path + 'RenevanWesten-AMOC-TEMP-Extremes-458bcf4/Data/CESM_1500_PI/Atmosphere/TEMP_2m_Europe_month_1-1.nc')
    amoc_vwb_1500_pi = xr.open_dataset(local_path + 'RenevanWesten-AMOC-TEMP-Extremes-458bcf4/Data/CESM_1500_PI/Ocean/AMOC_transport_depth_0-1000m.nc')

    tas_vwb_600_45 = xr.open_dataset(local_path + 'RenevanWesten-AMOC-TEMP-Extremes-458bcf4/Data/CESM_0600_RCP45/Atmosphere/TEMP_2m_Europe_month_1-1.nc')
    amoc_vwb_600_45 = xr.open_dataset(local_path + 'RenevanWesten-AMOC-TEMP-Extremes-458bcf4/Data/CESM_0600_RCP45/Ocean/AMOC_transport_depth_0-1000m.nc')
    tas_vwb_1500_45 = xr.open_dataset(local_path + 'RenevanWesten-AMOC-TEMP-Extremes-458bcf4/Data/CESM_1500_RCP45/Atmosphere/TEMP_2m_Europe_month_1-1.nc')
    amoc_vwb_1500_45 = xr.open_dataset(local_path + 'RenevanWesten-AMOC-TEMP-Extremes-458bcf4/Data/CESM_1500_RCP45/Ocean/AMOC_transport_depth_0-1000m.nc')

    mean_amoc_vwb_600_pi = amoc_vwb_600_pi.Transport.sel(time=slice(1001, 1100)).mean()
    mean_amoc_vwb_600_45 = amoc_vwb_600_45.Transport.sel(time=slice(2400, 2500)).mean()
    mean_amoc_vwb_1500_pi = amoc_vwb_1500_pi.Transport.sel(time=slice(1901, 2000)).mean()
    mean_amoc_vwb_1500_45 = amoc_vwb_1500_45.Transport.sel(time=slice(2400, 2500)).mean()

    vwb_mask = define_region_masks(tas_vwb_600_45.mean(dim='time').TEMP_2m)
    vwb_land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(tas_vwb_600_45.mean(dim='time').TEMP_2m)
    vwb_mask['LAND'] = vwb_land_mask

    tas_600_pi = tas_vwb_600_pi.TEMP_2m.mean(dim='time').where(vwb_mask['EU_buffer'], drop=True)
    tas_600_45 = tas_vwb_600_45.TEMP_2m.mean(dim='time').where(vwb_mask['EU_buffer'], drop=True)
    tas_1500_pi = tas_vwb_1500_pi.TEMP_2m.mean(dim='time').where(vwb_mask['EU_buffer'], drop=True)
    tas_1500_45 = tas_vwb_1500_45.TEMP_2m.mean(dim='time').where(vwb_mask['EU_buffer'], drop=True)

    tas_600_pi = tas_600_pi.assign_coords(scenar='pi', type='control')
    tas_600_45 = tas_600_45.assign_coords(scenar='ghg', type='control')
    tas_1500_pi = tas_1500_pi.assign_coords(scenar='pi', type='hosing')
    tas_1500_45 = tas_1500_45.assign_coords(scenar='ghg', type='hosing')

    tas_control = xr.concat([tas_600_pi, tas_600_45], dim='scenar')
    tas_hosing = xr.concat([tas_1500_pi, tas_1500_45], dim='scenar')
    vwb_data_tas = xr.concat([tas_control, tas_hosing], dim='type')
    vwb_data_tas = vwb_data_tas.transpose('lat', 'lon', 'scenar', 'type')
    vwb_data = vwb_data_tas.to_dataset(name='tas')

    amoc_600_pi = mean_amoc_vwb_600_pi.assign_coords(scenar='pi', type='control')
    amoc_600_45 = mean_amoc_vwb_600_45.assign_coords(scenar='ghg', type='control')
    amoc_1500_pi = mean_amoc_vwb_1500_pi.assign_coords(scenar='pi', type='hosing')
    amoc_1500_45 = mean_amoc_vwb_1500_45.assign_coords(scenar='ghg', type='hosing')

    amoc_control = xr.concat([amoc_600_pi, amoc_600_45], dim='scenar')
    amoc_hosing = xr.concat([amoc_1500_pi, amoc_1500_45], dim='scenar')
    vwb_data_amoc = xr.concat([amoc_control, amoc_hosing], dim='type')
    vwb_data_amoc = vwb_data_amoc.transpose('scenar', 'type')
    vwb_data['amoc'] = vwb_data_amoc

    vwb_data = vwb_data.assign_coords(season='')
    vwb_data = vwb_data.expand_dims('season', axis=-3)

    vwb_masks_ds = xr.concat([vwb_mask[k] for k in vwb_mask.keys()],
                    dim=pd.Index(list(vwb_mask.keys()), name='region'))
    vwb_masks_ds = vwb_masks_ds.where(vwb_mask['EU_buffer'], drop=True)

    vwb_data['mask'] = vwb_masks_ds

    ########################################

    print('Loading Boot et al. (2024) data...')

    boot_exps = ['CTL_126', 'CTL_585', 'HOS_126', 'HOS_585']
    boot_data_dict = {}

    for i, exp in enumerate(boot_exps):
        # AMOC strength data (MOC)
        amoc = xr.open_dataset(f"/work/uo1075/m300817/teu_amoc/data/CESM2_hosing/{exp}/{exp}_amoc26_yr.nc", use_cftime=True)

        ctrl_or_hos = 'control' if exp[:3] == 'CTL' else 'hosing_05'
        amoc = amoc.assign_coords(scenar=f'ssp{exp[4:]}', type=ctrl_or_hos if ctrl_or_hos=='control' else 'hosing')

        boot_data_dict[exp+'_amoc'] = amoc.MOC.drop_vars(['transport_reg', 'moc_comp', 'lat_aux_grid', 'moc_z'])

        # Surface Air Temperature at Reference Height (TREFHT)
        tas = xr.open_dataset(local_path + f'Boot2024/TREFHT_{ctrl_or_hos}_{exp[4:]}.nc', use_cftime=True).TREFHT - 273.15
        tas.coords['lon'] = (tas.coords['lon'] + 180) % 360 - 180
        tas = tas.sortby(tas.lon)
        
        tas = tas.where(masks['CESM2']['EU_buffer'], drop=True)

        tas = tas.assign_coords(scenar=f'ssp{exp[4:]}', type=ctrl_or_hos if ctrl_or_hos=='control' else 'hosing')
        
        boot_data_dict[exp+'_tas'] = tas.resample(time='1YS').mean(dim='time').isel(time=slice(1, None))

    boot_tas_control = xr.concat([boot_data_dict['CTL_126_tas'], boot_data_dict['CTL_585_tas']], dim='scenar')
    boot_tas_hosing = xr.concat([boot_data_dict['HOS_126_tas'], boot_data_dict['HOS_585_tas']], dim='scenar')
    boot_tas = xr.concat([boot_tas_control, boot_tas_hosing], dim='type')
    boot_tas = boot_tas.transpose('lat', 'lon', 'time', 'scenar', 'type')

    boot_tas = boot_tas.assign_coords(season='')
    boot_tas = boot_tas.expand_dims('season', axis=-3)
    boot_tas = boot_tas.transpose('lat', 'lon', 'time', 'season', 'scenar', 'type')
    # boot_tas = boot_tas
    boot_data = boot_tas.to_dataset(name='tas')

    boot_amoc_control = xr.concat([boot_data_dict['CTL_126_amoc'], boot_data_dict['CTL_585_amoc']], dim='scenar')
    boot_amoc_hosing = xr.concat([boot_data_dict['HOS_126_amoc'], boot_data_dict['HOS_585_amoc']], dim='scenar')
    boot_amoc = xr.concat([boot_amoc_control, boot_amoc_hosing], dim='type')
    boot_amoc = boot_amoc.transpose('time', 'scenar', 'type')

    boot_amoc = boot_amoc.assign_coords(season='')
    boot_amoc = boot_amoc.expand_dims('season', axis=-3)
    boot_amoc = boot_amoc.transpose('time', 'season', 'scenar', 'type')
    boot_data['amoc'] = boot_amoc

    ########################################

    print('Loading Liu et al. (2020) data...')

    liu_hist = xr.open_dataset(local_path + 'Liu2020/b40.20th.track1.1deg.ensm.cam2.h0.TS.1961-1980.annclim.nc').TS.drop_vars(['time']).squeeze()
    liu_85 = xr.open_dataset(local_path + 'Liu2020/b40.rcp8_5.1deg.ensm.cam2.h0.TS.2061-2080.annclim.nc').TS.drop_vars(['time']).squeeze()
    liu_85_stable = xr.open_dataset(local_path + 'Liu2020/b40.rcp8_5.1deg.ensm.dhos6.cam2.h0.TS.2061-2080.annclim.nc').TS.drop_vars(['time']).squeeze()

    liu_hist_ip = liu_hist.interp_like(liu_85)

    amoc_pi = xr.DataArray(25.84686852) # 1850-1900
    amoc_85 = xr.DataArray(25.19321251) # 2061-2080
    amoc_85_stable = xr.DataArray(19.09885025) # 2061-2080

    liu_hist_ip = liu_hist_ip.assign_coords(scenar='his', type='control')
    liu_85 = liu_85.assign_coords(scenar='ghg', type='control')
    liu_85_stable = liu_85_stable.assign_coords(scenar='ghg', type='hosing')

    ghg_both = xr.concat([liu_85, liu_85_stable], dim='type')
    liu_data = xr.concat([liu_hist_ip.expand_dims('type', axis=-1), ghg_both], dim='scenar')
    liu_data = liu_data.transpose('lat', 'lon', 'scenar', 'type')
    liu_data = liu_data.to_dataset(name='tas')

    amoc_pi = amoc_pi.assign_coords(scenar='pi', type='control')
    amoc_85 = amoc_85.assign_coords(scenar='ghg', type='control')
    amoc_85_stable = amoc_85_stable.assign_coords(scenar='ghg', type='hosing')

    ghg_amoc = xr.concat([amoc_85, amoc_85_stable], dim='type')
    liu_data = liu_data.reindex(scenar=['pi', 'his', 'ghg'])
    liu_data['amoc'] = xr.concat([amoc_pi.expand_dims('type', axis=-1), ghg_amoc], dim='scenar')

    liu_data = liu_data.assign_coords(season='')
    liu_data = liu_data.expand_dims('season', axis=-3)

    liu_data['tas'] = liu_data['tas'].transpose('lat', 'lon', 'season', 'scenar', 'type')

    liu_data.coords['lon'] = (liu_data.coords['lon'] + 180) % 360 - 180
    liu_data = liu_data.sortby(liu_data.lon)

    liu_mask = define_region_masks(liu_data.tas.sel(scenar='his', type='control', season=''))
    liu_land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(liu_data.tas.sel(scenar='his', type='control', season=''))
    liu_mask['LAND'] = liu_land_mask

    liu_masks_ds = xr.concat([liu_mask[k] for k in liu_mask.keys()], 
                    dim=pd.Index(list(liu_mask.keys()), name='region'))
    liu_masks_ds = liu_masks_ds.where(liu_mask['EU_buffer'], drop=True)
    liu_masks_ds = liu_masks_ds.drop_vars(['scenar', 'type', 'season'])

    liu_data = liu_data.transpose('lat', 'lon', 'season', 'scenar', 'type')

    liu_data['tas'] = liu_data['tas'].where(liu_masks_ds.sel(region='EU_buffer'), drop=True)

    liu_data['mask'] = liu_masks_ds

    ########################################

    return bm_data, vwb_data, boot_data, liu_data

def get_full_multi_model_dict():

    if os.path.exists(local_path + 'multi_model_dict.pkl'):
        print("Loading precomputed multi_model_dict...")
        with open(local_path + 'multi_model_dict.pkl', 'rb') as f:
            multi_model_dict = pickle.load(f)
        print("Loading precomputed HosMIP masks...")
        with open(local_path + 'hosmip_masks.pkl', 'rb') as f:
            masks = pickle.load(f)
    else:
        print("Precomputed multi_model_dict not found. Recomputing...")

        amoc_data, hosmip_amoc_pi_data, tas_data, hosmip_tas_pi_data, masks, data_cutoffs = load_hosmip_data()
        multi_model_dict = build_hosmip_dataset(amoc_data, hosmip_amoc_pi_data, tas_data, hosmip_tas_pi_data, masks, data_cutoffs)
        bm_data, vwb_data, boot_data, liu_data = get_other_studies_data(masks)

        multi_model_dict['CCSM4'] = liu_data
        multi_model_dict['CESM1'] = vwb_data
        multi_model_dict['EC-Earth3'] = xr.merge([multi_model_dict['EC-Earth3'], bm_data])
        multi_model_dict['CESM2'] = xr.merge([multi_model_dict['CESM2'], boot_data])

        print("Saving multi_model_dict to ../data/multi_model_dict.pkl")
        with open(local_path + 'multi_model_dict.pkl', 'wb') as f:
            pickle.dump(multi_model_dict, f)

        print("Saving HosMIP masks to ../data/hosmip_masks.pkl")
        with open(local_path + 'hosmip_masks.pkl', 'wb') as f:
            pickle.dump(masks, f)

    return multi_model_dict, masks

def get_cmip_projections(masks, data_dict=None):

    if data_dict is None:
        data_dict = load_mpi_esm_data(eur_only=True)

    cmip6_ctrl_data = {}
    model_names = ['MPI-ESM1-2-HR', 'CESM2', 'CanESM5', 'IPSL-CM6A-LR',
                'HadGEM3-GC3-1MM', 'HadGEM3-GC3-1LL', 'EC-Earth3']

    cmip6_data_path = '/work/uo1075/m300817/teu_amoc/data/CMIP6/'

    realiz_ids = {}
    for model in model_names:
        realiz_ids[model] = {}
        realiz_ids[model]['physics'] = []
        for ssp_i in ssps:
            realiz_ids[model][ssp_i] = []

    # HadGEM3-GC3-1LL !F3!
    realiz_ids['HadGEM3-GC3-1LL']['ssp126'] = [1]
    realiz_ids['HadGEM3-GC3-1LL']['ssp245'] = [1]
    realiz_ids['HadGEM3-GC3-1LL']['ssp370'] = []
    realiz_ids['HadGEM3-GC3-1LL']['physics'] = 'p1'
    realiz_ids['HadGEM3-GC3-1LL']['missing'] = ['ssp370']

    # HadGEM3-GC3-1MM !F3!
    realiz_ids['HadGEM3-GC3-1MM']['ssp126'] = [1]
    realiz_ids['HadGEM3-GC3-1MM']['ssp245'] = []
    realiz_ids['HadGEM3-GC3-1MM']['ssp370'] = []
    realiz_ids['HadGEM3-GC3-1MM']['physics'] = 'p1'
    realiz_ids['HadGEM3-GC3-1MM']['missing'] = ['ssp245', 'ssp370']

    # EC-Earth3
    realiz_ids['EC-Earth3']['ssp126'] = []
    realiz_ids['EC-Earth3']['ssp245'] = [2, 10, 21]
    realiz_ids['EC-Earth3']['ssp370'] = []
    realiz_ids['EC-Earth3']['physics'] = 'p1'
    realiz_ids['EC-Earth3']['missing'] = ['ssp126', 'ssp370']

    # CESM2
    realiz_ids['CESM2']['ssp126'] = [10]
    realiz_ids['CESM2']['ssp245'] = [4, 10, 11]
    realiz_ids['CESM2']['ssp370'] = [4, 10, 11]
    realiz_ids['CESM2']['physics'] = 'p1'
    realiz_ids['CESM2']['missing'] = []

    # MPI-ESM1-2-HR
    realiz_ids['MPI-ESM1-2-HR']['ssp126'] = [1, 2]
    realiz_ids['MPI-ESM1-2-HR']['ssp245'] = [1, 2]
    realiz_ids['MPI-ESM1-2-HR']['ssp370'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    realiz_ids['MPI-ESM1-2-HR']['physics'] = 'p1'
    realiz_ids['MPI-ESM1-2-HR']['missing'] = []

    # CanESM5
    realiz_ids['CanESM5']['ssp126'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    realiz_ids['CanESM5']['ssp370'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    realiz_ids['CanESM5']['ssp245'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    realiz_ids['CanESM5']['physics'] = 'p2'
    realiz_ids['CanESM5']['missing'] = []

    # IPSL-CM6A-LR
    realiz_ids['IPSL-CM6A-LR']['ssp126'] = [1, 2, 3, 4, 6, 14]
    realiz_ids['IPSL-CM6A-LR']['ssp245'] = [1, 2, 3, 4, 5, 6, 10, 11, 14, 22, 25]
    realiz_ids['IPSL-CM6A-LR']['ssp370'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14]
    realiz_ids['IPSL-CM6A-LR']['physics'] = 'p1'
    realiz_ids['IPSL-CM6A-LR']['missing'] = []

    # Add historical ensemble members for each model
    # Historical data uses full variant_id format: r{realiz}i1p{physics_num}f{forcing}
    realiz_ids['HadGEM3-GC3-1LL']['his'] = [1, 2, 3, 4]  # p1f3
    realiz_ids['HadGEM3-GC3-1MM']['his'] = [1]  # p1f3
    realiz_ids['EC-Earth3']['his'] = []  # Not available in historical
    realiz_ids['EC-Earth3']['missing'].append('his')
    realiz_ids['CESM2']['his'] = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]  # p1f1 (no r7)
    realiz_ids['MPI-ESM1-2-HR']['his'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # p1f1
    realiz_ids['CanESM5']['his'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]  # p2f1 to match SSP config
    realiz_ids['IPSL-CM6A-LR']['his'] = [1, 25]  # p1f1

    # Load SSP scenarios first (they share time coordinates 2015-2100)
    for model in model_names:
        for ssp_i in ssps:
            for season in ['', 'djf', 'jja']:
                if ssp_i in realiz_ids[model]['missing']:
                    continue
                print(f"Loading {model} {ssp_i} {'annual' if season=='' else season} data...")

                physics = realiz_ids[model]['physics']
                model_lower = model.lower()
                if model == 'HadGEM3-GC3-1LL' or model == 'HadGEM3-GC3-1MM':
                    f_number = 'f3'
                    model_lower = 'hadgem3-gc31-mm' if model == 'HadGEM3-GC3-1MM' else 'hadgem3-gc31-ll'
                else:
                    f_number = 'f1'

                data_path = cmip6_data_path + f'{ssp_i}/{model_lower}/'

                if season == '':
                    new_cmip_tas_data = xr.concat([
                        (xr.open_dataarray(data_path+f'{model_lower}_r{r}i1{physics}{f_number}_tas_yr.nc', use_cftime=True).to_dataset(name='tas') - 273.15).assign_coords(scenar=ssp_i, season=season) for r in realiz_ids[model][ssp_i]
                    ], dim='realiz')

                    if model in ['CESM2', 'CanESM5', 'HadGEM3-GC3-1LL', 'HadGEM3-GC3-1MM']:
                        new_cmip_tas_data = new_cmip_tas_data.assign_coords(time=cmip6_ctrl_data['MPI-ESM1-2-HR'].time.values)
                else:
                    new_cmip_tas_data = xr.concat([
                        (xr.open_dataset(data_path+f'tas_{season}/{model_lower}_r{r}i1{physics}{f_number}_tas_{season}.nc', use_cftime=True) - 273.15).assign_coords(scenar=ssp_i, season=season) for r in realiz_ids[model][ssp_i]
                    ], dim='realiz')
                    if 'dim' in drop_stuff[model].keys():
                        new_cmip_tas_data = new_cmip_tas_data.drop_dims(drop_stuff[model]['dim'])
                    if 'var' in drop_stuff[model].keys():
                        new_cmip_tas_data = new_cmip_tas_data.drop_vars(drop_stuff[model]['var'])

                new_cmip_tas_data.coords['lon'] = (new_cmip_tas_data.coords['lon'] + 180) % 360 - 180
                new_cmip_tas_data = new_cmip_tas_data.sortby(new_cmip_tas_data.lon)

                if model == 'EC-Earth3':
                    mask_ecearth_new = define_region_masks(new_cmip_tas_data.tas)
                    new_cmip_tas_data = new_cmip_tas_data.where(mask_ecearth_new['EU_buffer'], drop=True)
                else:
                    new_cmip_tas_data = new_cmip_tas_data.where(masks[model]['EU_buffer'], drop=True)

                if model not in cmip6_ctrl_data.keys():
                    cmip6_ctrl_data[model] = new_cmip_tas_data.mean(dim='realiz')
                else:
                    cmip6_ctrl_data[model]['tas'].loc[dict(scenar=ssp_i, season=season)] = new_cmip_tas_data.tas.mean(dim='realiz')

                if season != '':
                    continue

                new_cmip_amoc_data = xr.concat([
                    xr.open_dataarray(data_path+f'{model_lower}_r{r}i1{physics}{f_number}_amoc26_yr.nc', use_cftime=True).drop_vars('lev' if model != 'IPSL-CM6A-LR' else 'olevel').to_dataset(name='amoc') for r in realiz_ids[model][ssp_i]
                ], dim='realiz')
                if model in ['CESM2', 'CanESM5', 'HadGEM3-GC3-1LL', 'HadGEM3-GC3-1MM']:
                    if ssp_i == 'ssp126' and model == 'CESM2':
                        new_cmip_amoc_data = new_cmip_amoc_data.assign_coords(time=cmip6_ctrl_data['MPI-ESM1-2-HR'].time.values[50:])
                    else:
                        new_cmip_amoc_data = new_cmip_amoc_data.assign_coords(time=cmip6_ctrl_data['MPI-ESM1-2-HR'].time.values)

                if 'amoc' not in cmip6_ctrl_data[model].data_vars:
                    cmip6_ctrl_data[model]['amoc'] = new_cmip_amoc_data.amoc.mean(dim='realiz')
                    cmip6_ctrl_data[model] = cmip6_ctrl_data[model].expand_dims('scenar', axis=-1).reindex(scenar=ssps)
                    cmip6_ctrl_data[model]['tas'] = cmip6_ctrl_data[model]['tas'].expand_dims('season', axis=-1).reindex(season=['', 'djf', 'jja'])
                else:
                    cmip6_ctrl_data[model]['amoc'].loc[dict(scenar=ssp_i)] = new_cmip_amoc_data.amoc.mean(dim='realiz')

    # Load historical data separately (different time range: 1850-2015)
    # Then concatenate with SSP data using join='outer' to handle different time coordinates
    # Store MPI-ESM1-2-HR historical time coords for calendar alignment with other models
    mpi_hr_his_time = None

    for model in model_names:
        if 'his' in realiz_ids[model]['missing']:
            continue
        print(f"Loading {model} his annual data...")

        physics = realiz_ids[model]['physics']
        model_lower = model.lower()
        if model == 'HadGEM3-GC3-1LL' or model == 'HadGEM3-GC3-1MM':
            model_lower = 'hadgem3-gc31-mm' if model == 'HadGEM3-GC3-1MM' else 'hadgem3-gc31-ll'

        data_path = cmip6_data_path + f'historical/{model_lower}/'
        physics_num = '1' if physics == 'p1' else '2'
        f_num = '3' if model in ['HadGEM3-GC3-1LL', 'HadGEM3-GC3-1MM'] else '1'

        # Load historical TAS data
        new_cmip_tas_data = xr.concat([
            (xr.open_dataarray(data_path+f'{model_lower}_r{r}i1p{physics_num}f{f_num}_tas_yr.nc', use_cftime=True).to_dataset(name='tas') - 273.15).assign_coords(scenar='his', season='') for r in realiz_ids[model]['his']
        ], dim='realiz')

        new_cmip_tas_data.coords['lon'] = (new_cmip_tas_data.coords['lon'] + 180) % 360 - 180
        new_cmip_tas_data = new_cmip_tas_data.sortby(new_cmip_tas_data.lon)
        new_cmip_tas_data = new_cmip_tas_data.where(masks[model]['EU_buffer'], drop=True)

        # Load historical AMOC data
        new_cmip_amoc_data = xr.concat([
            xr.open_dataarray(data_path+f'{model_lower}_r{r}i1p{physics_num}f{f_num}_amoc26_yr.nc', use_cftime=True).drop_vars('lev' if model != 'IPSL-CM6A-LR' else 'olevel').to_dataset(name='amoc') for r in realiz_ids[model]['his']
        ], dim='realiz')

        # Store MPI-ESM1-2-HR time coords for calendar alignment
        if model == 'MPI-ESM1-2-HR':
            mpi_hr_his_time = new_cmip_tas_data.time.values

        # Align time coordinates for models with different calendars
        if model in ['CESM2', 'CanESM5', 'HadGEM3-GC3-1LL', 'HadGEM3-GC3-1MM']:
            new_cmip_tas_data = new_cmip_tas_data.assign_coords(time=mpi_hr_his_time)
            new_cmip_amoc_data = new_cmip_amoc_data.assign_coords(time=mpi_hr_his_time)

        # Build historical dataset with proper dimensions
        his_tas = new_cmip_tas_data.tas.mean(dim='realiz').expand_dims('scenar').assign_coords(scenar=['his'])
        his_tas = his_tas.expand_dims('season').assign_coords(season=[''])
        his_amoc = new_cmip_amoc_data.amoc.mean(dim='realiz').expand_dims('scenar').assign_coords(scenar=['his'])

        his_ds = xr.Dataset({'tas': his_tas, 'amoc': his_amoc})

        # Concatenate historical with SSP data using join='outer' to handle different time coordinates
        cmip6_ctrl_data[model] = xr.concat([cmip6_ctrl_data[model], his_ds], dim='scenar', join='outer')

    # add MPI-ESM1-2-LR data (SSP scenarios)
    for ssp_i in ssps:
        for season in ['', 'djf', 'jja']:
            print(f"Loading MPI-ESM1-2-LR {ssp_i} {'annual' if season=='' else season} data...")
            if 'MPI-ESM1-2-LR' not in cmip6_ctrl_data.keys():
                cmip6_ctrl_data['MPI-ESM1-2-LR'] = data_dict[season]['tas']['ssp'].sel(scenar=ssp_i).mean(dim='realiz').assign_coords(season=season)
                cmip6_ctrl_data['MPI-ESM1-2-LR']['amoc'] = data_dict['']['amoc']['ssp'].AMOC_strength.sel(scenar=ssp_i).mean(dim='realiz')
                cmip6_ctrl_data['MPI-ESM1-2-LR'] = cmip6_ctrl_data['MPI-ESM1-2-LR'].expand_dims('scenar', axis=-1).reindex(scenar=ssps)
                cmip6_ctrl_data['MPI-ESM1-2-LR']['tas'] = cmip6_ctrl_data['MPI-ESM1-2-LR']['tas'].expand_dims('season', axis=-1).reindex(season=['', 'djf', 'jja'])
            else:
                if season != '':
                    cmip6_ctrl_data['MPI-ESM1-2-LR']['tas'].loc[dict(scenar=ssp_i, season=season)] = data_dict[season]['tas']['ssp'].tas.sel(scenar=ssp_i).assign_coords(time=data_dict['']['tas']['ssp'].time.values).mean(dim='realiz')
                    continue
                else:
                    cmip6_ctrl_data['MPI-ESM1-2-LR']['tas'].loc[dict(scenar=ssp_i, season=season)] = data_dict[season]['tas']['ssp'].tas.sel(scenar=ssp_i).mean(dim='realiz')
                cmip6_ctrl_data['MPI-ESM1-2-LR']['amoc'].loc[dict(scenar=ssp_i)] = data_dict['']['amoc']['ssp'].AMOC_strength.sel(scenar=ssp_i).mean(dim='realiz')

    # add MPI-ESM1-2-LR historical data (annual only, different time range)
    print("Loading MPI-ESM1-2-LR his annual data...")
    his_tas = data_dict['']['tas']['his'].tas.mean(dim='realiz').expand_dims('scenar').assign_coords(scenar=['his'])
    his_tas = his_tas.expand_dims('season').assign_coords(season=[''])
    his_amoc = data_dict['']['amoc']['his'].AMOC_strength.mean(dim='realiz').expand_dims('scenar').assign_coords(scenar=['his'])
    his_ds = xr.Dataset({'tas': his_tas, 'amoc': his_amoc})

    # Concatenate historical with SSP data using join='outer' to handle different time coordinates
    cmip6_ctrl_data['MPI-ESM1-2-LR'] = xr.concat([cmip6_ctrl_data['MPI-ESM1-2-LR'], his_ds], dim='scenar', join='outer')

    return cmip6_ctrl_data

# TRANSFORMING MULTI-MODEL DATA

def net_cooling_point(coefs, T_2100, AMOC_2100, AMOC_pi_MPI, AMOC_metric='rel'):
    '''
        Mainly useful for quadratic regressions (quantile regressions). Not necessary for main (linear) analyses in paper.
        Calculate the net cooling point based on regression coefficients and temperature at 2100.
        Parameters:
            coefs (dict): Dictionary containing regression coefficients. Keys: 'coef_lin', coef_quad'. If linear regression, coef_quad should be 0.
            T_2100 (float): Temperature at 2100.
            AMOC_2100 (float): AMOC strength at 2100.
            AMOC_pi_MPI (float): Pre-industrial AMOC strength from MPI-ESM1.2-LR.
            AMOC_metric (str): Relative AMOC weakening in % ('rel') or absolute AMOC strength in Sv ('abs'). Default is 'rel'.
    '''

    quad_solution = np.roots([
        coefs['coef_quad'], 
        coefs['coef_lin'], 
        T_2100
        ]) if not np.isnan(coefs['coef_quad']) else np.nan

    if np.iscomplex(quad_solution).any():
        return np.nan
    elif coefs['coef_quad']==0 and coefs['coef_lin'] >= 0:
        return 110 if AMOC_metric=='rel' else -1
    else:
        if coefs['coef_quad'] < 0:
                if AMOC_metric == 'rel':
                    return (1 - AMOC_2100 / AMOC_pi_MPI) * 100 + np.max(quad_solution)
                elif AMOC_metric == 'abs':
                    return AMOC_2100 - AMOC_pi_MPI/100 * np.max(quad_solution)
                else:
                    raise ValueError("AMOC_metric must be 'rel' or 'abs'.")
        elif coefs['coef_quad'] >= 0:
                if AMOC_metric == 'rel':
                    return (1 - AMOC_2100 / AMOC_pi_MPI) * 100 + np.min(quad_solution)
                elif AMOC_metric == 'abs':
                    return AMOC_2100 - AMOC_pi_MPI/100 * np.min(quad_solution)
                else:
                    raise ValueError("AMOC_metric must be 'rel' or 'abs'.")

def get_hosmip_reg_ds(hosmip_ref_pi=True, recompute=False, data_dict=None):

    if recompute:  
        
        print('Recomputing HosMIP regression dataset...')

        if data_dict is None:
            data_dict = load_mpi_esm_data(eur_only=True)
        multi_model_dict, masks = get_full_multi_model_dict()
        cmip6_ctrl_data = get_cmip_projections(masks, data_dict)

        hosmip_reg_ds_dict = {}
        for model in hosmip_labels:
            static_reg_ds = multi_model_dict[model].isel(time=0, drop=True).copy(deep=True).drop_vars(['tas', 'amoc', 'mask', 'region', 'type']) if 'time' in list(multi_model_dict[model].coords) else multi_model_dict[model].copy(deep=True).drop_vars(['tas', 'amoc', 'mask', 'region', 'type'])

            static_reg_ds = static_reg_ds.assign_coords(scenar=list(set([str(s) for s in static_reg_ds.scenar.values]+ssps)))

            reg_ds = static_reg_ds.assign(
                lin_coef_hosmip=(['lat', 'lon', 'season'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),
                ste_hosmip=(['lat', 'lon', 'season'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),  # standard error of regression coefficient
                rsq_hosmip=(['lat', 'lon', 'season'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),  # R^2 of regression
                pvalue_hosmip=(['lat', 'lon', 'season'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),  # p-value of regression coefficient
                AMOC_2090_2100=(['scenar'], np.full(tuple(static_reg_ds.isel(lat=0, lon=0, season=0).sizes.values()), np.nan)),
                AMOC_pi=([], np.full(tuple(static_reg_ds.isel(lat=0, lon=0, scenar=0, season=0).sizes.values()), np.nan)),
                T_2090_2100=(['lat', 'lon', 'season', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
                T_pi=(['lat', 'lon', 'season'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),
                T_pd_245=(['lat', 'lon'], np.full(tuple(static_reg_ds.isel(scenar=0, season=0).sizes.values()), np.nan)),
                req_strength_pi=(['lat', 'lon', 'season', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
                req_strength_pd=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.isel(season=0).sizes.values()), np.nan)),
                req_weakening_pi=(['lat', 'lon', 'season', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
                req_weakening_pd=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.isel(season=0).sizes.values()), np.nan)),
            )

            hosmip_reg_ds_dict[model] = reg_ds

        for model in hosmip_labels:
            for season_idx, season in enumerate(['', 'djf', 'jja']):
                print(f"Calculating {'annual' if season=='' else season} regression coefficients for {model}...")
                i=0
                for lat in np.arange(len(hosmip_reg_ds_dict[model].lat)):
                    for lon in np.arange(len(hosmip_reg_ds_dict[model].lon)):
                        
                        model_reg_coefs = hosmip_regression_plot(multi_model_dict, season=season, region=None, lat=lat, lon=lon, only_model=model, hosmip_ref_pi=hosmip_ref_pi, single_model_linear=True, quantile_reg=False, no_plots=True)

                        hosmip_reg_ds_dict[model].lin_coef_hosmip.values[lat, lon, season_idx] = model_reg_coefs['model_coef_lin']
                        hosmip_reg_ds_dict[model].ste_hosmip.values[lat, lon, season_idx] = model_reg_coefs['model_ste_lin']
                        hosmip_reg_ds_dict[model].rsq_hosmip.values[lat, lon, season_idx] = model_reg_coefs['model_rsq']
                        hosmip_reg_ds_dict[model].pvalue_hosmip.values[lat, lon, season_idx] = model_reg_coefs['model_pvalue_lin']
                        i += 1

                        if i % 100 == 0:
                            print(f"{i}/{len(hosmip_reg_ds_dict[model].lat)*len(hosmip_reg_ds_dict[model].lon)} regression coefficients calculated.")

        for model in hosmip_labels:

            hosmip_reg_ds_dict[model].AMOC_pi.values = multi_model_dict[model].sel(scenar='pi', type='control', season='').amoc.isel(time=0)
            hosmip_reg_ds_dict[model].T_pi.values[:, :, :] = multi_model_dict[model].sel(scenar='pi', type='control').tas.isel(time=0)
            if model != 'EC-Earth3':
                hosmip_reg_ds_dict[model].T_pd_245.values[:, :] = xr.concat([cmip6_ctrl_data[model]['tas'].sel(scenar='his', season='').sel(time=slice('2000', '2014')), cmip6_ctrl_data[model]['tas'].sel(scenar='ssp245', season='').sel(time=slice('2015', '2029'))], dim='time').mean(dim='time').values

            for ssp_i in ssps:
                print(f"Calculating required AMOC changes for {model} under {ssp_i}...")

                if np.isnan(cmip6_ctrl_data[model]['amoc'].sel(scenar=ssp_i).isel(time=slice(-11, -1)).mean(dim='time').values):
                    print(f'Skipping model {model} for scenario {ssp_i} due to missing AMOC data.')
                    continue

                hosmip_reg_ds_dict[model].AMOC_2090_2100.loc[ssp_i] = cmip6_ctrl_data[model]['amoc'].sel(scenar=ssp_i).isel(time=slice(-11, -1)).mean(dim='time').values
                hosmip_reg_ds_dict[model].T_2090_2100.loc[:, :, :, ssp_i] = cmip6_ctrl_data[model]['tas'].sel(scenar=ssp_i).isel(time=slice(-11, -1)).mean(dim='time').values

                hosmip_reg_ds_dict[model].req_strength_pi.loc[:, :, :, ssp_i] = hosmip_reg_ds_dict[model].AMOC_2090_2100.loc[ssp_i] - (hosmip_reg_ds_dict[model].T_2090_2100.loc[:, :, :, ssp_i] - hosmip_reg_ds_dict[model].T_pi) / (hosmip_reg_ds_dict[model].lin_coef_hosmip.loc[:, :, :].where(hosmip_reg_ds_dict[model].lin_coef_hosmip.loc[:, :, :]<0)*(-100)/hosmip_reg_ds_dict[model].AMOC_pi)

                if model != 'EC-Earth3':
                    hosmip_reg_ds_dict[model].req_strength_pd.loc[:, :, ssp_i] = hosmip_reg_ds_dict[model].AMOC_2090_2100.loc[ssp_i] - (hosmip_reg_ds_dict[model].T_2090_2100.loc[:, :, '', ssp_i] - hosmip_reg_ds_dict[model].T_pd_245) / (hosmip_reg_ds_dict[model].lin_coef_hosmip.loc[:, :, ''].where(hosmip_reg_ds_dict[model].lin_coef_hosmip.loc[:, :, '']<0)*(-100)/hosmip_reg_ds_dict[model].AMOC_pi)

                hosmip_reg_ds_dict[model].req_weakening_pi.loc[:, :, :, ssp_i] = (hosmip_reg_ds_dict[model].AMOC_pi - hosmip_reg_ds_dict[model].req_strength_pi.loc[:, :, :, ssp_i]) / hosmip_reg_ds_dict[model].AMOC_pi * 100

                if model != 'EC-Earth3':
                    hosmip_reg_ds_dict[model].req_weakening_pd.loc[:, :, ssp_i] = (hosmip_reg_ds_dict[model].AMOC_pi - hosmip_reg_ds_dict[model].req_strength_pd.loc[:, :, ssp_i]) / hosmip_reg_ds_dict[model].AMOC_pi * 100

        # save to pickle
        with open(local_path + 'hosmip_reg_ds_dict.pkl', 'wb') as f:
            pickle.dump(hosmip_reg_ds_dict, f)
            print(f"HosMIP regression dataset dictionary saved to '{local_path}hosmip_reg_ds_dict.pkl'.")
    else:
        # open from to pickle
        with open(local_path + 'hosmip_reg_ds_dict.pkl', 'rb') as f:
            hosmip_reg_ds_dict = pickle.load(f)
            print(f"HosMIP regression dataset dictionary loaded from '{local_path}hosmip_reg_ds_dict.pkl'.")
    
    return hosmip_reg_ds_dict

def get_cesm_reg_ds(recompute=False, data_dict=None, multi_model_dict=None, masks=None, cmip6_ctrl_data=None):

    if recompute:
        print('Recomputing CESM2 regression dataset...')

        # load the necessary datasets
        print('############################################################')
        print('Loading necessary datasets...')
        print('############################################################')
        multi_model_dict, masks = get_full_multi_model_dict() if multi_model_dict is None or masks is None else (multi_model_dict, masks)
        cmip6_ctrl_data = get_cmip_projections(masks, data_dict=(data_dict if data_dict!=None else None)) if cmip6_ctrl_data is None else cmip6_ctrl_data
        bm_data, vwb_data, boot_data, liu_data = get_other_studies_data(masks)

        print('############################################################')
        print('Calculating CESM2 regression coefficients...')
        print('############################################################')
        static_reg_ds = multi_model_dict['CESM2'].isel(time=0, drop=True).copy(deep=True).drop_vars(['tas', 'amoc', 'mask', 'region', 'type', 'season'])
        static_reg_ds = static_reg_ds.assign_coords(scenar=ssps+['ssp585'])

        reg_ds_cesm = static_reg_ds.assign(
            coef_ensmean=(['lat', 'lon'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),
            ste_ensmean=(['lat', 'lon'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)), # standard error of regression coefficient
            rsq_ensmean=(['lat', 'lon'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)), # R^2 of regression
            coef_ensmean_intercept=(['lat', 'lon'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),
            ste_ensmean_intercept=(['lat', 'lon'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)), # standard error of regression coefficient
            rsq_ensmean_intercept=(['lat', 'lon'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)), # R^2 of regression
            coef_ssp=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
            ste_ssp=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),  # standard error for SSP-specific regressions
            rsq_ssp=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),  # R^2 for SSP-specific regressions
            AMOC_2090_2100=(['scenar'], np.full(tuple(static_reg_ds.isel(lat=0, lon=0).sizes.values()), np.nan)),
            AMOC_pi=([], np.full(tuple(static_reg_ds.isel(lat=0, lon=0, scenar=0).sizes.values()), np.nan)),
            T_2090_2100=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
            T_pi=(['lat', 'lon'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),
            T_pd_245=(['lat', 'lon'], np.full(tuple(static_reg_ds.isel(scenar=0).sizes.values()), np.nan)),
            req_strength_pi=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
            req_strength_pd=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
            req_weakening_pi=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
            req_weakening_pd=(['lat', 'lon', 'scenar'], np.full(tuple(static_reg_ds.sizes.values()), np.nan)),
        )

        i=0
        for lat in np.arange(len(static_reg_ds.lat)):
            for lon in np.arange(len(static_reg_ds.lon)):
                reg_quadruples = cesm_regressions(boot_data, multi_model_dict, masks, ssp='all', lat=lat, lon=lon, region=None, combined_reg=True, no_plots=True, add_combined_intercept=False)
                reg_quadruples_intercept = cesm_regressions(boot_data, multi_model_dict, masks, ssp='all', lat=lat, lon=lon, region=None, combined_reg=True, no_plots=True, add_combined_intercept=True)
                reg_ds_cesm.coef_ensmean.values[lat, lon] = reg_quadruples['combined'][1]
                reg_ds_cesm.ste_ensmean.values[lat, lon] = reg_quadruples['combined'][2]
                reg_ds_cesm.rsq_ensmean.values[lat, lon] = reg_quadruples['combined'][3]
                reg_ds_cesm.coef_ensmean_intercept.values[lat, lon] = reg_quadruples_intercept['combined'][1]
                reg_ds_cesm.ste_ensmean_intercept.values[lat, lon] = reg_quadruples_intercept['combined'][2]
                reg_ds_cesm.rsq_ensmean_intercept.values[lat, lon] = reg_quadruples_intercept['combined'][3]
                # SSP-specific: tuple is (coef, ste, rsq)
                reg_ds_cesm.coef_ssp.values[lat, lon, 0] = reg_quadruples['ssp126'][0]
                reg_ds_cesm.ste_ssp.values[lat, lon, 0] = reg_quadruples['ssp126'][1]
                reg_ds_cesm.rsq_ssp.values[lat, lon, 0] = reg_quadruples['ssp126'][2]
                reg_ds_cesm.coef_ssp.values[lat, lon, 3] = reg_quadruples['ssp585'][0]
                reg_ds_cesm.ste_ssp.values[lat, lon, 3] = reg_quadruples['ssp585'][1]
                reg_ds_cesm.rsq_ssp.values[lat, lon, 3] = reg_quadruples['ssp585'][2]
                i += 1

                if i % 100 == 0:
                    print(f"{i}/{len(static_reg_ds.lat)*len(static_reg_ds.lon)} regression coefficients calculated.")

        reg_ds_cesm.AMOC_pi.values = multi_model_dict['CESM2'].sel(scenar='pi', type='control', season='').amoc.isel(time=0).values
        reg_ds_cesm.T_pi.values[:, :] = multi_model_dict['CESM2'].sel(scenar='pi', type='control', season='').tas.isel(time=0).values
        # reg_ds_cesm.T_pd_245.loc[:, :] = xr.concat([.sel(time=slice('2000', '2014')),
        #                                             cmip6_ctrl_data['CESM2']['tas'].sel(scenar='ssp245', season='').sel(time=slice('2015', '2029'))], dim='time').mean(dim='time').values
        reg_ds_cesm.T_pd_245.loc[:, :] = xr.concat([cmip6_ctrl_data['CESM2']['tas'].sel(scenar='his', season='').sel(time=slice('2000', '2014')), cmip6_ctrl_data['CESM2']['tas'].sel(scenar='ssp245', season='').sel(time=slice('2015', '2029'))], dim='time').mean(dim='time').values

        for ssp_i in ssps:
            reg_ds_cesm.AMOC_2090_2100.loc[ssp_i] = cmip6_ctrl_data['CESM2']['amoc'].sel(scenar=ssp_i).isel(time=slice(-11, -1)).mean(dim='time').values
            reg_ds_cesm.T_2090_2100.loc[:, :, ssp_i] = cmip6_ctrl_data['CESM2']['tas'].sel(scenar=ssp_i, season='').isel(time=slice(-11, -1)).mean(dim='time').values

            reg_ds_cesm.req_strength_pi.loc[:, :, ssp_i] = reg_ds_cesm.AMOC_2090_2100.loc[ssp_i] - (reg_ds_cesm.T_2090_2100.loc[:, :, ssp_i] - reg_ds_cesm.T_pi) / (reg_ds_cesm.coef_ensmean*(-100)/reg_ds_cesm.AMOC_pi)
            reg_ds_cesm.req_strength_pd.loc[:, :, ssp_i] = reg_ds_cesm.AMOC_2090_2100.loc[ssp_i] - (reg_ds_cesm.T_2090_2100.loc[:, :, ssp_i] - reg_ds_cesm.T_pd_245) / (reg_ds_cesm.coef_ensmean*(-100)/reg_ds_cesm.AMOC_pi)
            reg_ds_cesm.req_weakening_pi.loc[:, :, ssp_i] = (reg_ds_cesm.AMOC_pi - reg_ds_cesm.req_strength_pi.loc[:, :, ssp_i]) / reg_ds_cesm.AMOC_pi * 100
            reg_ds_cesm.req_weakening_pd.loc[:, :, ssp_i] = (reg_ds_cesm.AMOC_pi - reg_ds_cesm.req_strength_pd.loc[:, :, ssp_i]) / reg_ds_cesm.AMOC_pi * 100

        print(f'Saving CESM2 regression dataset to {local_path}reg_ds_cesm.nc')
        reg_ds_cesm.to_netcdf(local_path + 'reg_ds_cesm.nc')
    else:
        print(f'Loading CESM2 regression dataset from {local_path}reg_ds_cesm.nc')
        reg_ds_cesm = xr.open_dataset(local_path + 'reg_ds_cesm.nc')
    return reg_ds_cesm

# PLOTTING MULTI-MODEL DATA

def hosmip_regression_plot(multi_model_dict, window=10, region='EU', season='', no_plots=False, plot_bg='white', only_model=None, only_model_label=None, low_ylim=-16, markersize=20, quantile_reg=False, linear_95_reg=False, linear_5_reg=False, central_reg=None, central_reg_intercept=True, single_model_linear=True, single_model_intercept=False, hosmip_ref_pi=True, lat=None, lon=None, cap_x_range=100, bm_data=False, vwb_data=False, boot_data=False, liu_data=False, boot_regression=False, boot_intercept=False, add_combined_results=False, combined_results_label=True, add_hosmip_mpi_regression=False, ax=None, savefig=False):
    
    if season != '':
        if not no_plots:
            print('No seasonal data for other studies.')
        bm_data, vwb_data, boot_data, liu_data = False, False, False, False

    if not no_plots:
        plt.style.use('default')
        if plot_bg == 'black':
            plt.style.use('dark_background')
            plt.rcParams['axes.facecolor'] = '#191919'
            plt.rcParams['figure.facecolor'] = '#191919'
        if ax is None:
            plt.rcParams.update({'font.size': 14,
                                 'axes.labelsize': 14,
                                 'xtick.labelsize': 14,
                                 'ytick.labelsize': 14,
                                 })
            fig, ax = plt.subplots(figsize=(9, 6))
            ext_ax = False
        else:
            ext_ax = True

    is_latlon, is_region, is_ocean = False, False, False
    if lat != None and lon != None:
        is_latlon = True
        if only_model is not None:
            if np.sum(~np.isnan(multi_model_dict[only_model].sel(type='control', season='', scenar='pi').tas.isel(time=0).values))==0:
                return {
                        '95pct_coef_lin': np.nan,
                        '95pct_coef_quad': np.nan,
                        '5pct_coef_lin': np.nan,
                        '5pct_coef_quad': np.nan,
                        'model_coef_lin': np.nan,
                        'model_coef_quad': np.nan
                        }
        else:
            print('Lat/lon only works for a single model.')
            return
    elif region != None:
        is_region = True
    else:
        print('No lat/lon or region provided.')
        return

    all_x = []
    all_y = []
    for model in hosmip_labels:
        if only_model is not None and model != only_model:
            continue
        amoc = multi_model_dict[model].sel(type='hosing', scenar='pi', season='').rolling(time=window, center=True).mean('time').amoc
        if hosmip_ref_pi:
            amoc_base = multi_model_dict[model].sel(type='control', scenar='pi', season='').amoc.isel(time=0)
            x = (amoc.values - amoc_base.values) / amoc_base.values * (-100)
        else:
            amoc_base = amoc.isel(time=slice(0, 10)).mean('time')
            x = (amoc.values - amoc_base.values) / amoc_base.values * (-100)
        if is_latlon:
            tas_data_unsorted = multi_model_dict[model].sel(type='hosing', scenar='pi', season=season).tas
            tas_lat_sorted = tas_data_unsorted.sortby(tas_data_unsorted.lat)
            tas = tas_lat_sorted.isel(lat=lat, lon=lon)
        else:
            tas = weighted_area_lat(multi_model_dict[model].sel(type='hosing', scenar='pi', season=season).tas.where(
                multi_model_dict[model].mask.sel(region=region)).where(multi_model_dict[model].mask.sel(region='LAND')==0)
                ).mean('lat').mean('lon')
        tas_rolling = tas.rolling(time=window, center=True).mean('time')
        if hosmip_ref_pi:
            if is_latlon:
                tas_base = multi_model_dict[model].sel(type='control', scenar='pi', season=season).tas.isel(time=0, lat=lat, lon=lon)
            else:
                tas_base = weighted_area_lat(multi_model_dict[model].sel(type='control', scenar='pi', season=season).tas.isel(time=0).where(
                    multi_model_dict[model].mask.sel(region=region)).where(multi_model_dict[model].mask.sel(region='LAND')==0)
                    ).mean('lat').mean('lon')
        else:
            tas_base = tas.isel(time=slice(0, 10)).mean('time')
        y = tas_rolling.values - tas_base.values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        all_x.append(x)
        all_y.append(y)
        if not no_plots:
            if only_model and only_model_label != None:
                ax.scatter(x, y, label=only_model_label, color=hosmip_colors[model], clip_on=False, s=markersize, alpha=0.9, marker='o') # '+' '$\\bigcirc$'
            else:
                ax.scatter(x, y, label=model, color=hosmip_colors[model], clip_on=False, s=markersize, alpha=0.9, marker='o') # '+' '$\\bigcirc$'
        
    all_x_flat = np.concatenate(all_x)
    all_y_flat = np.concatenate(all_y)

    # Sort x for plotting
    sort_idx = np.argsort(all_x_flat)
    restricted_x_range = np.where((all_x_flat[sort_idx] < cap_x_range) & (all_x_flat[sort_idx] > 0))
    x_sorted = (all_x_flat[sort_idx])[restricted_x_range]
    y_sorted = (all_y_flat[sort_idx])[restricted_x_range]
    Xq = np.column_stack([x_sorted, x_sorted**2])
    Xq_lin = np.column_stack([x_sorted])
    # Xq = sm.add_constant(Xq)  # add intercept

    # 5th percentile
    mod_5 = sm.QuantReg(y_sorted, Xq_lin) if linear_5_reg else sm.QuantReg(y_sorted, Xq)
    res_5 = mod_5.fit(q=0.05)
    yq_5 = res_5.predict(Xq_lin) if linear_5_reg else res_5.predict(Xq)
    
    # 95th percentile
    mod_95 = sm.QuantReg(y_sorted, Xq_lin) if linear_95_reg else sm.QuantReg(y_sorted, Xq)
    res_95 = mod_95.fit(q=0.95)
    yq_95 = res_95.predict(Xq_lin) if linear_95_reg else res_95.predict(Xq)

    if central_reg == 'linear':
        X = x_sorted
        if central_reg_intercept:
            X = sm.add_constant(x_sorted)  # Add intercept column to regressor
        mod_lin = sm.OLS(y_sorted, X)
        x_range = np.linspace(0, max(x_sorted), 100)
        X_pred = sm.add_constant(x_range) if central_reg_intercept else x_range
        pred_lin = mod_lin.fit(cov_type='HAC', cov_kwds={'maxlags': window-1}).predict(X_pred)
        if not no_plots:
            ax.plot(x_range, pred_lin, color='black' if plot_bg != 'black' else 'white', lw=2, ls='-', label='Linear regression')
            if central_reg_intercept:
                ax.text(52, -10, f"y = {mod_lin.fit(cov_type='HAC', cov_kwds={'maxlags': window-1}).params[0]:.5f} + {mod_lin.fit().params[1]:.5f} x (lin. reg.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')
            else:
                ax.text(52, -10, f"y = {mod_lin.fit(cov_type='HAC', cov_kwds={'maxlags': window-1}).params[0]:.5f} x (lin. reg.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')
    elif central_reg == 'quadratic':
        # Quadratic regression, linear + quadratic term, with optional intercept
        X_quad = np.column_stack([x_sorted, x_sorted**2])
        if central_reg_intercept:
            X_quad = sm.add_constant(X_quad)  # Adds intercept as first column
        mod_quad = sm.OLS(y_sorted, X_quad)
        x_range = np.linspace(0, max(x_sorted), 100)
        X_range_quad = np.column_stack([x_range, x_range**2])
        if central_reg_intercept:
            X_range_quad = sm.add_constant(X_range_quad)
        pred_quad = mod_quad.fit(cov_type='HAC', cov_kwds={'maxlags': window-1}).predict(X_range_quad)
        if not no_plots:
            ax.plot(x_range, pred_quad, color='black' if plot_bg != 'black' else 'white', lw=2, ls='-', label='Quadratic regression')
            params = mod_quad.fit(cov_type='HAC', cov_kwds={'maxlags': window-1}).params
            if central_reg_intercept:
                ax.text(52, -10, f"y = {params[0]:.5f} + {params[1]:.5f} x + {params[2]:.5f} x² (quad. reg.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')
            else:
                ax.text(52, -10, f"y = {params[0]:.5f} x + {params[1]:.5f} x² (quad. reg.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')

    # for a single model, also linear regression
    if only_model:
        if single_model_linear:
            # Linear regression, optionally with intercept
            if single_model_intercept:
                x_with_const = sm.add_constant(x_sorted)
                mod_lin = sm.OLS(y_sorted, x_with_const)
            else:
                mod_lin = sm.OLS(y_sorted, x_sorted)
            mod_lin_fit = mod_lin.fit(cov_type='HAC', cov_kwds={'maxlags': window-1})
            x_range = np.linspace(0, max(x_sorted), 100)
            if single_model_intercept:
                print(only_model, mod_lin_fit.summary())
                x_range_pred = sm.add_constant(x_range)
                mod_lin_intercept = mod_lin_fit.params[0]
                mod_lin_coef = mod_lin_fit.params[1]
                mod_lin_intercept_ste = mod_lin_fit.bse[0]
                mod_lin_ste = mod_lin_fit.bse[1]
                mod_lin_rsq = mod_lin_fit.rsquared
                mod_lin_intercept_pvalue = mod_lin_fit.pvalues[0]
                mod_lin_pvalue = mod_lin_fit.pvalues[1]
            else:
                x_range_pred = x_range
                mod_lin_intercept = None
                mod_lin_intercept_ste = None
                mod_lin_intercept_pvalue = None
                mod_lin_coef = mod_lin_fit.params[0]
                mod_lin_ste = mod_lin_fit.bse[0]
                mod_lin_rsq = mod_lin_fit.rsquared
                mod_lin_pvalue = mod_lin_fit.pvalues[0]
            pred_lin = mod_lin_fit.predict(x_range_pred)
            if not no_plots and central_reg != False:
                ax.plot(x_range, pred_lin, color='black' if plot_bg != 'black' else 'white', lw=2, ls='-', label='Linear regression')
                if single_model_intercept:
                    ax.text(52, -10, f"y = {mod_lin_intercept:.5f} + {mod_lin_coef:.5f} x (lin. reg.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')
                else:
                    ax.text(52, -10, f"y = {mod_lin_coef:.5f} x (lin. reg.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')
        else:
            # Quadratic regression, linear + quadratic term, no intercept
            X_quad = np.column_stack([x_sorted, x_sorted**2])
            mod_quad = sm.OLS(y_sorted, X_quad)
            mod_quad_fit = mod_quad.fit(cov_type='HAC', cov_kwds={'maxlags': window-1})
            x_range = np.linspace(0, max(x_sorted), 100)
            X_range_quad = np.column_stack([x_range, x_range**2])
            pred_quad = mod_quad_fit.predict(X_range_quad)
            # Extract coefficients, standard errors and R² for quadratic regression
            coef_lin = mod_quad_fit.params[0]
            coef_quad = mod_quad_fit.params[1]
            mod_quad_ste_lin = mod_quad_fit.bse[0]
            mod_quad_ste_quad = mod_quad_fit.bse[1]
            mod_quad_rsq = mod_quad_fit.rsquared
            mod_quad_pvalue_lin = mod_quad_fit.pvalues[0]
            mod_quad_pvalue_quad = mod_quad_fit.pvalues[1]
            if not no_plots and central_reg != False:
                ax.plot(x_range, pred_quad, color='black' if plot_bg != 'black' else 'white', lw=2, ls='-', label='Quadratic regression')
                ax.text(52, -10, f"y = {coef_lin:.5f} x + {coef_quad:.5f} x² (quad. reg.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')
    if not no_plots:
        # Plot quantile regression lines
        if quantile_reg:
            ax.plot(x_sorted, yq_5, color='black' if plot_bg != 'black' else 'white', lw=2, ls=':', clip_on=True)
            ax.plot(x_sorted, yq_95, color='black' if plot_bg != 'black' else 'white', lw=2, ls=':', clip_on=True)
        # ax.fill_between(x_sorted, yq_5, yq_95, color='black' if plot_bg != 'black' else 'white', alpha=0.1, zorder=-10, clip_on=False,
                        #  label='5th-95th percentile envelope'
                        #  )

        if ax is None:
            # plot both quadratic quantile regression equations at bottom of plot
            if linear_95_reg:
                ax.text(5, -9, fr"95th percentile: $\Delta \mathrm{{T}}_\mathrm{{{region}}}$ = {res_95.params[0]:.4f} $\Delta \mathrm{{AMOC}}$", fontsize=10, color='black' if plot_bg != 'black' else 'white')
            else:
                ax.text(5, -9, f"$\Delta \mathrm{{T}}_\mathrm{{{region}}}$ = {res_95.params[0]:.4f} $\Delta \mathrm{{AMOC}}$ + {res_95.params[1]:.5f} $\Delta \mathrm{{AMOC}}$² (95th percentile quad.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')
            if linear_5_reg:
                ax.text(5, -10, f"$\Delta \mathrm{{T}}_\mathrm{{{region}}}$ = {res_5.params[0]:.4f} $\Delta \mathrm{{AMOC}}$ (5th percentile lin.)", fontsize=10, color='black' if plot_bg != 'black' else 'white')
            else:
                ax.text(5, -10, f"5th percentile: $\Delta \mathrm{{T}}_\mathrm{{{region}}}$ = {res_5.params[0]:.4f} $\Delta \mathrm{{AMOC}}$" + (" + " if res_5.params[1]>0 else " ") + f"{res_5.params[1]:.5f} $\Delta \mathrm{{AMOC}}^2$", fontsize=10, color='black' if plot_bg != 'black' else 'white')

        if bm_data:

            Delta_AMOC_hosing = (multi_model_dict['EC-Earth3'].sel(type='diff', scenar='pi', season='').amoc.isel(time=0) / \
                                multi_model_dict['EC-Earth3'].sel(type='control', scenar='pi', season='').amoc.isel(time=0)).values * 100
            
            Delta_AMOC_4x = (multi_model_dict['EC-Earth3'].sel(type='diff', scenar='ghg', season='').amoc.isel(time=0) / \
                            multi_model_dict['EC-Earth3'].sel(type='control', scenar='pi', season='').amoc.isel(time=0)).values * 100            

            if is_region:
                Delta_tas_hosing = weighted_area_lat(
                    multi_model_dict['EC-Earth3'].sel(type='diff', scenar='pi', season='').tas.where(
                    multi_model_dict['EC-Earth3'].mask.sel(region=region)).where(multi_model_dict['EC-Earth3'].mask.sel(region='LAND')==0)
                    ).mean('lat').mean('lon').isel(time=0)
                Delta_tas_4x = weighted_area_lat(
                    multi_model_dict['EC-Earth3'].sel(type='diff', scenar='ghg', season='').tas.where(
                    multi_model_dict['EC-Earth3'].mask.sel(region=region)).where(multi_model_dict['EC-Earth3'].mask.sel(region='LAND')==0)
                    ).mean('lat').mean('lon').isel(time=0)
            elif is_latlon: # untested whether this works correctly (have to verify that it is fully the same grid ordering as the other EC-Earth3 data) --> should work but still untested
                Delta_tas_hosing = multi_model_dict['EC-Earth3'].sel(type='diff', scenar='pi', season='').tas.isel(lat=lat, lon=lon).isel(time=0)
                Delta_tas_4x = multi_model_dict['EC-Earth3'].sel(type='diff', scenar='ghg', season='').tas.isel(lat=lat, lon=lon).isel(time=0)
            
            ax.scatter(Delta_AMOC_hosing, Delta_tas_hosing, color=hosmip_colors['EC-Earth3'], edgecolors='k', linewidths=1.5, marker='P', s=10*markersize, label='Bellomo & Mehling\npreindustrial', zorder=10) # '$\ominus$'
            ax.scatter(Delta_AMOC_4x, Delta_tas_4x, color=hosmip_colors['EC-Earth3'], edgecolors='k', linewidths=1.5, marker='X', s=10*markersize, label='Bellomo & Mehling\n'+r'4x$\mathrm{{CO}}_2$ experiment', zorder=10) # '$\otimes$'

        if vwb_data:

            Delta_AMOC_600_1500 = (
                (multi_model_dict['CESM1'].sel(type='control', scenar='ghg', season='').amoc.values -
                multi_model_dict['CESM1'].sel(type='control', scenar='pi', season='').amoc.values) -
                (multi_model_dict['CESM1'].sel(type='hosing', scenar='ghg', season='').amoc.values -
                multi_model_dict['CESM1'].sel(type='hosing', scenar='pi', season='').amoc.values)) / \
                multi_model_dict['CESM1'].sel(type='control', scenar='pi', season='').amoc.values * 100

            if is_region:
                vwb_tas_1500 = weighted_area_lat(
                    (multi_model_dict['CESM1'].sel(type='hosing', scenar='ghg', season='') - multi_model_dict['CESM1'].sel(type='hosing', scenar='pi', season='')).tas.where(
                    multi_model_dict['CESM1'].mask.sel(region=region)).where(multi_model_dict['CESM1'].mask.sel(region='LAND')==0)
                    ).mean('lat').mean('lon').values
                vwb_tas_600 = weighted_area_lat(
                    (multi_model_dict['CESM1'].sel(type='control', scenar='ghg', season='') - multi_model_dict['CESM1'].sel(type='control', scenar='pi', season='')).tas.where(
                    multi_model_dict['CESM1'].mask.sel(region=region)).where(multi_model_dict['CESM1'].mask.sel(region='LAND')==0)
                    ).mean('lat').mean('lon').values

                # marker='$\\ddagger$'
                ax.scatter(Delta_AMOC_600_1500, vwb_tas_1500 - vwb_tas_600, color=hosmip_colors['CESM2'], edgecolors='k', linewidths=1.5, marker='v', s=10*markersize, label='v. Westen & Baatsen\nRCP4.5 in 2400-2500', zorder=10, clip_on=False)

        if liu_data:

            Delta_AMOC = (multi_model_dict['CCSM4'].sel(type='control', scenar='ghg', season='').amoc.values -
                          multi_model_dict['CCSM4'].sel(type='hosing', scenar='ghg', season='').amoc.values) / \
                          multi_model_dict['CCSM4'].sel(type='control', scenar='pi', season='').amoc.values * 100

            if is_region:
                liu_tas_rcp85 = weighted_area_lat(
                    (multi_model_dict['CCSM4'].sel(type='control', scenar='ghg', season='').tas.where(
                    multi_model_dict['CCSM4'].mask.sel(region=region)
                    ).where(multi_model_dict['CCSM4'].mask.sel(region='LAND')==0))
                    ).mean('lat').mean('lon')
                liu_tas_rcp85_stable = weighted_area_lat(
                    (multi_model_dict['CCSM4'].sel(type='hosing', scenar='ghg', season='').tas.where(
                    multi_model_dict['CCSM4'].mask.sel(region=region)
                    ).where(multi_model_dict['CCSM4'].mask.sel(region='LAND')==0))
                    ).mean('lat').mean('lon')
            
                ax.scatter(Delta_AMOC, liu_tas_rcp85 - liu_tas_rcp85_stable, color=hosmip_colors['CESM2'], edgecolors='k', linewidths=1.5, marker='^', s=10*markersize, label='Liu et al. (2020)\nRCP8.5 in 2061-2080', zorder=10, clip_on=False)

        if boot_data:

            if is_region:
                boot_x = []
                boot_y = []
                for ssp_i in ['ssp126', 'ssp585']:
                    x = (multi_model_dict['CESM2'].amoc.sel(type='hosing', scenar=ssp_i, season='').rolling(time=window, center=True).mean('time').values - 
                         multi_model_dict['CESM2'].amoc.sel(type='control', scenar=ssp_i, season='').rolling(time=window, center=True).mean('time').values) / (
                         -multi_model_dict['CESM2'].amoc.sel(type='control', scenar='pi', season='').isel(time=0)).values * 100

                    y = weighted_area_lat(
                        multi_model_dict['CESM2'].tas.sel(type='hosing', scenar=ssp_i, season='').rolling(time=window, center=True).mean('time').where(
                        multi_model_dict['CESM2'].mask.sel(region=region)
                        ).where(multi_model_dict['CESM2'].mask.sel(region='LAND')==0)).mean('lat').mean('lon').values - \
                        weighted_area_lat(
                        multi_model_dict['CESM2'].tas.sel(type='control', scenar=ssp_i, season='').rolling(time=window, center=True).mean('time').where(
                        multi_model_dict['CESM2'].mask.sel(region=region)
                        ).where(multi_model_dict['CESM2'].mask.sel(region='LAND')==0)).mean('lat').mean('lon').values

                    ax.scatter(x, y, color='dodgerblue', edgecolors='k', linewidths=0.7, marker='d', s=1.5*markersize, label=f'Boot et al. (2024)\nSSP1-2.6 & SSP5-8.5\n2015 - 2100' if ssp_i=='ssp126' else '', clip_on=False)

                    mask = ~np.isnan(x) & ~np.isnan(y)
                    x = x[mask]
                    y = y[mask]
                    boot_x.append(x)
                    boot_y.append(y)

                if boot_regression:

                    boot_x_flat = np.concatenate(boot_x)
                    boot_y_flat = np.concatenate(boot_y)

                    # Sort x for plotting
                    boot_sort_idx = np.argsort(boot_x_flat)
                    boot_x_sorted = boot_x_flat[boot_sort_idx]
                    boot_y_sorted = boot_y_flat[boot_sort_idx]

                    boot_X = boot_x_sorted
                    if boot_intercept:
                        boot_X = sm.add_constant(boot_x_sorted)  # Add intercept column to regressor
                    boot_mod_lin = sm.OLS(boot_y_sorted, boot_X)
                    boot_x_range = np.linspace(0, 50, 100) # use 50% as end of extrapolation range because SSP1-2.6 in CESM2 already has 49.8% implicit weakening by 2100
                    boot_X_pred = sm.add_constant(boot_x_range) if boot_intercept else boot_x_range
                    boot_pred_lin = boot_mod_lin.fit(cov_type='HAC', cov_kwds={'maxlags': window-1}).predict(boot_X_pred)
                    if not no_plots:
                        ax.plot(boot_x_range, boot_pred_lin, color='dodgerblue', lw=3, zorder=9)

        if add_combined_results:
            
            reg_ds_mpi = load_regression_ds_mpi()

            # with open('../data/dfs_countries.pkl', 'rb') as f:
            #     dfs_countries = pickle.load(f)
            
            # combined_coefs = dict(dfs_countries['']['ssp126'].coef_all_ssps_combined)
            regional_combined_coef = weighted_area_lat(
                                    reg_ds_mpi.coef_ensmean.sel(season=season).where(multi_model_dict['MPI-ESM1-2-LR'].mask.sel(region=region)).where(multi_model_dict['MPI-ESM1-2-LR'].mask.sel(region='LAND')==0)
                                    ).mean('lat').mean('lon').values

            x_plot = np.linspace(0, 82, 200)
            ax.plot(x_plot, x_plot * (-regional_combined_coef * AMOC_pi_MPI / 100), color='brown', lw=3, label='This study' if combined_results_label else '', zorder=9, clip_on=True)

        if add_hosmip_mpi_regression:

            hosmip_reg_ds_dict = get_hosmip_reg_ds()
            regional_hosmip_mpi_coef = weighted_area_lat(
                                    hosmip_reg_ds_dict['MPI-ESM1-2-LR'].lin_coef_hosmip.sel(season=season).where(multi_model_dict['MPI-ESM1-2-LR'].mask.sel(region=region)).where(multi_model_dict['MPI-ESM1-2-LR'].mask.sel(region='LAND')==0)
                                    ).mean('lat').mean('lon').values

            x_plot = np.linspace(0, 82, 100)
            ax.plot(x_plot, x_plot * regional_hosmip_mpi_coef, color='blueviolet', lw=3, zorder=9, clip_on=True)

        # Formatting
        ax.set_xlim(0, 80)
        ax.set_ylim(low_ylim, 0.)
        ax.set_yticks(np.arange(low_ylim, 2, 2))
        ax.set_xticks(np.arange(0, 90, 10))
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
        if hosmip_ref_pi:
            ax.set_xlabel("Additional AMOC weakening") # [%]
        else:
            ax.set_xlabel("Weakening wrt first 10 yrs AMOC strength [%]")
        if hosmip_ref_pi:
            # ax.set_ylabel(f"Deviation from preindustrial {region if is_region else (lat, lon)} temperature [°C]")
            ax.set_ylabel(f"$\Delta \mathrm{{T}}_\mathrm{{{region if is_region else (lat, lon)}}}$ [°C]"+(f' in {season.upper()}' if season != '' else ''))
        else:
            ax.set_ylabel(f"Deviation from first 10 yrs {region if is_region else (lat, lon)} temperature [°C]")
        ax.axhline(0, color='black' if not plot_bg=='black' else 'white', lw=1., ls='--', clip_on=False)
        ax.axvline(0, color='black' if not plot_bg=='black' else 'white', lw=1., ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines["left"].set_position(("axes", -0.0))

        if ext_ax:
            return
        
        handles, labels = ax.get_legend_handles_labels()
        num_phantoms = 7 - (2 if bm_data else 0) - (3 if liu_data and vwb_data else 2 if vwb_data or liu_data else 0) - (1 if boot_data else 0) # 7 without external data, 5 incl. Bellomo, 2 incl. Bellomo & vWB & Liu
        for i in range(num_phantoms):
            handles.append(plt.Line2D([0], [0], color='none', alpha=0))
            labels.append('')
        ax.legend(handles, labels, frameon=False, ncols=2, fontsize=10, loc='upper left', bbox_to_anchor=(0., 0.52))

        if savefig != False:
            plt.savefig(savefig, dpi=400, bbox_inches='tight', transparent=True if plot_bg=='black' else False)

    return {
            '95pct_coef_lin': res_95.params[0],
            '95pct_coef_quad': res_95.params[1] if not linear_95_reg else 0,
            '5pct_coef_lin': res_5.params[0],
            '5pct_coef_quad': res_5.params[1] if not linear_5_reg else 0,
            'model_coef_lin': mod_lin_coef if only_model and single_model_linear else coef_lin if only_model and not single_model_linear else None,
            'model_coef_quad': coef_quad if only_model and not single_model_linear else None,
            'model_ste_lin': mod_lin_ste if only_model and single_model_linear else mod_quad_ste_lin if only_model and not single_model_linear else None,
            'model_ste_quad': mod_quad_ste_quad if only_model and not single_model_linear else None,
            'model_rsq': mod_lin_rsq if only_model and single_model_linear else mod_quad_rsq if only_model and not single_model_linear else None,
            'model_pvalue_lin': mod_lin_pvalue if only_model and single_model_linear else mod_quad_pvalue_lin if only_model and not single_model_linear else None,
            'model_pvalue_quad': mod_quad_pvalue_quad if only_model and not single_model_linear else None,
            'model_intercept_lin': mod_lin_intercept if only_model and single_model_linear else None,
            'model_intercept_ste_lin': mod_lin_intercept_ste if only_model and single_model_linear else None,
            'model_intercept_pvalue_lin': mod_lin_intercept_pvalue if only_model and single_model_linear else None,
            }

def cesm_regressions(boot_data, multi_model_dict, masks, ssp='all', lat=None, lon=None, region='EU', window=10, regressions=True, combined_reg=True, add_combined_intercept=False, no_plots=False, plot_bg='white', low_ylim=-10, xlim=80, hosmip_cesm=False, savefig=False, ext_ax=None, equation_y_pos=0.18, equation_y_spacing=0.07):

    is_latlon, is_region = False, False
    if lat != None and lon != None:
        is_latlon = True
    elif region != None:
        is_region = True
    else:
        print('No lat/lon or region provided.')

    if not no_plots:
        if ext_ax is not None:
            ax = ext_ax
            fig = ax.figure
        else:
            plt.style.use('default')
            if plot_bg == 'black':
                plt.style.use('dark_background')
                plt.rcParams['axes.facecolor'] = '#191919'
                plt.rcParams['figure.facecolor'] = '#191919'

            fig, ax = plt.subplots(figsize=(9, 6))

    combined_x = []
    combined_y = []
    ssp_coefs = {}
    ssp_stes = {}
    ssp_rsqs = {}

    ssp_i = None
    for ssp_i in ['ssp126', 'ssp585']:
        if ssp != 'all' and ssp_i != ssp:
                continue

        x = (boot_data.sel(scenar=ssp_i, type='hosing', season='').amoc.rolling(time=window, center=True).mean("time").values - boot_data.sel(scenar=ssp_i, type='control', season='').amoc.rolling(time=window, center=True).mean("time").values) / (-multi_model_dict['CESM2'].sel(scenar='pi', type='control', season='').amoc.isel(time=0).values) * 100

        if is_latlon:
            y = boot_data.sel(scenar=ssp_i, type='hosing', season='').tas.rolling(time=window, center=True).mean("time").isel(lat=lat, lon=lon).values - boot_data.sel(scenar=ssp_i, type='control', season='').tas.rolling(time=window, center=True).mean("time").isel(lat=lat, lon=lon).values
        elif is_region:
            y = weighted_area_lat(boot_data.sel(scenar=ssp_i, type='hosing', season='').tas.rolling(time=window, center=True).mean("time").where(masks['CESM2'][region]).where(masks['CESM2']['LAND']==0)).mean('lat').mean('lon').values - weighted_area_lat(boot_data.sel(scenar=ssp_i, type='control', season='').tas.rolling(time=window, center=True).mean("time").where(masks['CESM2'][region]).where(masks['CESM2']['LAND']==0)).mean('lat').mean('lon').values
        else:
            print('No lat/lon or region provided.')

        if not no_plots:
            boot_label = f'Boot et al. (2024, CESM2): {ssp_i.upper()}'
            ax.scatter(x, y, label=boot_label[:-2]+'-'+boot_label[-2]+'.'+boot_label[-1], s=15, c=hosing_colors[ssp_i]['ge'])
            
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        
        combined_x.extend(x)
        combined_y.extend(y)

        # Linear regression for this SSP
        x_with_const = sm.add_constant(x)
        model = sm.OLS(y, x_with_const).fit(cov_type='HAC', cov_kwds={'maxlags': window-1})
        x_range = np.linspace(0, 50, 200)
        x_range_with_const = sm.add_constant(x_range)
        y_pred = model.predict(x_range_with_const)
        if regressions and not no_plots:
            ax.plot(x_range, y_pred,
                    color=hosing_colors[ssp_i]['ge'], lw=1.5, linestyle='-')
        
        # Calculate confidence intervals for the predictions
        predictions = model.get_prediction(x_range_with_const)
        conf_int = predictions.conf_int(alpha=0.05)
        lower_bound = conf_int[:, 0]
        upper_bound = conf_int[:, 1]
        if regressions and not no_plots:
            ax.fill_between(x_range, lower_bound, upper_bound,
                            color=hosing_colors[ssp_i]['ge'], alpha=0.4, linewidth=0)
        
        # Plot regression equation and coefficients with p-values
        coef = model.params[1]
        ste = model.bse[1]
        rsq = model.rsquared
        ssp_coefs[ssp_i] = coef
        ssp_stes[ssp_i] = ste
        ssp_rsqs[ssp_i] = rsq
        intercept = model.params[0]
        p_value = model.pvalues[1]
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = ''
        if is_latlon:
            equation = fr"$\Delta T_{{lat{lat}\_lon{lon}}}$ = {coef:.3f}$^{{{{\text{{{significance}}}}}}}$$\Delta AMOC$ + {intercept:.2f}"
        else:
            equation = fr"$\Delta T_{{{region}}}$ = {coef:.3f}$^{{{{\text{{{significance}}}}}}}$$\Delta AMOC$ + {intercept:.2f}"
        if regressions and not no_plots:
            ax.text(0.05, equation_y_pos,
                    f"{hosing_names[ssp_i]['ge'][11:]}: {equation}",
                    transform=ax.transAxes, fontsize=12, verticalalignment='top', color=hosing_colors[ssp_i]['ge'])
        equation_y_pos -= equation_y_spacing

    # Combine both datasets for a third regression line
    combined_x = np.array(combined_x)
    combined_y = np.array(combined_y)

    if add_combined_intercept:
        combined_x_with_const = sm.add_constant(combined_x)
        combined_model = sm.OLS(combined_y, combined_x_with_const).fit(cov_type='HAC', cov_kwds={'maxlags': window-1})
    else:
        combined_model = sm.OLS(combined_y, combined_x).fit(cov_type='HAC', cov_kwds={'maxlags': window-1})
    combined_x_range = np.linspace(0, 50, 200)
    combined_x_pred = sm.add_constant(combined_x_range) if add_combined_intercept else combined_x_range
    combined_y_pred = combined_model.predict(combined_x_pred)
    if combined_reg and not no_plots:
        ax.plot(combined_x_range, combined_y_pred,
                color='black' if not plot_bg=='black' else 'white', lw=1.5, linestyle='-')

    # Calculate confidence intervals for the combined predictions
    combined_predictions = combined_model.get_prediction(combined_x_pred)
    combined_conf_int = combined_predictions.conf_int(alpha=0.05)
    combined_lower_bound = combined_conf_int[:, 0]
    combined_upper_bound = combined_conf_int[:, 1]
    if combined_reg and not no_plots:
        ax.fill_between(combined_x_range, combined_lower_bound, combined_upper_bound,
                        color='black' if not plot_bg=='black' else 'white', alpha=0.2)

    # Plot combined regression equation and coefficients with p-values
    combined_coef = combined_model.params[1] if add_combined_intercept else combined_model.params[0]
    if add_combined_intercept:
        combined_intercept = combined_model.params[0]
    combined_p_value = combined_model.pvalues[1] if add_combined_intercept else combined_model.pvalues[0]
    combined_ste = combined_model.bse[1] if add_combined_intercept else combined_model.bse[0]
    combined_rsq = combined_model.rsquared
    if combined_p_value < 0.001:
        combined_significance = '***'
    elif combined_p_value < 0.01:
        combined_significance = '**'
    elif combined_p_value < 0.05:
        combined_significance = '*'
    else:
        combined_significance = ''
    if is_latlon:
        combined_equation = fr"$\Delta T_{{lat{lat}\_lon{lon}}}$ = {combined_coef:.3f}$^{{{{\text{{{combined_significance}}}}}}}$$\Delta AMOC$ + {combined_intercept:.2f}" if add_combined_intercept else fr"$\Delta T_{{lat{lat}\_lon{lon}}}$ = {combined_coef:.3f}$^{{{{\text{{{combined_significance}}}}}}}$$\Delta AMOC$"
    else: 
        combined_equation = fr"$\Delta T_{{{region}}}$ = {combined_coef:.3f}$^{{{{\text{{{combined_significance}}}}}}}$$\Delta AMOC$ + {combined_intercept:.2f}" if add_combined_intercept else fr"$\Delta T_{{{region}}}$ = {combined_coef:.3f}$^{{{{\text{{{combined_significance}}}}}}}$$\Delta AMOC$"
    if combined_reg and not no_plots:
        ax.text(0.05, equation_y_pos,
                f"Combined: {combined_equation}",
                transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black' if not plot_bg=='black' else 'white')


    if hosmip_cesm:
        hosmip_regression_plot(multi_model_dict, quantile_reg=False, plot_bg=plot_bg, ax=ax, region=region, window=window, only_model='CESM2', only_model_label='NAHosMIP CESM2 (preindustrial)', low_ylim=low_ylim, central_reg=False, lat=lat, lon=lon)

    if not no_plots:
        ax.set_xlim(0, xlim)
        ax.set_ylim(low_ylim, 0)
        ax.set_xlabel("Additional AMOC weakening [%]")
        if is_latlon:
            ax.set_ylabel(f"Deviation from no-hosing lat={lat}, lon={lon} temperature [°C]")
        elif is_region:
            ax.set_ylabel(f"Dev. from no-hosing {region} temp. [°C]")
        ax.axhline(0, color='black' if not plot_bg=='black' else 'white', lw=1.5, ls='--')
        ax.axvline(0, color='black' if not plot_bg=='black' else 'white', lw=1.5, ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(fontsize=12, frameon=False, loc='center left')

        if savefig is not False:
            plt.savefig(savefig, dpi=400, bbox_inches='tight', transparent=True if plot_bg=='black' else False)

    if combined_reg and add_combined_intercept:
        combined_quadruple = np.round(combined_intercept, 4), np.round(combined_coef, 5), np.round(combined_ste, 5), np.round(combined_rsq, 4)
    elif combined_reg and not add_combined_intercept:
        combined_quadruple = np.nan, np.round(combined_coef, 5), np.round(combined_ste, 5), np.round(combined_rsq, 4)

    # SSP-specific tuples: (coef, ste, rsq)
    ssp126_tuple = (np.round(ssp_coefs['ssp126'], 5), np.round(ssp_stes['ssp126'], 5), np.round(ssp_rsqs['ssp126'], 4)) if ssp == 'all' or ssp == 'ssp126' else None
    ssp585_tuple = (np.round(ssp_coefs['ssp585'], 5), np.round(ssp_stes['ssp585'], 5), np.round(ssp_rsqs['ssp585'], 4)) if ssp == 'all' or ssp == 'ssp585' else None

    return {
        'combined': combined_quadruple,
        'ssp126': ssp126_tuple,
        'ssp585': ssp585_tuple,
    }

def plot_net_cooling_ranges(dfs_countries, T_ref='pi', AMOC_metric='rel', season='', plot_bg='white', ext_ax=None, title=True):
    plt.style.use('default')
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    if ext_ax is None:
        fig, ax = plt.subplots(figsize=(10, 14))
    else:
        ax = ext_ax

    effect_sizes = {}
    for r in dfs_countries['']['ssp126'].index:
        effect_sizes[r] = dfs_countries['']['ssp245'][f'CP_pi_hosmip_5pct_{AMOC_metric}'][r]

    regions = sorted(effect_sizes.keys(), key=lambda x: effect_sizes[x], reverse=True if AMOC_metric == 'rel' else False)
    countries = [r for r in regions if len(r) == 2 and r != 'EU']

    group_centers = np.arange(len(countries))
    bar_height = 0.3

    bar_alpha = 0.7
    line_alpha = 1.0

    for (i, r) in enumerate(countries):
        
        ax.barh(group_centers[i] + bar_height, 
                dfs_countries[season]['ssp126'][f'CP_{T_ref}_hosmip_5pct_{AMOC_metric}'][r] - dfs_countries[season]['ssp126'][f'CP_{T_ref}_hosmip_95pct_{AMOC_metric}'][r],
                left = dfs_countries[season]['ssp126'][f'CP_{T_ref}_hosmip_95pct_{AMOC_metric}'][r], 
                height=bar_height * 0.8, alpha=bar_alpha, color=hosing_colors['ssp126']['ge'])
        for model in hosmip_labels:
            ax.barh(group_centers[i] + bar_height,
                    (0.4 if AMOC_metric == 'rel' else 0.08),
                    left = dfs_countries[season]['ssp126'][f'CP_{T_ref}_hosmip_'+model+f'_{AMOC_metric}'][r] - (0.2 if AMOC_metric == 'rel' else 0.04),
                    height=bar_height * 1.0, alpha=line_alpha, color=hosing_colors['ssp126']['ge'] if model=='MPI-ESM1-2-LR' else 'none' if model=='MPI-ESM1-2-HR' else 'none' if model=='CESM2' else 'none')
        if AMOC_metric == 'rel':
            ax.scatter((1 - dfs_countries[season]['ssp126'][f'CP_{T_ref}_all_ssps_combined'][r] / AMOC_pi_MPI) * 100, group_centers[i] + 1.1 * bar_height, alpha=line_alpha, color=hosing_colors['ssp126']['ge'], marker='*')
        else:
            ax.scatter(dfs_countries[season]['ssp126'][f'CP_{T_ref}_all_ssps_combined'][r], group_centers[i] + 1.1 * bar_height, alpha=line_alpha, color=hosing_colors['ssp126']['ge'], marker='*')

        ax.barh(group_centers[i], 
                dfs_countries[season]['ssp245'][f'CP_{T_ref}_hosmip_5pct_{AMOC_metric}'][r] - dfs_countries[season]['ssp245'][f'CP_{T_ref}_hosmip_95pct_{AMOC_metric}'][r],
                left = dfs_countries[season]['ssp245'][f'CP_{T_ref}_hosmip_95pct_{AMOC_metric}'][r], 
                height=bar_height * 0.8, alpha=bar_alpha, color=hosing_colors['ssp245']['ge'])
        for model in hosmip_labels:
            ax.barh(group_centers[i],
                    (0.4 if AMOC_metric == 'rel' else 0.08),
                    left = dfs_countries[season]['ssp245'][f'CP_{T_ref}_hosmip_'+model+f'_{AMOC_metric}'][r] - (0.2 if AMOC_metric == 'rel' else 0.04),
                    height=bar_height * 1.0, alpha=line_alpha, color='darkorange' if model=='MPI-ESM1-2-LR' else 'none' if model=='MPI-ESM1-2-HR' else 'none' if model=='CESM2' else 'none')
        if AMOC_metric == 'rel':
            ax.scatter((1 - dfs_countries[season]['ssp245'][f'CP_{T_ref}_all_ssps_combined'][r] / AMOC_pi_MPI) * 100, group_centers[i] + 0.1 * bar_height, alpha=line_alpha, color=hosing_colors['ssp245']['ge'], marker='*')
        else:
            ax.scatter(dfs_countries[season]['ssp245'][f'CP_{T_ref}_all_ssps_combined'][r], group_centers[i] + 0.1 * bar_height, alpha=line_alpha, color=hosing_colors['ssp245']['ge'], marker='*')

        ax.barh(group_centers[i] - bar_height, 
                dfs_countries[season]['ssp370'][f'CP_{T_ref}_hosmip_5pct_{AMOC_metric}'][r] - dfs_countries[season]['ssp370'][f'CP_{T_ref}_hosmip_95pct_{AMOC_metric}'][r],
                left = dfs_countries[season]['ssp370'][f'CP_{T_ref}_hosmip_95pct_{AMOC_metric}'][r], 
                height=bar_height * 0.8, alpha=bar_alpha, color=hosing_colors['ssp370']['ge'])
        for model in hosmip_labels:
            ax.barh(group_centers[i] - bar_height,
                    (0.4 if AMOC_metric == 'rel' else 0.08),
                    left = dfs_countries[season]['ssp370'][f'CP_{T_ref}_hosmip_'+model+f'_{AMOC_metric}'][r] - (0.2 if AMOC_metric == 'rel' else 0.04),
                    height=bar_height * 1.0, alpha=line_alpha, color='firebrick' if model=='MPI-ESM1-2-LR' else 'none' if model=='MPI-ESM1-2-HR' else 'none' if model=='CESM2' else 'none')
        if AMOC_metric == 'rel':
            ax.scatter((1 - dfs_countries[season]['ssp370'][f'CP_{T_ref}_all_ssps_combined'][r] / AMOC_pi_MPI) * 100, group_centers[i] - 0.9 * bar_height, alpha=line_alpha, color=hosing_colors['ssp370']['ge'], marker='*')
        else:
            ax.scatter(dfs_countries[season]['ssp370'][f'CP_{T_ref}_all_ssps_combined'][r], group_centers[i] - 0.9 * bar_height, alpha=line_alpha, color=hosing_colors['ssp370']['ge'], marker='*')

    # shade every second row in light gray
    for i in range(len(countries)):
        if i % 2 == 0:
            if AMOC_metric == 'rel':
                ax.add_patch(Rectangle((-20, group_centers[i] - 0.5), 120, 1, color='black', alpha=0.15, zorder=0, linewidth=0, clip_on=False))
            else:
                ax.add_patch(Rectangle((-3.5, group_centers[i] - 0.5), 23.5, 1, color='black', alpha=0.15, zorder=0, linewidth=0, clip_on=False))


    range_y = 0.98  # y position just above the plot (in axes coords)
    range_height = 0.004  # thickness of the colored range

    for idx, ssp in enumerate(ssps):
        # Get the AMOC_extent range for this SSP
        lower, upper = AMOC_extent[ssp]
        x_min = lower if AMOC_metric == 'rel' else (1 - lower / 100) * AMOC_pi_MPI
        x_max = upper if AMOC_metric == 'rel' else (1 - upper / 100) * AMOC_pi_MPI
        # Convert to axes fraction
        x_min_frac = x_min / (100 if AMOC_metric == 'rel' else 20)
        x_max_frac = x_max / (100 if AMOC_metric == 'rel' else 20)
        # Rectangle in axes coordinates
        rect = Rectangle(
            (x_min_frac, range_y - idx * 0.007),  # (x, y) in axes fraction
            x_max_frac - x_min_frac,  # width
            range_height,             # height
            transform=ax.transAxes,
            color=hosing_colors[ssp]['ge'],
            alpha=1.0,
            clip_on=False,
        )
        ax.add_patch(rect)

    arrow_start_x = 0.65 if AMOC_metric == 'rel' else 0.3      # right edge in axes fraction
    arrow_end_x = 0.55 if AMOC_metric == 'rel' else 0.4       # where the arrow points to
    arrow_y = range_y - 0.005

    # Add arrow (in axes coordinates)
    ax.annotate(
        '', xy=(arrow_end_x, arrow_y), xytext=(arrow_start_x, arrow_y),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='-|>', lw=1., color='black' if plot_bg != 'black' else 'white'),
        annotation_clip=False
    )

    # Add text box (in axes coordinates)
    ax.text(
        0.84 if AMOC_metric == 'rel' else 0.3,  # x position in axes fraction
        arrow_y, 'CMIP6 projections',
        va='center', ha='right',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', fc='white' if plot_bg != 'black' else '#222', ec='none', alpha=0.8),
        color='black' if plot_bg != 'black' else 'white',
        fontsize=10, weight='bold'
    )

    if AMOC_metric == 'rel':
        for x in [20, 40, 60, 80]:
            ax.axvline(x=x, ymin=0.03, color='black' if plot_bg != 'black' else 'white', linestyle=(0, (1, 4)), linewidth=0.8, alpha=1.0, zorder=0)

    if AMOC_metric == 'rel':
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
        ax.set_xlabel('AMOC weakening')
        ref_temperature = 'preindustrial' if T_ref == 'pi' else 'present-day'
        season_text = 'in winter ' if season == 'djf' else 'in summer ' if season == 'jja' else ''
        if title:
            ax.set_title(f'How much AMOC weakening would lead to net cooling {season_text}wrt {ref_temperature}?', fontsize=14, color='black' if plot_bg != 'black' else 'white')
        ax.xaxis.set_ticks_position('both')
        # ax.xaxis.set_label_position('bottom')  # keep label at bottom, or use 'top' for top
        ax.tick_params(axis='x', which='both', top=True, labeltop=True, bottom=True, labelbottom=True, labelsize=10)
    else:
        ax.set_xlim(0, 20)
        ax.set_xticks(np.arange(0, 21, 4))
        ax.set_xlabel('AMOC strength [Sv]')
        if title:
            ax.set_title('At which AMOC strength would net cooling emerge?', fontsize=14, color='black' if plot_bg != 'black' else 'white')

    ax.set_yticks(group_centers)
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_yticklabels(country_names[c] for c in countries)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines["left"].set_position(("axes", 0.0))
    ax.spines["bottom"].set_position(("axes", 0.03))

def plot_net_cooling_ranges_mpi_cesm(reg_ds_mpi, reg_ds_cesm, masks, hosmip_reg_ds_dict=None, hosmip_markers=False, T_ref='pi', season='', plot_bg='white', ext_ax=None, title=True, savefig=False):
    plt.style.use('default')
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    if ext_ax is None:
        fig, ax = plt.subplots(figsize=(10, 14))
    else:
        ax = ext_ax

    # if T_ref != 'pi':
    #     raise NotImplementedError("Only T_ref='pi' is implemented in this function.")
    # if season != '':
    #     raise NotImplementedError("Only annual data (season='') is implemented in this function.")

    cesm_effect_sizes = {}
    mpi_effect_sizes = {}
    for r in regions:
        cesm_effect_sizes[r] = weighted_area_lat(reg_ds_cesm.req_weakening_pi.sel(scenar='ssp126').where(masks['CESM2'][r])).mean('lat').mean('lon')
        mpi_effect_sizes[r] = weighted_area_lat(reg_ds_mpi.req_weakening_pi.sel(scenar='ssp126').where(masks['MPI-ESM1-2-LR'][r])).mean('lat').mean('lon')

    # countries = sorted(cesm_effect_sizes.keys(), key=lambda x: cesm_effect_sizes[x], reverse=True)
    # countries = sorted(mpi_effect_sizes.keys(), key=lambda x: mpi_effect_sizes[x], reverse=True)
    countries = list(reversed(regions))

    group_centers = np.arange(len(countries)+1)  # +1 for the EU aggregate
    bar_height = 0.25

    bar_alpha = 0.7
    line_alpha = 1.0

    vertical_offsets = {'ssp126': 1., 'ssp245': 0., 'ssp370': -1.}

    # Build custom legend handles
    legend_color = 'white' if plot_bg == 'black' else 'black'
    legend_handles = []
    legend_labels = []

    legend_handles.append(Line2D([], [], marker='*', color=legend_color, linestyle='None', markersize=8))
    legend_labels.append('MPI-ESM1.2-LR (this study)')

    if season == '':
        legend_handles.append(Line2D([], [], marker='o', color=legend_color, linestyle='None', markersize=5))
        legend_labels.append('CESM2 (Boot et al. 2024)')

        # Combined forcing: thin bar
        legend_handles.append(Line2D([], [], color=legend_color, alpha=bar_alpha, linewidth=5, solid_capstyle='butt'))
        legend_labels.append('Combined forcing range (MPI-ESM1.2-LR & CESM2)')

    # NAHosMIP: thin transparent bar
    hosmip_bar_h = Line2D([], [], color=legend_color, alpha=0.3, linewidth=5, solid_capstyle='butt')
    legend_handles.append(hosmip_bar_h)
    legend_labels.append('Preindustrial hosing range (NAHosMIP models)')

    if hosmip_markers:
        for mkr, sz, lbl in [('*', 4, 'NAHosMIP MPI-ESM1.2-LR'), ('^', 4, 'NAHosMIP MPI-ESM1.2-HR'),
                              ('o', 4, 'NAHosMIP CESM2'), ('<', 4, 'NAHosMIP HadGEM3-GC3.1-MM'),
                              ('>', 4, 'NAHosMIP HadGEM3-GC3.1-LL'), ('v', 4, 'NAHosMIP EC-Earth3')]:
            legend_handles.append(Line2D([], [], marker=mkr, color=legend_color, linestyle='None', markersize=sz, alpha=0.5))
            legend_labels.append(lbl)

    mpi_cesm_values_ssp126 = {}
    for (i, r) in enumerate(['EU']+countries):
        for ssp_i in ssps:

            # mpi_esm_value = (1 - dfs_countries[season][ssp_i][f'CP_{T_ref}_all_ssps_combined'][r] / AMOC_pi_MPI) * 100
            if T_ref == 'pi':
                mpi_value = weighted_area_lat(reg_ds_mpi.req_weakening_pi.sel(scenar=ssp_i, season=season).where(masks['MPI-ESM1-2-LR'][r]).where(masks['MPI-ESM1-2-LR']['LAND']==0)).mean('lat').mean('lon')
                cesm_value = weighted_area_lat(reg_ds_cesm.req_weakening_pi.sel(scenar=ssp_i).where(masks['CESM2'][r]).where(masks['CESM2']['LAND']==0)).mean('lat').mean('lon')
            elif T_ref == 'pd':
                mpi_value = weighted_area_lat(reg_ds_mpi.req_weakening_pd.sel(scenar=ssp_i, season=season).where(masks['MPI-ESM1-2-LR'][r])).mean('lat').mean('lon')
                cesm_value = weighted_area_lat(reg_ds_cesm.req_weakening_pd.sel(scenar=ssp_i).where(masks['CESM2'][r])).mean('lat').mean('lon')
            else:
                raise NotImplementedError("Only T_ref='pi' and T_ref='pd' are implemented in this function.")
            
            if ssp_i == 'ssp126':
                mpi_cesm_values_ssp126[r] = np.nanmin([mpi_value.values.item(), cesm_value.values.item()])

            hosmip_values = []
            for model in hosmip_labels:
                if T_ref == 'pi':
                    hosmip_value = weighted_area_lat(hosmip_reg_ds_dict[model].req_weakening_pi.sel(scenar=ssp_i, season=season).where(masks[model][r]).where(masks[model]['LAND']==0)).mean('lat').mean('lon')
                elif T_ref == 'pd':
                    hosmip_value = weighted_area_lat(hosmip_reg_ds_dict[model].req_weakening_pd.sel(scenar=ssp_i).where(masks[model][r]).where(masks[model]['LAND']==0)).mean('lat').mean('lon')
                    if season != '':
                        raise NotImplementedError("Seasonal data for T_ref='pd' HosMIP is not implemented.")
                else:
                    raise NotImplementedError("Only T_ref='pi' and T_ref='pd' are implemented in this function.")
                # if model in ['MPI-ESM1-2-LR', 'CESM2']:
                #     hosmip_values.append(hosmip_value.values.item())
                # else:
                #     continue
                hosmip_values.append(hosmip_value.values.item())
                
                if hosmip_markers:
                    hosmip_marker = 'x'
                    if model == 'MPI-ESM1-2-LR':
                        hosmip_marker = '*'
                    elif model == 'MPI-ESM1-2-HR':
                        hosmip_marker = '^'
                    elif model == 'CESM2':
                        hosmip_marker = 'o'
                    elif model == 'HadGEM3-GC3-1MM':
                        hosmip_marker = '<'
                    elif model == 'HadGEM3-GC3-1LL':
                        hosmip_marker = '>'
                    elif model == 'EC-Earth3':
                        hosmip_marker = 'v'
                    else:
                        hosmip_marker = ''
                    ax.scatter(hosmip_value.values,
                            group_centers[i] + vertical_offsets[ssp_i] * bar_height, alpha=bar_alpha, color=hosing_colors[ssp_i]['ge'],
                            marker=hosmip_marker, s=10, clip_on=True)

            ax.barh(group_centers[i] - 0.01 + vertical_offsets[ssp_i] * bar_height,
                    np.nanmax(hosmip_values) - np.nanmin(hosmip_values),
                    left = np.nanmin(hosmip_values),
                    height=bar_height * 0.8, alpha=0.3, color=hosing_colors[ssp_i]['ge'], clip_on=True)

            if season == '':
                ax.barh(group_centers[i] - 0.01 + vertical_offsets[ssp_i] * bar_height,
                        mpi_value - cesm_value,
                        left = cesm_value,
                        height=bar_height * 0.8, alpha=bar_alpha, color=hosing_colors[ssp_i]['ge'])

            ax.scatter(mpi_value,
                       group_centers[i] + vertical_offsets[ssp_i] * bar_height, alpha=line_alpha, color=hosing_colors[ssp_i]['ge'], marker='*', s=50)
            if season == '':
                ax.scatter(cesm_value,
                        group_centers[i] + vertical_offsets[ssp_i] * bar_height, alpha=line_alpha, color=hosing_colors[ssp_i]['ge'], marker='o', s=25)

    # shade every second row in light gray
    for i in range(len(countries)+1):
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, group_centers[i] - 0.5), 120, 1, color='black', alpha=0.15, zorder=0, linewidth=0, clip_on=False))

    range_y = 1.03  # y position above the upper x-axis (in axes coords)
    range_height = 0.006  # thickness of the colored range

    for idx, ssp in enumerate(ssps):
        # Get the AMOC_extent range for this SSP
        lower, upper = AMOC_extent[ssp]
        x_min = lower # 100 -
        x_max = upper
        # Convert to axes fraction
        x_min_frac = x_min / 100
        x_max_frac = x_max / 100
        # Rectangle in axes coordinates
        rect = Rectangle(
            (x_min_frac, range_y + idx * 0.01),  # (x, y) in axes fraction
            x_max_frac - x_min_frac,  # width
            range_height,             # height
            transform=ax.transAxes,
            color=hosing_colors[ssp]['ge'],
            alpha=1.0,
            clip_on=False,
        )
        ax.add_patch(rect)

        # MPI-ESM projected AMOC weakening as vertical line
        mpi_amoc_strength = float(reg_ds_mpi.AMOC_2090_2100.isel(season=0).sel(scenar=ssp))
        mpi_weakening_frac = convert_strength_to_weakening(mpi_amoc_strength) / 100
        base_rgb = plt.matplotlib.colors.to_rgb(hosing_colors[ssp]['ge'])
        h, l, s = colorsys.rgb_to_hls(*base_rgb)
        line_color = colorsys.hls_to_rgb(h, max(0, l * 0.7), min(1, s * 1.3))
        bar_y = range_y + idx * 0.01
        ax.plot(
            [mpi_weakening_frac, mpi_weakening_frac],
            [bar_y - 0.001, bar_y + range_height + 0.001],
            color=line_color, linewidth=2,
            transform=ax.transAxes, clip_on=False, zorder=11
        )

    arrow_start_x = 0.65      # right edge in axes fraction (0.365 when arrows on the left)
    arrow_end_x = 0.55       # where the arrow points to (0.465 when arrows on the left)
    arrow_y = range_y + 0.009

    # Add arrow (in axes coordinates)
    ax.annotate(
        '', xy=(arrow_end_x, arrow_y), xytext=(arrow_start_x, arrow_y),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='-|>', lw=1., color='black' if plot_bg != 'black' else 'white'),
        annotation_clip=False
    )

    ax.legend(legend_handles, legend_labels,
              frameon=True, bbox_to_anchor=(0.02, 0.055), loc='lower left', fontsize=10,
              title=('Annual' if season=='' else f'Seasonal ({season.upper()})') + ' net cooling points',
              title_fontproperties={'weight': 'bold', 'size': 10})

    # Add text box (in axes coordinates)
    ax.text(
        0.66,  # x position in axes fraction (0.18 when on the left)
        arrow_y, 'CMIP6 AMOC projections',
        va='center', ha='left',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', fc='white' if plot_bg != 'black' else '#222', ec='none', alpha=0.8),
        color='black' if plot_bg != 'black' else 'white',
        fontsize=10, weight='bold'
    )

    for x in np.arange(10, 100, 10):
        ax.axvline(x=x, ymin=0.03, color='black' if plot_bg != 'black' else 'white', linestyle=(0, (1, 4)), linewidth=0.8, alpha=1.0, zorder=0)

    ax.set_xlim(0, 100)
    ax.set_ylim(-1.7, len(countries)+0.5)
    ax.set_xticks(np.arange(0, 110, 10))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.set_xlabel('AMOC weakening')
    ref_temperature = 'preindustrial' if T_ref == 'pi' else 'present-day'
    season_text = 'in winter ' if season == 'djf' else 'in summer ' if season == 'jja' else ''
    if title:
        ax.set_title(f'How much AMOC weakening would lead to net cooling {season_text}wrt {ref_temperature}?', fontsize=14, color='black' if plot_bg != 'black' else 'white')
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_label_position('bottom')  # keep label at bottom, or use 'top' for top
    ax.tick_params(axis='x', which='both', top=True, labeltop=True, bottom=True, labelbottom=True, labelsize=10)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.set_yticks(group_centers)
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_yticklabels(['European mean']+[country_names[c] for c in countries])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.spines["left"].set_position(("axes", 0.0))
    ax.spines["bottom"].set_position(("axes", 0.03))

    if savefig:
        fig.savefig(f'../plots/cesm2_mpi_net_cooling_ranges_Tref-{T_ref}_season-{season}_plotbg-{plot_bg}_hosmip_markers-{hosmip_markers}.png', dpi=400, bbox_inches='tight', transparent=True if plot_bg=='black' else False)

def get_mask_plot(masks, region, ext_ax=None, alpha=0.15, plot_bg='white'):

    plt.style.use('default')
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    if ext_ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.Robinson(central_longitude=8)})
    else:
        ax = ext_ax
    ax.spines['geo'].set_visible(False)
    ax.set_rasterized(True)
    ax.coastlines(resolution='50m', color='black' if not plot_bg=='black' else 'white', linewidth=.5)
    # ax.add_feature(cfeature.BORDERS, edgecolor='black' if not plot_bg=='black' else 'white', linewidth=.5)
    ax.add_feature(cfeature.LAND, facecolor='white' if not plot_bg=='black' else '#191919')
    ax.add_feature(cfeature.OCEAN, facecolor='white' if not plot_bg=='black' else '#191919', zorder=2)

    masks['HadGEM3-GC3-1MM'][region].plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='Greys',
        alpha=alpha,
        add_colorbar=False,
        rasterized=True
    )

    # ax.set_extent([-9.5, 38.5, 34.8, 72], crs=ccrs.PlateCarree())
    ax.set_extent([-11.0, 38.5, 30, 72], crs=ccrs.PlateCarree())

    # add_square(ax, 39.5, 65,  64.5, 75.5, colour='white' if not plot_bg=='black' else '#191919') # cut off North Russia where previously there was no data
    add_square(ax, -15., 0,  60, 75.5, colour='white' if not plot_bg=='black' else '#191919') # get rid of Nordic sea small islands