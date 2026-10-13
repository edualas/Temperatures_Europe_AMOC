########################################
# %%
# LOAD PACKAGES

import io
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import sys, pathlib
if "__file__" in globals():  # standalone run: ensure scripts/ is importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import functions
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon


########################################
# %%
# MPI-ESM1.2-LR EUROPE MAPS W.R.T. PRESENT DAY (supplementary; shareable package)
#
# Columns = scenarios. Row 1: AMOC weakening at which end-of-century
# temperature falls back to its 1991-2020 level (net cooling, cf. Fig. 2 PD
# row). Rows 2-3: end-of-century ΔT w.r.t. 1991-2020 at fixed AMOC weakening.
#
# Formulas (w = AMOC weakening in % of the 1850-1899 AMOC):
#   ΔT(w)         = dT_future + slope * (w - w_future)
#   w_net_cooling = w_future - dT_future / slope        (the w at which ΔT(w) = 0; only where slope < 0)
#
# Variables in the NetCDF (dims season, scenario, lat, lon as applicable):
#   dT_future      temperature change 2091-2100 vs 1991-2020 at the model's own projected weakening w_future [K]
#   w_future       MPI-ESM AMOC weakening 2091-2100 [%]
#   slope          temperature change per additional % of AMOC weakening [K/%]; negative = cooling as the AMOC weakens
#   slope_se       standard error of slope [K/%] (not used in the figure)
#   cmip6_w_range  CMIP6 multi-model range of AMOC weakening 2091-2100 [%], dashed contours in row 1
#   land_mask      1 = land, 0 = ocean
#
# Names in make_figure:
#   w_net          w_net_cooling (row 1)
#   dT             ΔT(w) for each w in w_levels (rows 2-3)
#   nc_*           colour map, levels and norm of row 1 (net cooling)
#   dT_cmap, dT_norm, bounds   colour map, norm and bin edges of rows 2-3
#
# Self-contained: needs only numpy, xarray, matplotlib, cartopy and the NetCDF
# written by processing/process_mpi_pd_maps.py. Default paths assume the run
# starts in the repo's scripts/ folder; pass datapath/savedir otherwise.

DATAPATH = '../data/mpi_esm_europe_amoc_pd_maps.nc' if functions.DATA_MODE == 'raw' else functions.local_path + 'mpi_esm_europe_amoc_pd_maps.nc'
SAVEDIR = '../plots'


########################################
# %%
# FIGURE FUNCTION

SSP_LABELS = {'ssp126': 'SSP1-2.6', 'ssp245': 'SSP2-4.5', 'ssp370': 'SSP3-7.0'}
RC = {'font.family': ['Nimbus Sans TT', 'Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 7, 'axes.titlesize': 7,
      'pdf.fonttype': 42, 'axes.linewidth': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2,
      'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'ytick.major.pad': 2, 'savefig.pad_inches': 0}


def _square(ax, lon0, lon1, lat0, lat1, colour):
    ax.add_geometries([Polygon([(lon0, lat0), (lon0, lat1), (lon1, lat1), (lon1, lat0)])],
                      crs=ccrs.PlateCarree(), facecolor=colour, edgecolor=colour, zorder=4)


def make_figure(datapath=DATAPATH, savedir=SAVEDIR, season='', w_levels=(25, 75), plot_bg='white', dT_step=0.5):
    plt.style.use('dark_background' if plot_bg == 'black' else 'default')
    plt.rcParams.update(RC)
    bg = '#191919' if plot_bg == 'black' else 'white'
    fg = 'white' if plot_bg == 'black' else 'black'
    grey = '#b0b0b0'
    season = season.lower() or 'annual'  # repo convention '' = annual; the NetCDF labels it 'annual'
    ds = xr.open_dataset(datapath).sel(season=season)
    w_net = (ds.w_future - ds.dT_future / ds.slope).where(ds.slope < 0)
    dT = xr.concat([ds.dT_future + ds.slope * (w - ds.w_future) for w in w_levels], dim=xr.DataArray(list(w_levels), dims='w_level'))

    # net-cooling colours as in Fig. 2 (YlGnBu from 0.17, reversed; grey beyond 0-100 %)
    nc_cmap = LinearSegmentedColormap.from_list('nc', cm.YlGnBu(np.linspace(0.17, 1.0, 256))).reversed()
    nc_cmap.set_under(grey); nc_cmap.set_over(grey)
    nc_levels = np.linspace(0, 100, 11)
    nc_norm = BoundaryNorm(nc_levels, ncolors=nc_cmap.N, clip=False)

    # ΔT: discrete diverging, 0 on a bin edge, sampled symmetrically (red = warmer, blue = cooler)
    land = ds.land_mask == 1
    vmin = np.floor(float(dT.where(land).min()) / dT_step) * dT_step
    vmax = np.ceil(float(dT.where(land).max()) / dT_step) * dT_step
    bounds = np.round(np.arange(vmin, vmax + dT_step / 2, dT_step), 6)
    sym = max(abs(vmin), abs(vmax)) + dT_step
    cw = plt.get_cmap('RdBu_r')
    gap = 0.06  # skip the near-white centre of the cmap so the bins either side of 0 differ clearly
    pos = lambda c: 0.5 + np.sign(c) * (gap + (0.5 - gap) * abs(c) / sym)
    dT_cmap = ListedColormap([cw(pos(c)) for c in (bounds[:-1] + bounds[1:]) / 2])
    dT_cmap.set_under(cw(pos(vmin - dT_step / 2)))
    dT_cmap.set_over(cw(pos(vmax + dT_step / 2)))
    dT_norm = BoundaryNorm(bounds, ncolors=dT_cmap.N, clip=False)

    fig, axes = plt.subplots(3, 3, figsize=(183 / 25.4, 5.0), subplot_kw={'projection': ccrs.Robinson(central_longitude=8)})
    fig.patch.set_facecolor(bg)
    plt.subplots_adjust(left=0.02, right=0.86, top=0.90, bottom=0.02, wspace=0.03, hspace=0.18)
    for j, ssp in enumerate(ds.scenario.values):
        fields = [(w_net.sel(scenario=ssp), nc_cmap, nc_norm, nc_levels)] + \
                 [(dT.sel(scenario=ssp, w_level=w), dT_cmap, dT_norm, bounds) for w in w_levels]
        for i, (field, cmap, norm, levels) in enumerate(fields):
            ax = axes[i, j]
            ax.spines['geo'].set_visible(False)
            ax.coastlines(color=fg, linewidth=0.33, zorder=3)
            ax.add_feature(cfeature.BORDERS, edgecolor=fg, linewidth=0.33, zorder=3)
            ax.add_feature(cfeature.LAND, facecolor=grey)
            ax.add_feature(cfeature.OCEAN, facecolor=bg, zorder=2)
            m = ax.contourf(ds.lon, ds.lat, field, levels=levels, cmap=cmap, norm=norm,
                            extend='neither' if i == 0 else 'both', transform=ccrs.PlateCarree())
            functions.rasterize_map(ax, m)
            if i == 0:
                for lw, colour, ls in [(1.66, 'white', '-'), (0.55, 'black', '--')]:
                    ax.contour(ds.lon, ds.lat, field, levels=sorted(ds.cmip6_w_range.sel(scenario=ssp).values),
                               colors=colour, linewidths=lw, linestyles=ls, transform=ccrs.PlateCarree(), zorder=1)
            ax.set_extent([-18.5, 38.5, 34.8, 72], crs=ccrs.PlateCarree())
            _square(ax, -13.5, 0, 60, 75.5, bg)    # Nordic-sea islands
            _square(ax, -30, -18, 68, 75, bg)      # Greenland
            ax.text(0.02, 0.98, 'abcdefghi'[3 * i + j], transform=ax.transAxes, fontsize=8, fontweight='bold',
                    ha='left', va='top', color=fg, zorder=10)
            if i == 0: nc_map = m
            else: dT_map = m

    season_txt = {'annual': 'annual mean', 'djf': 'winter mean (DJF)', 'jja': 'summer mean (JJA)'}[season]
    row_labels = ['Net-cooling AMOC weakening w.r.t. 1991–2020 temperatures'] + \
                 [f'Temperature change 2091–2100 vs 1991–2020 at {w:.0f}% AMOC weakening' for w in w_levels]
    for i, lab in enumerate(row_labels):
        p = axes[i, 0].get_position()
        fig.text(p.x0, p.y1 + 0.004, lab + f', {season_txt}', fontsize=7, fontweight='bold', color=fg, ha='left', va='bottom')
    p = axes[0, 2].get_position()
    fig.legend([tuple(Line2D([], [], color=c, lw=lw, ls=ls) for lw, c, ls in [(1.66, 'white', '-'), (0.55, 'black', '--')])],
               ['CMIP6 range of AMOC weakening'], handler_map={tuple: HandlerTuple(ndivide=None)},
               loc='lower right', bbox_to_anchor=(p.x1, p.y1 + 0.004), frameon=False, fontsize=7, labelcolor=fg,
               handlelength=2.5, borderpad=0, borderaxespad=0)
    for j, ssp in enumerate(ds.scenario.values):
        p = axes[0, j].get_position()
        fig.text((p.x0 + p.x1) / 2, p.y1 + 0.032, SSP_LABELS[ssp], color=fg, fontweight='bold', fontsize=7, ha='center', va='bottom')  # no coloured text (Nature)

    x_cb = axes[0, 2].get_position().x1 + 0.02
    p0, p2 = axes[0, 2].get_position(), axes[2, 2].get_position()
    cb1 = fig.colorbar(nc_map, cax=fig.add_axes([x_cb, p0.y0, 0.015, p0.height]))
    cb1.ax.invert_yaxis()
    cb1.set_ticks(nc_levels)
    cb1.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.0f}%'))
    cb1.set_label('AMOC weakening', fontsize=7, color=fg)
    cb2 = fig.colorbar(dT_map, cax=fig.add_axes([x_cb, p2.y0, 0.015, axes[1, 2].get_position().y1 - p2.y0]))
    cb2.set_ticks(bounds[::2] if len(bounds) > 12 else bounds)
    cb2.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:+.0f} °C'.replace('-', '−') if x else '0 °C'))
    cb2.set_label('Temperature change vs 1991–2020', fontsize=7, color=fg)

    savepath = f"{savedir}/FigSupp_mpi_pd_maps_season-{season}_w-{'-'.join(f'{w:.0f}' for w in w_levels)}_plotbg-{plot_bg}"
    for _ in range(6):  # scale the canvas until the zero-pad tight crop is 183 mm wide; text and lines keep their pt sizes
        buf = io.BytesIO()
        fig.savefig(buf, format='pdf', bbox_inches='tight', facecolor=bg)
        w = float(re.search(rb'/MediaBox \[ *0 0 ([\d.]+)', buf.getvalue()).group(1)) / 72 * 25.4
        if abs(w - 183) < 0.01:
            break
        fig.set_size_inches(fig.get_size_inches() * 183 / w, forward=False)
    fig.savefig(savepath + '.png', dpi=300, bbox_inches='tight', facecolor=bg)
    fig.savefig(savepath + '.pdf', dpi=600, bbox_inches='tight', facecolor=bg)
    return fig, savepath


########################################
# %%
# RUN

if __name__ == '__main__':
    datapath = DATAPATH    # path to mpi_esm_europe_amoc_pd_maps.nc
    savedir = SAVEDIR      # folder the .png and .pdf are written to
    season = ''            # options: '' (annual mean), 'JJA' (summer), 'DJF' (winter)
    w_levels = (25, 75)    # AMOC weakening [% of 1850-1899] for rows 2 and 3; exactly two values
    plot_bg = 'white'      # options: 'white', 'black'
    dT_step = 0.5          # temperature colourbar step [°C]
    fig, savepath = make_figure(datapath=datapath, savedir=savedir, season=season, w_levels=w_levels,
                                plot_bg=plot_bg, dT_step=dT_step)
    print(f"Saved {savepath}")

# %%
