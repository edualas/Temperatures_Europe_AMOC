########################################
# %%
# LOAD PACKAGES

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from matplotlib.transforms import Bbox, TransformedBbox
plt.rcParams.update({'font.size': 12})
import numpy as np

import importlib
import functions
importlib.reload(functions)
import cmip_cooling  # type: ignore
importlib.reload(cmip_cooling)

########################################
# %%
# LOAD DATA (mirrors Fig3.py)

if __name__ == '__main__':
    multi_model_dict, masks = functions.get_full_multi_model_dict()
    reg_ds_mpi = functions.load_regression_ds_mpi()
    reg_ds_cesm = functions.get_cesm_reg_ds(recompute=False)
    reg_ds_giss = functions.get_giss_reg_ds(recompute=False)
    reg_ds_giss_panel = {
        s: functions.get_giss_panel_data(masks=masks, season=s, recompute=False)
        for s in ['', 'djf', 'jja']
    }
    hosmip_reg_ds_dict = functions.get_hosmip_reg_ds(recompute=False)
    cmip_cooling_ds = cmip_cooling.make_cmip_cooling_ds(recompute=False)


########################################
# %%
# Regression-line entries used by the bottom-half "Regression lines" sub-legend.

REG_LINE_ROWS = [
    ('blueviolet', ':',  'Preindustrial hosing MPI-ESM1.2-LR (NAHosMIP)'),
    ('brown',      '-', 'Combined-forcing MPI-ESM1.2-LR (this study)'),
    ('dodgerblue', '-',  'Combined-forcing CESM2 (Boot et al. 2024)'),
    (functions.hosing_colors['ssp245']['ge'], '-',
     'SSP2-4.5 GISS-E2-1-G (Romanou et al. 2023)'),
]


def _restyle_regression_line(ax, target_color, *, linestyle='-', zorder=10, lw=3):
    """Find the lw-3 line of target_color drawn by hosmip_regression_plot and
    bump its zorder / set its linestyle. Used to surface the brown 'this study'
    line that otherwise hides behind the violet NAHosMIP line."""
    tgt = mcolors.to_rgba(target_color)
    for line in ax.get_lines():
        if line.get_linewidth() == lw and np.allclose(
                mcolors.to_rgba(line.get_color()), tgt, atol=1e-3):
            line.set_linestyle(linestyle)
            line.set_zorder(zorder)
            return line
    return None


def _set_marker_clip_to_spines(ax, *, left=-0.02, top=1.05):
    """Clip every scatter PathCollection on ax to a box bounded by the visible
    spines (left at axes -0.02, top at axes 1.05) rather than the default
    data-axes box. Markers can now extend into the gap between the data area
    and the displaced spines, so points right at the edges (x=0 or y=0) don't
    get half-cut. Lines are left alone — only marker collections are touched."""
    clip_box = TransformedBbox(Bbox([[left, 0.0], [1.0, top]]), ax.transAxes)
    for coll in ax.collections:
        coll.set_clip_box(clip_box)
        coll.set_clip_on(True)


########################################
# %%
# FIGURE FUNCTION

def make_figure(multi_model_dict, masks, reg_ds_mpi, reg_ds_cesm, hosmip_reg_ds_dict,
                season='', window=10, T_ref='pi', hosmip_markers=False, plot_bg='white',
                reg_ds_giss=None, giss_time_period='2101-2300',
                giss_panel_data=None, cmip_cooling_ds=None,
                cmip_cooling_show=False, aggregate_first=True, weakening_unit='pct',
                future_window=None):
    plt.style.use('default')
    sv_xmax_e = sv_xmax_ad = None
    if weakening_unit == 'sv':
        pis = [float(ds.AMOC_pi.values) for ds in hosmip_reg_ds_dict.values()]
        pis.append(float(reg_ds_cesm.AMOC_pi.values))
        pis.append(functions.AMOC_pi_MPI)
        if reg_ds_giss is not None:
            pis.append(functions.AMOC_pi_GISS)
        sv_xmax_e = int(np.ceil(max(pis) / 5) * 5)
        sv_xmax_ad = int(np.ceil(0.8 * max(pis) / 5) * 5)
    _ext = functions.get_amoc_extent(future_window, unit=weakening_unit)
    if plot_bg == 'black':
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] = '#191919'

    # Figsize: width matches Fig3 (16 in), but height shrinks by 35/42 = 5/6
    # so each grey row in panel b has the same line height it had before
    # dropping the 7 sub-30k-km² countries (LU, CY, XK, ME, SI, MK, AL).
    # Original Fig3 had 41 countries + EU = 42 rows in 14 in. Now 34 + EU = 35
    # rows; same per-row height implies 14 × 35/42 = 11.667 in tall.
    fig = plt.figure(figsize=(16, 14 * 35 / 42))
    gs = GridSpec(2, 2, width_ratios=[0.35, 0.65], height_ratios=[1.0, 1.0],
                  wspace=0.4, hspace=0.04)
    ax_left   = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_legend.set_axis_off()
    ax_right  = fig.add_subplot(gs[:, 1])

    text_color = 'black' if plot_bg != 'black' else 'white'

    if isinstance(giss_panel_data, dict):
        giss_panel_for_season = giss_panel_data.get(season)
    else:
        giss_panel_for_season = giss_panel_data

    # ── Panel a: EU regression ────────────────────────────────────────────────
    # low_ylim tightened from Fig3's -14 to -10: EU regression data extends to
    # ~-8 at the negative-slope end of the NAHosMIP MPI line at x=80%, plus a
    # 2°C safety margin below. Wider range wastes vertical space.
    # x- and y-axis labels intentionally LEFT TO hosmip_regression_plot — its
    # default y-label carries the {EU} region subscript and the season qualifier
    # that an override would silently drop.
    functions.hosmip_regression_plot(
        multi_model_dict, region='EU', season=season, window=window, plot_bg=plot_bg,
        quantile_reg=False, no_plots=False, low_ylim=-8, linear_95_reg=True,
        linear_5_reg=False, central_reg=None, central_reg_intercept=False,
        hosmip_ref_pi=True, cap_x_range=100,
        bm_data=True, liu_data=True, vwb_data=True, boot_data=True, boot_regression=True,
        giss_data=giss_panel_for_season,
        giss_regression=(giss_panel_for_season is not None),
        giss_intercept=False, giss_time_period=giss_time_period,
        add_combined_results=True, combined_results_label=False,
        add_hosmip_mpi_regression=True, hosmip_reg_ds_dict=hosmip_reg_ds_dict,
        weakening_unit=weakening_unit, sv_xmax=sv_xmax_ad,
        ax=ax_left, markersize=14)

    # X-axis on top, aligned visually with panel b's CMIP6 envelope strip.
    # Bottom spine + bottom xticks hidden so the panel reads as a single
    # top-aligned strip-axis layout. Top spine re-enabled because
    # hosmip_regression_plot disables it at functions.py:5760.
    ax_left.xaxis.tick_top()
    ax_left.xaxis.set_label_position('top')
    ax_left.tick_params(axis='x', which='both', bottom=False, labelbottom=False,
                        top=True, labeltop=True)
    ax_left.tick_params(axis='y')
    ax_left.spines['bottom'].set_visible(False)
    ax_left.spines['top'].set_visible(True)
    ax_left.spines['top'].set_position(("axes", 1.05))
    ax_left.spines['left'].set_position(("axes", -0.02))
    if weakening_unit == 'pct':
        ax_left.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))

    # Remove the dashed y=0 and x=0 reference lines drawn by
    # hosmip_regression_plot (functions.py:5750, 5751). The y=0 axhline
    # overlaps the top spine at ylim_max=0 and reads as a redundant
    # horizontal stripe; the x=0 axvline sits on the left spine.
    for line in list(ax_left.get_lines()):
        xd = np.asarray(line.get_xdata())
        yd = np.asarray(line.get_ydata())
        if line.get_linestyle() == '--':
            if yd.size > 0 and np.all(yd == 0):
                line.remove(); continue
            if xd.size > 0 and np.all(xd == 0):
                line.remove(); continue

    # Brown 'this study' line otherwise hides behind the violet NAHosMIP line
    # (similar negative slope). Dash it and bump zorder so it's distinguishable.
    _restyle_regression_line(ax_left, 'brown',      linestyle='-', zorder=2)
    _restyle_regression_line(ax_left, 'blueviolet', linestyle=':',  zorder=3)
    _restyle_regression_line(ax_left, 'dodgerblue', linestyle='-',  zorder=1)

    # Clip all marker PathCollections to a box matching the visible spines:
    # extends 0.02 axes-fraction further LEFT and 0.05 further UP than the
    # default data-axes clip. Markers right at x=0 or y=0 are no longer
    # half-clipped, while points wandering far outside stay bounded.
    _set_marker_clip_to_spines(ax_left)

    # ── Legend (bottom-half of left column, two rows of sub-legends) ─────────
    # Top row (full width):  Regression lines  — 3 entries, single-line labels.
    # Bottom row (split):    Models | Other studies.
    # Each sub-legend keeps its bold title. `ax_legend.legend` overwrites the
    # axes' active legend, so persisted ones are wrapped in `add_artist`.
    handles, labels = ax_left.get_legend_handles_labels()
    label_renames = {'HadGEM3-GC3-1MM': 'HadGEM3-GC3.1-MM',
                     'HadGEM3-GC3-1LL': 'HadGEM3-GC3.1-LL',
                     'MPI-ESM1-2-HR':   'MPI-ESM1.2-HR',
                     'MPI-ESM1-2-LR':   'MPI-ESM1.2-LR'}
    h_models  = handles[:8]
    l_models  = [label_renames.get(l, l) for l in labels[:8]]
    h_studies = [h for h, l in zip(handles[8:], labels[8:]) if l]
    l_studies = [l for l in labels[8:] if l]

    # Hand-defined linebreak rewrites for the studies sub-legend. Upstream
    # hosmip_regression_plot bakes `\n` into the labels at fixed positions
    # that read awkwardly when the legend column has horizontal room. Prefix
    # match so the Romanou entry binds to whatever giss_time_period was
    # passed; en-dash for scenario windows (range convention).
    _gw = giss_time_period.replace('-', '–')
    _study_rewrites = [
        ('Bellomo & Mehling\npreindustrial',
         'Bellomo & Mehling (2024): preindustrial'),
        ('Bellomo & Mehling\n4x',
         'Bellomo & Mehling (2024): 4×CO₂'),
        ('v. Westen & Baatsen',
         'van Westen & Baatsen (2025)\nRCP4.5 (2400–2500)'),
        ('Liu et al.',
         'Liu et al. (2020): RCP8.5 (2061–2080)'),
        ('Boot et al.',
         'Boot et al. (2024):\nSSP1-2.6 & SSP5-8.5 (2015–2100)'),
        ('Romanou et al.',
         f'Romanou et al. (2023):\nSSP2-4.5 ({_gw})'),
    ]
    def _restyle_study_label(l):
        for prefix, new in _study_rewrites:
            if l.startswith(prefix):
                return new
        return l
    l_studies = [_restyle_study_label(l) for l in l_studies]

    h_lines = [Line2D([], [], color=c, lw=3, linestyle=s) for c, s, _ in REG_LINE_ROWS]
    l_lines = [t for _, _, t in REG_LINE_ROWS]

    leg_lines = ax_legend.legend(
        h_lines, l_lines, frameon=False, fontsize=10, ncols=1,
        loc='upper left', bbox_to_anchor=(-0.02, 0.2),
        title='Regression lines',
        title_fontproperties={'weight': 'bold', 'size': 11},
        handlelength=2.6, handletextpad=0.7, labelspacing=0.55,
        borderaxespad=0.0)
    for t, (c, _, _) in zip(leg_lines.get_texts(), REG_LINE_ROWS):
        t.set_color(c); t.set_weight('bold')
    leg_lines._legend_box.align = 'left'
    ax_legend.add_artist(leg_lines)

    leg_models = ax_legend.legend(
        h_models, l_models, frameon=False, fontsize=10, ncols=1,
        loc='upper left', bbox_to_anchor=(-0.02, 0.9),
        title='NAHosMIP models',
        title_fontproperties={'weight': 'bold', 'size': 11},
        labelcolor=text_color, borderaxespad=0.0,
        handletextpad=0.5, labelspacing=0.35)
    leg_models._legend_box.align = 'left'
    ax_legend.add_artist(leg_models)

    leg_studies = ax_legend.legend(
        h_studies, l_studies, frameon=False, fontsize=10, ncols=1,
        loc='upper left', bbox_to_anchor=(0.4, 0.9),
        title='Other studies',
        title_fontproperties={'weight': 'bold', 'size': 11},
        labelcolor=text_color, borderaxespad=0.0,
        handletextpad=0.5, labelspacing=0.35)
    leg_studies._legend_box.align = 'left'

    # ── Panel b (right): net-cooling ranges ─────────────────────────────────
    cc_ds_for_plot = cmip_cooling_ds if cmip_cooling_show else None
    functions.plot_net_cooling_ranges_mpi_cesm(
        reg_ds_mpi, reg_ds_cesm, masks,
        hosmip_reg_ds_dict=hosmip_reg_ds_dict, season=season,
        hosmip_markers=hosmip_markers, T_ref=T_ref, plot_bg=plot_bg,
        ext_ax=ax_right, title=False,
        reg_ds_giss=reg_ds_giss, giss_time_period=giss_time_period,
        cmip_cooling_ds=cc_ds_for_plot, cmip_cooling_decade=None,
        aggregate_first=aggregate_first, amoc_extent=_ext,
        weakening_unit=weakening_unit, sv_xmax=sv_xmax_e, amoc_extent_y=1.04)

    # Right panel vertical alignment with the left column (mirrors Fig3.py:170–180)
    pos_left_top = ax_left.get_position()
    pos_left_bot = ax_legend.get_position()
    pos_right    = ax_right.get_position()
    extend_down = 0.02
    new_bottom = pos_left_bot.y0 - extend_down
    new_top    = pos_left_top.y1
    new_height = new_top - new_bottom
    overflow_frac = 0.05
    new_height_adjusted = new_height / (1 + overflow_frac)
    ax_right.set_position([pos_right.x0, new_bottom, pos_right.width, new_height_adjusted])

    # Align panel a's top axis with panel b's CMIP6 envelope band: shrink
    # ax_left's bbox so its top spine sits at panel b's *data-axis top* (the
    # boundary BELOW the envelope strip). The xticks then read in the same
    # vertical band as panel b's CMIP6 percentile labels.
    # Also widen ax_left (and ax_legend below it) to fill the inter-column
    # gutter — wspace=0.4 reserves ~1.4 inch of unused space between panel a
    # and panel b, which used to host Fig3's arrow callouts.
    ax_right_top = ax_right.get_position().y1
    pos_right    = ax_right.get_position()
    gutter_gap   = 0.03
    new_right_edge = pos_right.x0 - gutter_gap
    new_width_left = new_right_edge - pos_left_top.x0
    ax_left.set_position([pos_left_top.x0, pos_left_top.y0,
                          new_width_left, ax_right_top - pos_left_top.y0])
    pos_legend = ax_legend.get_position()
    ax_legend.set_position([pos_left_top.x0, pos_legend.y0,
                            new_width_left, pos_legend.height])

    # Centre the left column vertically with whitespace padding. Panel a's
    # *axes-1.05 top spine* is aligned with panel b's top spine at fig y =
    # ax_right.y1 ≈ 0.842 (data-axis top, just below the CMIP6 envelope
    # strip). Solving spine = y0 + 1.05 × height with panel a's bottom held
    # fixed at panel_a_y0 = 0.484:
    #   spine target = 0.842   →   height = 0.341   →   band_top = 0.825
    # Legend height stays as 0.260 (its previous each_h), so the panel a /
    # legend gap stays at hspace = 0.04 (= 0.484 − 0.445).
    band_top      = 0.825
    band_bot      = 0.185
    panel_a_y0    = 0.484
    legend_h      = 0.260
    pos_a_cur = ax_left.get_position()
    pos_l_cur = ax_legend.get_position()
    ax_left.set_position([pos_a_cur.x0, panel_a_y0,
                          pos_a_cur.width, band_top - panel_a_y0])
    ax_legend.set_position([pos_l_cur.x0, band_bot,
                            pos_l_cur.width, legend_h])

    # Subplot letters — 'a' stays at its current absolute fig-y position (the
    # top of the figure, aligned with panel b's CMIP6 envelope band). 'b'
    # lives at the same y.
    pos_a = ax_left.get_position()
    pos_b = ax_right.get_position()
    label_y = ax_right_top + 0.5 * (overflow_frac * new_height_adjusted) + 0.012
    fig.text(pos_a.x0 - 0.015, label_y, 'a',
             fontsize=16, fontweight='bold', ha='right', va='bottom',
             color=text_color)
    fig.text(pos_b.x0, label_y, 'b',
             fontsize=16, fontweight='bold', ha='right', va='bottom',
             color=text_color)

    # Map-axes white-border patches (mirrors Fig3.py:185–200)
    border_width = 0.001
    border_color = 'white' if plot_bg == 'white' else '#191919'
    for ax in fig.get_axes():
        if hasattr(ax, 'projection'):
            bbox = ax.get_position()
            fig.add_artist(plt.Rectangle((bbox.x0, bbox.y1 - border_width), bbox.width, border_width,
                                         transform=fig.transFigure, color=border_color, zorder=1000, clip_on=False))
            fig.add_artist(plt.Rectangle((bbox.x0, bbox.y0), bbox.width, border_width,
                                         transform=fig.transFigure, color=border_color, zorder=1000, clip_on=False))
            fig.add_artist(plt.Rectangle((bbox.x0, bbox.y0), border_width, bbox.height,
                                         transform=fig.transFigure, color=border_color, zorder=1000, clip_on=False))
            fig.add_artist(plt.Rectangle((bbox.x1 - border_width, bbox.y0), border_width, bbox.height,
                                         transform=fig.transFigure, color=border_color, zorder=1000, clip_on=False))

    # ── Save ────────────────────────────────────────────────────────────────
    giss_tag  = f'_giss-{giss_time_period}' if reg_ds_giss is not None else '_giss-off'
    cc_tag    = ('_cc-on' if (cmip_cooling_ds is not None and cmip_cooling_show)
                 else '_cc-off')
    agg_tag   = '_aggfirst-on' if aggregate_first else '_aggfirst-off'
    wunit_tag = f'_wunit-{weakening_unit}'
    savepath = (f"../plots/Fig3_simple_plotbg-{plot_bg}_season-{season or 'annual'}"
                f"_Tref-{T_ref}_markers-{hosmip_markers}{giss_tag}{cc_tag}{agg_tag}"
                f"{wunit_tag}{functions.fw_suffix(future_window)}")
    fig.savefig(savepath + '.png', dpi=200, bbox_inches='tight',
                transparent=True if plot_bg == 'black' else False)
    fig.savefig(savepath + '.pdf', dpi=400, bbox_inches='tight',
                transparent=True if plot_bg == 'black' else False)
    return fig, savepath


########################################
# %%
# RUN

if __name__ == '__main__':
    plot_bg = 'white'
    window = 10
    season = ''
    T_ref = 'pi'
    hosmip_markers = False
    giss_time_period = '2101-2300'
    cmip_cooling_show = False
    aggregate_first = True
    weakening_unit = 'pct'
    future_window = None

    fig, savepath = make_figure(multi_model_dict, masks, reg_ds_mpi, reg_ds_cesm,
                                hosmip_reg_ds_dict, season=season, window=window,
                                T_ref=T_ref, hosmip_markers=hosmip_markers,
                                plot_bg=plot_bg, reg_ds_giss=reg_ds_giss,
                                giss_panel_data=reg_ds_giss_panel,
                                giss_time_period=giss_time_period,
                                cmip_cooling_ds=cmip_cooling_ds,
                                cmip_cooling_show=cmip_cooling_show,
                                aggregate_first=aggregate_first,
                                weakening_unit=weakening_unit,
                                future_window=future_window)
    print(f"Saved {savepath}")
# %%
