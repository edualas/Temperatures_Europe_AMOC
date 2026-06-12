"""Tier 1 — structural assertions on config dicts in ``functions.py``.

These guard against silent edits that drop a model, retune a baseline,
or break the per-SSP correspondence between colour/name/marker
dictionaries.

If one of these tests fires after an intentional change, update the
assertion.
"""

from __future__ import annotations


# --- Scenarios -------------------------------------------------------------

def test_ssps_match_paper(functions):
    """Main paper uses three SSPs in a fixed order."""
    assert functions.ssps == ["ssp126", "ssp245", "ssp370"]


def test_ssp_labels_cover_ssps_and_ssp585(functions):
    """``ssp_labels`` provides display strings for every SSP we reference,
    including the opt-in ssp585 (used downstream by amoc-uncertainty)."""
    expected = {"ssp126", "ssp245", "ssp370", "ssp585"}
    assert set(functions.ssp_labels.keys()) == expected


# --- AMOC reference values -------------------------------------------------

def test_amoc_pi_mpi_value(functions):
    """Pre-industrial AMOC strength: 1850–1899 CMIP6 historical
    ensemble mean for MPI-ESM1.2-LR (MPI-GE).

    Computed at module load by ``_compute_amoc_pi_hist_scalar``
    (functions.py). Previously a piControl scalar (19.0154 Sv);
    harmonised to 1850–1899 hist on 2026-05-25.
    Tolerance: 1e-4 to absorb numerical noise; value should be ~19.011 Sv.
    """
    assert abs(functions.AMOC_pi_MPI - 19.010973) < 1e-4


def test_amoc_extent_keys_match_main_paper_ssps(functions):
    """The CMIP-range envelopes used in Fig2's colorbar exist exactly for
    the three main-paper SSPs."""
    assert set(functions.AMOC_extent.keys()) == set(functions.ssps)


def test_amoc_extent_values_are_max_then_min(functions):
    """Each tuple is (max_weakening_pct, min_weakening_pct) with max > min."""
    for ssp, (hi, lo) in functions.AMOC_extent.items():
        assert hi > lo, f"AMOC_extent[{ssp}] should be (max, min), got ({hi}, {lo})"
        assert 0 < lo < hi < 100, (
            f"AMOC_extent[{ssp}] = ({hi}, {lo}) out of plausible weakening range"
        )


# --- HosMIP models ---------------------------------------------------------

EXPECTED_HOSMIP_LABELS = [
    "CanESM5",
    "EC-Earth3",
    "CESM2",
    "IPSL-CM6A-LR",
    "HadGEM3-GC3-1MM",
    "HadGEM3-GC3-1LL",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
]


def test_hosmip_labels_canonical_order(functions):
    """Order is load-bearing for plot legends and colour assignments."""
    assert functions.hosmip_labels == EXPECTED_HOSMIP_LABELS


def test_hosmip_colors_cover_all_labels(functions):
    """Every HosMIP model must have a plot colour assigned."""
    missing = set(functions.hosmip_labels) - set(functions.hosmip_colors)
    assert not missing, f"Models without colour assignment: {missing}"


# --- Hosing variants -------------------------------------------------------

def test_hosing_names_subset_of_hosing_colors_per_ssp(functions):
    """For each SSP, every named hosing variant has a colour.

    The reverse is not required: ``hosing_colors['ssp370']`` carries a
    ``'1'`` key that ``hosing_names['ssp370']`` deliberately omits
    (no +1.0 Sv variant for ssp370 in the published figures).
    """
    for ssp, names in functions.hosing_names.items():
        colors = functions.hosing_colors.get(ssp, {})
        missing = set(names.keys()) - set(colors.keys())
        assert not missing, (
            f"hosing_names[{ssp}] has variants without colour assignment: {missing}"
        )


def test_hosing_symbols_cover_all_named_variants(functions):
    """Flat ``hosing_symbols`` must define a symbol for every variant
    used in any per-SSP names dict."""
    all_variants = set()
    for names in functions.hosing_names.values():
        all_variants.update(names.keys())
    missing = all_variants - set(functions.hosing_symbols.keys())
    assert not missing, f"Variants without symbol: {missing}"


def test_hosing_markers_cover_all_named_variants(functions):
    all_variants = set()
    for names in functions.hosing_names.values():
        all_variants.update(names.keys())
    missing = all_variants - set(functions.hosing_markers.keys())
    assert not missing, f"Variants without marker: {missing}"


def test_hosing_symbols_and_markers_same_keys(functions):
    """The two parallel dicts must declare the same variant set."""
    assert set(functions.hosing_symbols.keys()) == set(functions.hosing_markers.keys())


# --- Country dictionaries --------------------------------------------------

def test_country_names_and_codes_have_same_keys(functions):
    names = set(functions.country_names.keys())
    codes = set(functions.country_codes.keys())
    assert names == codes, (
        f"country_names vs country_codes mismatch: "
        f"only in names = {names - codes}, only in codes = {codes - names}"
    )


def test_regions_list_matches_country_dict_keys(functions):
    """The plot-order ``regions`` list must enumerate exactly the
    countries defined in the lookup dicts."""
    assert set(functions.regions) == set(functions.country_names.keys())
    # No duplicates: list length equals dict size.
    assert len(functions.regions) == len(functions.country_names)


def test_country_count_matches_paper(functions):
    """The publication set includes 41 European countries."""
    assert len(functions.country_names) == 41
