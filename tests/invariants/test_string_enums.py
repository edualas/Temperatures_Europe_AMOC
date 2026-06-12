"""Tier 1 — static check that string kwargs in figure callsites carry
recognised values.

Walks the AST of every ``Fig*.py`` and the orchestrator
``produce_all_paper_figures.py``. For each call expression, inspects
keyword arguments whose name is in :data:`KWARGS_TO_CHECK`. If the
value is a string literal, it must belong to the corresponding allowed
set; non-literal values (variables, expressions) are skipped — they
can only be checked at runtime.

This is the silent-typo guard. It is intentionally narrow: it does not
check ``region=`` (too many valid values, some dynamic), and it does
not try to reason about which function each kwarg is being passed to —
the allowed sets are unioned across all known consumers.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


SEASON_ALLOWED = {"", "djf", "jja"}
VAR_ALLOWED = {"tas", "tmn", "pr"}
# Union of (a) dispatch keywords used by simulations_plot's hos_type
# argument and Fig1's hosing variants, and (b) individual hosing
# variant keys from hosing_colors/hosing_names.
HOSING_ALLOWED = {
    # Dispatch keywords:
    "all", "constant", "linear", "all1Sv",
    # Individual variants:
    "ge", "neg01", "01", "03", "05", "1",
    "linneg02", "lin02", "lin06", "lin10",
}
T_REF_ALLOWED = {"pi", "pd"}
PLOT_BG_ALLOWED = {"white", "black"}

KWARGS_TO_CHECK = {
    "season": SEASON_ALLOWED,
    "plot_season": SEASON_ALLOWED,
    "var": VAR_ALLOWED,
    "hosing": HOSING_ALLOWED,
    "hos_type": HOSING_ALLOWED,
    "T_ref": T_REF_ALLOWED,
    "plot_bg": PLOT_BG_ALLOWED,
}


def _check_file(path: Path) -> list[str]:
    """Return a list of human-readable offender strings for one file."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as e:
        return [f"{path}: SyntaxError: {e}"]
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg not in KWARGS_TO_CHECK:
                continue
            value = kw.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                if value.value not in KWARGS_TO_CHECK[kw.arg]:
                    offenders.append(
                        f"{path.name}:{value.lineno}: "
                        f"{kw.arg}={value.value!r} not in allowed set "
                        f"{sorted(KWARGS_TO_CHECK[kw.arg])}"
                    )
    return offenders


def _figure_scripts(scripts_dir: Path) -> list[Path]:
    """Every ``Fig*.py`` (recursing into subfolders such as ``supplementary/``)
    plus the orchestrator. Excludes archive / __pycache__."""
    paths: list[Path] = [
        path for path in sorted(scripts_dir.rglob("Fig*.py"))
        if not {"archive", "_archive", "__pycache__"} & set(path.parts)
    ]
    orch = scripts_dir / "produce_all_paper_figures.py"
    if orch.exists():
        paths.append(orch)
    return paths


@pytest.mark.parametrize("path", _figure_scripts(Path(__file__).resolve().parents[2] / "scripts"))
def test_string_kwargs_in_figure_callsites(path):
    offenders = _check_file(path)
    assert not offenders, "Unrecognised string kwargs:\n" + "\n".join(offenders)


def test_functions_py_has_no_typo_in_internal_callsites(scripts_dir):
    """Same check on the central library. Looser scope — many internal
    callsites here pass variables rather than literals; we only flag
    literal mismatches.
    """
    functions_py = scripts_dir.parent / "src" / "teu_functions.py"
    offenders = _check_file(functions_py)
    assert not offenders, "Unrecognised string kwargs in teu_functions.py:\n" + "\n".join(offenders)
