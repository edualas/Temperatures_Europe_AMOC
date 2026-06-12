"""Tier 1 — no ``plt.show()`` in production scripts.

``plt.show()`` blocks non-interactive runs (SLURM, headless background
jobs). teu_amoc itself runs interactively today, but the same scripts
are imported by the amoc-uncertainty SLURM pipeline. Keep them
non-blocking.

Notebooks (``*.ipynb``) and files under ``scripts/archive/`` are
exempted — the former are interactive by definition; the latter are
not on any execution path.
"""

from __future__ import annotations

import re
from pathlib import Path


PLT_SHOW_RE = re.compile(r"\bplt\.show\s*\(")

# Comment-stripped detection: ignore lines whose ``plt.show(`` sits in a
# trailing ``#`` comment.
COMMENT_RE = re.compile(r"#.*$")


def test_no_plt_show_in_scripts(scripts_dir):
    offenders = []
    # The library now lives in ``src/`` (imported via the scripts/functions.py
    # shim); scan it alongside scripts/.
    for root in (scripts_dir, scripts_dir.parent / "src"):
        for path in root.rglob("*.py"):
            # Skip __pycache__ and archive dirs.
            rel = path.relative_to(root)
            if "__pycache__" in rel.parts or "archive" in rel.parts:
                continue
            for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
                line = COMMENT_RE.sub("", raw)
                if PLT_SHOW_RE.search(line):
                    offenders.append(f"{path.relative_to(scripts_dir.parent)}:{lineno}: {raw.strip()}")
    assert not offenders, (
        "plt.show() blocks background runs; remove or guard with "
        "`if __name__ == '__main__'` plus an interactive check. Offenders:\n"
        + "\n".join(offenders)
    )
