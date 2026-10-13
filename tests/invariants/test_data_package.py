"""Tier 2 — the data package (data/package/) carries what package mode reads.

Catches:
- A loader reading a group the builder (tools/repro_data/build_package.py)
  no longer writes, or under another name.
- A shipped cache group without the ``filename``/``kind`` attrs that
  ``unpack_package`` needs.
- A package file reaching GitHub's 100 MB per-file limit.

Skips when data/package/ is absent (it is untracked on main).
"""

from __future__ import annotations

import re

import netCDF4
import pytest

from conftest import require_cached

READ_CALL = re.compile(r"read_package\('([\w.]+)',\s*f?(['\"])(.*?)\2")


def _groups(path):
    """All group paths of a netCDF file, with their attribute names."""
    out = {}
    def walk(g, prefix):
        for name, sub in g.groups.items():
            p = f'{prefix}/{name}' if prefix else name
            out[p] = sub.ncattrs()
            walk(sub, p)
    with netCDF4.Dataset(path) as nc:
        walk(nc, '')
    return out


@pytest.fixture(scope='module')
def package(cached_data_path):
    pkg = cached_data_path / 'package'
    require_cached(pkg)
    return pkg


def test_every_read_group_exists(package, repo_root):
    sources = [repo_root / 'src' / 'teu_functions.py']
    calls = {(f, g) for s in sources for f, _, g in READ_CALL.findall(s.read_text())}
    assert calls, 'no read_package calls found'
    missing = []
    for f, g in sorted(calls):
        require_cached(package / f)
        pattern = re.compile('^' + re.sub(r'\\\{.*?\\\}', '[^/]+', re.escape(g)) + '$')  # f-string fields match any name
        if not any(pattern.match(p) for p in _groups(package / f)):
            missing.append(f'{f}:{g}')
    assert not missing, f'groups read by the loaders but absent from the package: {missing}'


def test_cache_groups_unpackable(package):
    bad = [f'{f.name}:{g}' for f in sorted(package.glob('*.nc')) for g, attrs in _groups(f).items()
           if 'filename' in attrs and 'kind' not in attrs]
    assert not bad, f'cache groups without kind attr: {bad}'


def test_file_size_below_github_limit(package):
    big = {f.name: f.stat().st_size / 1e6 for f in package.glob('*.nc') if f.stat().st_size >= 95e6}
    assert not big, f'package files near or above 100 MB: {big}'
