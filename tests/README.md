# Tests

Lightweight pytest harness for catching methodological drift during the
paper revision. Two tiers, both required to be green before any commit
on `main`.

## Running

```bash
conda activate /work/mh0033/m300940/miniconda3/envs/teu
cd /home/m/m300940/teu_amoc
pytest tests/invariants/                # Tier 1 + Tier 2; ~1–2 min (~60s is the one-time cartopy/regionmask import)
pytest tests/invariants/ -m slow        # Tier 3 (deferred; not implemented yet)
```

`addopts` in `pyproject.toml` excludes `-m slow` by default, so a bare
`pytest` runs only the fast tiers.

## Tiers

### Tier 1 — structural (no data access)

- `test_config_dicts.py` — `hosmip_labels`, `ssps`, `AMOC_pi_MPI`,
  `hosing_names ⊆ hosing_colors` per SSP, country dict consistency.
- `test_string_enums.py` — static scan of every `Fig*.py` callsite for
  unrecognised `season=`, `hosing=`, `var=`, `region=` values.
- `test_no_plt_show.py` — grep for stray `plt.show(` in `scripts/`.

Runs without `data/` present.

### Tier 2 — light data (cached pickles & netCDFs)

- `test_region_masks.py` — load `hosmip_masks.pkl`, assert EU is a
  single connected region, `EU ⊂ EU_EEU`, `EU_buffer ⊃ EU`, every
  country mask intersects `LAND` in at least one cell.
- `test_regression_ds_shapes.py` — open `reg_ds_mpi_tas.nc` and
  `reg_ds_cesm.nc`, assert dims/coords and that all required variables
  exist and are finite over Europe.
- `test_cache_versions.py` — compare a stored shape + numeric
  fingerprint per cached file against `baselines/cache_fingerprints.json`.

If a cached file is missing, the relevant test **skips** rather than
fails — the harness verifies *contents*, not presence.

### Tier 3 — figure-regen baselines (deferred)

Per-`make_figure` PNG comparison against stored baselines. Not yet
implemented; revisit after the B1 audit pass lands.

## Updating baselines

Cache fingerprints in `baselines/cache_fingerprints.json` are
write-once. To update them after an intentional methodology change:

```bash
TEU_AMOC_UPDATE_BASELINES=1 pytest tests/invariants/test_cache_versions.py
```

The env-var gate exists so a baseline can never be silently rewritten
by a typo'd `pytest -k baseline`. After updating, commit the
new fingerprint together with the methodology change.

## Adding a test

Tests must be **fast** (<1 s each) and **deterministic**. If you need
data, use the `cached_data_path` fixture and call `require_cached`
from `conftest.py` to skip cleanly when the file is absent. If you
need `scripts/functions.py`, take the `functions` fixture rather than
importing directly.
