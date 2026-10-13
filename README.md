# Temperatures_Europe_AMOC

Repository with all scripts for the analysis and figures of the manuscript "_AMOC weakening unlikely to cool large parts of Europe in a warming climate_" (URL/DOI to be provided upon publication).

The code is written in Python. The data needed to reproduce every figure ship with the repository in `data/package/` (see Data below); no path needs adjusting. On Levante, DKRZ's supercomputer, the same code runs from the full model output instead.

## Setup

The stack (cartopy, regionmask, xarray, …) is conda-forge. Create the environment from the pinned spec:

```bash
conda env create -f environment.yml
conda activate teu
```

No install step is needed: the `scripts/functions.py` shim puts `src/` and the `scripts/` subfolders on `sys.path` when imported.

The figures use the Helvetica clone Nimbus Sans (URW++), shipped in `fonts/` and registered with matplotlib on import, so no font needs to be installed (see License below). `plots/` is created on first use.

## Running — always with the working directory set to `scripts/`

**The pipeline must be run with `cwd = scripts/`.** Two things depend on it:
- the scripts read `../data` and write `../plots` via paths relative to `scripts/`;
- `import functions` finds the shim because `scripts/` is on `sys.path`, after which the shim adds `src/` and the subfolders so the cross-folder figure imports resolve.

- **Terminal:** `cd scripts && python produce_all_paper_figures.py` (optionally with figure names, e.g. `… Fig1 Fig3`).
- **VS Code interactive (`# %%` cells):** set `jupyter.notebookFileRoot` to `${workspaceFolder}/scripts` so every cell-run uses `cwd = scripts/`.

## Data

`data/package/` holds one netCDF file per data source, cut to a European window that includes the surrounding ocean (on the MPI-ESM1.2-LR grid 30.8–71.8°N, 37.5°W–48.8°E). When the Levante input paths are absent, the code runs in package mode: it reads these files, unpacks the shipped intermediate results into `data/package_cache/`, and recomputes the MPI-ESM, NAHosMIP and GISS-E2-1-G regressions from the fields there. Set `TEU_DATA_MODE=package` or `raw` to choose explicitly. `tools/repro_data/build_package.py` builds the package from the Levante sources.

| File | Content | Recomputed in package mode |
|---|---|---|
| `mpi_ge.nc` | MPI-ESM1.2-LR grand ensemble: ensemble-mean tas (historical, SSP1-2.6/2-4.5/3-7.0; annual, DJF, JJA), regional land-mean tas and AMOC at 26°N per member, region masks | MPI-ESM regressions, scenario- and state-independence and linearity diagnostics |
| `mpi_hosing.nc` | MPI-ESM1.2-LR SSP hosing experiments, 2015–2115: tas and AMOC per member | (as above) |
| `nahosmip.nc` | NAHosMIP control and hosing fields and AMOC per model, region masks, CMIP6 ensemble means of each model (AMOC and tas in 1850–1899, 1991–2020, 2091–2100) | NAHosMIP regressions |
| `giss.nc` | GISS-E2-1-G SSP2-4.5 to 2500: tas and AMOC per member, region masks, CMIP6 ensemble means | GISS regressions and diagnostics |
| `cesm2_boot.nc` | Boot et al. (2024) CESM2: regional land-mean tas and AMOC; CESM2 regression results | – (the fields are not public) |
| `other_studies.nc` | Bellomo & Mehling (2024), van Westen & Baatsen (2025), Liu et al. (2020): the temperature differences and AMOC changes used in the figures | – |
| `cmip.nc` | CMIP6 projection ensemble: synthetic CMIP6 ranges, (w, t) targets, CMIP-projected cooling, AMOC extent, global-warming-level diagnostic | – |

Not reproducible from the package: Supplementary Fig. (MPI-ESM tas/tmn anomalies in 2300) and the +1 Sv variant of Fig. 1, which need MPI-ESM output beyond 2100. `produce_all_paper_figures.py` skips them in package mode.

## Layout

- `src/teu_functions.py` — central library defining all routines for loading data, analysing it, and plotting; imported everywhere as `import functions` via the `scripts/functions.py` re-export shim.
- `scripts/Fig1.py`, `Fig2.py`, `Fig3_simple.py` — main figures; each defines a `make_figure(...)` taking preloaded data and returning a matplotlib figure. `Fig3_simple.py` is Figure 3 of the paper; `Fig3.py` is its extended variant shown in the supplementary material.
- `scripts/produce_all_paper_figures.py` — generates all paper figures (main + supplementary) in one go.
- `scripts/supplementary/` — one standalone script per supplementary figure (`FigSupp_*.py`). `FigSupp_synthetic_scaling_factors.py` also builds the synthetic CMIP6 ranges used by Figure 3 and several supplementary figures.
- `scripts/exploratory/` — analysis modules shared by the supplementary figures (CMIP6 projected-cooling pipeline, warming-at-weakening diagnostics).
- `scripts/processing/` — preprocessing of the raw model output (AMOC strength and tas from MPI-ESM1.2-LR simulations, CMIP6 scenarios, and NAHosMIP hosing experiments) into the cached inputs used by the figure scripts.
- `scripts/cmip6_inventory.py` — filesystem inventory of the CMIP6 input files; single source of truth for which model/scenario/realisation files the projection scripts use.
- `scripts/data_process_paper.ipynb`, `analysis_paper.ipynb`, `create_hosing_fields.ipynb` — notebooks with the remaining preprocessing steps, paper analysis (including three supplementary figures not covered by the scripts above), and generation of the hosing perturbation fields.
- `tests/` — pytest invariant suite guarding the methodological defaults (see `tests/README.md`).

## License

The code is under the BSD 3-Clause License (`LICENSE`). This does not cover `fonts/`: those files are URW++'s Nimbus Sans, redistributed under the GNU AGPL v3 with URW's PostScript/PDF embedding exemption, see `fonts/LICENSE`, `fonts/COPYING` and `fonts/README.md` (origin, licence and the modification made). The code does not depend on them; without them matplotlib falls back to Arial, Helvetica or DejaVu Sans.
