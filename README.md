# Temperatures_Europe_AMOC

Repository with all scripts for the analysis and figures of the submission of the revised manuscript "_How much AMOC weakening would cool Europe in a warming climate?_" (URL/DOI to be provided upon publication)

The code is written in Python. Input files can either be obtained from the accompanying Zenodo dataset (link to be made public upon publication) or from ESGF (https://esgf-metagrid.cloud.dkrz.de/search). Please note that the scripts originally work with paths and environments on Levante, DKRZ's supercomputer; data paths need to be adjusted to the user's environment.

## Setup

The stack (cartopy, regionmask, xarray, …) is conda-forge. Create the environment from the pinned spec:

```bash
conda env create -f environment.yml
conda activate teu
```
No install step is needed: the `scripts/functions.py` shim puts `src/` and the `scripts/` subfolders on `sys.path` when imported.

## Running — always with the working directory set to `scripts/`
**The pipeline must be run with `cwd = scripts/`.** Two things depend on it:
- the scripts read `../data` and write `../plots` via paths relative to `scripts/`;
- `import functions` finds the shim because `scripts/` is on `sys.path`, after which the shim adds `src/` and the subfolders so the cross-folder figure imports resolve.

- **Terminal:** `cd scripts && python produce_all_paper_figures.py` (optionally with figure names, e.g. `… Fig1 Fig3`).
- **VS Code interactive (`# %%` cells):** set `jupyter.notebookFileRoot` to `${workspaceFolder}/scripts` so every cell-run uses `cwd = scripts/`.

## Layout
- `src/teu_functions.py` — central library defining all routines for loading data, analysing it, and plotting; imported everywhere as `import functions` via the `scripts/functions.py` re-export shim.
- `scripts/Fig1.py`, `Fig2.py`, `Fig3.py`, `Fig3_simple.py` — main figures; each defines a `make_figure(...)` taking preloaded data and returning a matplotlib figure.
- `scripts/produce_all_paper_figures.py` — generates all paper figures (main + supplementary) in one go.
- `scripts/supplementary/` — one standalone script per supplementary figure (`FigSupp_*.py`).
- `scripts/exploratory/` — analysis modules shared by the supplementary figures (CMIP6 projected-cooling pipeline, warming-at-weakening diagnostics).
- `scripts/processing/` — preprocessing of the raw model output (AMOC strength and tas from MPI-ESM1.2-LR simulations, CMIP6 scenarios, and NAHosMIP hosing experiments) into the cached inputs used by the figure scripts.
- `scripts/cmip6_inventory.py` — filesystem inventory of the CMIP6 input files; single source of truth for which model/scenario/realisation files the projection scripts use.
- `scripts/data_process_paper.ipynb`, `additional_supplementary_plots.ipynb` — notebooks with the remaining preprocessing steps, and supplementary figures with data not covered by the standalone .py scripts
- `tests/` — pytest invariant suite guarding the methodological defaults (see `tests/README.md`).
