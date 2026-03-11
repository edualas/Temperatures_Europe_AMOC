# Temperatures_Europe_AMOC

Repository with all scripts for the analysis and figures of the submission of the manuscript "_How much AMOC weakening would cool Europe in a warming climate?_" (URL/DOI to be provided upon publication)

The code is written in Python. The notebooks (`.ipynb`) and scripts (`.py`) here included are the following:

- `data_process_paper.ipynb`: Includes all preprocessing steps of the amoc strength and tas data from our own new MPI-ESM1.2-LR simulations, as well as all CMIP6 scenarios and NAHosMIP hosing experiments.
- `functions.py`: central module defining all routines for loading data, analysing it, and plotting helpers
- `Fig1.py`, `Fig2.py`, `Fig3.py`, `FigS9.py`, `FigS10.py`, `FigS11.py`, `FigS15.py`: each figure script defines a function taking preloaded data and returning a matplotlib figure.
- `produce_all_paper_figures.py`: generates all paper figures in one go.
- `additional_supplementary_plots.ipynb`: produces supplementary figures with data not covered by the standalone .py scripts.

Please note that the scripts originally work with paths and environments on Levante, DKRZ's supercomputer, and therefore those need to be adjusted to work with the user's environments and paths for the relevant files. Files can either be obtained from the accompanying Zenodo datasete (doi: ), or from ESGF nodes.
