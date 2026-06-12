"""Compatibility shim — the real library lives in ``src/teu_functions.py``.

Every script imports the library as ``import functions`` (qualified
``functions.foo()`` access); this thin re-export keeps that working now that the
library has been relocated to ``src/``, so no call site changed. A future split
of ``teu_functions`` into a package only touches that package's ``__init__`` —
this shim and all call sites stay frozen.

This shim is the universal first import, so it also puts ``src/`` and the
``scripts/`` subfolders on ``sys.path`` — that way the figure pipeline's
cross-folder imports (``produce_all`` → ``supplementary/`` + ``exploratory/``;
``cmip_cooling`` → ``cmip6_inventory``; …) resolve by bare name from
any cwd, with no install. ``scripts/`` itself is already importable — it is how
this shim was found (the run cwd, or the entry script's own directory).

Interactive reload: scripts do ``import functions; importlib.reload(functions)``
to pick up edits. Re-executing this shim (that ``reload``) propagates the reload
to ``teu_functions`` so edits to ``src/teu_functions.py`` take effect — matching
the pre-relocation behaviour. The ``_RELOADED`` sentinel skips the redundant
reload on the first import.
"""
import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # …/scripts
_REPO = os.path.dirname(_HERE)                              # repo root
for _d in [os.path.join(_REPO, 'src'),
           *(os.path.join(_HERE, _sub) for _sub in
             ('supplementary', 'exploratory', 'processing', 'inventory'))]:
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

import teu_functions as _teu_functions                      # noqa: E402

if globals().get('_RELOADED'):
    importlib.reload(_teu_functions)
_RELOADED = True

from teu_functions import *                                 # noqa: E402,F401,F403
