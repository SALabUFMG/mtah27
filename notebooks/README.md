# Notebooks

Exploration and diagnostics. Code that goes into the submission lives in `src/mtah27/`, not here.

- **Name:** `<initials>_<NN>_<topic>.ipynb`, e.g. `jl_01_eda_tracking.ipynb`. Avoids merge conflicts between members.
- **No outputs in git:** the `nbstripout` hook (pre-commit) strips them automatically. Outputs contain data (tables, tracking plots) and the repo is public.
- Import the package: `from mtah27 import paths`.
