"""Roda o validador OFICIAL (mesmas checagens do scorer) num CSV.

uv run python scripts/validate.py outputs/submissions/SALab_UFMG.csv
"""

import subprocess
import sys

from mtah27.paths import OFFICIAL

validator = OFFICIAL / "evaluation" / "validate_submission.py"
sys.exit(subprocess.call([sys.executable, str(validator), *sys.argv[1:]]))
