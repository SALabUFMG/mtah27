"""Roda o scorer OFICIAL sobre todos os CSVs de outputs/submissions/ contra os 5 min públicos.

    uv run python scripts/score.py                 # placar
    uv run python scripts/score.py --bootstrap 1000

ATENÇÃO: os 5 minutos públicos servem para checar se o pipeline está CORRETO (alinhamento,
formato, significado dos labels). Usá-los para escolher entre versões ou tunar parâmetros é
proibido pelas regras e, como o próprio repo oficial mostra, escolhe a versão errada.
"""

import subprocess
import sys

from mtah27.paths import OFFICIAL, OUTPUTS_DIR, SUBMISSIONS_DIR

cmd = [
    sys.executable,
    str(OFFICIAL / "evaluation" / "run_evaluation.py"),
    "--results",
    str(SUBMISSIONS_DIR),
    "--out",
    str(OUTPUTS_DIR / "eval"),
    *sys.argv[1:],
]
sys.exit(subprocess.call(cmd))
