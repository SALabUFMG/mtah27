"""Caminhos do projeto e ponte para o repositório oficial (submódulo).

Tudo que precisa de um caminho passa por aqui, para que ninguém escreva caminhos à mão.

- ROOT              raiz deste repositório
- OFFICIAL          external/mtah-official (submódulo, commit fixado)
- DATA_DIR          $MTAH_DATA_DIR, ou ./data deste repositório
- INTERIM_DIR       caches/parquet derivados (gitignored)
- OUTPUTS_DIR       submissões, relatórios, figuras (gitignored)

Importar este módulo define MTAH_DATA_DIR (se ainda não estiver definido) e coloca o repo
oficial no sys.path, então `games`, `possession_metrics` e os loaders de `starter/` passam a
ler os dados deste repositório.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "external" / "mtah-official"

DATA_DIR = Path(os.environ.get("MTAH_DATA_DIR") or ROOT / "data").resolve()
os.environ.setdefault("MTAH_DATA_DIR", str(DATA_DIR))

INTERIM_DIR = DATA_DIR / "interim"
OUTPUTS_DIR = ROOT / "outputs"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
CONFIGS_DIR = ROOT / "configs"

if not (OFFICIAL / "games.py").exists():
    raise RuntimeError("Submódulo oficial ausente. Rode:  git submodule update --init --recursive")

for p in (OFFICIAL, OFFICIAL / "evaluation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
