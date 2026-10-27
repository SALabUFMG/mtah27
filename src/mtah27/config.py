"""Leitura de configs YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mtah27.paths import CONFIGS_DIR


def load_config(path: str | Path = "default.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = CONFIGS_DIR / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)
