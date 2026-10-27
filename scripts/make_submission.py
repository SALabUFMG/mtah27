"""Gera o arquivo de submissão a partir dos dados. Este é o comando de reprodução oficial.

uv run python scripts/make_submission.py                    # configs/default.yaml
uv run python scripts/make_submission.py --config outra.yaml
"""

from __future__ import annotations

import argparse
import hashlib

from mtah27.config import load_config
from mtah27.paths import ROOT, SUBMISSIONS_DIR
from mtah27.pipeline import predict
from mtah27.submission import write_submission


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--out", default=None, help="caminho do CSV (padrão: outputs/submissions/<team>.csv)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    labels = predict(cfg)
    out = args.out or SUBMISSIONS_DIR / f"{cfg['team_name']}.csv"
    path = write_submission(labels, out, cfg.get("game", "BAR-ATH"))

    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"escrito: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    print(f"sha256:  {sha}   <- deve ser idêntico em qualquer máquina")


if __name__ == "__main__":
    main()
