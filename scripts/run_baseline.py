"""Roda a baseline OFICIAL com os dados deste repositório e pontua nos 5 min públicos.

    uv run python scripts/run_baseline.py

Serve como "primeiro contato" com os dados: se os scores públicos baterem com os publicados
abaixo, os dados estão no lugar certo e o ambiente está correto.

    baseline_smooth125   macroF1 0.5812   spellF1 0.4242
    baseline_smooth50    macroF1 0.6207   spellF1 0.3256
    baseline_smooth25    macroF1 0.6200   spellF1 0.2868
    baseline_raw         macroF1 0.5972   spellF1 0.0378

Os CSVs vão para outputs/baselines/ (gitignored). Leva 1–2 minutos: lê o tracking inteiro.
"""

# ruff: noqa: I001  (a ordem importa: mtah27.paths precisa vir antes de `games`)
import os
import subprocess
import sys

from mtah27.paths import DATA_DIR, OFFICIAL, OUTPUTS_DIR  # define MTAH_DATA_DIR para os subprocessos

from games import asset_paths  # type: ignore  # noqa: E402  (vem do repo oficial, via mtah27.paths)

BASELINES_DIR = OUTPUTS_DIR / "baselines"


def main() -> int:
    wanted = [asset_paths()[k] for k in ("atd_meta", "atd_raw")]
    missing = [p for p in wanted if not os.path.isfile(p)]
    if missing:
        print(f"Dados não encontrados em {DATA_DIR}. Faltando:")
        for p in missing:
            print(f"  - {p}")
        print("Coloque o bundle em data/ (ver data/README.md) ou defina MTAH_DATA_DIR.")
        return 1

    rc = subprocess.call(
        [sys.executable, str(OFFICIAL / "starter" / "baseline.py"), "--out", str(BASELINES_DIR)]
    )
    if rc:
        return rc

    print("\n=== placar nos 5 min públicos (compare com o docstring deste script) ===\n")
    return subprocess.call(
        [
            sys.executable,
            str(OFFICIAL / "evaluation" / "run_evaluation.py"),
            "--results",
            str(BASELINES_DIR),
            "--out",
            str(OUTPUTS_DIR / "eval_baselines"),
            "--no-detail",
            "--bootstrap",
            "0",
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
