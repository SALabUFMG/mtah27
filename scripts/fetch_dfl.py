"""Baixa os 7 jogos públicos da DFL (Bundesliga 1 e 2, 2022/23) para treino e validação.

Uso:
    uv run python scripts/fetch_dfl.py

Destino: ``DATA_DIR/external/dfl/``, com os nomes originais dos arquivos. Cada arquivo é verificado por
SHA-256 e fixado a um commit do espelho; arquivos já presentes e íntegros não são baixados de novo. São
cerca de 2,6 GB.

Fonte: Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of spatiotemporal and event data in
elite soccer", Scientific Data, https://doi.org/10.1038/s41597-025-04505-y. Licença CC-BY 4.0; os dados
são da Deutsche Fußball Liga (DFL). O original está no figshare, que barra download automático; baixamos
do espelho da PySport, https://huggingface.co/datasets/pysport/idsse-data.
"""

# ruff: noqa: E501  (hashes SHA-256 longos em FILES)
import os
import tempfile
import urllib.request
from pathlib import Path

from fetch_metrica_sample import _sha256

from mtah27.paths import DATA_DIR

COMMIT = "a715a38dfbaf5f58e431727c2b78d174101a703c"
BASE_URL = f"https://huggingface.co/datasets/pysport/idsse-data/resolve/{COMMIT}"
ROOT_DIR = DATA_DIR / "external" / "dfl"
MATCHES = {  # match_id: competição (DFL-COM-...)
    "J03WMX": "000001",
    "J03WN1": "000001",
    "J03WOH": "000002",
    "J03WOY": "000002",
    "J03WPY": "000002",
    "J03WQQ": "000002",
    "J03WR9": "000002",
}
META = "DFL_02_01_matchinformation_DFL-COM-{com}_DFL-MAT-{mid}.xml"
TRACKING = "DFL_04_03_positions_raw_observed_DFL-COM-{com}_DFL-MAT-{mid}.xml"
SHA = {  # (metadados, tracking) por match_id
    "J03WMX": (
        "a884728647a564cf8b20e5223a1d04a5da6f6725e1e506c8651505327c197ba9",
        "d3dd92e509421a6466acb0a083c009035f1751bc5fd9ef42ffa8f20d01cf7d49",
    ),
    "J03WN1": (
        "5490286436c86e6f80a538d2b1b379d0efd016e0defcaf594bc59d9ef5079f4c",
        "56ef948961a18e150dba0b7b16869d0008ff66f3050a0c355d31fa5b8ec2fcec",
    ),
    "J03WOH": (
        "f23b3ee13b988b1ff9a0bce438c416dac47e71355bbc7ee1dab21aaf3e33c18c",
        "ec4dcf30200aad694dea263d5289485fcd0ca42badddaeb7c81f5cd5a4229d4f",
    ),
    "J03WOY": (
        "4c497fac683eee8292901c18ca02c0be1d9609d0b9d4bd8aa092c61f9e0b19e9",
        "a228ef0e676d009cefff1580fcb721b53f598454eb120ec347e0a628a3a4646b",
    ),
    "J03WPY": (
        "566dd5e4c86e124ab90fcdba51100881f0436e22afa71832945a451a2d95a420",
        "845202c29ad93d696653436da426a7cdaefb848298e07b7343de6529581671e1",
    ),
    "J03WQQ": (
        "e976aa0c5dba862cfa898ea57ec226efce9d1bc04b23b8d775908db86aa6fa41",
        "cbefcb0bd7f39f3b94ee69a7bc97aac0f936425062c1cc1b144733df36804d6f",
    ),
    "J03WR9": (
        "429b584e89df3e3e8b81f4bd0d004b38bd0b4298bb390b8fb43efccb60963e01",
        "70b63069189019066fb7e3d9db891122edfe787f3a43ba522ebe45ed89ea0269",
    ),
}
FILES = {
    tpl.format(com=com, mid=mid): sha
    for mid, com in MATCHES.items()
    for tpl, sha in zip((META, TRACKING), SHA[mid], strict=True)
}


def fetch(name: str, expected: str) -> str:
    """Garante que ``name`` existe em ROOT_DIR com o hash esperado. Retorna o status."""
    dest = ROOT_DIR / name
    if dest.exists() and _sha256(dest) == expected:
        return "ok (cached)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(f"{BASE_URL}/{name}") as resp:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        got = _sha256(tmp)
        if got != expected:
            raise RuntimeError(f"SHA-256 diferente para {name}: esperado {expected}, obtido {got}")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return "ok (downloaded)"


def main() -> None:
    for name, expected in FILES.items():
        status = fetch(name, expected)
        print(f"{name}  {(ROOT_DIR / name).stat().st_size} bytes  {status}", flush=True)


if __name__ == "__main__":
    main()
