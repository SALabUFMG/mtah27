"""Extrai os dados públicos da PFF (Copa do Mundo 2022, 64 jogos) dos zips baixados do Google Drive.

Uso:
    uv run python scripts/extract_pff.py <zip> [<zip> ...]

Destino: ``DATA_DIR/external/pff/{tracking,events,metadata}/<game_id>.*``. Só a versão raiz dos
eventos é extraída (as pastas datadas são versões antigas). Arquivos já presentes são pulados.
"""

import re
import shutil
import sys
import zipfile

from mtah27.paths import DATA_DIR

ROOT_DIR = DATA_DIR / "external" / "pff"
WANTED = {
    "Tracking Data": ("tracking", ".jsonl.bz2"),
    "Event Data": ("events", ".json"),
    "Metadata": ("metadata", ".json"),
}
PATTERN = re.compile(r"^[^/]+/(Tracking Data|Event Data|Metadata)/(\d+)(\.jsonl\.bz2|\.json)$")


def main(zips: list[str]) -> None:
    n = 0
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for name in sorted(z.namelist()):
                m = PATTERN.match(name)
                if not m or m.group(3) != WANTED[m.group(1)][1]:
                    continue
                sub, ext = WANTED[m.group(1)]
                dest = ROOT_DIR / sub / f"{m.group(2)}{ext}"
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                with z.open(name) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                tmp.replace(dest)
                n += 1
    counts = {sub: len(list((ROOT_DIR / sub).glob("*"))) for sub, _ in WANTED.values()}
    print(f"extraídos {n} arquivos; em {ROOT_DIR}: {counts}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
