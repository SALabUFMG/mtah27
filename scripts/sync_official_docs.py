"""Copia os .md da raiz do repo oficial (submódulo) para docs/official/.

    uv run python scripts/sync_official_docs.py

Rode depois de `git submodule update --remote external/mtah-official` e commite junto.
O README oficial vira OFFICIAL_README.md para não colidir com o índice desta pasta.
"""

import shutil
import subprocess

from mtah27.paths import OFFICIAL, ROOT

DEST = ROOT / "docs" / "official"

DESCRIPTIONS = {
    "OFFICIAL_README.md": "visão geral da Fase 1, comandos, timeline",
    "OVERVIEW.md": "a competição como um todo, mudanças em relação a 2026",
    "CHALLENGE.md": "o problema, os quatro rótulos, o que conta como posse",
    "DATA.md": "cada asset, formatos, alinhamento de frames",
    "EVALUATION.md": "métricas (spellF1, macroF1), abstenção, por que não tunar nos 5 min",
    "SUBMISSION.md": "contrato do CSV, entregáveis, formulário",
    "RULES.md": "o que é permitido e proibido",
    "FAQ.md": "perguntas frequentes",
}
ORDER = list(DESCRIPTIONS)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "-C", str(OFFICIAL), "log", "-1", "--format=%h %cs %s"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    copied = []
    for src in sorted(OFFICIAL.glob("*.md")):
        name = "OFFICIAL_README.md" if src.name == "README.md" else src.name
        shutil.copyfile(src, DEST / name)
        copied.append(name)
    copied.sort(key=lambda n: (ORDER.index(n) if n in ORDER else len(ORDER), n))

    lines = [
        "# Documentação oficial (cópia de referência)",
        "",
        "Cópia dos `.md` do repositório oficial "
        "[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027) "
        "(licença MIT), para leitura rápida.",
        "",
        f"- **Commit de origem:** `{commit}`",
        "- **Não edite estes arquivos.** A fonte de verdade é o submódulo `external/mtah-official/`. "
        "Para atualizar: `uv run python scripts/sync_official_docs.py`.",
        "- Links para código (`evaluation/`, `starter/`, `games.py`) e para `README.md` referem-se ao "
        "repo oficial: abra-os em `external/mtah-official/`.",
        "",
        "| arquivo | conteúdo |",
        "|---|---|",
        *[f"| [{n}]({n}) | {DESCRIPTIONS.get(n, '')} |" for n in copied],
    ]
    (DEST / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(copied)} arquivos copiados para docs/official/ (origem: {commit})")


if __name__ == "__main__":
    main()
