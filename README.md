# mtah27 — More than a Hack 2027

Pipeline do time para a **Fase 1** do [More than a Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027)
(Metrica Sports · FC Barcelona · Mobile World Capital Barcelona): para cada frame do jogo
Barcelona × Athletic, dizer quem tem a posse (`H` / `A`) ou se a bola está morta (`D`).

Regras, dados e métrica: repositório oficial, incluído como submódulo em
[`external/mtah-official`](https://github.com/metrica-sports/more-than-a-hack-2027).

---

## Reproduzir a submissão

Requisitos: [uv](https://docs.astral.sh/uv/) e git. O uv instala o Python 3.11 sozinho.

```bash
git clone --recurse-submodules <url-deste-repo>
cd mtah27
uv sync                                   # ambiente idêntico ao uv.lock
# coloque o bundle de dados em ./data  (ou defina MTAH_DATA_DIR) — ver data/README.md
uv run python scripts/make_submission.py  # -> outputs/submissions/<team>.csv
```

O script imprime o SHA-256 do CSV; ele deve ser idêntico em qualquer máquina.

## Comandos do dia a dia

| tarefa | comando | `make` |
|---|---|---|
| preparar ambiente | `git submodule update --init --recursive && uv sync && uv run pre-commit install` | `make setup` |
| gerar submissão | `uv run python scripts/make_submission.py` | `make submission` |
| validar (checagem oficial) | `uv run python scripts/validate.py outputs/submissions/mtah27.csv` | `make validate` |
| pontuar nos 5 min públicos | `uv run python scripts/score.py` | `make score` |
| testes | `uv run pytest -q` | `make test` |
| notebooks | `uv run jupyter lab` | |

> Os 5 minutos públicos servem **só** para checar se o pipeline está correto, nunca para escolher
> entre versões ou tunar (regra da competição).

## Estrutura

```
mtah27/
├─ configs/               parâmetros (YAML) — nada amarrado a frames específicos
├─ data/                  dados da competição (gitignored)
├─ docs/                  documento técnico da submissão
├─ experiments/log.csv    registro de execuções
├─ external/mtah-official repo oficial (submódulo): scorer, validador, loaders
├─ notebooks/             exploração (sem outputs no git)
├─ outputs/               submissões, relatórios, figuras (gitignored)
├─ scripts/               entrypoints (make_submission, validate, score)
├─ src/mtah27/            o pacote: pipeline.predict() é o ponto de entrada do modelo
└─ tests/
```

## Atualizar o repo oficial

```bash
git submodule update --remote external/mtah-official
git add external/mtah-official && git commit -m "Atualiza repo oficial"
```

## Time

Ver [CONTRIBUTING.md](CONTRIBUTING.md) para o fluxo de trabalho e as regras de dados.
