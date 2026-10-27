# mtah27 — More than a Hack 2027

Pipeline do time para a **Fase 1** do [More than a Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027)
(Metrica Sports · FC Barcelona · Mobile World Capital Barcelona): para cada frame do jogo
Barcelona × Athletic, dizer quem tem a posse (`H` / `A`) ou se a bola está morta (`D`).

Regras, dados e métrica: repositório oficial, incluído como submódulo em
[`external/mtah-official`](https://github.com/metrica-sports/more-than-a-hack-2027).
Cópia da documentação oficial para consulta rápida em **[docs/official/](docs/official/README.md)**;
resumo em português em [RULES.md](RULES.md).

---

## Instalação local (membros do time)

Passo a passo para deixar o projeto rodando na sua máquina. Leva uns 5 minutos.

### 1. Pré-requisitos

- **Git** — https://git-scm.com/downloads
- **uv** (gerencia Python e dependências; instala o Python 3.11 sozinho, não precisa ter Python antes):

  ```powershell
  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  Feche e reabra o terminal depois e confira com `uv --version`.
- **Acesso ao repo** `SALabUFMG/mtah27` (privado) — peça para alguém do time te adicionar.

### 2. Clonar (com o submódulo oficial)

```bash
git clone --recurse-submodules https://github.com/SALabUFMG/mtah27.git
cd mtah27
```

Já clonou sem `--recurse-submodules` e a pasta `external/mtah-official` está vazia? Rode:

```bash
git submodule update --init --recursive
```

### 3. Criar o ambiente

```bash
uv sync                     # cria .venv com as versões exatas do uv.lock
uv run pre-commit install   # ganchos de commit: limpa outputs de notebooks, bloqueia arquivos grandes
```

O `pre-commit install` é **obrigatório para cada membro**: é ele que impede que outputs de
notebook (que contêm dados) entrem no git.

### 4. Colocar os dados

Descompacte o bundle da competição em `data/` (estrutura em [data/README.md](data/README.md)),
ou guarde em outro lugar e aponte a variável `MTAH_DATA_DIR` para lá:

```powershell
$env:MTAH_DATA_DIR = "D:\mtah-data"          # Windows (só na sessão atual)
setx MTAH_DATA_DIR "D:\mtah-data"            # Windows (permanente; reabra o terminal)
```
```bash
export MTAH_DATA_DIR=/caminho/para/mtah-data # macOS / Linux
```

**Nunca** commite os dados nem cole trechos deles em LLMs na nuvem — ver [CONTRIBUTING.md](CONTRIBUTING.md).

### 5. Conferir se está tudo certo

```bash
uv run pytest -q                                                   # testes passam
uv run python scripts/make_submission.py                           # gera outputs/submissions/SALab_UFMG.csv
uv run python scripts/validate.py outputs/submissions/SALab_UFMG.csv   # "Valid."
uv run python scripts/score.py --no-detail                         # placar nos 5 min públicos
```

Se os quatro rodarem sem erro, você está pronto. Com os dados no lugar, rode também
`uv run python scripts/run_baseline.py`: os scores públicos devem bater com os publicados (spellF1 0,3256 na `baseline_smooth50`). Para abrir notebooks: `uv run jupyter lab`.

### Dicas e problemas comuns

- **VS Code / PyCharm:** selecione o interpretador em `.venv` (Windows: `.venv\Scripts\python.exe`;
  macOS/Linux: `.venv/bin/python`).
- **Repo dentro do OneDrive/Dropbox:** a `.venv` tem milhares de arquivos e vai sincronizar.
  Para criá-la fora, defina `UV_PROJECT_ENVIRONMENT` antes do `uv sync`
  (ex.: `setx UV_PROJECT_ENVIRONMENT "%USERPROFILE%\venvs\mtah27"`).
- **`ModuleNotFoundError: games` ou "Submódulo oficial ausente":** falta o passo do submódulo (item 2).
- **Adicionar uma dependência:** `uv add <pacote>` e commite `pyproject.toml` + `uv.lock` juntos.
  Depois de um `git pull` que mude o `uv.lock`, rode `uv sync` de novo.
- **`make`** é opcional; no Windows use os comandos `uv run ...` diretamente.

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
| validar (checagem oficial) | `uv run python scripts/validate.py outputs/submissions/SALab_UFMG.csv` | `make validate` |
| baseline oficial (1º contato com os dados) | `uv run python scripts/run_baseline.py` | |
| pontuar nos 5 min públicos | `uv run python scripts/score.py` | `make score` |
| testes | `uv run pytest -q` | `make test` |
| notebooks | `uv run jupyter lab` | |

> Os 5 minutos públicos servem **só** para checar se o pipeline está correto, nunca para escolher
> entre versões ou tunar (regra da competição).

## Estrutura

```
mtah27/
├─ RULES.md              resumo das regras da competição
├─ CLAUDE.md             instruções para agentes LLM
├─ configs/               parâmetros (YAML) — nada amarrado a frames específicos
├─ data/                  dados da competição (gitignored)
├─ docs/                  documento técnico da submissão
│  └─ official/           cópia dos .md do repo oficial (referência, não editar)
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
uv run python scripts/sync_official_docs.py          # atualiza a cópia em docs/official/
git add external/mtah-official docs/official && git commit -m "Atualiza repo oficial"
```

## Time

- [RULES.md](RULES.md): resumo das regras, métrica, entregáveis e checklist de submissão.
- [docs/official/](docs/official/README.md): documentação oficial completa (DATA, EVALUATION, RULES, FAQ…).
- [CONTRIBUTING.md](CONTRIBUTING.md): fluxo de trabalho e regras de dados.
- [CLAUDE.md](CLAUDE.md): instruções para agentes LLM (Claude Code, Codex, Copilot…).
