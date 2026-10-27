# Instruções para agentes LLM

Projeto: pipeline do time para a Fase 1 do **More than a Hack 2027**. A saída é uma previsão de
posse de bola por frame (`H`/`A`/`D`/`X`) do jogo Barcelona × Athletic. Regras completas em
[RULES.md](RULES.md). Fonte oficial em `external/mtah-official/`.

**Documentação de referência — consulte antes de escrever código que toque dados ou métrica:**

| arquivo | quando ler |
|---|---|
| [docs/official/DATA.md](docs/official/DATA.md) | formato de cada asset, alinhamento de frames, frames avaliados |
| [docs/official/CHALLENGE.md](docs/official/CHALLENGE.md) | definição de posse, rótulos, o que conta como troca |
| [docs/official/EVALUATION.md](docs/official/EVALUATION.md) | spellF1/macroF1, como o `X` é pontuado |
| [docs/official/SUBMISSION.md](docs/official/SUBMISSION.md) | contrato exato do CSV |
| [docs/official/RULES.md](docs/official/RULES.md) | o que é permitido e proibido |
| [docs/official/FAQ.md](docs/official/FAQ.md) | dúvidas comuns |
| [docs/official/README.md](docs/official/README.md) | índice e commit de origem |

Esses arquivos são cópias: **não edite**. Se divergirem do submódulo, o submódulo vence
(`uv run python scripts/sync_official_docs.py` sincroniza). A definição exata das métricas está em
`external/mtah-official/evaluation/possession_metrics.py`.

---

## 1. Confidencialidade dos dados: regra nº 1, sem exceções

Os termos da competição **proíbem enviar vídeos, dados ou extratos para LLMs hospedados na
nuvem**. Você é um deles. Portanto:

- **NUNCA leia, abra, liste o conteúdo, imprima ou resuma arquivos de dados**: nada dentro de
  `data/` (ou de `$MTAH_DATA_DIR`), `*.mp4`, `*_atd.txt`, `*_atd.xml`, `*_homography.csv`,
  `*_smart_tagging.xml`, `*.parquet`, `*.pkl`, `*.npy`, nem CSVs de submissão em
  `outputs/submissions/`.
- **NUNCA rode comandos cuja saída mostre dados**: `head`, `cat`, `df.head()`, `print(df)`,
  `df.describe()` em tracking, plots renderizados para você ver etc.
- **NUNCA leia outputs de notebooks** que tenham sido executados com dados.
- **Pode** listar nomes de arquivos (`ls`) para checar se existem, e **pode** rodar scripts cuja
  saída seja só: status (ok/erro), tracebacks, shapes/contagens de linhas, mensagens do validador
  e scores agregados (`spellF1`, `macroF1`, nº e duração média de posses).
- Escreva código **sem ver os dados**, a partir da documentação oficial ([docs/official/DATA.md](docs/official/DATA.md))
  e dos loaders oficiais (`external/mtah-official/starter/`). Na dúvida sobre formato, pergunte ao humano.
- Se um traceback ou log incluir valores de dados, não os repita nem os analise além do necessário
  para corrigir o erro.

`.claude/settings.json` bloqueia leituras desses caminhos no Claude Code. Isso é uma rede de
segurança, não substitui esta regra.

## 2. Regras da competição que afetam o código

- **Nada amarrado a frames específicos** do jogo (ex.: `if frame in range(45000, 45100)`). Parâmetros
  devem ser genéricos e ficar em `configs/*.yaml`.
- **Nenhum rótulo manual** do jogo da competição, nem para treino ou validação. Pseudo-rótulos
  gerados por algoritmo são permitidos.
- **Os 5 min públicos** (`external/mtah-official/evaluation/gt/`, frames 11185–18684) servem **só
  para checagem de sanidade**. Não use para tunar hiperparâmetros, escolher entre variantes ou treinar.
  Não proponha isso.
- **Proibido** usar dados humanos sobre este jogo (Opta, StatsBomb, Wyscout etc.). Dados de outros
  jogos são permitidos.
- **Reprodutibilidade exata**: o CSV final precisa sair idêntico byte a byte em qualquer máquina.
  Fixe seeds, evite dependência de ordem de dicionário/sistema de arquivos, não use paralelismo
  não determinístico no caminho da submissão.

## 3. Estrutura e contratos

```
src/mtah27/
  paths.py        caminhos + coloca o repo oficial no sys.path (importe antes de `games`)
  config.py       load_config()
  submission.py   write_submission(): contrato exato do CSV (199.319 linhas, LF)
  pipeline.py     predict(config) -> np.ndarray[199319] de {H,A,D,X}   <- ponto de entrada do modelo
scripts/          make_submission.py · validate.py · score.py
configs/          default.yaml (team_name, parâmetros)
external/mtah-official/   submódulo oficial: NÃO EDITAR
docs/official/            cópia dos .md oficiais: NÃO EDITAR (sync_official_docs.py)
```

- Indexação: **frame do vídeo tático**, 0…199318. ATD usa `frame_id` 1-based (`frame = frame_id - 1`);
  Smart Tagging usa segundos (`round(s * 25)`).
- Home = Barcelona = `H`; away = Athletic = `A`.
- Coordenadas do tracking são normalizadas 0–1; converta para metros (105,3 × 68,0 m em `games.py`)
  antes de calcular distâncias.
- Use os helpers oficiais (`games.asset_paths()`, `games.half_video_blocks()`) em vez de caminhos
  ou números escritos à mão.
- Código novo do modelo fica em módulos dentro de `src/mtah27/`, chamados por `pipeline.predict`.
  Notebooks são só exploração.

## 4. Comandos

Sempre via `uv` (nunca `pip install`):

```bash
uv sync                                                   # ambiente
uv run pytest -q                                          # testes
uv run ruff check . && uv run ruff format .               # lint/format (linha 110)
uv run python scripts/make_submission.py                  # gera outputs/submissions/<team>.csv
uv run python scripts/validate.py outputs/submissions/<team>.csv
uv run python scripts/score.py --no-detail                # placar nos 5 min públicos (só sanidade)
uv add <pacote>                                           # dependência nova (commitar pyproject + uv.lock)
```

Antes de dizer que uma tarefa está pronta: testes e lint passando, e, se mexeu no pipeline,
`make_submission` + `validate` rodando sem erro.

## 5. Convenções

- Código e nomes em inglês; docstrings, comentários, docs e mensagens de commit em **português**.
- Commits no imperativo ("Adiciona detector de bola morta"), pequenos e focados.
- Branches `<iniciais>/<assunto>`; PR para `main`. Não faça push nem commit sem o humano pedir.
- Não commite nada de `data/` ou `outputs/`, nem notebooks com output.
- Experimentos relevantes: uma linha em `experiments/log.csv` (ver `experiments/README.md`).
- Não edite `external/mtah-official/` nem `docs/official/`. Para atualizar: `git submodule update --remote`
  e depois `uv run python scripts/sync_official_docs.py`.
- O time usa Windows, macOS e Linux, e o repo pode estar no OneDrive: use `pathlib`, nada de caminhos
  com `\` ou `/` fixos, e escreva arquivos com `newline="\n"`.
