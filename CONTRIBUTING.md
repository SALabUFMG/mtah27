# Como trabalhamos

## Regras inegociáveis (vêm dos termos da competição)

1. **Dados nunca vão para o git.** Nem vídeo, nem tracking, nem parquet derivado, nem CSV de submissão. O `.gitignore` e o pre-commit protegem, mas a responsabilidade é de cada um.
2. **Dados nunca vão para LLMs na nuvem** (ChatGPT, Claude, Gemini, Copilot Chat…) — nem extratos, nem prints de tabela/plot do tracking. Pode pedir ajuda com código; cole mensagens de erro e scores, não dados.
3. **Nenhum label feito à mão** no jogo da competição, nem para treinar/validar. Rótulos manuais em **outros** jogos são permitidos (documentar).
4. **Nenhum parâmetro amarrado a frames específicos** do jogo.
5. **Os 5 min públicos não são conjunto de validação.** Só checagem de sanidade.
6. **Notebooks sem output** no git (o `nbstripout` faz isso no commit).

## Fluxo git

- `main` sempre gera uma submissão válida (`make submission && make validate`).
- Trabalho em branches: `<iniciais>/<assunto>` — ex.: `jl/deadball-features`.
- Pull request para `main`, com pelo menos 1 revisão de outro membro.
- Antes de abrir o PR: `make test` e `make lint`.
- Mensagens de commit no imperativo, em português: "Adiciona detector de bola morta".

## Experimentos

Toda execução que alguém usaria para tomar decisão vira uma linha em `experiments/log.csv`, rodada a partir de um commit limpo (sem alterações pendentes).

## Dependências

`uv add <pacote>` (ou `uv add --group dev <pacote>`) e commitar `pyproject.toml` + `uv.lock` juntos. Nunca `pip install` solto.

## Antes de tornar o repo público

- [ ] `git log --all --stat` sem nada de `data/`, `.mp4`, `.parquet`, `.csv` grande
- [ ] notebooks sem outputs
- [ ] clone limpo + `uv sync` + `make submission` reproduz o mesmo SHA-256
