# Experimentos

Cada execução relevante vira uma linha em `log.csv`:

| coluna | o que é |
|---|---|
| `date` | AAAA-MM-DD |
| `author` | quem rodou |
| `git_commit` | `git rev-parse --short HEAD` (commit limpo!) |
| `config` | arquivo em `configs/` |
| `description` | o que mudou em relação ao anterior |
| `n_possessions` | nº de posses previstas (impresso pelo validador) |
| `mean_poss_s` | duração média das posses previstas |
| `val_spellF1` / `val_macroF1` | scores na validação **própria** do time |
| `public_spellF1` / `public_macroF1` | scores nos 5 min públicos — **só sanidade, nunca para escolher versão** |
| `notes` | |

Não colocar aqui nada derivado frame a frame dos dados.
