# SALab UFMG — More than a Hack 2027, Fase 1

> Documento técnico da submissão (poucas páginas). Estrutura pedida no SUBMISSION.md oficial.
> Exportar como `SALab_UFMG.pdf` para o formulário.

## 1. Abordagem
O que construímos e por que isso, e não outra coisa.

## 2. Dados utilizados
Quais assets usamos e como; quais testamos e descartamos.

## 3. Dados de treino
Todo dataset rotulado usado: fonte, quem rotulou, quanto. Nenhum do jogo fornecido.

## 4. Regras de posse implementadas
Onde traçamos a linha em bolas divididas, disputas, reinícios.

## 5. Tratamento de tracking ausente/ruidoso

## 6. Limitações
Onde falha e o que faríamos com mais tempo.

## 7. Reprodução
```bash
git clone --recurse-submodules <url>
uv sync
uv run python scripts/make_submission.py
```
SHA-256 esperado do CSV: `<preencher>`
