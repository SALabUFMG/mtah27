# Regras da competição — resumo para o time

> Resumo em português do que importa no dia a dia. **A fonte oficial é o repositório da
> organização** ([`external/mtah-official`](external/mtah-official), principalmente `RULES.md`,
> `EVALUATION.md`, `SUBMISSION.md` e `DATA.md`) e os termos aceitos na inscrição. Em caso de
> divergência, o oficial vence. Dúvida perto de uma linha cinzenta → abrir issue no repo oficial
> **antes** de construir em cima.

## Calendário

| | |
|---|---|
| Inscrições | até **10/nov/2026** |
| Dados liberados | 20/out/2026 |
| **Prazo de submissão (Fase 1)** | **17/nov/2026** — meta interna: **15/nov** |
| Resultado da Fase 1 | antes de 26/nov/2026 |
| Fase 2 — Innovation Challenge (top 20) | 26/nov/2026 – 14/jan/2027 · tema livre, 10 jogos |
| Final Pitch (top 5) | 2/mar/2027, presencial, Talent Arena (Barcelona) |

Time: **3 a 5 membros**. Com menos de 3, não avança para a Fase 2.

## A tarefa (Fase 1)

Jogo: **FC Barcelona (home) × Athletic Club (away)**, LaLiga 2024-25, rodada 2.

Para **cada frame do vídeo tático** (199.319 frames, 25 fps), um rótulo:

| label | significado |
|---|---|
| `H` | Barcelona com a posse, bola viva |
| `A` | Athletic com a posse, bola viva |
| `D` | bola morta (fora de jogo) |
| `X` | abstenção |

Frames avaliados: **11.185–90.651** (1º tempo) e **110.808–190.197** (2º tempo). O resto é
ignorado. Atenção: de 110.808 a 113.741 (antes do pontapé do 2º tempo) é tudo `D` e **é avaliado**.

Definições que já estão fechadas:

- Só três estados no gabarito: `H`, `A`, `D`. Não existe "disputada" nem "transição".
- **Posse = sequência máxima de um time com bola viva.** `H H D D H H` são **duas** posses.
- Bola dividida/solta: a posse continua com o último time que a tinha.
- Troca de posse exige **controle** pelo adversário. Toque sem controle, desvio que não intercepta
  e duelo aéreo sem controle posterior **não** trocam a posse.
- Bola morta é sempre `D`, independente de quem vai repor.
- O gabarito vem de tracking **corrigido** (não de eventos nem de vídeo), sem suavização.

## Métrica

- **Ranking: `spellF1`**. Cada posse é um objeto; posses previstas são casadas 1-a-1 com as reais
  por IoU temporal, F1 médio nos limiares 0,3–0,7. Oscilar entre rótulos cria posses falsas e
  derruba o score, mesmo com boa acurácia por frame. Diferenças < ~0,04 são ruído.
- **Desempate: `macroF1`** (F1 médio de H, A, D por frame). Diferenças < ~0,01 são ruído.
- **Baseline a superar: `spellF1` 0,2433.**
- **`X` no spellF1 = repetir o último rótulo** do mesmo tempo. Grátis no meio de uma posse, caro
  numa troca. No macroF1, `X` conta como erro.
- Não há leaderboard ao vivo nem feedback antes do prazo.

## Dados fornecidos

| | tático (**referência**) | broadcast (opcional) |
|---|---|---|
| vídeo | 199.319 frames, 1920×1080 | outra linha do tempo, com cortes e replays, **não sincronizado** |
| tracking automático (ATD) | sim, com bola | sim, com bola |
| homografia | 100% válida | 65,8% válida |
| Smart Tagging | 638 instâncias | — |

- Tracking **automático e não corrigido**: sem identidade de jogador (46 slots), falhas, trocas de
  ID. **Bola ausente em ~41% dos frames avaliados.**
- Smart Tagging é automático (tem falsos positivos/negativos). Fases ≠ posses. Laterais têm `Team = N/A`.
- Alinhamento: ATD `frame_id - 1` = frame do vídeo; Smart Tagging `round(segundos * 25)`;
  homografia já está em frame do vídeo.

## O que é permitido

- **Qualquer método**, desde que (1) os organizadores possam acessar e inspecionar e (2) o documento
  técnico explique. Qualquer linguagem, framework, modelo pré-treinado, LLM.
- Melhorar ou substituir o tracking/tagging **por algoritmo** (detector próprio etc.).
- Dados externos de **outros jogos**, inclusive rotulados à mão pelo time (documentar: o quê,
  como, quem, quanto).
- Aprendizado não supervisionado/auto-supervisionado e pseudo-rótulos gerados por algoritmo no
  jogo da competição.
- Assistir ao vídeo e inspecionar os dados para entender e depurar.

## O que é proibido

1. **Rotular à mão o jogo da competição**, em qualquer quantidade e para qualquer fim (submissão,
   treino ou validação). Inclui corrigir à mão a saída do algoritmo ou o tracking.
2. **Regras ou constantes amarradas a frames específicos** do jogo.
3. **Usar o gabarito privado** ou qualquer dado humano sobre o jogo (Opta, StatsBomb, Wyscout,
   LaLiga, tracking corrigido, anotações…).
4. **Treinar ou tunar nos 5 minutos públicos** (frames 11.185–18.684). Eles servem só para
   checar se o pipeline está correto.
5. **Enviar um arquivo que o repositório não regenera exatamente** (mesmo arquivo, byte a byte).
6. **Redistribuir os dados**: nada no git, nada publicado que permita reconstruí-los (exports
   de tracking frame a frame, recortes de vídeo).
7. **Enviar vídeos, dados ou extratos para LLMs hospedados na nuvem** (ChatGPT, Claude, Gemini,
   Copilot…). Usar LLM para código é permitido; mandar os dados para ele, não.
8. Depender de modelo privado ou dataset interno que não possa ser compartilhado (perguntar antes).

## Entregáveis (um formulário Google por time; vale a última resposta antes do prazo)

1. **Arquivo de posse** — `<team_name>.csv`: cabeçalho `frame,label`, exatamente 199.319 linhas,
   frames 0…199318 em ordem, sem células vazias (~1,9 MB).
2. **Repositório público** — código, dependências pinadas, instruções e o comando que regenera o CSV.
3. **Documento técnico** — `<team_name>.pdf` ou `.md`, poucas páginas: abordagem, assets usados e
   descartados, dados de treino, regras de posse implementadas, tratamento de tracking ausente,
   limitações, reprodução. Template em [`docs/technical_report.md`](docs/technical_report.md).

Os organizadores podem rodar o repositório do time vencedor para confirmar o resultado.

## Checklist de submissão

- [ ] `team_name` correto em `configs/default.yaml` (mesmo nome da inscrição: **SALab UFMG**)
- [ ] `main` limpo; `uv run python scripts/make_submission.py` a partir de clone novo
- [ ] SHA-256 do CSV igual ao gerado em outra máquina do time
- [ ] `uv run python scripts/validate.py outputs/submissions/<team_name>.csv` → "Valid."
- [ ] histórico do git sem dados (`git log --all --stat`), notebooks sem outputs
- [ ] repo tornado **público**
- [ ] documento técnico exportado como `<team_name>.pdf`
- [ ] formulário enviado (precisa de conta Google) até 15/nov
