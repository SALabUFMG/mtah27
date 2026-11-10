# SALab UFMG — More than a Hack 2027, Fase 1

> Documento técnico da submissão (poucas páginas). Estrutura pedida no SUBMISSION.md oficial.
> Exportar como `SALab_UFMG.pdf` para o formulário.

## 1. Abordagem

A submissão é o caminho C1: um HMM de três estados (H, A, D) decodificado por Viterbi com duração
mínima por estado, em que a emissão vem de um classificador de gradient boosting. Cada tempo do jogo
é decodificado separadamente, e os frames fora dos dois tempos recebem `X`.

O modelo roda numa grade de 5 Hz: de cada 0,2 s entra um frame, e cada frame nativo recebe o
rótulo da amostra mais próxima dentro do mesmo tempo. A grade é a mesma para todos os providers
(25 fps no alvo e na Metrica, 29,97 na PFF), então a duração mínima e as transições valem o mesmo
em segundos em todos.

Para cada amostra calculamos 33 features do tracking numa janela centrada de 2 s
(`src/mtah27/features.py`):

- bola: presença, velocidade, fora do campo, distância à linha lateral e à linha de fundo;
- disputa: distância dos dois jogadores mais próximos da bola em cada time, a diferença entre os
  mais próximos, quantos jogadores de cada time estão a menos de 3 m e de 8 m, e a fração da janela
  em que o mais próximo é do Barcelona;
- times: para cada um, posição do centróide, dispersão, velocidade média dos jogadores, número de
  jogadores visíveis e distância do centróide à bola;
- movimento: mediana da velocidade dos jogadores, a mesma dividida pela mediana do tempo de jogo, e
  a fração de jogadores quase parados.

Nenhuma feature depende do índice do slot, só do time, porque os 46 slots do ATD não seguem uma
pessoa. Nenhuma depende do lado que cada time ataca: a posição em x entra como distância ao meio do
campo.

Um `HistGradientBoostingClassifier` do scikit-learn (300 iterações, taxa 0,05, 31 folhas, seed 0)
estima P(estado | features). O HMM precisa de P(features | estado), que pela regra de Bayes é
proporcional a P(estado | features) / P(estado). Essa razão entra como emissão no mesmo Viterbi do
C0, com a transição contada nos rótulos de treino e duração mínima de 2 s para H e A e de 4 s para
D. A duração mínima segura a fragmentação que o spellF1 pune: uma posse quebrada em três pedaços
conta como erro.

O C1 veio de uma análise de erro do C0 (`scripts/error_analysis_c0.py`), o HMM categórico que foi
nossa primeira submissão. O C0 via só o time do jogador mais próximo da bola e uma faixa de
velocidade dos jogadores. Na validação, quase metade das posses verdadeiras dura menos de 4 s, e o
C0 casava só 11% delas, porque a sua duração mínima era de 4 s. A acurácia por frame caía de 0,81
com bola visível para 0,66 sem bola. Uma emissão mais informativa deixa usar durações mínimas
menores sem fragmentar.

## 2. Dados utilizados

Do bundle oficial usamos só o ATD tático (`atd_meta` e `atd_raw`): posições de jogadores e bola,
o time de cada slot e o tamanho do campo (105,3 × 68,0 m em `games.py`). O vídeo, a homografia e o
broadcast não entram.

O Smart Tagging ficou fora de propósito. Ele só existe para o jogo da competição, então o peso dele
não pode ser calibrado nem validado nos jogos externos.

## 3. Dados de treino

O modelo é treinado nos três jogos públicos da Metrica Sports
([metrica-sports/sample-data](https://github.com/metrica-sports/sample-data), commit `e706dd5`).
São de outra partida e foram baixados por `scripts/fetch_metrica_sample.py`, com o SHA-256 de cada
arquivo verificado. O repositório não declara licença; o README pede uso responsável e atribuição
à Metrica Sports. Nenhum dado do jogo da competição entra no treino ou na validação.

Os rótulos H/A/D por frame saem dos eventos desses jogos, que foram anotados por pessoas da
Metrica, por uma regra fixa (`labels_from_events` em `src/mtah27/external.py`):

- vale o time do último evento com controle de bola (passe, set piece, recuperação, chute,
  condução);
- perda de bola e disputa não trocam a posse;
- D vai de uma bola fora, falta ou gol até o próximo evento de controle, que é o reinício. O tiro
  de meta aparece como passe, não como set piece.

Cada jogo entra no treino em duas cópias com a bola apagada em lacunas sorteadas com seeds
diferentes (seção 5). O modelo não fica salvo no repositório: o `make_submission` o treina de novo
nos três jogos, o que leva cerca de dois minutos.

A validação é leave-one-game-out: treina em dois jogos e avalia no terceiro com a métrica oficial,
girando os três. O modelo é treinado uma vez por jogo de teste, e a grade (duração mínima de H e A
de 1 a 6 s, de D de 2 a 8 s, e peso 0,5 ou 1 na emissão) varia só a decodificação
(`scripts/calibrate_c1.py`). Resultado do ponto escolhido, médio nos três jogos:

| | spellF1 | macroF1 |
| --- | --- | --- |
| C1 (submissão) | 0,592 | 0,793 |
| C0, HMM categórico | 0,495 | 0,748 |

Os dois números são o máximo das respectivas grades, então são um pouco otimistas, mas a
comparação entre eles é justa. O LightGBM no lugar do scikit-learn deu 0,589, dentro do ruído, e
ficamos com o scikit-learn para ter uma dependência a menos.

### Dados públicos da PFF (Copa de 2022)

Testamos também os 64 jogos públicos da PFF FC como fonte de treino. Os rótulos saem dos eventos
pela mesma regra, com ajustes para o formato da PFF:

- um evento de bola só troca a posse se mostra controle (recepção, condução, cruzamento, chute, ou
  passe depois de outra ação). Duelo, corte, rebote ou toque isolado não trocam, e um passe de
  primeira só troca se for completo;
- o JSON repete o evento uma vez por ação de posse; juntamos as linhas antes de aplicar a regra;
- a disputa de pênaltis, que vem depois do fim da prorrogação, fica fora. Pênaltis durante o jogo
  ficam.

A bola da PFF usada é a bruta (`balls`). A versão suavizada (`ballsSmoothed`) é reposicionada a
partir dos eventos e vazaria o rótulo. A PFF marca algumas posições como estimadas, e as trocamos
por ausentes no caso da bola, já que o tracking do alvo não estima.

A PFF não ajudou em nenhum dos dois modelos. No C0, ajustar na PFF e avaliar na Metrica deu
spellF1 entre 0,417 e 0,452 (`scripts/cross_eval_c0.py`). No C1, somar a PFF ao treino baixou o
spellF1 na Metrica de 0,592 para 0,563, embora o macroF1 tenha subido de 0,793 para 0,812. A PFF
tem menos posses por jogo que a Metrica (226 contra 347), o que sugere que as duas anotações contam
posse de jeito diferente.

## 4. Regras de posse implementadas

Seguimos o CHALLENGE.md. A posse só muda quando o outro time ganha controle; bola disputada ou
solta fica com o último time. O classificador aprende isso dos rótulos, em que perda de bola e
disputa não trocam a posse, e a duração mínima impede trocas mais curtas que 2 s.

Bola morta é sempre D, sem dono, como pede o desafio.

## 5. Tratamento de tracking ausente/ruidoso

A bola some em 41% dos frames avaliados do alvo, a taxa publicada no DATA.md. Para que o treino e
a validação vejam a mesma ausência, a bola dos jogos externos é apagada em lacunas aleatórias até
essa taxa, contada só dentro dos tempos de jogo. A duração média das lacunas (2 s) é um chute. No
treino, cada jogo entra com duas cópias degradadas com seeds diferentes; a avaliação usa sempre a
mesma cópia.

As features aceitam valores ausentes: sem bola, as distâncias à bola ficam vazias e o classificador
usa o que sobra, principalmente o movimento dos jogadores. Em bola morta os jogadores param, e isso
continua visível no tracking mesmo quando a bola não está.

Na velocidade dos jogadores, saltos acima de 12 m/s são tratados como troca de identidade do
tracklet e descartados. A velocidade de cada time é a média dos deslocamentos dos jogadores
presentes em duas amostras seguidas, e não o deslocamento do centróide, que salta quando um
tracklet aparece ou some.

No C0 também testamos aprender a parte de bola morta por EM no próprio jogo, sem rótulo. O EM
sempre convergiu para "bola ausente = D" e foi descartado.

## 6. Limitações

- Três jogos externos é pouco. O ruído do spellF1 é de cerca de ±0,04. O ganho do C1 sobre o C0
  (0,10) passa disso, mas as diferenças dentro da grade do C1 não.
- Nos jogos externos a bola quase só some em bola morta (91% a 99,8% dos frames D). A degradação
  apaga a bola também com o jogo vivo, mas no treino a ausência continua mais comum em D do que
  provavelmente é no alvo. O C0 neutralizava isso com uma emissão igual nos três estados; o C1 vê a
  presença da bola como feature e pode ter aprendido parte desse artefato.
- Os rótulos vêm de eventos, não do tracking corrigido que gera o gabarito oficial. As fronteiras
  de bola morta são aproximadas. Na Metrica, metade das posses rotuladas dura menos de 4 s, e a
  duração mínima de 2 s escolhida pela validação segue esses rótulos. No jogo da competição o
  modelo prevê 396 posses com 10,6 s de média e marca D em 33,7% dos frames avaliados.
- O formato das lacunas de bola do alvo é desconhecido; só a taxa é conhecida.
- A Metrica também é outro provider. Ela é o alvo mais próximo que temos, mas a escolha entre
  Metrica e PFF supõe que o alvo se parece mais com ela.
- O modelo é treinado de novo a cada execução. O CSV sai igual com uma ou várias threads no
  Windows, mas ainda não conferimos o SHA-256 em macOS e Linux.

## 7. Reprodução

```bash
git clone --recurse-submodules <url>
uv sync
uv run python scripts/fetch_metrica_sample.py
uv run python scripts/make_submission.py
```

Os hiperparâmetros estão em `configs/c1_fitted.yaml`, e o modelo é treinado nos jogos da Metrica
dentro do `make_submission`. Para refazer a validação e a escolha dos hiperparâmetros:

```bash
uv run python scripts/calibrate_c1.py
```

O C0 continua disponível com `model: c0` em `configs/default.yaml`
(`scripts/calibrate_c0.py` recalcula `configs/c0_fitted.yaml`).

SHA-256 esperado do CSV: `fd0cb8337f6d20d6a39340a7dcc7d59fef9e20733970b4c03e999eba745f98d9`
