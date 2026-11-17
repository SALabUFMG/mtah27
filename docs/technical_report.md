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

Para cada amostra calculamos 40 features do tracking numa janela centrada de 2 s
(`src/mtah27/features.py`):

- bola: presença, velocidade, fora do campo, distância à linha lateral e à linha de fundo;
- disputa: distância dos dois jogadores mais próximos da bola em cada time, a diferença entre os
  mais próximos, quantos jogadores de cada time estão a menos de 3 m e de 8 m, e a fração da janela
  em que o mais próximo é do Barcelona;
- times: para cada um, posição do centróide, dispersão, velocidade média dos jogadores, número de
  jogadores visíveis e distância do centróide à bola;
- movimento: mediana da velocidade dos jogadores, a mesma dividida pela mediana do tempo de jogo, e
  a fração de jogadores quase parados;
- lado de ataque (5): posição em x do centróide de cada time, velocidade em x de cada time e posição
  em x da bola, todas no referencial do gol que o time ataca;
- lacuna de bola (2): tamanho da lacuna de bola em que a amostra está e tempo desde a última bola
  vista (seção 5).

Nenhuma feature depende do índice do slot, só do time, porque os 46 slots do ATD não seguem uma
pessoa. O lado que cada time ataca é inferido em cada tempo só pelo tracking daquele tempo: a casa
ataca para +x se a mediana do x do seu centróide fica à esquerda da mediana do visitante. Os
limites dos tempos vêm do helper oficial `games.half_video_blocks`; nada está amarrado a um frame.

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

Do bundle oficial o modelo usa só o ATD tático (`atd_meta` e `atd_raw`): posições de jogadores e
bola, o time de cada slot e o tamanho do campo (105,3 × 68,0 m em `games.py`). O vídeo, a homografia
e o broadcast não entram.

O Smart Tagging não entra no modelo. Ele só existe para o jogo da competição, então o peso dele não
pode ser calibrado nem validado nos jogos externos. Usamos o Smart Tagging uma vez como diagnóstico
(`scripts/check_smart_tagging.py`): para cada código, a fração de frames que a submissão marca como D
e a concordância entre o time previsto e o time da tag, só em agregados. Ele é automático, não é
gabarito, e serviu para comparar variantes já escolhidas pela validação, não para escolher.

Também medimos, sem rótulo, a estatística das lacunas de bola do alvo (taxa e duração por faixa de
velocidade dos jogadores, `scripts/measure_ball_gaps.py`) para degradar o treino com lacunas
parecidas. Não houve ganho (seção 3) e isso ficou desligado.

Os 5 minutos públicos só foram usados para checar que o pipeline está alinhado e bem-formado.
Nenhum script de calibração ou de comparação de variantes os lê.

## 3. Dados de treino e validação

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

A validação é leave-one-game-out (LOGO): treina em dois jogos e avalia no terceiro com a métrica
oficial, girando os três. Usamos dois protocolos, e os números de um não se comparam com os do outro:

- **grade**: o modelo é treinado uma vez por jogo de teste e a grade (duração mínima de H e A de 1 a
  6 s, de D de 2 a 8 s, peso 0,5 ou 1 na emissão) varia só a decodificação. O número é o máximo da
  grade, com uma semente de degradação, e por isso é otimista;
- **pareado**: ponto de decodificação fixo e 15 pares jogo × semente, com as duas variantes
  avaliadas nos mesmos pares. Dá a diferença entre variantes e em quantos pares uma vence a outra.

| variante | protocolo | spellF1 | macroF1 |
| --- | --- | --- | --- |
| C0, HMM categórico | grade | 0,495 | 0,748 |
| C1 original (33 features) | grade | 0,592 | 0,793 |
| C1 + lado de ataque | grade | 0,618 | 0,799 |
| **C1 + ataque + interpolação, 2/4/1 (submissão)** | pareado, média | **0,621** | **0,817** |

O que foi medido de forma pareada:

| mudança | ΔspellF1 | pares em que ganha | decisão |
| --- | --- | --- | --- |
| interpolação de 1 s sobre o C1 original | +0,019 | 14/15 | adotada |
| interpolação de 1 s sobre o C1 + ataque | +0,013 | 13/15 | adotada |
| lacunas medidas no alvo no lugar das uniformes | +0,003 | 9–11/15 | não adotada |
| melhor ponto da grade (1/4/1) contra 2/4/1 | +0,005 | 10/15 | não adotada (máximo de 30 pontos) |
| features em janelas de 1 + 2 + 5 s | −0,004 | 6/15 | não adotada |

O ruído entre jogos é de cerca de ±0,04 no spellF1. O ganho do C1 sobre o C0 (0,10) passa disso. Os
ganhos da interpolação e das features de ataque são consistentes nos pares, mas menores que esse
ruído; os pares repetem os mesmos três jogos com sementes diferentes, então não são independentes.
O LightGBM no lugar do scikit-learn deu 0,589 no C1 original, dentro do ruído, e ficamos com o
scikit-learn para ter uma dependência a menos.

### Dados públicos da DFL (Bundesliga 2022/23)

Testamos os 7 jogos públicos da DFL (Bassek, Rein, Weber & Memmert, 2025, *Scientific Data*,
[doi:10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y), CC-BY 4.0), baixados
por `scripts/fetch_dfl.py` do espelho da PySport no Hugging Face, com SHA-256 fixado (cerca de 2,6 GB).
Ao contrário da Metrica e da PFF, o rótulo da DFL não sai de eventos: o próprio tracking traz, por
frame, bola em jogo ou não e o time com a posse, que é a mesma construção do gabarito da competição.

Avaliação cruzada por fonte, sem o jogo de teste no treino, com a interpolação ligada (salvo onde indicado) e sem as
features de ataque:

| treino | teste na Metrica (spell / macro) | teste na DFL (spell / macro) |
| --- | --- | --- |
| só Metrica | 0,605 / 0,812 | 0,333 / 0,680 (sem interpolação) |
| Metrica + DFL | 0,605 / 0,817 | 0,473 / 0,871 |

Somar a DFL ao treino empata na Metrica e melhora muito na DFL. A submissão treina só na Metrica.
**[A DEFINIR pelo time: motivo de não usar a DFL. Ex.: a combinação DFL + features de ataque não foi
avaliada a tempo, e o download de 2,6 GB tornaria a reprodução mais pesada.]**

### Dados públicos da PFF (Copa de 2022)

Testamos também os 64 jogos públicos da PFF FC como fonte de treino. Os dados vêm dos arquivos
públicos da PFF FC [**A DEFINIR: link de onde se obtêm os zips**], e `scripts/extract_pff.py` os
converte. Os rótulos saem dos eventos pela mesma regra, com ajustes para o formato da PFF:

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
spellF1 entre 0,417 e 0,452 (`scripts/cross_eval_c0.py`). No C1 original, somar a PFF ao treino
baixou o spellF1 na Metrica de 0,592 para 0,563, embora o macroF1 tenha subido de 0,793 para 0,812.
A PFF tem menos posses por jogo que a Metrica (226 contra 347), o que sugere que as duas anotações
contam posse de jeito diferente.

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
mesma cópia. Degradar com lacunas no formato medido no alvo (seção 2) não mudou o resultado.

Depois da degradação, nos jogos externos e no alvo do mesmo jeito, a bola é interpolada em linha
reta nas lacunas de até 1 s, entre a última e a próxima posição vistas. As features de distância e
velocidade usam a bola interpolada; as de presença continuam olhando só a bola vista. Duas features
dizem ao classificador onde ele está: o tamanho da lacuna e o tempo desde a última bola vista.
Interpolar até 2 s foi um pouco pior que 1 s.

Nas lacunas mais longas, as distâncias à bola ficam vazias e o classificador usa o que sobra,
principalmente o movimento dos jogadores. Em bola morta os jogadores param, e isso continua visível
no tracking mesmo quando a bola não está.

Na velocidade dos jogadores, saltos acima de 12 m/s são tratados como troca de identidade do
tracklet e descartados. A velocidade de cada time é a média dos deslocamentos dos jogadores
presentes em duas amostras seguidas, e não o deslocamento do centróide, que salta quando um
tracklet aparece ou some.

No C0 também testamos aprender a parte de bola morta por EM no próprio jogo, sem rótulo. O EM
sempre convergiu para "bola ausente = D" e foi descartado.

## 6. Limitações

- Três jogos externos é pouco. O ruído do spellF1 é de cerca de ±0,04, maior que os ganhos das
  últimas mudanças (seção 3).
- Os rótulos da Metrica vêm de eventos, não do tracking corrigido que gera o gabarito oficial. As
  fronteiras de bola morta são aproximadas, e metade das posses rotuladas dura menos de 4 s. Num
  rótulo construído como o oficial (a DFL), o mesmo modelo treinado só na Metrica cai para cerca de
  0,33 de spellF1. Por isso não esperamos que o placar no alvo fique perto do 0,62 da validação.
- Nos jogos externos a bola quase só some em bola morta (91% a 99,8% dos frames D). A degradação
  apaga a bola também com o jogo vivo, mas no treino a ausência continua mais comum em D do que
  provavelmente é no alvo. O C1 vê a presença da bola como feature e pode ter aprendido parte desse
  artefato.
- No jogo da competição o modelo prevê 363 posses com 11,8 s de média e marca D em 32,4% dos
  frames avaliados.
- A Metrica também é outro provider. Ela é o alvo mais próximo que temos em formato, mas a DFL é a
  fonte cujo rótulo mais se parece com o oficial.
- O modelo é treinado de novo a cada execução. O CSV sai idêntico no Windows e no Linux x86_64,
  com uma ou várias threads, desde que o ambiente seja o do `uv.lock` com Python 3.11. Não testamos
  em ARM (Mac M).

**[A DEFINIR: um parágrafo sobre o C2 (FC-JEPA), resultado negativo, sem código no repositório.]**

## 7. Reprodução

```bash
git clone --recurse-submodules https://github.com/SALabUFMG/mtah27.git
cd mtah27
uv sync                                          # Python 3.11 e versões exatas do uv.lock
# bundle de dados em ./data ou MTAH_DATA_DIR (ver data/README.md)
uv run python scripts/fetch_metrica_sample.py
uv run python scripts/make_submission.py
```

Os hiperparâmetros estão em `configs/c1_fitted.yaml` e as chaves do C1 (interpolação, DFL, lacunas
medidas) em `configs/default.yaml`. O modelo é treinado nos jogos da Metrica dentro do
`make_submission`. Para refazer as comparações pareadas: `scripts/eval_c1_variants.py`; por fonte de
treino: `scripts/eval_c1_sources.py`. O C0 continua disponível com `model: c0` em
`configs/default.yaml`.

SHA-256 esperado do CSV (tag `fase1`):
`327c282a03fb566381bff3f14cb5d55939f71aa718ed921b42227a6f5199ced8`
