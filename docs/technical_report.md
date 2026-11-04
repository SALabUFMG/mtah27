# SALab UFMG — More than a Hack 2027, Fase 1

> Documento técnico da submissão (poucas páginas). Estrutura pedida no SUBMISSION.md oficial.
> Exportar como `SALab_UFMG.pdf` para o formulário.

## 1. Abordagem

A submissão atual é o caminho C0, um HMM de três estados (H, A, D) decodificado por Viterbi com
duração mínima por estado. Cada tempo do jogo é decodificado separadamente, e os frames fora dos
dois tempos recebem `X`.

O modelo roda numa grade de 5 Hz: de cada 0,2 s entra um frame, e cada frame nativo recebe o
rótulo da amostra mais próxima dentro do mesmo tempo. A grade é a mesma para todos os providers
(25 fps no alvo e na Metrica, 29,97 na PFF), então a duração mínima e as transições valem o mesmo
em segundos nos três.

O modelo vê dois canais por amostra, tratados como independentes dado o estado:

1. A observação da bola: o time do jogador mais próximo, combinado com perto ou longe (raio de
   1 m), além de "bola fora do campo" e "bola ausente".
2. A velocidade mediana dos jogadores, suavizada em 1 s e dividida pela mediana do próprio tempo
   de jogo, em três faixas com cortes em 0,75 e 1,0. A escala absoluta muda com o provider; a
   relativa fica parecida (D perto de 0,7, jogo vivo perto de 1,2).

O segundo canal existe porque a bola some em 41% dos frames avaliados. Em bola morta os jogadores
param, e isso continua visível no tracking mesmo quando a bola não está.

A duração mínima é o que controla a fragmentação, que o spellF1 pune: uma posse quebrada em três
pedaços conta como erro. Com 4 s para H e A e 8 s para D, o modelo não troca de estado por causa
de um toque isolado.

Escolhemos começar por um modelo simples porque ele exercita o caminho inteiro (leitura do ATD,
validação externa, métrica oficial, submissão byte a byte) antes de qualquer investimento em
modelos treinados.

## 2. Dados utilizados

Do bundle oficial usamos só o ATD tático (`atd_meta` e `atd_raw`): posições de jogadores e bola,
o time de cada slot e o tamanho do campo (105,3 × 68,0 m em `games.py`). O vídeo, a homografia e o
broadcast não entram.

O Smart Tagging ficou fora de propósito. Ele só existe para o jogo da competição, então o peso dele
não pode ser calibrado nem validado nos jogos externos.

## 3. Dados de treino

Os parâmetros vêm dos três jogos públicos da Metrica Sports
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

A calibração é leave-one-game-out: ajusta em dois jogos e avalia no terceiro com a métrica
oficial, girando os três. As emissões e transições são contagens com suavização de Laplace. A grade
pequena (raio de 1 ou 2 m, duração mínima de 2 a 8 s) foi escolhida pelo spellF1 médio. Resultado
do ponto escolhido: spellF1 0,495 e macroF1 0,748, médios nos três jogos. É o máximo da grade, então
é um pouco otimista.

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
partir dos eventos e vazaria o rótulo. A PFF marca algumas posições como estimadas, e testamos
trocá-las por ausentes, já que o tracking do alvo não estima.

No CV de 8 folds dentro da PFF, trocar a bola estimada por ausente melhora o spellF1 de 0,450 para
0,469; trocar os jogadores estimados piora para 0,388. A pergunta que importa é se a PFF ajuda no
alvo, e para isso ajustamos na PFF e avaliamos nos três jogos da Metrica
(`scripts/cross_eval_c0.py`). Com os parâmetros que o CV da própria PFF escolhe, o spellF1 na
Metrica fica em 0,417 (base) e 0,452 (sem bola estimada), contra 0,495 da calibração na Metrica.
Mesmo a melhor célula da grade, escolhida olhando o teste, chega só a 0,472. Juntar PFF e Metrica
no ajuste dá o mesmo que a PFF sozinha, porque os 64 jogos dominam as contagens.

A PFF escolhe durações mínimas maiores (6 a 8 s) e tem menos posses por jogo que a Metrica (226
contra 347), o que sugere que as duas anotações contam posse de jeito diferente. A submissão segue
com a calibração da Metrica.

## 4. Regras de posse implementadas

Seguimos o CHALLENGE.md. A posse só muda quando o outro time ganha controle; bola disputada ou
solta fica com o último time. No HMM isso aparece de duas formas: a duração mínima impede trocas
mais curtas que 4 s, e o "perto" (jogador a menos de 1 m da bola) pesa muito mais que o "longe".

Bola morta é sempre D, sem dono, como pede o desafio.

## 5. Tratamento de tracking ausente/ruidoso

Bola ausente é uma observação neutra: tem a mesma probabilidade nos três estados. Nos jogos
externos a bola quase só some em bola morta (91% a 99,8% dos frames D), enquanto no jogo da
competição ela some também com o jogo vivo. Aprender esse padrão dos externos transferiria um
artefato do provider. Pelo mesmo motivo, a posição da bola não informa D: nos externos a pouca bola
visível em bola morta fica parada num ponto fixo. A emissão da bola para D é a média das de H e A,
e só o canal de movimento separa D de jogo vivo.

Para que a validação reflita a ausência do alvo, a bola dos jogos externos é apagada em lacunas
aleatórias até 41% dos frames avaliados, a taxa publicada no DATA.md. A taxa é medida e as lacunas
são sorteadas só dentro dos tempos de jogo, como no alvo. A duração média das lacunas (2 s) é um
chute.

Os 46 slots do ATD não seguem uma pessoa, então nada no modelo depende da identidade do slot, só
do time. Na velocidade, saltos acima de 12 m/s são tratados como troca de identidade do tracklet e
descartados, e a mediana sobre os jogadores amortece o resto.

Também testamos aprender a parte de bola morta por EM no próprio jogo, sem rótulo. Nos jogos
externos o EM sempre convergiu para "bola ausente = D", com P(ausente | vivo) = 0 contra 0,15 a
0,22 contados, e por isso foi descartado.

## 6. Limitações

- Três jogos externos é pouco. O ruído do spellF1 é de cerca de ±0,04, e várias combinações da
  grade ficam dentro dele.
- Os rótulos vêm de eventos, não do tracking corrigido que gera o gabarito oficial. As fronteiras
  de bola morta são aproximadas.
- No jogo da competição, o modelo marca D em 38,5% dos frames avaliados, dentro dos 35% a 40% de
  bola parada típicos de uma partida.
- O formato das lacunas de bola do alvo é desconhecido; só a taxa é conhecida.
- A Metrica também é outro provider. Ela é o alvo mais próximo que temos, mas a escolha entre
  Metrica e PFF supõe que o alvo se parece mais com ela.
- Próximo passo: testar o FC-JEPA como extrator de representação.

## 7. Reprodução

```bash
git clone --recurse-submodules <url>
uv sync
uv run python scripts/make_submission.py
```

Os parâmetros já estão em `configs/c0_fitted.yaml`. Para recalculá-los a partir dos dados públicos:

```bash
uv run python scripts/fetch_metrica_sample.py
uv run python scripts/calibrate_c0.py
```

SHA-256 esperado do CSV: `9d2097d713026f67dbdc4375154c6da5b6905c7dcc32b631730f4ff92cba5fbd`
