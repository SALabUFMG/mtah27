# SALab UFMG: More than a Hack 2027, Phase 1

> Technical document for the submission, following the structure asked for in the official
> SUBMISSION.md.

## 1. Approach

The model is a three-state HMM (H, A, D) decoded with Viterbi under a minimum duration per state,
whose emission comes from a gradient boosting classifier. Each half is decoded separately, and the
frames outside the two halves get `X`.

The model runs on a 5 Hz grid: one frame every 0.2 s goes in, and each native frame gets the label
of the nearest sample within the same half. The grid is the same for every provider (25 fps in the
target game and in Metrica, 29.97 in PFF), so minimum durations and transitions mean the same
number of seconds everywhere.

### How the model got here

We built the model in four steps. Each step answers a weakness measured on the one before, always
on external games. The spellF1 figures below are leave-one-game-out on the three Metrica games;
section 3 explains the two protocols and why the last figure is not directly comparable with the
first three.

We started from the metric. spellF1 counts possessions, so a possession split into three pieces is
an error, and a classifier that decides frame by frame flickers. The first version was therefore a
state model: the same three-state HMM and Viterbi decoding, with an observation designed by hand.
Each sample was reduced to a category built from the team of the player nearest to the ball and a
band of player speed, and the minimum duration was 4 s for H and A and 8 s for D. It scored 0.495.

An error analysis of that version (`scripts/error_analysis_c0.py`) showed where it lost. Almost
half of the true possessions in validation last less than 4 s, and the model matched 11% of them,
because it could not produce a possession shorter than its own minimum duration. Per-frame accuracy
fell from 0.81 with the ball visible to 0.66 without it. A more informative emission lets the model
use shorter minimum durations without fragmenting.

The second step replaced the hand-built category with a learnt emission. A gradient boosting
classifier reads 33 features of a 2 s window and its output becomes the emission of the same HMM,
now with minimum durations of 2 s and 4 s. spellF1 rose to 0.592, a step of 0.10 that is larger
than the noise between games (about ±0.04).

The third step added the direction of play. The 33 features described distances and speeds, with
nothing about which way each team was playing. Five more features give the position and velocity
of each team's centroid and the position of the ball in the frame of reference of the goal that
team attacks. spellF1 went to 0.618.

The fourth step dealt with the ball, which is missing in 41% of the evaluated frames of the target
game. When the ball disappears for up to 1 s and comes back, we interpolate its position, and two
features tell the classifier how long the gap is and how long ago the ball was last seen
(section 5). In paired comparisons this won 13 of 15 game × seed pairs. The submitted model scores
0.621 spellF1 and 0.817 macroF1 at the fixed decoding point.

Two other routes were tried and left out. One was a self-supervised encoder: a transformer
pre-trained from scratch by JEPA-style masked prediction on windows of tracking (the Metrica and
PFF games and, without labels, the target game), with a classification head trained on the external
labels. Its best spellF1 in the same validation was 0.506, below the boosting emission, so we
dropped it. That code is in the branch `jl/c2-fcjepa` and is not part of the submission. The other
route was more training games from other providers, DFL and PFF, covered in section 3.

### The submitted model in detail

For each sample we compute 40 tracking features over a centred 2 s window
(`src/mtah27/features.py`):

- ball: presence, speed, out of the pitch, distance to the touchline and to the goal line;
- contest: distance of the two players nearest to the ball in each team, the difference between
  the nearest of each team, how many players of each team are within 3 m and 8 m, and the fraction
  of the window in which the nearest player is from Barcelona;
- teams: for each team, centroid position, spread, mean player speed, number of visible players and
  centroid distance to the ball;
- movement: median player speed, the same divided by the median over the half, and the fraction of
  nearly stationary players;
- attacking direction (5): x position of each team's centroid, x velocity of each team and x
  position of the ball, all in the frame of reference of the goal that team attacks;
- ball gap (2): length of the ball gap the sample is in, and time since the ball was last seen
  (section 5).

No feature depends on the slot index, only on the team, because the 46 ATD slots do not follow a
person. The direction each team attacks is inferred per half from that half's tracking only: the
home team attacks towards +x if the median x of its centroid lies to the left of the away team's.
The half boundaries come from the official helper `games.half_video_blocks`; nothing is tied to a
specific frame.

A scikit-learn `HistGradientBoostingClassifier` (300 iterations, learning rate 0.05, 31 leaves,
seed 0) estimates P(state | features). The HMM needs P(features | state), which by Bayes' rule is
proportional to P(state | features) / P(state). That ratio is the Viterbi emission. Transitions are
counted on the training labels, and the minimum duration is 2 s for H and A and 4 s for D. The
minimum duration holds back the fragmentation that spellF1 penalises: a possession broken into
three pieces counts as an error.

## 2. Assets used

From the official bundle the model uses only the tactical ATD (`atd_meta` and `atd_raw`): player
and ball positions, the team of each slot and the pitch size (105.3 × 68.0 m, in `games.py`). The
video, the homography and the broadcast are not used.

Smart Tagging is not part of the model. It exists only for the competition game, so its weight
could not be calibrated or validated on external games. We used it once as a diagnostic
(`scripts/check_smart_tagging.py`): per code, the fraction of frames the submission marks as D and
the agreement between the predicted team and the tag's team, as aggregates only. It is automatic,
not ground truth, and it was used to look at variants already chosen by validation, not to choose
between them.

We also measured, without labels, the ball-gap statistics of the target game (rate and length per
player-speed bin, `scripts/measure_ball_gaps.py`) to degrade the training games with similar gaps.
It brought no gain (section 3) and is switched off.

The public 5 minutes were used only to check that the pipeline is aligned and well formed. No
calibration or variant-comparison script reads them.

## 3. Training data and validation

The model is trained on the three public Metrica Sports games
([metrica-sports/sample-data](https://github.com/metrica-sports/sample-data), commit `e706dd5`).
They are other matches, downloaded by `scripts/fetch_metrica_sample.py` with the SHA-256 of every
file checked. The repository declares no licence; its README asks for responsible use and
attribution to Metrica Sports. No data from the competition game is used for training or
validation.

The per-frame H/A/D labels come from these games' events, annotated by people at Metrica, through a
fixed rule (`labels_from_events` in `src/mtah27/external.py`):

- possession belongs to the team of the last event with ball control (pass, set piece, recovery,
  shot, carry);
- ball losses and challenges do not change possession;
- D runs from a ball out, foul or goal until the next control event, which is the restart. Goal
  kicks appear as passes, not as set pieces.

Each game enters training twice, with the ball removed in gaps drawn with different seeds
(section 5). The model is not stored in the repository: `make_submission` retrains it on the three
games, which takes about two minutes.

Validation is leave-one-game-out (LOGO): train on two games and score the third with the official
metric, rotating over the three. We used two protocols, and numbers from one are not comparable with
numbers from the other:

- **grid**: the model is trained once per test game and the grid (minimum duration of H and A from
  1 to 6 s, of D from 2 to 8 s, emission weight 0.5 or 1) changes only the decoding. The number is
  the maximum over the grid, with one degradation seed, so it is optimistic;
- **paired**: fixed decoding point and 15 game × seed pairs, with both variants scored on the same
  pairs. It gives the difference between variants and in how many pairs one beats the other.

| step | protocol | spellF1 | macroF1 |
| --- | --- | --- | --- |
| state model with hand-built observation | grid | 0.495 | 0.748 |
| learnt emission, 33 features | grid | 0.592 | 0.793 |
| + attacking direction | grid | 0.618 | 0.799 |
| **+ short-gap interpolation, 2/4/1 (submission)** | paired, mean | **0.621** | **0.817** |

Paired measurements:

| change | ΔspellF1 | pairs won | decision |
| --- | --- | --- | --- |
| 1 s interpolation on the 33-feature model | +0.019 | 14/15 | adopted |
| 1 s interpolation with attacking direction | +0.013 | 13/15 | adopted |
| gaps measured on the target instead of uniform gaps | +0.003 | 9 to 11 of 15 | not adopted |
| best grid point (1/4/1) vs 2/4/1 | +0.005 | 10/15 | not adopted (maximum of 30 points) |
| features over 1 + 2 + 5 s windows | −0.004 | 6/15 | not adopted |

The between-game noise of spellF1 is about ±0.04. The step from the hand-built observation to the
learnt emission (0.10) is larger than that. The gains from interpolation and from the
attacking-direction features are consistent across pairs but smaller than that noise, and the
pairs repeat the same three games with different seeds, so they are not independent. LightGBM in
place of scikit-learn scored 0.589 on the 33-feature model, within noise, and we kept scikit-learn
to have one dependency less.

### Public DFL data (Bundesliga 2022/23)

We tested the 7 public DFL games (Bassek, Rein, Weber & Memmert, 2025, *Scientific Data*,
[doi:10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y), CC-BY 4.0),
downloaded by `scripts/fetch_dfl.py` from the PySport mirror on Hugging Face with pinned SHA-256
(about 2.6 GB). Unlike Metrica and PFF, the DFL labels do not come from events: the tracking itself
carries, per frame, whether the ball is in play and which team has possession, annotated by the
DFL. That is the same construction as the competition's ground truth.

Cross-evaluation by source, with the test game never in training, interpolation on and without the
attacking-direction features:

| training | test on Metrica (spell / macro) | test on DFL (spell / macro) |
| --- | --- | --- |
| Metrica only | 0.605 / 0.812 | 0.330 / 0.648 |
| Metrica + DFL | 0.605 / 0.817 | 0.473 / 0.871 |

Adding DFL to training ties on Metrica and improves a lot on DFL. Trained on Metrica only and
without interpolation, the model scores 0.333 / 0.680 on DFL, so interpolation costs macroF1 there
unless DFL is in training.

The submission still trains on Metrica only. We measured DFL on the model without the
attacking-direction features and did not evaluate the combination before the deadline, so adding
it would have changed the training set of the submitted model on an untested configuration. The
2.6 GB download would also make reproduction heavier. It is the first thing we would add
(section 6).

### Public PFF data (2022 World Cup)

We also tested the 64 public PFF FC games as a training source. PFF distributes the dataset for
free through a form at
[blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset);
the page states no licence. The data is not in the repository and is not needed to reproduce the
submission; `scripts/extract_pff.py` converts it. Labels come from PFF's own event annotation
through the same rule, adjusted to the PFF format:

- a ball event changes possession only if it shows control (reception, carry, cross, shot, or a
  pass after another action). Duels, clearances, rebounds or isolated touches do not change it, and
  a first-time pass changes it only if completed;
- the JSON repeats the event once per possession action; we merge those rows before applying the
  rule;
- the penalty shoot-out, after the end of extra time, is left out. Penalties during play are kept.

We use the raw PFF ball (`balls`). The smoothed version (`ballsSmoothed`) is repositioned from the
events and would leak the label. PFF marks some positions as estimated; for the ball we treat them
as missing, since the target tracking does not estimate.

PFF did not help. On the 33-feature model, adding PFF to training lowered spellF1 on Metrica from 0.592
to 0.563, although macroF1 rose from 0.793 to 0.812. PFF has fewer possessions per game than
Metrica (226 vs 347), which suggests the two annotations count possession differently.

## 4. Possession rules implemented

We follow CHALLENGE.md. Possession changes only when the other team gains control; a contested or
loose ball stays with the last team. The classifier learns this from labels in which ball losses
and challenges do not change possession, and the minimum duration prevents changes shorter than
2 s.

A dead ball is always D, with no owner, as the challenge asks. In the training labels it starts at
a ball out, a foul or a goal and ends at the restart, and the team that takes the restart has
possession from that event on. That is the boundary the classifier learns.

## 5. Handling missing and noisy tracking

The ball is missing in 41% of the evaluated frames of the target game, the rate published in
DATA.md. So that training and validation see the same absence, the ball in the external games is
removed in random gaps up to that rate, counted only within the halves. The mean gap length (2 s)
is a guess. In training each game enters as two degraded copies with different seeds; evaluation
always uses the same copy. Degrading with gaps shaped like those measured on the target (section 2)
did not change the result.

After degradation, in the external games and in the target alike, the ball is linearly
interpolated across gaps of up to 1 s, between the last and the next seen positions. The distance
and speed features use the interpolated ball; the presence features still look only at the seen
ball. Two features tell the classifier where it is: the gap length and the time since the ball was
last seen. Interpolating up to 2 s was slightly worse than 1 s.

In longer gaps the ball distances are empty and the classifier uses what is left, mainly player
movement. When the ball is dead the players stop, and that stays visible in the tracking even when
the ball is not.

For player speed, jumps above 12 m/s are treated as tracklet identity switches and dropped. Each
team's speed is the mean displacement of the players present in two consecutive samples, not the
centroid displacement, which jumps when a tracklet appears or disappears.

## 6. Limitations

- Three external games is little. The spellF1 noise is about ±0.04, larger than the gains of the
  latest changes (section 3).
- Metrica labels come from events, not from the corrected tracking that produces the official
  ground truth. Dead-ball boundaries are approximate, and half of the labelled possessions last
  less than 4 s. On labels built like the official ones (DFL), the same model trained on Metrica
  only drops to about 0.33 spellF1. We therefore do not expect the target score to be close to the
  0.62 of validation.
- In the external games the ball is missing almost only when it is dead (91% to 99.8% of D frames).
  Degradation also removes the ball during live play, but in training absence is still more common
  in D than it probably is in the target. The model sees ball presence as a feature and may have
  learnt part of that artefact.
- On the competition game the model predicts 363 possessions with a mean of 11.8 s and marks D in
  32.4% of the evaluated frames.
- Metrica is also a different provider. It is the closest we have to the target in format, but DFL
  is the source whose labels most resemble the official ones.
- The model is retrained on every run. The CSV comes out byte-identical on Windows and Linux x86_64,
  with one or several threads, and on macOS with Apple Silicon (arm64), as long as the environment
  is the one in `uv.lock` with Python 3.11.

With more time we would first retrain with the DFL games together with the attacking-direction
features, since DFL is the only source whose labels are built like the official ones. We would also
try the ball from the broadcast tracking to fill the long gaps that interpolation leaves empty,
which needs a time alignment between the two videos that the bundle does not provide.

## 7. Reproduction

```bash
git clone --recurse-submodules https://github.com/SALabUFMG/mtah27.git
cd mtah27
uv sync                                          # Python 3.11 and the exact versions in uv.lock
# data bundle in ./data or MTAH_DATA_DIR (see data/README.md)
uv run python scripts/fetch_metrica_sample.py
uv run python scripts/make_submission.py
```

Hyperparameters are in `configs/c1_fitted.yaml` and the model switches (interpolation, DFL,
measured gaps) in `configs/default.yaml`. The model is trained on the Metrica games inside
`make_submission`. To redo the paired comparisons: `scripts/eval_c1_variants.py`; by training
source: `scripts/eval_c1_sources.py`.

Expected SHA-256 of the CSV (tag `fase1`):
`327c282a03fb566381bff3f14cb5d55939f71aa718ed921b42227a6f5199ced8`
