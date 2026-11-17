# Experiments

Every relevant run becomes a line in `log.csv`:

| column | meaning |
|---|---|
| `date` | YYYY-MM-DD |
| `author` | who ran it |
| `git_commit` | `git rev-parse --short HEAD` (clean commit!) |
| `config` | file in `configs/` |
| `description` | what changed with respect to the previous run |
| `n_possessions` | number of predicted possessions (printed by the validator) |
| `mean_poss_s` | mean duration of the predicted possessions |
| `val_spellF1` / `val_macroF1` | scores on the team's **own** validation (leave-one-game-out on external games) |
| `public_spellF1` / `public_macroF1` | scores on the 5 public minutes, **sanity check only, never used to choose a version** |
| `notes` | |

Descriptions and notes are in Portuguese. Nothing derived frame by frame from the data goes here.
