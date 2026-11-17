# mtah27 — More than a Hack 2027

Team SALab UFMG's pipeline for **Phase 1** of [More than a Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027)
(Metrica Sports · FC Barcelona · Mobile World Capital Barcelona): for every frame of the
Barcelona × Athletic match, say who has possession (`H` / `A`) or whether the ball is dead (`D`).

Rules, data and metric: the official repository, included as a submodule in
[`external/mtah-official`](https://github.com/metrica-sports/more-than-a-hack-2027).
A copy of the official documentation for quick reference is in **[docs/official/](docs/official/README.md)**.
The technical document of the submission is **[docs/technical_report.md](docs/technical_report.md)**.

---

## Reproducing the submission

Requirements: [uv](https://docs.astral.sh/uv/) and git. uv installs **Python 3.11** (pinned in
`.python-version`) and the exact versions in `uv.lock`. Run everything through `uv run`: with another
Python version (for example 3.12) the dependencies resolve to other versions and the CSV may differ.

```bash
git clone --recurse-submodules https://github.com/SALabUFMG/mtah27.git
cd mtah27
uv sync                                          # environment identical to uv.lock
# put the data bundle in ./data (or set MTAH_DATA_DIR); see data/README.md
uv run python scripts/fetch_metrica_sample.py    # 3 public Metrica games (model training data)
uv run python scripts/make_submission.py         # trains the model (~2 min) and writes outputs/submissions/SALab_UFMG.csv
uv run python scripts/validate.py outputs/submissions/SALab_UFMG.csv
```

`make_submission` prints the SHA-256 of the CSV. For the Phase 1 submission (tag `fase1`) it is

```
327c282a03fb566381bff3f14cb5d55939f71aa718ed921b42227a6f5199ced8
```

checked on Windows, Linux x86_64 and macOS on Apple Silicon (arm64). The seed is fixed and the
submission path uses no cache and no non-deterministic parallelism. The DFL and PFF data only appear in
the experiments (`scripts/fetch_dfl.py`, `scripts/extract_pff.py`) and are not needed to reproduce.

---

## Local setup

Step by step to get the project running on your machine. It takes about 5 minutes.

### 1. Prerequisites

- **Git** — https://git-scm.com/downloads
- **uv** (manages Python and dependencies; it installs Python 3.11 by itself, no Python needed beforehand):

  ```powershell
  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  Close and reopen the terminal afterwards and check with `uv --version`.

### 2. Clone (with the official submodule)

```bash
git clone --recurse-submodules https://github.com/SALabUFMG/mtah27.git
cd mtah27
```

Already cloned without `--recurse-submodules` and `external/mtah-official` is empty? Run:

```bash
git submodule update --init --recursive
```

### 3. Create the environment

```bash
uv sync                     # creates .venv with the exact versions in uv.lock
uv run pre-commit install   # commit hooks: strip notebook outputs, block large files
```

For contributors, `pre-commit install` is **mandatory**: it is what keeps notebook outputs (which
contain data) out of git.

### 4. Add the data

Unpack the competition bundle into `data/` (layout in [data/README.md](data/README.md)), or keep it
elsewhere and point the `MTAH_DATA_DIR` variable at it. Then download the 3 public Metrica games used
for training (the download checks the SHA-256 of each file):

```bash
uv run python scripts/fetch_metrica_sample.py
```

```powershell
$env:MTAH_DATA_DIR = "D:\mtah-data"          # Windows (current session only)
setx MTAH_DATA_DIR "D:\mtah-data"            # Windows (permanent; reopen the terminal)
```
```bash
export MTAH_DATA_DIR=/path/to/mtah-data      # macOS / Linux
```

**Never** commit the data or paste extracts of it into cloud-hosted LLMs — see [CONTRIBUTING.md](CONTRIBUTING.md).

### 5. Check that everything works

```bash
uv run pytest -q                                                   # tests pass
uv run python scripts/make_submission.py                           # writes outputs/submissions/SALab_UFMG.csv
uv run python scripts/validate.py outputs/submissions/SALab_UFMG.csv   # "Valid."
uv run python scripts/score.py --no-detail                         # score on the 5 public minutes
```

If all four run without errors, you are ready. With the data in place, also run
`uv run python scripts/run_baseline.py`: the public scores should match the published ones (spellF1
0.3256 for `baseline_smooth50`). To open notebooks: `uv run jupyter lab`.

### Tips and common problems

- **VS Code / PyCharm:** select the interpreter in `.venv` (Windows: `.venv\Scripts\python.exe`;
  macOS/Linux: `.venv/bin/python`).
- **Repo inside OneDrive/Dropbox:** `.venv` has thousands of files and will sync.
  To create it elsewhere, set `UV_PROJECT_ENVIRONMENT` before `uv sync`
  (e.g. `setx UV_PROJECT_ENVIRONMENT "%USERPROFILE%\venvs\mtah27"`).
- **`ModuleNotFoundError: games` or "Submódulo oficial ausente" (official submodule missing):** the
  submodule step (item 2) is missing.
- **Adding a dependency:** `uv add <package>` and commit `pyproject.toml` + `uv.lock` together.
  After a `git pull` that changes `uv.lock`, run `uv sync` again.
- **`make`** is optional; on Windows use the `uv run ...` commands directly.

## Everyday commands

| task | command | `make` |
|---|---|---|
| set up the environment | `git submodule update --init --recursive && uv sync && uv run pre-commit install` | `make setup` |
| build the submission | `uv run python scripts/make_submission.py` | `make submission` |
| validate (official checks) | `uv run python scripts/validate.py outputs/submissions/SALab_UFMG.csv` | `make validate` |
| official baseline (first contact with the data) | `uv run python scripts/run_baseline.py` | |
| score on the 5 public minutes | `uv run python scripts/score.py` | `make score` |
| tests | `uv run pytest -q` | `make test` |
| notebooks | `uv run jupyter lab` | |

> The 5 public minutes are **only** for checking that the pipeline is correct, never for choosing
> between versions or tuning (competition rule).

## Layout

```
mtah27/
├─ CLAUDE.md             instructions for LLM agents (team-internal, in Portuguese)
├─ RULES.md              team summary of the competition rules (in Portuguese)
├─ configs/               parameters (YAML), nothing tied to specific frames
├─ data/                  competition data (gitignored)
├─ docs/                  technical document of the submission
│  └─ official/           copy of the official repo's .md files (reference, do not edit)
├─ experiments/log.csv    log of runs
├─ external/mtah-official official repo (submodule): scorer, validator, loaders
├─ notebooks/             exploration (no outputs in git)
├─ outputs/               submissions, reports, figures (gitignored)
├─ scripts/               entry points (make_submission, validate, score)
├─ src/mtah27/            the package: pipeline.predict() is the model's entry point
└─ tests/
```

Code comments, docstrings and commit messages are in Portuguese; identifiers are in English.

## Updating the official repo

```bash
git submodule update --remote external/mtah-official
uv run python scripts/sync_official_docs.py          # refreshes the copy in docs/official/
git add external/mtah-official docs/official && git commit -m "Atualiza repo oficial"
```

## Team

SALab UFMG: Bruno Martins de Sá Freire, Gabriel Valadão Meira, João Lucas Lage Gonçalves, Ricardo Furbino Marques do Nascimento.

Documentation:

- [docs/technical_report.md](docs/technical_report.md): technical document of the submission.
- [docs/official/](docs/official/README.md): full official documentation (DATA, EVALUATION, RULES, FAQ…).
- [CONTRIBUTING.md](CONTRIBUTING.md): workflow and data rules.
