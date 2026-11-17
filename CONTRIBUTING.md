# How we work

## Non-negotiable rules (from the competition terms)

Full rules in [docs/official/RULES.md](docs/official/RULES.md). LLM agents (Claude Code, Codex, Copilot…) must follow [CLAUDE.md](CLAUDE.md).

1. **Data never goes into git.** No video, no tracking, no derived parquet, no submission CSV. `.gitignore` and pre-commit protect against it, but everyone is responsible.
2. **Data never goes into cloud-hosted LLMs** (ChatGPT, Claude, Gemini, Copilot Chat…), not even extracts or screenshots of tracking tables or plots. Asking for help with code is fine; paste error messages and scores, not data.
3. **No hand labels** on the competition game, not even to train or validate. Hand labels on **other** games are allowed (and must be documented).
4. **No parameter tied to specific frames** of the game.
5. **The 5 public minutes are not a validation set.** Sanity checks only.
6. **Notebooks without outputs** in git (`nbstripout` does it on commit).

## Git workflow

- `main` always produces a valid submission (`make submission && make validate`).
- Work on branches: `<initials>/<topic>`, e.g. `jl/deadball-features`.
- Pull request to `main`, with at least 1 review from another member.
- Before opening the PR: `make test` and `make lint`.
- Commit messages in the imperative, in Portuguese: "Adiciona detector de bola morta".

## Experiments

Every run someone would use to make a decision becomes a line in `experiments/log.csv`, run from a clean commit (no pending changes).

## Dependencies

`uv add <package>` (or `uv add --group dev <package>`) and commit `pyproject.toml` + `uv.lock` together. Never a loose `pip install`.

## Before making the repo public

- [ ] `git log --all --stat` shows nothing from `data/`, `.mp4`, `.parquet`, large `.csv`
- [ ] notebooks without outputs
- [ ] clean clone + `uv sync` + `make submission` reproduces the same SHA-256
