# Repo layout

This repository follows a standard “src layout” Python project structure.

## Top-level

- `README.md`: quickstart, demos, and main commands.
- `pyproject.toml`: Python package metadata and optional extras.
- `src/imc25/`: library code (importable package).
- `scripts/`: thin wrappers for running common entrypoints without installation.
- `notebooks/`: exploratory and curated notebooks (see `notebooks/README.md`).
- `docker/`: optional Docker build assets (e.g. COLMAP/Jupyter image).

## Local artifacts (not committed)

These directories are expected to exist locally but are ignored by git:

- `data/`: competition datasets.
- `cache/`, `cache_retrieval/`: computed caches.
- `outputs/`, `outputs_*`: reconstruction outputs and trained checkpoints.

