# Repo layout

This is a standard Python “src layout” project: importable code lives under `src/`, and `scripts/` contains small
wrappers so you can run things without installing.

## Top level

- `README.md`: quick start and pointers.
- `PIPELINE.md`: end-to-end runbook.
- `pyproject.toml`: package metadata + console entry points.
- `src/imc25/`: the actual library code.
- `scripts/`: wrappers around the library entry points.
- `notebooks/`: exploration, ablations, and comparison studies (see `notebooks/README.md`).
- `docker/`: Docker bits (COLMAP/Jupyter image).
- `LightGlue-main/`: vendored LightGlue (upstream matcher).

## Local-only folders (ignored by git)

- `data/`: competition data.
- `cache*/`: embeddings, clustering outputs, and other computed caches.
- `outputs*/`: reconstructions, checkpoints, demo HTML.
- `logs/`, `tmp/`: scratch space.
