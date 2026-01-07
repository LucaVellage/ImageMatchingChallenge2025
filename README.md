# Robust Scene Clustering and 3D Reconstruction from Messy Image Collections

This started as a practical pipeline for the Kaggle Image Matching Challenge 2025, but it goes beyond the
competition baseline. In addition to producing a valid IMC-style `submission.csv`, the repo includes:

- advanced matcher fallbacks (DINOv2 and diffusion features),
- learned edge scoring for improved clustering,
- visualization tools (Jigsaw/Pose Explorer and HTML demo export),
- ablation and comparison notebooks aimed at analysis and paper-ready results.

Most artifacts are generated into `cache*/` and `outputs*/` and are ignored by git. The only tracked artifacts under
`outputs/` are small model files in `outputs/edge_models/`.

## Setup

- Install (recommended): `pip install -e .`
- Extra dependencies:
  - training: `pip install -e '.[train]'`
  - diffusion matching: `pip install -e '.[diffusion]'`
  - pointcloud viewing (Open3D): `pip install -e '.[pointcloud]'`
- No install: you can also run the wrappers in `scripts/` directly.

## Quick start

For a full end-to-end walkthrough, see `PIPELINE.md`.

End-to-end (train → cache → cluster → SfM → `submission.csv`):

- `HF_ENDPOINT=https://hf-mirror.com python scripts/run_pipeline.py --runner docker --overwrite`

A heavier run (adds matcher fallbacks + dense MVS + depth fallback + HTML demos):

- `HF_ENDPOINT=https://hf-mirror.com python scripts/run_pipeline.py --preset sota --runner docker --overwrite`

## Demos

### Pose Explorer

Interactive 3D viewer for a submission CSV (camera centers + optional pointcloud overlay).

- Run the local app:
  - `python scripts/pose_explorer.py --csv submission.csv --images-root outputs_retrieval_test/ETs_cluster_0001/images`
  - after install: `imc25-pose-explorer --csv submission.csv --images-root outputs_retrieval_test/ETs_cluster_0001/images`
  - optional (only if you ran dense): add `--pointcloud outputs_retrieval_test/ETs_cluster_0001/dense_points.ply`
- Export a standalone HTML:
  - `python scripts/pose_explorer.py --csv submission.csv --images-root outputs_retrieval_test/ETs_cluster_0001/images --export-html outputs/pose_explorer_demo.html --no-server`

### Jigsaw Explorer

A small dashboard to inspect the retrieval graph (neighbors, clusters, outliers).

- Start it: `python scripts/jigsaw_explorer.py` (or `imc25-jigsaw-explorer` after install)
- Open `http://127.0.0.1:8060`

## Notebooks

See `notebooks/README.md` for a suggested order and what each notebook covers.
