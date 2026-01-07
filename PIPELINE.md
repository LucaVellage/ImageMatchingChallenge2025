# IMC25 pipeline

This document is a straight, reproducible path from “raw IMC folders” to a `submission.csv`.
It’s written around the scripts in `scripts/` (thin wrappers around `src/imc25/`).
While the pipeline is competition-ready, the repo goes beyond a Kaggle baseline with learned edge
scoring, matcher fallbacks (DINO + diffusion), visualization dashboards, and analysis notebooks.

Local data and generated artifacts are intentionally not committed:

- `data/` holds the competition dataset.
- `cache*/` holds computed embeddings and cluster assignments.
- `outputs*/` holds reconstructions, checkpoints, and demo HTML.

## What you’re solving

Each IMC test dataset folder contains images from multiple scenes mixed together.
You need to:

1) group images into scene clusters (plus an `outliers` bucket), and
2) estimate a camera pose per image inside each cluster (when possible).

This repo does that with retrieval embeddings + clustering + COLMAP.

## Overview

```
data/train + data/train_labels.csv
        │
        ├─ (optional) train retrieval model        → outputs/retrieval_finetune/best.pt
        │
data/test
        │
        ├─ embed images                            → cache_retrieval/<dataset>/embeddings.npy
        ├─ cluster per dataset                     → cache/clusters_retrieval.csv
        └─ SfM (+ optional MVS) per cluster        → outputs_retrieval_test/<dataset>_<scene>/...

Then: export submission.csv from clusters + best COLMAP model per cluster
```

## Prereqs

- Python 3.10+
- Docker (recommended) if you want COLMAP to “just work” with GPU support
- Disk space: recon outputs add up quickly

If you’re using a Hugging Face mirror, set `HF_ENDPOINT` (examples below use `https://hf-mirror.com`).
If you don’t need a mirror, drop the `HF_ENDPOINT=...` prefix.

## Environment

Conda:

```bash
conda env create -f environment_no_builds.yml
conda activate imc25
```

Pip editable install:

```bash
python -m pip install -U pip
pip install -e .
```

Optional extras:

- training: `pip install -e '.[train]'`
- diffusion matching: `pip install -e '.[diffusion]'`
- pointcloud viewing (Open3D): `pip install -e '.[pointcloud]'`

## Data layout

Put the competition data under `data/`:

```
data/
  train/
    <datasetA>/
    <datasetB>/
  test/
    <datasetA>/
    <datasetB>/
  train_labels.csv
```

Quick check:

```bash
ls data/train_labels.csv
find data/test -maxdepth 2 -type f | head
```

## One-command run

This runs train → cache → cluster → COLMAP → `submission.csv`:

```bash
HF_ENDPOINT=https://hf-mirror.com \
python scripts/run_pipeline.py \
  --runner docker \
  --overwrite
```

If you want a “kitchen sink” run, use the built-in preset:

```bash
HF_ENDPOINT=https://hf-mirror.com \
python scripts/run_pipeline.py \
  --preset sota \
  --runner docker \
  --overwrite
```

Outputs:

- `cache/clusters_retrieval.csv`
- `outputs_retrieval_test/<dataset>_<scene>/...`
- `submission.csv`

## Manual run (same steps, more control)

### 1) Train a retrieval model (recommended)

```bash
HF_ENDPOINT=https://hf-mirror.com \
python scripts/train_retrieval.py \
  --model-id facebook/dinov2-small \
  --output-dir outputs/retrieval_finetune
```

### 2) Build an embeddings cache for test

```bash
HF_ENDPOINT=https://hf-mirror.com \
python scripts/build_retrieval_cache.py \
  --data-root data/test \
  --cache-root cache_retrieval \
  --checkpoint outputs/retrieval_finetune/best.pt \
  --topk 30 \
  --mutual
```

### 3) Cluster into scenes + outliers

```bash
python scripts/cluster_retrieval.py \
  --cache-root cache_retrieval \
  --out-csv cache/clusters_retrieval.csv
```

Useful knobs:

- `--min-sim`: raise it to split scenes more aggressively
- `--min-cluster-size`: small clusters become `outliers`
- `--method`: `louvain`, `greedy`, `components`

### 4) Reconstruct per cluster (COLMAP)

Recommended:

```bash
HF_ENDPOINT=https://hf-mirror.com \
python scripts/colmap_dense_clusters.py \
  --clusters-csv cache/clusters_retrieval.csv \
  --data-root data/test \
  --output-root outputs_retrieval_test \
  --runner docker \
  --matcher auto \
  --overwrite
```

Classic COLMAP only:

```bash
python scripts/colmap_dense_clusters.py \
  --clusters-csv cache/clusters_retrieval.csv \
  --data-root data/test \
  --output-root outputs_retrieval_test \
  --runner docker \
  --matcher colmap \
  --overwrite
```

If you’re using diffusion matching or DINO matching directly:

```bash
HF_ENDPOINT=https://hf-mirror.com \
python scripts/colmap_dense_clusters.py \
  --clusters-csv cache/clusters_retrieval.csv \
  --data-root data/test \
  --output-root outputs_retrieval_test \
  --runner docker \
  --matcher diffusion \
  --diffusion-hf-endpoint https://hf-mirror.com \
  --overwrite
```

```bash
HF_ENDPOINT=https://hf-mirror.com \
python scripts/colmap_dense_clusters.py \
  --clusters-csv cache/clusters_retrieval.csv \
  --data-root data/test \
  --output-root outputs_retrieval_test \
  --runner docker \
  --matcher dino \
  --dino-hf-endpoint https://hf-mirror.com \
  --overwrite
```

Depth fallback (when dense point clouds are empty/tiny):

```bash
HF_ENDPOINT=https://hf-mirror.com \
python scripts/colmap_dense_clusters.py \
  --clusters-csv cache/clusters_retrieval.csv \
  --data-root data/test \
  --output-root outputs_retrieval_test \
  --runner docker \
  --matcher auto \
  --dense-min-vertices 5000 \
  --depth-fallback \
  --depth-hf-endpoint https://hf-mirror.com \
  --overwrite
```

## Visuals / demos

Jigsaw Explorer (graph + clusters):

```bash
python scripts/jigsaw_explorer.py
```

Open `http://127.0.0.1:8060`.

Pose Explorer (cameras + optional pointcloud):

```bash
python scripts/pose_explorer.py \
  --csv submission.csv \
  --images-root outputs_retrieval_test/ETs_cluster_0001/images
```

If you ran dense MVS and have a point cloud, add:

```bash
--pointcloud outputs_retrieval_test/ETs_cluster_0001/dense_points.ply
```

Export a set of HTML pages (one per cluster + an index):

```bash
python scripts/export_demo.py \
  --submission-csv submission.csv \
  --recon-root outputs_retrieval_test \
  --out-dir outputs/demo
```

## Troubleshooting

### `argument --scene: expected one argument`

This is usually a line-break / quoting issue. Keep the value on the same line:

```bash
--scene cluster_0006
```

### Permission errors when deleting recon outputs

If you run COLMAP through Docker, some files may be created as `root`. Two options:

1) Prefer `--overwrite` and let the scripts manage clean rebuilds.
2) If you need to delete a folder by hand, you may need elevated permissions:
   `sudo rm -rf outputs_retrieval_test/<dataset>_<scene>`

### `unrecognised option --Mapper.local_ba_min_tri_angle`

Some COLMAP builds don’t support every flag. The script tries to detect this and disable unsupported flags, but if
you’re pinning an older image, update it or switch to a newer COLMAP image.

### “No good initial image pair found” / “Discarding reconstruction due to bad initial pair”

SfM couldn’t bootstrap (no overlap, or too few good matches). Try:

- `--matcher auto` (adds fallbacks)
- relaxing mapper params:
  - `--mapper-min-model-size 2`
  - `--mapper-init-min-num-inliers 8`
  - `--mapper-init-min-tri-angle 0.5`

### Dense output has 0 points

If `dense_points.ply` says `element vertex 0`, dense stereo/fusion didn’t converge. On some clusters that’s expected.
If you want a best-effort result, try `--depth-fallback`.

## Short checklist

```bash
conda env create -f environment_no_builds.yml
conda activate imc25
pip install -e '.[train]'

HF_ENDPOINT=https://hf-mirror.com python scripts/run_pipeline.py --runner docker --overwrite
```
