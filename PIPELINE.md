# IMC25 Pipeline (End-to-end, reproducible)

This repo contains an end-to-end pipeline for the Image Matching Challenge 2025 (IMC25):

1) **Learn** an image embedding model from the IMC25 training labels (optional but recommended).
2) **Cluster** the test images into scenes (and mark outliers).
3) **Reconstruct** each scene with COLMAP (SfM) and optionally compute **dense** point clouds.
4) **Visualize** clusters and poses for presentations and debugging.

The goal of this document is that someone with *no prior computer vision background* can reproduce the results.

---

## 0) What is IMC25 asking you to do?

IMC25 mixes images from multiple different scenes in the same dataset folder (like mixing multiple jigsaw puzzles).

For each dataset you must:

- **Partition images** into clusters (each cluster = one scene) + an **outliers** bucket.
- For each cluster, **estimate the camera pose** (rotation + translation) for each image if possible.

This repo focuses on a practical approach:

- Use a **retrieval model** (an image embedder) to group similar images together → clusters.
- Use **Structure-from-Motion (SfM)** (COLMAP) to estimate camera poses inside each cluster.
- Use **Multi-view Stereo (MVS)** (COLMAP dense) to produce a **dense point cloud** (`dense_points.ply`) per cluster.

---

## 1) High-level pipeline diagram

```
data/train + data/train_labels.csv
        │
        ├─ (optional) train retrieval embedder  → outputs/retrieval_finetune/best.pt
        │
data/test (images only)
        │
        ├─ embed all test images               → cache_retrieval/<dataset>/embeddings.npy
        │
        ├─ cluster embeddings per dataset      → cache/clusters_retrieval.csv
        │
        └─ SfM+MVS per cluster (COLMAP)        → outputs_*/<dataset>_<cluster>/dense_points.ply
```

---

## 2) Prerequisites (what you need installed)

### Hardware (recommended)
- **NVIDIA GPU** with CUDA for:
  - fast training (retrieval fine-tuning)
  - dense COLMAP (PatchMatch stereo)
  - diffusion matching (optional, very heavy)
- Enough disk space:
  - COLMAP outputs can be large.
  - Diffusion model download can be multiple GB.

### Software
- **Python 3.10+**
- (Recommended) **conda** or **mamba**
- **Docker** (recommended for COLMAP GPU)
  - Linux: Docker + NVIDIA Container Toolkit
  - Windows: WSL2 + Docker Desktop with GPU enabled

Quick Docker GPU check (should print your GPU):
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

---

## 3) Get the code

```bash
git clone https://github.com/LucaVellage/ImageMatchingChallenge2025
cd ImageMatchingChallenge2025
```

---

## 4) Python environment setup

### Option A (recommended): conda env file

This repo includes `environment_no_builds.yml` which is known to work in this project’s environment:

```bash
conda env create -f environment_no_builds.yml
conda activate imc25
```

### Option B: pip editable install

```bash
python -m pip install -U pip
pip install -e .
```

Enable optional capabilities:

- Training: `pip install -e .[train]`
- Diffusion matching: `pip install -e .[diffusion]`
- Pointcloud loading (Open3D): `pip install -e .[pointcloud]`

---

## 5) Download / place the data

IMC25 data is not committed to git (see `.gitignore`). You should place it under `data/` like:

```
data/
  train/
    <datasetA>/
      <images...>
    <datasetB>/
      <images...>
  test/
    <datasetA>/
      <images...>
    <datasetB>/
      <images...>
  train_labels.csv
```

Notes:
- `train_labels.csv` contains the training scene labels (and training poses).
- Test has **no labels**; we predict them.

Sanity checks:
```bash
ls data/train_labels.csv
find data/test -maxdepth 2 -type f | head
```

---

## 6) (Recommended) Train a retrieval model using training labels

Why?
- IMC25 gives you training labels; training a retrieval embedder helps clustering a lot.

This script fine-tunes a backbone (default: DINOv2) using **Supervised Contrastive learning** on `(dataset, scene)` labels:

```bash
HF_ENDPOINT=https://hf-mirror.com \
python scripts/train_retrieval.py \
  --model-id facebook/dinov2-small \
  --output-dir outputs/retrieval_finetune
```

Outputs:
- `outputs/retrieval_finetune/best.pt` (checkpoint)
- `outputs/retrieval_finetune/train_config.json` (training metadata)

GPU note:
- The trainer automatically uses CUDA if available.
- You can force device: `--device cuda` or `--device cpu`.

---

## 7) Build embeddings cache for the test split

This runs the trained checkpoint over every image in `data/test` and saves embeddings + an approximate retrieval graph:

```bash
HF_ENDPOINT=https://hf-mirror.com \
python scripts/build_retrieval_cache.py \
  --data-root data/test \
  --cache-root cache_retrieval \
  --checkpoint outputs/retrieval_finetune/best.pt \
  --topk 30 \
  --mutual
```

Outputs (per dataset):
- `cache_retrieval/<dataset>/embeddings.npy`
- `cache_retrieval/<dataset>/meta.json` (image_id list + paths)

---

## 8) Cluster test images into scenes + outliers

We cluster per dataset using a kNN graph built from embeddings.

```bash
python scripts/cluster_retrieval.py \
  --cache-root cache_retrieval \
  --out-csv cache/clusters_retrieval.csv
```

Output:
- `cache/clusters_retrieval.csv` with columns:
  - `dataset`: dataset name
  - `image_id`: image identifier (filename stem in practice)
  - `scene`: `cluster_0001`, `cluster_0002`, … or `outliers`

Useful knobs:
- `--min-sim`: similarity threshold (higher = fewer edges = more/smaller clusters)
- `--min-cluster-size`: smaller clusters become `outliers`
- `--method louvain|greedy|components`

---

## 9) Reconstruct each cluster and produce dense point clouds (COLMAP)

This step runs SfM (camera poses) + dense stereo (point cloud) per cluster.

### 9.1 Recommended command (robust)

`--matcher auto` tries:
1) COLMAP SIFT features/matching
2) if that fails, DINOv2 patch-descriptor matching
3) if that fails, diffusion-feature matching (optional and heavy)

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

Outputs (per cluster directory `outputs_retrieval_test/<dataset>_<scene>/`):
- `dense_points.ply` (dense point cloud)
- `sparse/<model_id>/` (SfM model)
- `dense/` (MVS workspace)

If you see tiny / empty pointclouds (e.g. a ~200-byte PLY or only a few points), use the depth fallback:

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

### 9.2 If you only want classic COLMAP (fastest)

```bash
python scripts/colmap_dense_clusters.py \
  --clusters-csv cache/clusters_retrieval.csv \
  --data-root data/test \
  --output-root outputs_retrieval_test \
  --runner docker \
  --matcher colmap \
  --overwrite
```

### 9.3 If you want “wow factor” diffusion matching

This uses Stable Diffusion U-Net features as dense descriptors to propose correspondences and writes them into a COLMAP DB.

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

Important:
- First run will download GBs of model weights.
- Diffusion is compute-heavy. Use it selectively for “hard” clusters or demos.

### 9.4 If you want “wow factor” DINOv2 matching

This uses a foundation vision model (DINOv2) as a dense descriptor grid to propose correspondences, then verifies them with geometry and writes them into a COLMAP DB.

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

---

## 10) Visualize (for debugging and presentations)

### 10.1 Jigsaw Explorer (cluster graph browser)

```bash
python scripts/jigsaw_explorer.py
```
Open: `http://127.0.0.1:8060`

### 10.2 Pose Explorer (3D camera visualization)

Pose Explorer visualizes an IMC-style submission CSV and optionally overlays a point cloud.

```bash
python scripts/pose_explorer.py \
  --csv submission.csv \
  --images-root outputs_retrieval_test/ETs_cluster_0001/images \
  --pointcloud outputs_retrieval_test/ETs_cluster_0001/dense_points.ply
```

### 10.3 Export HTML demos (presentation-ready)

This exports one interactive Plotly HTML per cluster + an `index.html` linking them:

```bash
python scripts/export_demo.py \
  --submission-csv submission.csv \
  --recon-root outputs_retrieval_test \
  --out-dir outputs/demo
```

---

## 11) Troubleshooting (common issues)

### “`argument --scene: expected one argument`”
That’s a shell line-break issue. Put the value on the same line:
```bash
--scene cluster_0006
```

### “Permission denied … .bin” when using Docker outputs
Docker often writes root-owned files. The dense script includes a safe Docker-based cleanup, but if you manually delete:
```bash
sudo rm -rf outputs_retrieval_test/<dataset>_<scene>
```

### “unrecognised option --Mapper.local_ba_min_tri_angle”
Some COLMAP Docker images don’t support that flag. The script auto-detects support and disables it when needed.

### “No good initial image pair found” / “Discarding reconstruction due to bad initial pair”
This means SfM could not bootstrap (images might not overlap, or matching is too weak).
Try:
- `--matcher auto` (falls back to DINO then diffusion)
- More permissive mapper params:
  - `--mapper-min-model-size 2`
  - `--mapper-init-min-num-inliers 8`
  - `--mapper-init-min-tri-angle 0.5`

### Dense output has 0 points
If `dense_points.ply` has `element vertex 0`, it usually means dense stereo/fusion didn’t find consistent depth.
The script will try photometric fusion fallback; if still empty, the cluster likely has poor geometry.

---

## 12) Repro checklist (quick)

If you want a minimal “I can run it end-to-end” checklist:

```bash
# 1) env
conda env create -f environment_no_builds.yml
conda activate imc25
pip install -e .[train]

# 2) run everything end-to-end (writes `submission.csv`)
HF_ENDPOINT=https://hf-mirror.com python scripts/run_pipeline.py --runner docker --overwrite

# Optional: also run dense MVS (writes `dense_points.ply` per cluster)
HF_ENDPOINT=https://hf-mirror.com python scripts/run_pipeline.py --runner docker --dense --overwrite

# 2) train retrieval
HF_ENDPOINT=https://hf-mirror.com python scripts/train_retrieval.py --output-dir outputs/retrieval_finetune

# 3) embed test
HF_ENDPOINT=https://hf-mirror.com python scripts/build_retrieval_cache.py --data-root data/test --cache-root cache_retrieval --checkpoint outputs/retrieval_finetune/best.pt --topk 30 --mutual

# 4) cluster test
python scripts/cluster_retrieval.py --cache-root cache_retrieval --out-csv cache/clusters_retrieval.csv

# 5) dense reconstruction
HF_ENDPOINT=https://hf-mirror.com python scripts/colmap_dense_clusters.py --clusters-csv cache/clusters_retrieval.csv --data-root data/test --output-root outputs_retrieval_test --runner docker --matcher colmap --overwrite
```
