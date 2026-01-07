# Notebooks

These notebooks go beyond the competition baseline: they include ablations, metric studies, matcher comparisons, and
pipeline experiments. Start Jupyter from the repo root so relative paths like `data/train` resolve correctly.

Suggested order:

- `notebooks/00_clustering_exploration.ipynb`: exploratory clustering workbench (legacy/ad-hoc).
- `notebooks/00_sfm_exploration.ipynb`: exploratory SfM workbench (legacy/ad-hoc).
- `notebooks/01_sfm_mvs_reconstruction.ipynb`: curated SfM → MVS reconstruction walkthrough.
- `notebooks/02_nerf_or_3dgs_reconstruction.ipynb`: NeRF / 3DGS reconstruction notes.
- `notebooks/03_diffusion_postprocess.ipynb`: diffusion-based post-processing experiments.
- `notebooks/04_gpu_colmap_docker_pipeline.ipynb`: GPU COLMAP + Docker pipeline walkthrough.
- `notebooks/05_retrieval_finetune_vs_pretrained.ipynb`: quantitative + visual comparison of retrieval embeddings (pretrained vs fine-tuned).
- `notebooks/99_submission_notebook.ipynb`: single end-to-end notebook to generate `submission.csv`.
- `notebooks/15_end_to_end_sota_pipeline.ipynb`: comprehensive end-to-end SOTA runner (train→cluster→SfM/MVS) with diagnostics, visualizations, and optional GT scoring on labeled splits.
