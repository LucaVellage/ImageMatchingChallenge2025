import shutil
from pathlib import Path

from .prep import prepare_cluster_workspace
from .colmap_runner import (
    get_headless_env,
    colmap_feature_extractor,
    colmap_exhaustive_matcher,
    colmap_mapper,
)

def reconstruct_scene(dataset, scene, image_ids, cfg, pbar=None):
    env = get_headless_env()

    scene_dir, img_dir, db_path, sparse_dir = prepare_cluster_workspace(
        cfg, dataset, scene
    )

    if pbar:
        pbar.update(1)  # prepare_images

    # Copy images (same logic as notebook)
    for img_id in image_ids:
        matches = list((cfg.data_root / dataset).rglob(f"{img_id}.*"))
        if not matches:
            raise RuntimeError(f"Image not found: {img_id}")
        shutil.copy(matches[0], img_dir / matches[0].name)

    if not any(img_dir.iterdir()):
        raise RuntimeError("No images copied")

    colmap_feature_extractor(db_path, img_dir, cfg, env)
    if pbar: pbar.update(1)

    colmap_exhaustive_matcher(db_path, cfg, env)
    if pbar: pbar.update(1)

    print("Images in scene:", len(list(img_dir.iterdir())))
    print("Image path exists:", img_dir.exists())

    colmap_mapper(db_path, img_dir, sparse_dir, cfg, env)
    if pbar: pbar.update(1)

    model_dir = sparse_dir / "0"
    return (model_dir if model_dir.exists() else None), scene_dir, env
