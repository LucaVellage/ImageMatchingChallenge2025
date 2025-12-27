from pathlib import Path
import shutil

def prepare_cluster_workspace(cfg, dataset, scene):
    scene_dir = cfg.sfm_root / dataset / scene
    img_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse"
    db_path = scene_dir / "database.db"

    if cfg.reset_scene_dir and scene_dir.exists():
        shutil.rmtree(scene_dir)

    scene_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(exist_ok=True)

    # ensuring clean sparse dir
    if sparse_dir.exists() and not sparse_dir.is_dir():
        sparse_dir.unlink()
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    return scene_dir, img_dir, db_path, sparse_dir
