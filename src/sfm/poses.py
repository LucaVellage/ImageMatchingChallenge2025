from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd

from .colmap_runner import colmap_model_converter

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])

def extract_poses(model_dir, scene_dir, dataset, scene, cfg, env):
    txt_dir = scene_dir / "sparse_txt"
    if txt_dir.exists():
        shutil.rmtree(txt_dir)
    txt_dir.mkdir(parents=True, exist_ok=True)

    colmap_model_converter(model_dir, txt_dir, cfg, env)

    images_txt = txt_dir / "images.txt"
    if not images_txt.exists():
        return pd.DataFrame([])

    poses = []
    lines = images_txt.read_text().splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or not line:
            i += 1
            continue

        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue

        _, qw, qx, qy, qz, tx, ty, tz, _, name = parts[:10]

        qvec = np.array([float(qw), float(qx), float(qy), float(qz)])
        tvec = np.array([float(tx), float(ty), float(tz)])

        R = qvec2rotmat(qvec)
        C = -R.T @ tvec

        poses.append({
            "image_id": Path(name).stem,
            "dataset": dataset,
            "scene": scene,
            "image": name,
            "rotation_matrix": ";".join(map(str, R.T.flatten())),
            "translation_vector": ";".join(map(str, C)),
        })

        i += 2  # skip next line (2D points)

    return pd.DataFrame(poses)

def save_scene_outputs(scene_dir, model_dir, dataset, scene, image_ids, cfg, env):
    poses_df = extract_poses(model_dir, scene_dir, dataset, scene, cfg, env)
    poses_df.to_csv(scene_dir / "poses.csv", index=False)

    stats = {
        "dataset": dataset,
        "scene": scene,
        "num_images_total": len(image_ids),
        "num_images_registered": len(poses_df),
        "success": len(poses_df) >= 2,
    }
    (scene_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return poses_df
