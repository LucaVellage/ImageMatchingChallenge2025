from pathlib import Path
import pandas as pd
from tqdm import tqdm

from .reconstruction import reconstruct_scene
from .poses import save_scene_outputs

SCENE_STEPS = [
    "prepare_images",
    "feature_extraction",
    "matching",
    "mapping",
    "pose_extraction",
]

def run_sfm(clusters_df: pd.DataFrame, cfg, submission_csv: Path):
    clusters_df = clusters_df[clusters_df.scene != "outliers"]

    groups = list(clusters_df.groupby(["dataset", "scene"]))
    all_rows = []

    for i, ((dataset, scene), g) in enumerate(groups, 1):
        image_ids = g.image_id.tolist()

        print(f"\n[{i}/{len(groups)}] {dataset} / {scene}")

        if len(image_ids) < 2:
            print("Skipping: too few images")
            continue

        with tqdm(total=len(SCENE_STEPS), desc=f"{dataset}/{scene}", leave=False) as pbar:
            try:
                model_dir, scene_dir, env = reconstruct_scene(dataset, scene, image_ids, cfg, pbar)
                if model_dir is None:
                    print("Reconstruction failed")
                    continue

                poses_df = save_scene_outputs(scene_dir, model_dir, dataset, scene, image_ids, cfg, env)
                pbar.update(1)  # pose_extraction

                all_rows.append(poses_df)
                print(f"Finished {dataset}/{scene}")

            except Exception as e:
                print(f"Skipping {dataset}/{scene} due to error:")
                print(e)

    if not all_rows:
        raise RuntimeError("No successful reconstructions; submission would be empty.")

    submission_df = pd.concat(all_rows, ignore_index=True)
    submission_df = submission_df[
        ["image_id", "dataset", "scene", "image", "rotation_matrix", "translation_vector"]
    ]
    submission_csv.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_csv, index=False)
    print(f"Saved {submission_csv}")
    return submission_df
