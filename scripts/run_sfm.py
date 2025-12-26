import argparse
from pathlib import Path
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sfm.config import SfMConfig
from src.sfm.pipeline import run_sfm

def parse_args():
    p = argparse.ArgumentParser("Run COLMAP SfM")
    p.add_argument("--config", required=True)
    p.add_argument("--clusters_csv", required=True)
    p.add_argument("--submission_csv", default="submission.csv")

    # overrides arguments defined in YAML file
    p.add_argument("--sfm_root")
    p.add_argument("--data_root")

    p.add_argument("--min_num_matches", type=int)
    p.add_argument("--init_num_trials", type=int)
    p.add_argument("--max_reg_trials", type=int)
    p.add_argument("--ba_global_max_num_iterations", type=int)
    p.add_argument("--ba_global_max_refinements", type=int)

    p.add_argument("--max_image_size", type=int)
    p.add_argument("--num_threads", type=int)
    p.add_argument("--max_num_features", type=int)

    return p.parse_args()

def main():
    args = parse_args()
    clusters_df = pd.read_csv(args.clusters_csv)

    overrides = {
        k: v for k, v in vars(args).items()
        if v is not None
    }

    cfg = SfMConfig.from_yaml(Path(args.config), overrides=overrides)
    submission_csv = Path(args.submission_csv)

    # All relative paths mus resolve from project root
    cfg.data_root = (PROJECT_ROOT / cfg.data_root).resolve()
    cfg.sfm_root = (PROJECT_ROOT / cfg.sfm_root).resolve()


    submission_df = run_sfm(
        clusters_df=clusters_df,
        cfg=cfg,
        submission_csv=submission_csv,
    )

if __name__ == "__main__":
    main()
