from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class SfMConfig:
    # paths
    data_root: Path          
    sfm_root: Path           

    # colmap basics
    colmap_bin: str = "colmap"
    camera_model: str = "SIMPLE_RADIAL"
    single_camera: bool = True

    # compute / sift
    use_gpu: bool = False
    max_image_size: int = 2000
    num_threads: int = 4
    max_num_features: int = 4096
    first_octave: int = 0

    # matcher
    sift_matching_num_threads: int = 4

    # mapper (these must match your notebook intent)
    min_num_matches: int = 8
    init_num_trials: int = 50
    max_reg_trials: int = 2
    ba_global_max_num_iterations: int = 5
    ba_global_max_refinements: int = 1

    # timeouts (seconds)
    timeout_default: int = 900
    timeout_mapper: int = 3600

    # behavior
    reset_scene_dir: bool = True

    @classmethod
    def from_yaml(cls, path: Path, overrides: dict | None = None):
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        if overrides:
            for k, v in overrides.items():
                if v is not None and k in data or hasattr(cls, k):
                    data[k] = v

        # Path coercion
        data["data_root"] = Path(data["data_root"])
        data["sfm_root"] = Path(data["sfm_root"])
        return cls(**data)
