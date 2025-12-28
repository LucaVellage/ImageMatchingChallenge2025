import os
import subprocess

def get_headless_env():
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["DISPLAY"] = ""
    env["XDG_RUNTIME_DIR"] = "/tmp"
    return env

def run(cmd, cwd=None, timeout=900, env=None):
    print(" ".join(map(str, cmd)))
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")

def colmap_feature_extractor(db_path, img_dir, cfg, env):
    run([
        cfg.colmap_bin, "feature_extractor",
        "--database_path", db_path,
        "--image_path", img_dir,
        "--ImageReader.camera_model", cfg.camera_model,
        "--ImageReader.single_camera", "1" if cfg.single_camera else "0",
        "--SiftExtraction.use_gpu", "1" if cfg.use_gpu else "0",
        "--SiftExtraction.max_image_size", str(cfg.max_image_size),
        "--SiftExtraction.num_threads", str(cfg.num_threads),
        "--SiftExtraction.max_num_features", str(cfg.max_num_features),
        "--SiftExtraction.first_octave", str(cfg.first_octave),
    ], timeout=cfg.timeout_default, env=env)

def colmap_matches_importer(db_path, pairs_txt, cfg, env):
    run([
        cfg.colmap_bin, "matches_importer",
        "--database_path", db_path,
        "--match_list_path", pairs_txt,
        "--match_type", "pairs",
        "--SiftMatching.use_gpu", "1" if cfg.use_gpu else "0",
        "--SiftMatching.num_threads", str(cfg.sift_matching_num_threads),
    ], timeout=cfg.timeout_default, env=env)

def colmap_mapper(db_path, img_dir, sparse_dir, cfg, env):
    run([
        cfg.colmap_bin, "mapper",
        "--database_path", db_path,
        "--image_path", img_dir,
        "--output_path", sparse_dir,
        "--Mapper.min_num_matches", str(cfg.min_num_matches),
        "--Mapper.init_num_trials", str(cfg.init_num_trials),
        "--Mapper.max_reg_trials", str(cfg.max_reg_trials),
        "--Mapper.ba_global_max_num_iterations", str(cfg.ba_global_max_num_iterations),
        "--Mapper.ba_global_max_refinements", str(cfg.ba_global_max_refinements),
    ], timeout=cfg.timeout_mapper, env=env)

def colmap_model_converter(model_dir, txt_dir, cfg, env):
    run([
        cfg.colmap_bin, "model_converter",
        "--input_path", model_dir,
        "--output_path", txt_dir,
        "--output_type", "TXT",
    ], timeout=cfg.timeout_default, env=env)
