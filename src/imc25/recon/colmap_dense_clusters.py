from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sqlite3
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ClusterJob:
    dataset: str
    scene: str
    image_ids: list[str]
    out_dir: Path


def _read_clusters(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"dataset", "image_id", "scene"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.copy()
    df["dataset"] = df["dataset"].astype(str)
    df["scene"] = df["scene"].astype(str)
    df["image_id"] = df["image_id"].astype(str)
    return df


def _load_meta_map(cache_root: Path, dataset: str) -> dict[str, Path]:
    meta_path = cache_root / dataset / "meta.json"
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text())
    image_ids = meta.get("image_ids", [])
    paths = meta.get("paths", [])
    if not isinstance(image_ids, list) or not isinstance(paths, list) or len(image_ids) != len(paths):
        return {}
    out: dict[str, Path] = {}
    for image_id, p in zip(image_ids, paths, strict=False):
        out[str(image_id)] = Path(str(p))
    return out


def _resolve_image_path(
    *,
    repo_root: Path,
    data_root: Path,
    dataset: str,
    image_id: str,
    meta_map: dict[str, Path],
) -> Path | None:
    if image_id in meta_map:
        p = meta_map[image_id]
        p = p if p.is_absolute() else (repo_root / p)
        return p if p.exists() else None

    # Fallback: search by common extensions under data_root/dataset/.
    base = data_root / dataset / image_id
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = base.with_suffix(ext)
        if p.exists():
            return p
    return None


def _symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    rel = os.path.relpath(src, start=dst.parent)
    try:
        dst.symlink_to(rel)
    except Exception:
        shutil.copy2(src, dst)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            dst.unlink()
        elif dst.is_file():
            return
        else:
            raise ValueError(f"Refusing to overwrite non-file path: {dst}")
    shutil.copy2(src, dst)


def _safe_rmtree(path: Path) -> None:
    def _onerror(func, p, exc_info):
        try:
            os.chmod(p, 0o700)
            func(p)
        except Exception:
            raise

    shutil.rmtree(path, onerror=_onerror)


def _colmap_bin_count(path: Path) -> int | None:
    try:
        with path.open("rb") as f:
            head = f.read(8)
        if len(head) != 8:
            return None
        (n,) = struct.unpack("<Q", head)
        return int(n)
    except Exception:
        return None


def _pick_best_sparse_model(sparse_root: Path) -> Path | None:
    if not sparse_root.exists():
        return None
    candidates: list[Path] = [p for p in sparse_root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not candidates:
        return None

    best: tuple[int, int, str] | None = None
    best_path: Path | None = None
    for p in sorted(candidates, key=lambda x: int(x.name)):
        n_images = _colmap_bin_count(p / "images.bin") or 0
        n_points = _colmap_bin_count(p / "points3D.bin") or 0
        score = (n_images, n_points, p.name)
        if best is None or score > best:
            best = score
            best_path = p
    return best_path


def _db_table_count(db_path: Path, table: str) -> int:
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
        row = cur.fetchone()
    return int(row[0] if row else 0)


def _db_ready(db_path: Path) -> tuple[bool, bool]:
    """
    Returns (has_features, has_pair_geometries).

    - features: keypoints/descriptors present
    - pair geometries: two_view_geometries present (from matching)
    """
    try:
        images = _db_table_count(db_path, "images")
        keypoints = _db_table_count(db_path, "keypoints")
        two_view = _db_table_count(db_path, "two_view_geometries")
    except Exception:
        return False, False
    has_features = images > 0 and keypoints > 0
    has_pair_geom = two_view > 0
    return has_features, has_pair_geom


def _db_has_unsafe_image_names(db_path: Path) -> bool:
    """
    Detect path-traversal image names such as '../../../data/test/...png' stored in the COLMAP DB.

    This commonly happens when images in the cluster folder are symlinks and COLMAP stores the
    resolved symlink target as the image name.
    """
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute("SELECT name FROM images").fetchall()
    except Exception:
        return False

    for (name,) in rows:
        if not name:
            continue
        s = str(name)
        if s.startswith(("/", "\\")):
            return True
        if any(part == ".." for part in Path(s).parts):
            return True
    return False


def _db_image_name_set(db_path: Path) -> set[str] | None:
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute("SELECT name FROM images").fetchall()
    except Exception:
        return None
    return {str(r[0]) for r in rows if r and r[0]}


def _ply_vertex_count(path: Path) -> int | None:
    try:
        max_header_bytes = 1024 * 1024  # 1 MiB safety cap
        chunk_size = 64 * 1024
        buf = bytearray()
        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                buf.extend(chunk)
                if b"end_header" in buf:
                    break
                if len(buf) > max_header_bytes:
                    return None
        if b"end_header" not in buf:
            return None
        for line in bytes(buf).splitlines():
            if line.startswith(b"element vertex "):
                return int(line.split()[-1])
        return None
    except Exception:
        return None


class ColmapRunner:
    def __init__(
        self,
        *,
        repo_root: Path,
        mode: str,
        docker_image: str,
        docker_gpus: str | None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.mode = mode
        self.docker_image = docker_image
        self.docker_gpus = docker_gpus

    def _rel(self, path: Path) -> str:
        p = path.resolve()
        try:
            rel = p.relative_to(self.repo_root)
        except ValueError as e:
            raise ValueError(f"Path is outside repo root ({self.repo_root}): {p}") from e
        return rel.as_posix()

    def run(self, args: list[str]) -> None:
        if self.mode == "local":
            cmd = ["colmap", *args]
        elif self.mode == "docker":
            if shutil.which("docker") is None:
                raise RuntimeError("docker not found in PATH (runner=docker)")
            cmd = ["docker", "run", "--rm"]
            if self.docker_gpus:
                cmd += ["--gpus", self.docker_gpus]
            cmd += [
                "-v",
                f"{self.repo_root}:/workspace",
                "-w",
                "/workspace",
                self.docker_image,
                "colmap",
                *args,
            ]
        else:
            raise ValueError(f"Unknown runner mode: {self.mode}")

        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=self.repo_root, check=True)

    def run_capture(self, args: list[str]) -> str:
        """
        Run a COLMAP command and return combined stdout/stderr as text.

        Intended for lightweight capability checks (e.g., `colmap mapper --help`),
        not for long-running reconstruction steps.
        """
        if self.mode == "local":
            cmd = ["colmap", *args]
        elif self.mode == "docker":
            if shutil.which("docker") is None:
                raise RuntimeError("docker not found in PATH (runner=docker)")
            cmd = ["docker", "run", "--rm"]
            if self.docker_gpus:
                cmd += ["--gpus", self.docker_gpus]
            cmd += [
                "-v",
                f"{self.repo_root}:/workspace",
                "-w",
                "/workspace",
                self.docker_image,
                "colmap",
                *args,
            ]
        else:
            raise ValueError(f"Unknown runner mode: {self.mode}")

        proc = subprocess.run(
            cmd,
            cwd=self.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return str(proc.stdout or "")

    def rm_rf(self, path: Path, *, dry_run: bool = False) -> None:
        if not path.exists():
            return

        try:
            shutil.rmtree(path)
            return
        except Exception:
            pass

        if shutil.which("docker") is None:
            raise

        rel = self._rel(path)
        if rel in {"", "."}:
            raise ValueError(f"Refusing to remove unsafe path: {path}")
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{self.repo_root}:/workspace",
            "-w",
            "/workspace",
            self.docker_image,
            "sh",
            "-lc",
            f"rm -rf {shlex.quote(rel)}",
        ]
        print("+", " ".join(cmd), flush=True)
        if dry_run:
            return
        subprocess.run(cmd, cwd=self.repo_root, check=True)

    def rm_f(self, path: Path, *, dry_run: bool = False) -> None:
        if not path.exists():
            return

        try:
            path.unlink()
            return
        except Exception:
            pass

        if shutil.which("docker") is None:
            raise

        rel = self._rel(path)
        if rel in {"", "."}:
            raise ValueError(f"Refusing to remove unsafe path: {path}")
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{self.repo_root}:/workspace",
            "-w",
            "/workspace",
            self.docker_image,
            "sh",
            "-lc",
            f"rm -f {shlex.quote(rel)}",
        ]
        print("+", " ".join(cmd), flush=True)
        if dry_run:
            return
        subprocess.run(cmd, cwd=self.repo_root, check=True)


def _auto_runner_mode(prefer: str) -> str:
    if prefer in {"local", "docker"}:
        return prefer

    def local_dense_available() -> bool:
        if shutil.which("colmap") is None:
            return False
        try:
            proc = subprocess.run(
                ["colmap", "patch_match_stereo", "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return proc.returncode == 0
        except Exception:
            return False

    if local_dense_available():
        return "local"
    if shutil.which("docker") is not None:
        return "docker"
    if shutil.which("colmap") is not None:
        raise RuntimeError("Local COLMAP is available but dense stereo requires CUDA; use --runner docker")
    raise RuntimeError("Neither `colmap` nor `docker` found; install COLMAP or use Docker")


def _iter_jobs(
    df: pd.DataFrame,
    *,
    output_root: Path,
    outliers_label: str,
    min_images: int,
    dataset_filter: set[str] | None,
    scene_filter: set[str] | None,
) -> Iterable[ClusterJob]:
    for (dataset, scene), grp in df.groupby(["dataset", "scene"], sort=True):
        if dataset_filter and dataset not in dataset_filter:
            continue
        if scene_filter and scene not in scene_filter:
            continue
        if scene.lower() == outliers_label.lower():
            continue
        image_ids = grp["image_id"].astype(str).tolist()
        if len(image_ids) < min_images:
            continue
        out_dir = output_root / f"{dataset}_{scene}"
        yield ClusterJob(dataset=dataset, scene=scene, image_ids=image_ids, out_dir=out_dir)


def run_dense_for_cluster(
    job: ClusterJob,
    *,
    repo_root: Path,
    data_root: Path,
    cache_root: Path,
    runner: ColmapRunner,
    overwrite: bool,
    dry_run: bool,
    camera_model: str,
    single_camera: bool,
    patchmatch_gpu_index: int | None,
    matcher: str,
    image_mode: str,
    dino_args: dict | None,
    diffusion_args: dict | None,
    dense_min_vertices: int,
    depth_fallback: bool,
    depth_args: dict | None,
    sparse_only: bool,
    mapper_min_model_size: int,
    mapper_min_num_matches: int | None,
    mapper_init_min_num_inliers: int,
    mapper_init_min_tri_angle: float,
    mapper_tri_min_angle: float,
    mapper_filter_min_tri_angle: float,
    mapper_local_ba_min_tri_angle: float,
    mapper_tri_ignore_two_view_tracks: int,
    mapper_disable_local_ba_min_tri_angle: bool,
    patchmatch_min_tri_angle: float | None,
    patchmatch_filter_min_tri_angle: float | None,
    patchmatch_filter_min_ncc: float | None,
    patchmatch_relax_on_failure: bool,
    patchmatch_relaxed_min_tri_angle: float | None,
    patchmatch_relaxed_filter_min_tri_angle: float | None,
    patchmatch_relaxed_filter_min_ncc: float | None,
) -> str:
    out_dir = job.out_dir
    images_dir = out_dir / "images"
    sparse_root = out_dir / "sparse"
    dense_root = out_dir / "dense"
    dense_ply = out_dir / "dense_points.ply"

    if sparse_only:
        best = _pick_best_sparse_model(sparse_root)
        if best is not None and (best / "images.bin").exists() and not overwrite:
            print(f"[skip] {job.dataset}/{job.scene}: sparse model exists at {best}")
            return "skip"
    else:
        if dense_ply.exists() and not overwrite:
            vtx_existing = _ply_vertex_count(dense_ply)
            if vtx_existing is not None and vtx_existing < int(dense_min_vertices):
                print(
                    f"[warn] {job.dataset}/{job.scene}: existing {dense_ply} has {vtx_existing} points (<{dense_min_vertices}); rebuilding",
                    flush=True,
                )
            else:
                print(f"[skip] {job.dataset}/{job.scene}: {dense_ply} exists")
                return "skip"
    if overwrite and out_dir.exists() and not dry_run:
        runner.rm_rf(out_dir)

    if not dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)
        sparse_root.mkdir(parents=True, exist_ok=True)

    meta_map = _load_meta_map(cache_root, job.dataset)
    resolved: list[tuple[str, Path]] = []
    for image_id in job.image_ids:
        p = _resolve_image_path(
            repo_root=repo_root,
            data_root=data_root,
            dataset=job.dataset,
            image_id=image_id,
            meta_map=meta_map,
        )
        if p is None:
            print(f"[warn] missing image for {job.dataset}/{job.scene}: {image_id}")
            continue
        resolved.append((image_id, p))

    if len(resolved) < 2:
        print(f"[skip] {job.dataset}/{job.scene}: not enough images found ({len(resolved)})")
        return "skip"

    print(f"[run] {job.dataset}/{job.scene}: images={len(resolved)} matcher={matcher}", flush=True)

    if not dry_run:
        desired_names = {src.name for _, src in resolved}
        if images_dir.exists():
            for entry in images_dir.iterdir():
                if entry.is_dir():
                    continue
                if entry.name not in desired_names:
                    entry.unlink(missing_ok=True)
        for image_id, src in resolved:
            dst = images_dir / src.name
            mode = str(image_mode).lower().strip()
            if mode == "auto":
                # COLMAP feature_extractor resolves symlinks and stores the symlink target as the
                # image "name", which can contain "../" and breaks the downstream dense workspace.
                # For the classic COLMAP pipeline we therefore copy images into the cluster folder.
                mode = "copy" if matcher in {"colmap", "auto"} else "symlink"
            if mode == "copy":
                _copy(src, dst)
            elif mode == "symlink":
                _symlink_or_copy(src, dst)
            else:
                raise ValueError(f"Unknown image_mode: {image_mode}")

    def p(path: Path) -> str:
        return runner._rel(path) if runner.mode == "docker" else str(path)

    matcher_sequence = ["colmap", "dino", "diffusion"] if matcher == "auto" else [matcher]
    last_status: str = "fail"
    for seq_idx, matcher_try in enumerate(matcher_sequence):
        if matcher_try == "diffusion":
            db_path = out_dir / "colmap_diffusion.db"
        elif matcher_try == "dino":
            db_path = out_dir / "colmap_dino.db"
        else:
            db_path = out_dir / "colmap.db"
        if seq_idx > 0:
            print(f"[fallback] {job.dataset}/{job.scene}: trying matcher={matcher_try}", flush=True)

        feature_args = [
            "feature_extractor",
            "--database_path",
            p(db_path),
            "--image_path",
            p(images_dir),
            "--ImageReader.camera_model",
            camera_model,
        ]
        if single_camera:
            feature_args += ["--ImageReader.single_camera", "1"]

        match_args = ["exhaustive_matcher", "--database_path", p(db_path)]

        mapper_args = [
            "mapper",
            "--database_path",
            p(db_path),
            "--image_path",
            p(images_dir),
            "--output_path",
            p(sparse_root),
            "--Mapper.ba_refine_principal_point",
            "0",
        ]

        if dry_run:
            print(f"[dry-run] {job.dataset}/{job.scene} -> {out_dir}")
            if matcher_try == "diffusion":
                print("  [python] build colmap.db via diffusion features + RANSAC inliers")
            elif matcher_try == "dino":
                print("  [python] build colmap.db via DINOv2 patch descriptors + RANSAC inliers")
            else:
                print("  colmap", " ".join(feature_args))
                print("  colmap", " ".join(match_args))
            dry_min_num_matches = (
                int(mapper_min_num_matches) if mapper_min_num_matches is not None else int(mapper_init_min_num_inliers)
            )
            dry_mapper_args = [
                *mapper_args,
                "--Mapper.min_num_matches",
                str(int(dry_min_num_matches)),
                "--Mapper.min_model_size",
                str(int(mapper_min_model_size)),
                "--Mapper.init_min_num_inliers",
                str(int(mapper_init_min_num_inliers)),
                "--Mapper.init_min_tri_angle",
                str(float(mapper_init_min_tri_angle)),
                "--Mapper.tri_min_angle",
                str(float(mapper_tri_min_angle)),
                "--Mapper.filter_min_tri_angle",
                str(float(mapper_filter_min_tri_angle)),
                "--Mapper.tri_ignore_two_view_tracks",
                str(int(mapper_tri_ignore_two_view_tracks)),
            ]
            if not mapper_disable_local_ba_min_tri_angle:
                dry_mapper_args += [
                    "--Mapper.local_ba_min_tri_angle",
                    str(float(mapper_local_ba_min_tri_angle)),
                ]
            print("  colmap", " ".join(dry_mapper_args))
            print("  (then: image_undistorter, patch_match_stereo, stereo_fusion)")
            return "skip"

        has_features = False
        has_pair_geom = False
        if db_path.exists():
            if _db_has_unsafe_image_names(db_path):
                print(
                    f"[warn] {job.dataset}/{job.scene}: COLMAP DB has unsafe image names; rebuilding ({db_path.name})",
                    flush=True,
                )
                db_path.unlink(missing_ok=True)
                if sparse_root.exists():
                    runner.rm_rf(sparse_root)
                    sparse_root.mkdir(parents=True, exist_ok=True)
                if dense_root.exists():
                    runner.rm_rf(dense_root)
                dense_ply.unlink(missing_ok=True)
            else:
                # If the cluster membership changed, the DB may reference images that no longer
                # exist under images_dir; rebuild in that case.
                desired = {p.name for p in images_dir.iterdir() if p.is_file() or p.is_symlink()}
                names = _db_image_name_set(db_path)
                if names is not None and names != desired:
                    print(
                        f"[warn] {job.dataset}/{job.scene}: COLMAP DB image set differs from images/; rebuilding ({db_path.name})",
                        flush=True,
                    )
                    db_path.unlink(missing_ok=True)
                    if sparse_root.exists():
                        runner.rm_rf(sparse_root)
                        sparse_root.mkdir(parents=True, exist_ok=True)
                    if dense_root.exists():
                        runner.rm_rf(dense_root)
                    dense_ply.unlink(missing_ok=True)
                else:
                    has_features, has_pair_geom = _db_ready(db_path)

        if matcher_try == "dino":
            if (not db_path.exists()) or (not has_features) or (not has_pair_geom):
                if db_path.exists():
                    db_path.unlink(missing_ok=True)
                if dino_args is None:
                    raise RuntimeError("dino_args missing (internal error)")

                from imc25.matching.dino_features import DinoFeatureConfig, DinoPatchFeatureExtractor
                from imc25.matching.dino_matcher import KeypointConfig, MatchConfig
                from imc25.matching.dino_to_colmap import PairingConfig, build_colmap_db_from_dino

                feat_cfg = dino_args.get("feat_cfg")
                kp_cfg = dino_args.get("kp_cfg")
                match_cfg = dino_args.get("match_cfg")
                pairing_cfg = dino_args.get("pairing_cfg")
                extractor = dino_args.get("extractor")

                if feat_cfg is None:
                    feat_cfg = DinoFeatureConfig(
                        model_id=str(dino_args["model_id"]),
                        max_side=int(dino_args["max_side"]),
                        layer=int(dino_args["layer"]) if dino_args.get("layer") is not None else None,
                        use_fp16=bool(dino_args["fp16"]),
                    )
                    dino_args["feat_cfg"] = feat_cfg
                if kp_cfg is None:
                    kp_cfg = KeypointConfig(method=str(dino_args["keypoints"]), max_keypoints=int(dino_args["max_keypoints"]))
                    dino_args["kp_cfg"] = kp_cfg
                if match_cfg is None:
                    match_cfg = MatchConfig(
                        min_similarity=float(dino_args["min_similarity"]),
                        max_matches=int(dino_args["max_matches"]),
                        ransac_thresh_px=float(dino_args["ransac_thresh_px"]),
                        min_inliers=int(dino_args["min_inliers"]),
                    )
                    dino_args["match_cfg"] = match_cfg
                if pairing_cfg is None:
                    pairing_cfg = PairingConfig(mode=str(dino_args["pairing"]), topk=int(dino_args["topk"]))
                    dino_args["pairing_cfg"] = pairing_cfg
                if extractor is None:
                    if dino_args.get("hf_endpoint"):
                        os.environ["HF_ENDPOINT"] = str(dino_args["hf_endpoint"])
                    print("[dino] lazy loading model...", flush=True)
                    extractor = DinoPatchFeatureExtractor(
                        feat_cfg,
                        device=str(dino_args.get("device")) if dino_args.get("device") else None,
                        hf_endpoint=str(dino_args.get("hf_endpoint")) if dino_args.get("hf_endpoint") else None,
                    )
                    dino_args["extractor"] = extractor

                cache_dir = dino_args.get("cache_dir")
                cache_dir = Path(cache_dir) if cache_dir else None

                build_colmap_db_from_dino(
                    images_dir=images_dir,
                    db_path=db_path,
                    overwrite=True,
                    camera_model=camera_model,
                    single_camera=single_camera,
                    feat_cfg=feat_cfg,
                    kp_cfg=kp_cfg,
                    match_cfg=match_cfg,
                    pairing_cfg=pairing_cfg,
                    cache_dir=cache_dir,
                    dino_device=str(dino_args.get("device")) if dino_args.get("device") else None,
                    hf_endpoint=str(dino_args.get("hf_endpoint")) if dino_args.get("hf_endpoint") else None,
                    extractor=extractor,
                    verbose=True,
                )

            has_features, has_pair_geom = _db_ready(db_path) if db_path.exists() else (False, False)
            if not has_pair_geom:
                print(
                    f"[skip] {job.dataset}/{job.scene}: DINO matching produced no verified pairs "
                    f"(try --dino-min-inliers 8 or --dino-min-similarity 0.7)",
                    flush=True,
                )
                last_status = "fail"
                continue

        elif matcher_try == "diffusion":
            if (not db_path.exists()) or (not has_features) or (not has_pair_geom):
                if db_path.exists():
                    db_path.unlink(missing_ok=True)
                if diffusion_args is None:
                    raise RuntimeError("diffusion_args missing (internal error)")
                from imc25.matching.diffusion_features import DiffusionFeatureConfig, DiffusionUNetFeatureExtractor
                from imc25.matching.diffusion_matcher import KeypointConfig, MatchConfig
                from imc25.matching.diffusion_to_colmap import PairingConfig, build_colmap_db_from_diffusion

                feat_cfg = diffusion_args.get("feat_cfg")
                kp_cfg = diffusion_args.get("kp_cfg")
                match_cfg = diffusion_args.get("match_cfg")
                pairing_cfg = diffusion_args.get("pairing_cfg")
                extractor = diffusion_args.get("extractor")

                if feat_cfg is None:
                    feat_cfg = DiffusionFeatureConfig(
                        model_id=str(diffusion_args["model_id"]),
                        timestep=int(diffusion_args["timestep"]),
                        max_side=int(diffusion_args["max_side"]),
                        noise_seed=int(diffusion_args["noise_seed"]),
                        layers=tuple(diffusion_args["layers"]),
                        use_fp16=bool(diffusion_args["fp16"]),
                    )
                    diffusion_args["feat_cfg"] = feat_cfg
                if kp_cfg is None:
                    kp_cfg = KeypointConfig(
                        method=str(diffusion_args["keypoints"]), max_keypoints=int(diffusion_args["max_keypoints"])
                    )
                    diffusion_args["kp_cfg"] = kp_cfg
                if match_cfg is None:
                    match_cfg = MatchConfig(
                        min_similarity=float(diffusion_args["min_similarity"]),
                        max_matches=int(diffusion_args["max_matches"]),
                        ransac_thresh_px=float(diffusion_args["ransac_thresh_px"]),
                        min_inliers=int(diffusion_args["min_inliers"]),
                    )
                    diffusion_args["match_cfg"] = match_cfg
                if pairing_cfg is None:
                    pairing_cfg = PairingConfig(mode=str(diffusion_args["pairing"]), topk=int(diffusion_args["topk"]))
                    diffusion_args["pairing_cfg"] = pairing_cfg
                if extractor is None:
                    if diffusion_args.get("hf_endpoint"):
                        os.environ["HF_ENDPOINT"] = str(diffusion_args["hf_endpoint"])
                    print("[diffusion] lazy loading model...", flush=True)
                    extractor = DiffusionUNetFeatureExtractor(
                        feat_cfg,
                        device=str(diffusion_args.get("device")) if diffusion_args.get("device") else None,
                    )
                    diffusion_args["extractor"] = extractor

                cache_dir = diffusion_args.get("cache_dir")
                cache_dir = Path(cache_dir) if cache_dir else None

                build_colmap_db_from_diffusion(
                    images_dir=images_dir,
                    db_path=db_path,
                    overwrite=True,
                    camera_model=camera_model,
                    single_camera=single_camera,
                    feat_cfg=feat_cfg,
                    kp_cfg=kp_cfg,
                    match_cfg=match_cfg,
                    pairing_cfg=pairing_cfg,
                    cache_dir=cache_dir,
                    diffusion_device=str(diffusion_args.get("device")) if diffusion_args.get("device") else None,
                    hf_endpoint=str(diffusion_args.get("hf_endpoint")) if diffusion_args.get("hf_endpoint") else None,
                    extractor=extractor,
                    verbose=True,
                )

            # If diffusion matching produced no verified pair geometries, skip mapper early.
            has_features, has_pair_geom = _db_ready(db_path) if db_path.exists() else (False, False)
            if not has_pair_geom:
                print(
                    f"[skip] {job.dataset}/{job.scene}: diffusion matching produced no verified pairs "
                    f"(try --diffusion-min-inliers 8 or --diffusion-min-similarity 0.6)",
                    flush=True,
                )
                last_status = "fail"
                continue
        else:
            if not db_path.exists() or not has_features:
                if db_path.exists():
                    db_path.unlink(missing_ok=True)
                runner.run(feature_args)

            if not has_pair_geom:
                runner.run(match_args)

        # Always rebuild sparse from scratch for this attempt to avoid selecting a stale model
        # when switching matchers or re-running after previous failures.
        if sparse_root.exists():
            runner.rm_rf(sparse_root)
        sparse_root.mkdir(parents=True, exist_ok=True)

        mapper_profiles = [
            (
                int(mapper_min_model_size),
                int(mapper_init_min_num_inliers),
                float(mapper_init_min_tri_angle),
                float(mapper_tri_min_angle),
                float(mapper_filter_min_tri_angle),
                float(mapper_local_ba_min_tri_angle),
                int(mapper_tri_ignore_two_view_tracks),
            ),
            (2, 15, 1.0, 0.25, 0.25, 0.25, 0),
            (2, 8, 0.5, 0.05, 0.05, 0.05, 0),
            # Some COLMAP builds require triangulation angles to be strictly > 0.
            (2, 4, 0.1, 0.01, 0.01, 0.01, 0),
        ]
        mapper_ok = False
        for i, (
            min_model_size,
            init_min_inliers,
            init_min_tri_angle,
            tri_min_angle,
            filter_min_tri_angle,
            local_ba_min_tri_angle,
            tri_ignore_two_view_tracks,
        ) in enumerate(mapper_profiles, start=1):
            min_num_matches = int(mapper_min_num_matches) if mapper_min_num_matches is not None else int(init_min_inliers)
            attempt_args = [
                *mapper_args,
                "--Mapper.min_num_matches",
                str(int(min_num_matches)),
                "--Mapper.min_model_size",
                str(int(min_model_size)),
                "--Mapper.init_min_num_inliers",
                str(int(init_min_inliers)),
                "--Mapper.init_min_tri_angle",
                str(float(init_min_tri_angle)),
                "--Mapper.tri_min_angle",
                str(float(tri_min_angle)),
                "--Mapper.filter_min_tri_angle",
                str(float(filter_min_tri_angle)),
                "--Mapper.tri_ignore_two_view_tracks",
                str(int(tri_ignore_two_view_tracks)),
            ]
            if not mapper_disable_local_ba_min_tri_angle:
                attempt_args += [
                    "--Mapper.local_ba_min_tri_angle",
                    str(float(local_ba_min_tri_angle)),
                ]
            try:
                if i > 1:
                    print(
                        f"[retry] {job.dataset}/{job.scene}: mapper attempt {i} "
                        f"(min_model_size={min_model_size}, init_min_inliers={init_min_inliers}, init_min_tri_angle={init_min_tri_angle}, "
                        f"tri_min_angle={tri_min_angle}, local_ba_min_tri_angle={local_ba_min_tri_angle}, "
                        f"tri_ignore_two_view_tracks={tri_ignore_two_view_tracks})"
                    )
                runner.run(attempt_args)
                mapper_ok = True
                break
            except subprocess.CalledProcessError:
                continue

        if not mapper_ok:
            print(f"[fail] {job.dataset}/{job.scene}: COLMAP mapper failed after {len(mapper_profiles)} attempts")
            if (not sparse_only) and depth_fallback and depth_args is not None and not dry_run:
                try:
                    from imc25.recon.depth_fusion import build_dense_points_from_depth

                    if dense_ply.exists():
                        runner.rm_f(dense_ply, dry_run=dry_run)
                    pts = build_dense_points_from_depth(
                        images_dir=images_dir,
                        out_ply=dense_ply,
                        model_dir=None,
                        depth_model_id=str(depth_args["model_id"]),
                        hf_endpoint=str(depth_args.get("hf_endpoint")) if depth_args.get("hf_endpoint") else None,
                        device=str(depth_args.get("device")) if depth_args.get("device") else None,
                        fp16=bool(depth_args.get("fp16", True)),
                        stride=int(depth_args.get("stride", 3)),
                        max_points=int(depth_args.get("max_points", 200_000)),
                        align_scale=bool(depth_args.get("align_scale", False)),
                        with_color=bool(depth_args.get("with_color", True)),
                        verbose=True,
                    )
                    if pts >= int(dense_min_vertices):
                        print(
                            f"[ok] {job.dataset}/{job.scene}: wrote {dense_ply} via depth fallback (points={pts}) "
                            "(SfM failed; poses will be NaN)",
                            flush=True,
                        )
                        return "ok"
                except Exception as e:
                    print(f"[warn] {job.dataset}/{job.scene}: depth fallback failed after SfM failure: {e}", flush=True)
            last_status = "fail"
            continue

        best_model = _pick_best_sparse_model(sparse_root)
        if best_model is None:
            print(f"[fail] {job.dataset}/{job.scene}: no sparse model produced at {sparse_root}")
            last_status = "fail"
            continue

        if sparse_only:
            print(f"[ok] {job.dataset}/{job.scene}: sparse model ready at {best_model}", flush=True)
            return "ok"

        undistort_args = [
            "image_undistorter",
            "--image_path",
            p(images_dir),
            "--input_path",
            p(best_model),
            "--output_path",
            p(dense_root),
            "--output_type",
            "COLMAP",
        ]
        if dense_root.exists():
            runner.rm_rf(dense_root)
        runner.run(undistort_args)

        patchmatch_args = [
            "patch_match_stereo",
            "--workspace_path",
            p(dense_root),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "true",
        ]
        if patchmatch_gpu_index is not None:
            patchmatch_args += ["--PatchMatchStereo.gpu_index", str(patchmatch_gpu_index)]
        if patchmatch_min_tri_angle is not None:
            patchmatch_args += ["--PatchMatchStereo.min_triangulation_angle", str(float(patchmatch_min_tri_angle))]
        if patchmatch_filter_min_tri_angle is not None:
            patchmatch_args += ["--PatchMatchStereo.filter_min_triangulation_angle", str(float(patchmatch_filter_min_tri_angle))]
        if patchmatch_filter_min_ncc is not None:
            patchmatch_args += ["--PatchMatchStereo.filter_min_ncc", str(float(patchmatch_filter_min_ncc))]
        runner.run(patchmatch_args)

        fusion_geometric = [
            "stereo_fusion",
            "--workspace_path",
            p(dense_root),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            p(dense_ply),
        ]
        runner.run(fusion_geometric)

        vtx = _ply_vertex_count(dense_ply) if dense_ply.exists() else None
        if vtx is not None and vtx < int(dense_min_vertices):
            print(f"[warn] {job.dataset}/{job.scene}: geometric fusion has {vtx} points; trying photometric fusion", flush=True)
            fusion_photometric = [
                "stereo_fusion",
                "--workspace_path",
                p(dense_root),
                "--workspace_format",
                "COLMAP",
                "--input_type",
                "photometric",
                "--output_path",
                p(dense_ply),
            ]
            runner.run(fusion_photometric)
            vtx = _ply_vertex_count(dense_ply) if dense_ply.exists() else vtx

        if dense_ply.exists() and (vtx is None or vtx >= int(dense_min_vertices)):
            pts = f"{vtx}" if vtx is not None else "?"
            print(f"[ok] {job.dataset}/{job.scene}: wrote {dense_ply} (points={pts})")
            return "ok"

        if patchmatch_relax_on_failure:
            print(
                f"[warn] {job.dataset}/{job.scene}: dense fusion still has {vtx if vtx is not None else '?'} points; "
                "retrying PatchMatch with relaxed thresholds",
                flush=True,
            )
            if not dry_run:
                # Keep configs but remove outputs to ensure a clean retry.
                for sub in ["depth_maps", "normal_maps", "consistency_graphs"]:
                    runner.rm_rf(dense_root / "stereo" / sub, dry_run=dry_run)

            retry_args = [
                "patch_match_stereo",
                "--workspace_path",
                p(dense_root),
                "--workspace_format",
                "COLMAP",
                "--PatchMatchStereo.geom_consistency",
                "true",
            ]
            if patchmatch_gpu_index is not None:
                retry_args += ["--PatchMatchStereo.gpu_index", str(patchmatch_gpu_index)]
            if patchmatch_relaxed_min_tri_angle is not None:
                retry_args += ["--PatchMatchStereo.min_triangulation_angle", str(float(patchmatch_relaxed_min_tri_angle))]
            if patchmatch_relaxed_filter_min_tri_angle is not None:
                retry_args += [
                    "--PatchMatchStereo.filter_min_triangulation_angle",
                    str(float(patchmatch_relaxed_filter_min_tri_angle)),
                ]
            if patchmatch_relaxed_filter_min_ncc is not None:
                retry_args += ["--PatchMatchStereo.filter_min_ncc", str(float(patchmatch_relaxed_filter_min_ncc))]
            runner.run(retry_args)

            runner.run(fusion_geometric)
            vtx = _ply_vertex_count(dense_ply) if dense_ply.exists() else None
            if vtx is not None and vtx < int(dense_min_vertices):
                print(
                    f"[warn] {job.dataset}/{job.scene}: relaxed geometric fusion has {vtx} points; trying photometric fusion",
                    flush=True,
                )
                runner.run(fusion_photometric)
                vtx = _ply_vertex_count(dense_ply) if dense_ply.exists() else vtx

            if dense_ply.exists() and (vtx is None or vtx >= int(dense_min_vertices)):
                pts = f"{vtx}" if vtx is not None else "?"
                print(f"[ok] {job.dataset}/{job.scene}: wrote {dense_ply} (points={pts})")
                return "ok"

        vtx_msg = f"{vtx}" if vtx is not None else "?"
        print(f"[fail] {job.dataset}/{job.scene}: dense fusion produced too few points (points={vtx_msg})", flush=True)

        if depth_fallback and depth_args is not None and not dry_run:
            try:
                from imc25.recon.depth_fusion import build_dense_points_from_depth

                # Prefer undistorted workspace images + reconstruction for consistent intrinsics/poses.
                depth_images = dense_root / "images"
                depth_model_dir = dense_root / "sparse"
                use_images = depth_images if depth_images.exists() else images_dir
                use_model = depth_model_dir if depth_model_dir.exists() else None

                if dense_ply.exists():
                    runner.rm_f(dense_ply, dry_run=dry_run)

                pts = build_dense_points_from_depth(
                    images_dir=use_images,
                    out_ply=dense_ply,
                    model_dir=use_model,
                    depth_model_id=str(depth_args["model_id"]),
                    hf_endpoint=str(depth_args.get("hf_endpoint")) if depth_args.get("hf_endpoint") else None,
                    device=str(depth_args.get("device")) if depth_args.get("device") else None,
                    fp16=bool(depth_args.get("fp16", True)),
                    stride=int(depth_args.get("stride", 3)),
                    max_points=int(depth_args.get("max_points", 200_000)),
                    align_scale=bool(depth_args.get("align_scale", True)),
                    with_color=bool(depth_args.get("with_color", True)),
                    verbose=True,
                )
                if pts >= int(dense_min_vertices):
                    print(f"[ok] {job.dataset}/{job.scene}: wrote {dense_ply} via depth fallback (points={pts})", flush=True)
                    return "ok"
            except Exception as e:
                print(f"[warn] {job.dataset}/{job.scene}: depth fallback failed: {e}", flush=True)

        if dense_ply.exists() and not dry_run:
            runner.rm_f(dense_ply, dry_run=dry_run)
        last_status = "fail"
        continue

    return last_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COLMAP dense reconstruction for every cluster in clusters.csv")
    parser.add_argument("--clusters-csv", type=Path, default=Path("cache/clusters.csv"))
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--data-root", type=Path, default=Path("data/train"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--outliers-label", default="outliers")
    parser.add_argument("--min-images", type=int, default=4, help="Skip clusters smaller than this")
    parser.add_argument("--dataset", action="append", default=None, help="Only run for this dataset (repeatable)")
    parser.add_argument("--scene", action="append", default=None, help="Only run for this scene/cluster (repeatable)")
    parser.add_argument(
        "--runner",
        default="auto",
        choices=["auto", "local", "docker"],
        help="How to run COLMAP (auto tries local then docker)",
    )
    parser.add_argument("--docker-image", default="colmap/colmap:latest")
    parser.add_argument(
        "--docker-gpus",
        default="all",
        help="Passed to `docker run --gpus ...` when runner=docker (set empty to disable)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild outputs/<dataset>_<scene> if it exists")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running COLMAP")
    parser.add_argument("--camera-model", default="SIMPLE_RADIAL")
    parser.add_argument(
        "--single-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assume all images share intrinsics (recommended for quick demos)",
    )
    parser.add_argument("--patchmatch-gpu-index", type=int, default=0)
    parser.add_argument(
        "--patchmatch-min-tri-angle",
        type=float,
        default=0.5,
        help="COLMAP PatchMatch: minimum triangulation angle (degrees). Lower for tiny parallax scenes.",
    )
    parser.add_argument(
        "--patchmatch-filter-min-tri-angle",
        type=float,
        default=0.5,
        help="COLMAP PatchMatch: filter minimum triangulation angle (degrees). Lower for tiny parallax scenes.",
    )
    parser.add_argument(
        "--patchmatch-filter-min-ncc",
        type=float,
        default=0.1,
        help="COLMAP PatchMatch: filter minimum NCC. Lower to keep more points (may add noise).",
    )
    parser.add_argument(
        "--patchmatch-relax-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If fusion has too few points, rerun PatchMatch with even more permissive thresholds.",
    )
    parser.add_argument("--patchmatch-relaxed-min-tri-angle", type=float, default=0.1)
    parser.add_argument("--patchmatch-relaxed-filter-min-tri-angle", type=float, default=0.1)
    parser.add_argument("--patchmatch-relaxed-filter-min-ncc", type=float, default=0.05)
    parser.add_argument(
        "--image-mode",
        default="auto",
        choices=["auto", "symlink", "copy"],
        help="How to place images into outputs/<dataset>_<scene>/images (auto copies for matcher=colmap to avoid symlink issues)",
    )
    parser.add_argument(
        "--sparse-only",
        action="store_true",
        help="Stop after SfM (COLMAP mapper) and skip dense MVS (faster; useful for submission generation).",
    )
    parser.add_argument(
        "--mapper-min-model-size",
        type=int,
        default=3,
        help="COLMAP: minimum registered images for a model to be kept (default lowered for small clusters)",
    )
    parser.add_argument(
        "--mapper-init-min-num-inliers",
        type=int,
        default=15,
        help="COLMAP: minimum inliers for initial pair (default lowered for small clusters)",
    )
    parser.add_argument(
        "--mapper-min-num-matches",
        type=int,
        default=None,
        help="COLMAP: minimum matches per image pair (defaults to --mapper-init-min-num-inliers)",
    )
    parser.add_argument(
        "--mapper-init-min-tri-angle",
        type=float,
        default=2.0,
        help="COLMAP: minimum triangulation angle for initial pair in degrees (default lowered for small-baseline scenes)",
    )
    parser.add_argument(
        "--mapper-tri-min-angle",
        type=float,
        default=1.5,
        help="COLMAP: minimum triangulation angle for creating points (lower for tiny parallax)",
    )
    parser.add_argument(
        "--mapper-filter-min-tri-angle",
        type=float,
        default=1.5,
        help="COLMAP: minimum triangulation angle for filtering (lower for tiny parallax)",
    )
    parser.add_argument(
        "--mapper-local-ba-min-tri-angle",
        type=float,
        default=6.0,
        help="COLMAP: minimum triangulation angle for BA (lower for tiny parallax)",
    )
    parser.add_argument(
        "--mapper-disable-local-ba-min-tri-angle",
        action="store_true",
        help="Disable --Mapper.local_ba_min_tri_angle for older COLMAP builds",
    )
    parser.add_argument(
        "--mapper-tri-ignore-two-view-tracks",
        type=int,
        default=1,
        choices=[0, 1],
        help="COLMAP: ignore two-view tracks (set 0 to allow recon with only two connected images)",
    )
    parser.add_argument(
        "--matcher",
        default="colmap",
        choices=["colmap", "dino", "diffusion", "auto"],
        help="How to populate colmap.db (colmap=SIFT, dino=DINOv2 patch descriptors + RANSAC, diffusion=diffusion features + RANSAC, auto=try colmap then dino then diffusion)",
    )
    parser.add_argument(
        "--dense-min-vertices",
        type=int,
        default=1,
        help="Treat dense reconstruction as failed if dense_points.ply has fewer than this many points (0 often indicates a broken workspace).",
    )

    parser.add_argument("--diffusion-cache-dir", type=Path, default=Path("cache/diffusion_features"))
    parser.add_argument("--diffusion-model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--diffusion-timestep", type=int, default=200)
    parser.add_argument("--diffusion-max-side", type=int, default=512)
    parser.add_argument("--diffusion-noise-seed", type=int, default=0)
    parser.add_argument("--diffusion-layers", default="d0,d1,d2,mid")
    parser.add_argument("--diffusion-fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diffusion-device", default=None, help="Override diffusion device (e.g. cuda, cuda:0, cpu)")
    parser.add_argument("--diffusion-hf-endpoint", default=None, help="Hugging Face Hub endpoint (mirror), e.g. https://hf-mirror.com")
    parser.add_argument("--diffusion-keypoints", choices=["gftt", "grid"], default="gftt")
    parser.add_argument("--diffusion-max-keypoints", type=int, default=1024)
    parser.add_argument("--diffusion-min-similarity", type=float, default=0.7)
    parser.add_argument("--diffusion-max-matches", type=int, default=4096)
    parser.add_argument("--diffusion-ransac-thresh-px", type=float, default=1.0)
    parser.add_argument("--diffusion-min-inliers", type=int, default=15)
    parser.add_argument("--diffusion-pairing", choices=["auto", "exhaustive", "topk"], default="auto")
    parser.add_argument("--diffusion-topk", type=int, default=10)

    parser.add_argument("--dino-cache-dir", type=Path, default=Path("cache/dino_features"))
    parser.add_argument("--dino-model-id", default="facebook/dinov2-small")
    parser.add_argument("--dino-max-side", type=int, default=512)
    parser.add_argument("--dino-layer", type=int, default=None, help="Use hidden_states[layer] instead of last_hidden_state")
    parser.add_argument("--dino-fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dino-device", default=None, help="Override DINO device (e.g. cuda, cuda:0, cpu)")
    parser.add_argument("--dino-hf-endpoint", default=None, help="Hugging Face Hub endpoint (mirror), e.g. https://hf-mirror.com")
    parser.add_argument("--dino-keypoints", choices=["patch", "gftt"], default="patch")
    parser.add_argument("--dino-max-keypoints", type=int, default=2048)
    parser.add_argument("--dino-min-similarity", type=float, default=0.75)
    parser.add_argument("--dino-max-matches", type=int, default=4096)
    parser.add_argument("--dino-ransac-thresh-px", type=float, default=1.5)
    parser.add_argument("--dino-min-inliers", type=int, default=15)
    parser.add_argument("--dino-pairing", choices=["auto", "exhaustive", "topk"], default="auto")
    parser.add_argument("--dino-topk", type=int, default=10)

    parser.add_argument(
        "--depth-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If dense fusion is empty/tiny, generate a dense pointcloud via monocular depth estimation.",
    )
    parser.add_argument("--depth-model-id", default="Intel/dpt-hybrid-midas", help="Depth model on Hugging Face Hub")
    parser.add_argument("--depth-hf-endpoint", default=None, help="Hugging Face Hub endpoint (mirror), e.g. https://hf-mirror.com")
    parser.add_argument("--depth-device", default=None, help="Override depth device (e.g. cuda, cuda:0, cpu)")
    parser.add_argument("--depth-fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--depth-align-scale", action=argparse.BooleanOptionalAction, default=True, help="Align depth scale to SfM sparse points when available")
    parser.add_argument("--depth-with-color", action=argparse.BooleanOptionalAction, default=True, help="Store RGB colors in the PLY")
    parser.add_argument("--depth-stride", type=int, default=3, help="Pixel stride when backprojecting depth (lower = denser, slower)")
    parser.add_argument("--depth-max-points", type=int, default=200_000, help="Max points written to dense_points.ply (random subsample)")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    clusters_csv = args.clusters_csv
    if not clusters_csv.exists():
        raise SystemExit(f"clusters CSV not found: {clusters_csv}")

    df = _read_clusters(clusters_csv)
    dataset_filter = set(args.dataset) if args.dataset else None
    scene_filter = set(args.scene) if args.scene else None

    runner_mode = _auto_runner_mode(args.runner)
    docker_gpus = args.docker_gpus.strip() if isinstance(args.docker_gpus, str) else ""
    docker_gpus = docker_gpus if docker_gpus else None
    runner = ColmapRunner(
        repo_root=repo_root,
        mode=runner_mode,
        docker_image=args.docker_image,
        docker_gpus=docker_gpus,
    )

    camera_model = str(args.camera_model)
    single_camera = bool(args.single_camera)
    patchmatch_gpu_index = int(args.patchmatch_gpu_index) if args.patchmatch_gpu_index is not None else None
    patchmatch_min_tri_angle = float(args.patchmatch_min_tri_angle) if args.patchmatch_min_tri_angle is not None else None
    patchmatch_filter_min_tri_angle = (
        float(args.patchmatch_filter_min_tri_angle) if args.patchmatch_filter_min_tri_angle is not None else None
    )
    patchmatch_filter_min_ncc = float(args.patchmatch_filter_min_ncc) if args.patchmatch_filter_min_ncc is not None else None
    patchmatch_relax_on_failure = bool(args.patchmatch_relax_on_failure)
    patchmatch_relaxed_min_tri_angle = (
        float(args.patchmatch_relaxed_min_tri_angle) if args.patchmatch_relaxed_min_tri_angle is not None else None
    )
    patchmatch_relaxed_filter_min_tri_angle = (
        float(args.patchmatch_relaxed_filter_min_tri_angle) if args.patchmatch_relaxed_filter_min_tri_angle is not None else None
    )
    patchmatch_relaxed_filter_min_ncc = (
        float(args.patchmatch_relaxed_filter_min_ncc) if args.patchmatch_relaxed_filter_min_ncc is not None else None
    )
    image_mode = str(args.image_mode)
    matcher = str(args.matcher)
    mapper_min_model_size = int(args.mapper_min_model_size)
    mapper_min_num_matches = int(args.mapper_min_num_matches) if args.mapper_min_num_matches is not None else None
    mapper_init_min_num_inliers = int(args.mapper_init_min_num_inliers)
    mapper_init_min_tri_angle = float(args.mapper_init_min_tri_angle)
    mapper_tri_min_angle = float(args.mapper_tri_min_angle)
    mapper_filter_min_tri_angle = float(args.mapper_filter_min_tri_angle)
    mapper_local_ba_min_tri_angle = float(args.mapper_local_ba_min_tri_angle)
    mapper_tri_ignore_two_view_tracks = int(args.mapper_tri_ignore_two_view_tracks)
    mapper_disable_local_ba_min_tri_angle = bool(args.mapper_disable_local_ba_min_tri_angle)

    if runner_mode == "docker" and not mapper_disable_local_ba_min_tri_angle:
        # Some Docker COLMAP images don't support this flag; auto-disable to avoid
        # failing the entire batch with "unrecognised option".
        help_out = runner.run_capture(["mapper", "--help"])
        if "Mapper.local_ba_min_tri_angle" not in help_out:
            mapper_disable_local_ba_min_tri_angle = True
            print("[info] Docker COLMAP does not support --Mapper.local_ba_min_tri_angle; disabling it.", flush=True)

    # PatchMatch flags are stable in modern COLMAP, but some images can be older. Auto-disable if not supported.
    if runner_mode == "docker":
        pm_help = runner.run_capture(["patch_match_stereo", "--help"])
        if "PatchMatchStereo.min_triangulation_angle" not in pm_help:
            patchmatch_min_tri_angle = None
            patchmatch_relaxed_min_tri_angle = None
            print("[info] Docker COLMAP does not support --PatchMatchStereo.min_triangulation_angle; disabling it.", flush=True)
        if "PatchMatchStereo.filter_min_triangulation_angle" not in pm_help:
            patchmatch_filter_min_tri_angle = None
            patchmatch_relaxed_filter_min_tri_angle = None
            print(
                "[info] Docker COLMAP does not support --PatchMatchStereo.filter_min_triangulation_angle; disabling it.",
                flush=True,
            )
        if "PatchMatchStereo.filter_min_ncc" not in pm_help:
            patchmatch_filter_min_ncc = None
            patchmatch_relaxed_filter_min_ncc = None
            print("[info] Docker COLMAP does not support --PatchMatchStereo.filter_min_ncc; disabling it.", flush=True)

    diffusion_args = None
    if matcher in {"diffusion", "auto"}:
        diffusion_min_inliers = int(args.diffusion_min_inliers)
        default_diffusion_min_inliers = int(parser.get_default("diffusion_min_inliers"))
        default_mapper_init_min_inliers = int(parser.get_default("mapper_init_min_num_inliers"))
        if (
            diffusion_min_inliers == default_diffusion_min_inliers
            and mapper_init_min_num_inliers != default_mapper_init_min_inliers
        ):
            diffusion_min_inliers = mapper_init_min_num_inliers

        diffusion_args = {
            "cache_dir": str(args.diffusion_cache_dir) if args.diffusion_cache_dir else None,
            "model_id": str(args.diffusion_model_id),
            "timestep": int(args.diffusion_timestep),
            "max_side": int(args.diffusion_max_side),
            "noise_seed": int(args.diffusion_noise_seed),
            "layers": tuple([s.strip() for s in str(args.diffusion_layers).split(",") if s.strip()]),
            "fp16": bool(args.diffusion_fp16),
            "device": str(args.diffusion_device) if args.diffusion_device else None,
            "hf_endpoint": str(args.diffusion_hf_endpoint) if args.diffusion_hf_endpoint else None,
            "keypoints": str(args.diffusion_keypoints),
            "max_keypoints": int(args.diffusion_max_keypoints),
            "min_similarity": float(args.diffusion_min_similarity),
            "max_matches": int(args.diffusion_max_matches),
            "ransac_thresh_px": float(args.diffusion_ransac_thresh_px),
            "min_inliers": int(diffusion_min_inliers),
            "pairing": str(args.diffusion_pairing),
            "topk": int(args.diffusion_topk),
        }

    dino_args = None
    if matcher in {"dino", "auto"}:
        dino_min_inliers = int(args.dino_min_inliers)
        default_dino_min_inliers = int(parser.get_default("dino_min_inliers"))
        default_mapper_init_min_inliers = int(parser.get_default("mapper_init_min_num_inliers"))
        if dino_min_inliers == default_dino_min_inliers and mapper_init_min_num_inliers != default_mapper_init_min_inliers:
            dino_min_inliers = mapper_init_min_num_inliers

        dino_args = {
            "cache_dir": str(args.dino_cache_dir) if args.dino_cache_dir else None,
            "model_id": str(args.dino_model_id),
            "max_side": int(args.dino_max_side),
            "layer": int(args.dino_layer) if args.dino_layer is not None else None,
            "fp16": bool(args.dino_fp16),
            "device": str(args.dino_device) if args.dino_device else None,
            "hf_endpoint": str(args.dino_hf_endpoint) if args.dino_hf_endpoint else None,
            "keypoints": str(args.dino_keypoints),
            "max_keypoints": int(args.dino_max_keypoints),
            "min_similarity": float(args.dino_min_similarity),
            "max_matches": int(args.dino_max_matches),
            "ransac_thresh_px": float(args.dino_ransac_thresh_px),
            "min_inliers": int(dino_min_inliers),
            "pairing": str(args.dino_pairing),
            "topk": int(args.dino_topk),
        }

    depth_args = None
    if bool(args.depth_fallback):
        depth_args = {
            "model_id": str(args.depth_model_id),
            "hf_endpoint": str(args.depth_hf_endpoint) if args.depth_hf_endpoint else None,
            "device": str(args.depth_device) if args.depth_device else None,
            "fp16": bool(args.depth_fp16),
            "align_scale": bool(args.depth_align_scale),
            "with_color": bool(args.depth_with_color),
            "stride": int(args.depth_stride),
            "max_points": int(args.depth_max_points),
        }

    jobs = list(
        _iter_jobs(
            df,
            output_root=args.output_root,
            outliers_label=args.outliers_label,
            min_images=int(args.min_images),
            dataset_filter=dataset_filter,
            scene_filter=scene_filter,
        )
    )
    if not jobs:
        print("No clusters to run (check filters/min-images/outliers-label).")
        return

    print(f"Runner: {runner_mode}")
    print(f"Clusters: {len(jobs)}")
    if matcher == "diffusion" and diffusion_args is not None and not bool(args.dry_run):
        if diffusion_args.get("hf_endpoint"):
            os.environ["HF_ENDPOINT"] = str(diffusion_args["hf_endpoint"])
        from imc25.matching.diffusion_features import DiffusionFeatureConfig, DiffusionUNetFeatureExtractor
        from imc25.matching.diffusion_matcher import KeypointConfig, MatchConfig
        from imc25.matching.diffusion_to_colmap import PairingConfig

        feat_cfg = DiffusionFeatureConfig(
            model_id=str(diffusion_args["model_id"]),
            timestep=int(diffusion_args["timestep"]),
            max_side=int(diffusion_args["max_side"]),
            noise_seed=int(diffusion_args["noise_seed"]),
            layers=tuple(diffusion_args["layers"]),
            use_fp16=bool(diffusion_args["fp16"]),
        )
        kp_cfg = KeypointConfig(method=str(diffusion_args["keypoints"]), max_keypoints=int(diffusion_args["max_keypoints"]))
        match_cfg = MatchConfig(
            min_similarity=float(diffusion_args["min_similarity"]),
            max_matches=int(diffusion_args["max_matches"]),
            ransac_thresh_px=float(diffusion_args["ransac_thresh_px"]),
            min_inliers=int(diffusion_args["min_inliers"]),
        )
        pairing_cfg = PairingConfig(mode=str(diffusion_args["pairing"]), topk=int(diffusion_args["topk"]))

        print("[diffusion] preloading model once for all clusters...", flush=True)
        extractor = DiffusionUNetFeatureExtractor(feat_cfg, device=str(diffusion_args["device"]) if diffusion_args.get("device") else None)
        diffusion_args["feat_cfg"] = feat_cfg
        diffusion_args["kp_cfg"] = kp_cfg
        diffusion_args["match_cfg"] = match_cfg
        diffusion_args["pairing_cfg"] = pairing_cfg
        diffusion_args["extractor"] = extractor

    failures: list[str] = []
    ok = 0
    skipped = 0
    for i, job in enumerate(jobs, start=1):
        print(f"[cluster] {i}/{len(jobs)}: {job.dataset}/{job.scene}", flush=True)
        try:
            status = run_dense_for_cluster(
                job,
                repo_root=repo_root,
                data_root=args.data_root,
                cache_root=args.cache_root,
                runner=runner,
                overwrite=bool(args.overwrite),
                dry_run=bool(args.dry_run),
                camera_model=camera_model,
                single_camera=single_camera,
                patchmatch_gpu_index=patchmatch_gpu_index,
                matcher=matcher,
                image_mode=image_mode,
                dino_args=dino_args,
                diffusion_args=diffusion_args,
                dense_min_vertices=int(args.dense_min_vertices),
                depth_fallback=bool(args.depth_fallback),
                depth_args=depth_args,
                sparse_only=bool(args.sparse_only),
                mapper_min_model_size=mapper_min_model_size,
                mapper_min_num_matches=mapper_min_num_matches,
                mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                mapper_init_min_tri_angle=mapper_init_min_tri_angle,
                mapper_tri_min_angle=mapper_tri_min_angle,
                mapper_filter_min_tri_angle=mapper_filter_min_tri_angle,
                mapper_local_ba_min_tri_angle=mapper_local_ba_min_tri_angle,
                mapper_tri_ignore_two_view_tracks=mapper_tri_ignore_two_view_tracks,
                mapper_disable_local_ba_min_tri_angle=mapper_disable_local_ba_min_tri_angle,
                patchmatch_min_tri_angle=patchmatch_min_tri_angle,
                patchmatch_filter_min_tri_angle=patchmatch_filter_min_tri_angle,
                patchmatch_filter_min_ncc=patchmatch_filter_min_ncc,
                patchmatch_relax_on_failure=patchmatch_relax_on_failure,
                patchmatch_relaxed_min_tri_angle=patchmatch_relaxed_min_tri_angle,
                patchmatch_relaxed_filter_min_tri_angle=patchmatch_relaxed_filter_min_tri_angle,
                patchmatch_relaxed_filter_min_ncc=patchmatch_relaxed_filter_min_ncc,
            )
            if status == "ok":
                ok += 1
            elif status.startswith("skip"):
                skipped += 1
            else:
                failures.append(f"{job.dataset}/{job.scene}")
        except Exception as e:
            msg = f"{job.dataset}/{job.scene}: {type(e).__name__}: {e}"
            failures.append(msg)
            print(f"[error] {msg}", file=sys.stderr, flush=True)
            continue

    if failures:
        print(f"[done] ok={ok} skipped={skipped} failed={len(failures)}", flush=True)
        for msg in failures:
            print(f"[fail] {msg}", flush=True)
    else:
        print(f"[done] ok={ok} skipped={skipped} failed=0", flush=True)


if __name__ == "__main__":
    main()
