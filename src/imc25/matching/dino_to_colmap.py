from __future__ import annotations

import argparse
import itertools
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from imc25.matching.colmap_db import CameraSpec, create_empty_colmap_db, guess_simple_radial, insert_camera
from imc25.matching.colmap_db import insert_dummy_descriptors, insert_image, insert_keypoints, insert_matches, insert_two_view_geometry
from imc25.matching.dino_features import DinoFeatureConfig, DinoPatchFeatureExtractor
from imc25.matching.dino_matcher import KeypointConfig, MatchConfig, extract_image_features, match_pair


@dataclass(frozen=True)
class PairingConfig:
    mode: str = "auto"  # auto | exhaustive | topk
    topk: int = 10


def _iter_images(images_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    return [p for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def _camera_from_model(model: str, *, width: int, height: int) -> CameraSpec:
    model = model.strip().upper()
    if model == "SIMPLE_RADIAL":
        return guess_simple_radial(width, height)
    if model == "SIMPLE_PINHOLE":
        f = 1.2 * float(max(width, height))
        cx = float(width) / 2.0
        cy = float(height) / 2.0
        params = np.array([f, cx, cy], dtype=np.float64)
        return CameraSpec(model="SIMPLE_PINHOLE", width=width, height=height, params=params, prior_focal_length=0)
    if model == "PINHOLE":
        f = 1.2 * float(max(width, height))
        cx = float(width) / 2.0
        cy = float(height) / 2.0
        params = np.array([f, f, cx, cy], dtype=np.float64)
        return CameraSpec(model="PINHOLE", width=width, height=height, params=params, prior_focal_length=0)
    raise ValueError(f"Unsupported camera model for DINO DB builder: {model}")


def _K_from_camera(camera: CameraSpec) -> np.ndarray:
    p = np.asarray(camera.params, dtype=np.float64).reshape(-1)
    m = camera.model.upper()
    if m in {"SIMPLE_RADIAL", "RADIAL", "SIMPLE_PINHOLE"}:
        f, cx, cy = float(p[0]), float(p[1]), float(p[2])
        fx, fy = f, f
    elif m in {"PINHOLE", "OPENCV", "FULL_OPENCV"}:
        fx, fy, cx, cy = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    else:
        fx = fy = 1.2 * float(max(camera.width, camera.height))
        cx = float(camera.width) / 2.0
        cy = float(camera.height) / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rank2(E: np.ndarray) -> np.ndarray:
    U, S, Vt = np.linalg.svd(E)
    S[-1] = 0.0
    return (U @ np.diag(S) @ Vt).astype(np.float64)


def _global_hist_descriptor(image_path: Path, *, size: int = 160, bins: int = 16) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    hists = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 256), density=False)
        hists.append(h.astype(np.float32))
    v = np.concatenate(hists, axis=0)
    v /= np.linalg.norm(v) + 1e-6
    return v


def _iter_pairs(image_paths: list[Path], *, pairing: PairingConfig) -> Iterable[tuple[int, int]]:
    n = len(image_paths)
    mode = pairing.mode.lower().strip()
    if mode == "auto":
        mode = "exhaustive" if n <= 20 else "topk"

    if mode == "exhaustive":
        yield from itertools.combinations(range(n), 2)
        return

    if mode != "topk":
        raise ValueError(f"Unknown pairing mode: {pairing.mode}")

    topk = int(max(1, pairing.topk))
    descs = np.stack([_global_hist_descriptor(p) for p in image_paths], axis=0)
    sims = descs @ descs.T
    np.fill_diagonal(sims, -1.0)

    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        nn = np.argsort(-sims[i])[:topk]
        for j in nn.tolist():
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))
    for a, b in sorted(pairs):
        yield a, b


def build_colmap_db_from_dino(
    *,
    images_dir: Path,
    db_path: Path,
    overwrite: bool,
    camera_model: str,
    single_camera: bool,
    feat_cfg: DinoFeatureConfig,
    kp_cfg: KeypointConfig,
    match_cfg: MatchConfig,
    pairing_cfg: PairingConfig,
    cache_dir: Path | None,
    dino_device: str | None = None,
    hf_endpoint: str | None = None,
    extractor: DinoPatchFeatureExtractor | None = None,
    verbose: bool = True,
) -> None:
    image_paths = _iter_images(images_dir)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = str(hf_endpoint)
        if verbose:
            print(f"[dino] HF_ENDPOINT={os.environ['HF_ENDPOINT']}", flush=True)

    if verbose:
        print(
            f"[dino] build db={db_path} images={len(image_paths)} pairing={pairing_cfg.mode} topk={pairing_cfg.topk}",
            flush=True,
        )
        if cache_dir is not None:
            print(f"[dino] cache_dir={cache_dir}", flush=True)
        print(
            f"[dino] model={feat_cfg.model_id} layer={feat_cfg.layer if feat_cfg.layer is not None else 'last'} "
            f"max_side={feat_cfg.max_side} fp16={feat_cfg.use_fp16}",
            flush=True,
        )
        print(
            f"[dino] matcher: keypoints={kp_cfg.method} max_keypoints={kp_cfg.max_keypoints} "
            f"min_similarity={match_cfg.min_similarity} min_inliers={match_cfg.min_inliers}",
            flush=True,
        )

    create_empty_colmap_db(db_path, overwrite=overwrite)

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        sizes = []
        for p in image_paths:
            w, h = Image.open(p).size
            sizes.append((w, h))
        unique_sizes = sorted(set(sizes))

        cameras_by_index: list[int] = []
        camera_specs_by_id: dict[int, CameraSpec] = {}
        if single_camera and len(unique_sizes) != 1:
            print(f"[warn] --single-camera requested but image sizes differ {unique_sizes}; using per-image cameras.")
            single_camera = False

        if single_camera:
            w, h = unique_sizes[0]
            cam = _camera_from_model(camera_model, width=w, height=h)
            cam_id = insert_camera(conn, cam)
            camera_specs_by_id[cam_id] = cam
            cameras_by_index = [cam_id for _ in image_paths]
        else:
            for (w, h) in sizes:
                cam = _camera_from_model(camera_model, width=w, height=h)
                cam_id = insert_camera(conn, cam)
                camera_specs_by_id[cam_id] = cam
                cameras_by_index.append(cam_id)

        image_ids: list[int] = []
        image_id_by_name: dict[str, int] = {}
        for path, cam_id in zip(image_paths, cameras_by_index, strict=False):
            image_id = insert_image(conn, name=path.name, camera_id=cam_id)
            image_ids.append(image_id)
            image_id_by_name[path.name] = image_id

        if extractor is None:
            if verbose:
                print("[dino] loading DINO model (first run may download from Hugging Face)...", flush=True)
            extractor = DinoPatchFeatureExtractor(feat_cfg, device=dino_device, hf_endpoint=hf_endpoint)
            if verbose:
                print(f"[dino] model loaded on device={extractor.device} patch={extractor.patch_size}", flush=True)
        else:
            if extractor.config != feat_cfg:
                raise ValueError("Provided extractor config does not match feat_cfg")
            if verbose:
                print(f"[dino] using preloaded model on device={extractor.device} patch={extractor.patch_size}", flush=True)

        feats_by_name: dict[str, object] = {}
        for i, p in enumerate(image_paths, start=1):
            if verbose:
                print(f"[dino] image {i}/{len(image_paths)}: {p.name}", flush=True)
            feats = extract_image_features(p, extractor=extractor, keypoints=kp_cfg, cache_dir=cache_dir, verbose=verbose)
            feats_by_name[p.name] = feats
            image_id = image_id_by_name[p.name]
            insert_keypoints(conn, image_id=image_id, keypoints_xy=feats.keypoints_xy)
            insert_dummy_descriptors(conn, image_id=image_id, num_keypoints=int(feats.keypoints_xy.shape[0]))

        pairs = list(_iter_pairs(image_paths, pairing=pairing_cfg))
        if verbose:
            print(f"[dino] matching pairs={len(pairs)} (RANSAC)...", flush=True)

        pair_count = 0
        good_pairs = 0
        for idx, (ia, ib) in enumerate(pairs, start=1):
            name_a = image_paths[ia].name
            name_b = image_paths[ib].name
            fa = feats_by_name[name_a]
            fb = feats_by_name[name_b]
            pair_count += 1

            r = match_pair(fa, fb, match_cfg=match_cfg, device=extractor.device)
            if r.F is None:
                if verbose and (len(pairs) <= 20 or idx % 25 == 0):
                    print(f"[dino] pair {idx}/{len(pairs)}: {name_a} ↔ {name_b} (no F)", flush=True)
                continue
            if r.inlier_matches.shape[0] < int(match_cfg.min_inliers):
                if verbose and (len(pairs) <= 20 or idx % 25 == 0):
                    print(
                        f"[dino] pair {idx}/{len(pairs)}: {name_a} ↔ {name_b} "
                        f"(inliers={r.inlier_matches.shape[0]} < min_inliers={match_cfg.min_inliers})",
                        flush=True,
                    )
                continue
            if verbose and (len(pairs) <= 20 or idx % 25 == 0):
                print(f"[dino] pair {idx}/{len(pairs)}: {name_a} ↔ {name_b} (inliers={r.inlier_matches.shape[0]})", flush=True)

            image_id1 = image_id_by_name[name_a]
            image_id2 = image_id_by_name[name_b]
            insert_matches(conn, image_id1=image_id1, image_id2=image_id2, matches=r.inlier_matches)

            cam1 = camera_specs_by_id[cameras_by_index[ia]]
            cam2 = camera_specs_by_id[cameras_by_index[ib]]
            K1 = _K_from_camera(cam1)
            K2 = _K_from_camera(cam2)
            E = _rank2(K2.T @ r.F @ K1)

            insert_two_view_geometry(
                conn,
                image_id1=image_id1,
                image_id2=image_id2,
                inlier_matches=r.inlier_matches,
                config=3,  # uncalibrated / fundamental
                F_mat=r.F,
                E_mat=E,
                H_mat=None,
                qvec=None,
                tvec=None,
            )
            good_pairs += 1

        print(
            f"[ok] wrote {db_path} (images={len(image_paths)} pairs={pair_count} "
            f"good_pairs={good_pairs}, min_inliers={match_cfg.min_inliers})",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a COLMAP database using DINOv2 patch descriptors + RANSAC inliers")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/dino_features"))
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda, cuda:0, cpu)")
    parser.add_argument("--hf-endpoint", default=None, help="Hugging Face Hub endpoint (mirror), e.g. https://hf-mirror.com")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--camera-model", default="SIMPLE_RADIAL")
    parser.add_argument("--single-camera", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--model-id", default=DinoFeatureConfig.model_id)
    parser.add_argument("--max-side", type=int, default=DinoFeatureConfig.max_side)
    parser.add_argument("--layer", type=int, default=None, help="Use transformer hidden_states[layer] instead of last_hidden_state")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--keypoints", default="patch", choices=["patch", "gftt"])
    parser.add_argument("--max-keypoints", type=int, default=KeypointConfig.max_keypoints)
    parser.add_argument("--min-similarity", type=float, default=MatchConfig.min_similarity)
    parser.add_argument("--max-matches", type=int, default=MatchConfig.max_matches)
    parser.add_argument("--ransac-thresh-px", type=float, default=MatchConfig.ransac_thresh_px)
    parser.add_argument("--min-inliers", type=int, default=MatchConfig.min_inliers)

    parser.add_argument("--pairing", default="auto", choices=["auto", "exhaustive", "topk"])
    parser.add_argument("--topk", type=int, default=PairingConfig.topk)
    args = parser.parse_args()

    feat_cfg = DinoFeatureConfig(
        model_id=str(args.model_id),
        max_side=int(args.max_side),
        layer=int(args.layer) if args.layer is not None else None,
        use_fp16=bool(args.fp16),
    )
    kp_cfg = KeypointConfig(method=str(args.keypoints), max_keypoints=int(args.max_keypoints))
    match_cfg = MatchConfig(
        min_similarity=float(args.min_similarity),
        max_matches=int(args.max_matches),
        ransac_thresh_px=float(args.ransac_thresh_px),
        min_inliers=int(args.min_inliers),
    )
    pairing_cfg = PairingConfig(mode=str(args.pairing), topk=int(args.topk))
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    build_colmap_db_from_dino(
        images_dir=args.images_dir,
        db_path=args.db_path,
        overwrite=bool(args.overwrite),
        camera_model=str(args.camera_model),
        single_camera=bool(args.single_camera),
        feat_cfg=feat_cfg,
        kp_cfg=kp_cfg,
        match_cfg=match_cfg,
        pairing_cfg=pairing_cfg,
        cache_dir=cache_dir,
        dino_device=str(args.device) if args.device else None,
        hf_endpoint=str(args.hf_endpoint) if args.hf_endpoint else None,
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main()

