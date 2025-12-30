from __future__ import annotations

import argparse
import itertools
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import cv2  # Required for geometric verification
from PIL import Image

# --- NEW IMPORTS: LightGlue & ALIKED ---
from lightglue import ALIKED, LightGlue
from lightglue.utils import load_image, rbd

from imc25.matching.colmap_db import (
    CameraSpec, create_empty_colmap_db, guess_simple_radial, insert_camera,
    insert_dummy_descriptors, insert_image, insert_keypoints, insert_matches, insert_two_view_geometry
)

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
    raise ValueError(f"Unsupported camera model: {model}")

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

def build_colmap_db_aliked(
    *,
    images_dir: Path,
    db_path: Path,
    overwrite: bool,
    camera_model: str,
    single_camera: bool,
    max_keypoints: int,
    min_inliers: int,
    pairing_cfg: PairingConfig,
    device_str: str | None = None,
    verbose: bool = True,
) -> None:
    image_paths = _iter_images(images_dir)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    # --- 1. Initialize ALIKED & LightGlue ---
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    if verbose:
        print(f"[aliked] Loading models on {device} (max_kpts={max_keypoints})...", flush=True)

    # detection_threshold adjusted for high recall
    extractor = ALIKED(max_num_keypoints=max_keypoints, detection_threshold=0.01).eval().to(device)
    matcher = LightGlue(features='aliked').eval().to(device)

    # --- 2. Setup Database ---
    create_empty_colmap_db(db_path, overwrite=overwrite)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        # --- 3. Camera & Image Registration ---
        sizes = []
        for p in image_paths:
            with Image.open(p) as img:
                sizes.append(img.size) # (w, h)
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

        # --- 4. Extract Features (ALIKED) ---
        # We store features in a CPU dictionary to save GPU VRAM
        feats_cache = {} 
        
        if verbose:
            print(f"[aliked] Extracting features for {len(image_paths)} images...", flush=True)

        for i, p in enumerate(image_paths, start=1):
            # Load image (LightGlue utils handles resizing/normalization)
            image_tensor = load_image(p).to(device)
            
            with torch.inference_mode():
                feats = extractor.extract(image_tensor)
            
            # Move to CPU for storage
            feats_cpu = {k: v.cpu() for k, v in feats.items()}
            feats_cache[p.name] = feats_cpu
            
            # Insert into DB immediately
            kpts = feats_cpu['keypoints'][0].numpy()
            image_id = image_id_by_name[p.name]
            insert_keypoints(conn, image_id=image_id, keypoints_xy=kpts)
            # Dummy descriptors required by some COLMAP versions/viewers
            insert_dummy_descriptors(conn, image_id=image_id, num_keypoints=len(kpts))
            
            if verbose and i % 10 == 0:
                print(f"[aliked] Extracted {i}/{len(image_paths)}: {p.name}", flush=True)

        # --- 5. Match Pairs (LightGlue) ---
        pairs = list(_iter_pairs(image_paths, pairing=pairing_cfg))
        if verbose:
            print(f"[aliked] Matching {len(pairs)} pairs (LightGlue + MAGSAC)...", flush=True)

        good_pairs = 0
        for idx, (ia, ib) in enumerate(pairs, start=1):
            name_a = image_paths[ia].name
            name_b = image_paths[ib].name
            
            # Move features back to GPU for matching
            feats0 = {k: v.to(device) for k, v in feats_cache[name_a].items()}
            feats1 = {k: v.to(device) for k, v in feats_cache[name_b].items()}

            with torch.inference_mode():
                matches01 = matcher({'image0': feats0, 'image1': feats1})
            
            # Remove batch dim & get indices
            matches_dict = rbd(matches01)
            match_idx = matches_dict['matches'].cpu().numpy()

            if len(match_idx) < min_inliers:
                continue

            # Geometric Verification (MAGSAC++)
            kpts0 = feats_cache[name_a]['keypoints'][0].numpy()
            kpts1 = feats_cache[name_b]['keypoints'][0].numpy()
            
            p0 = kpts0[match_idx[:, 0]]
            p1 = kpts1[match_idx[:, 1]]
            
            # USAC_MAGSAC is faster/better than standard RANSAC
            F, inliers = cv2.findFundamentalMat(p0, p1, cv2.USAC_MAGSAC, 1.0, 0.999, 10000)
            
            if F is None or inliers.sum() < min_inliers:
                continue
                
            # Filter matches
            good_matches = match_idx[inliers.ravel() == 1]
            
            # Insert into DB
            image_id1 = image_id_by_name[name_a]
            image_id2 = image_id_by_name[name_b]
            
            # 1. Insert Matches (Use keyword arguments!)
            insert_matches(conn, image_id1=image_id1, image_id2=image_id2, matches=good_matches)
            
            # 2. Calc Essential Matrix
            cam1 = camera_specs_by_id[cameras_by_index[ia]]
            cam2 = camera_specs_by_id[cameras_by_index[ib]]
            K1 = _K_from_camera(cam1)
            K2 = _K_from_camera(cam2)
            E = _rank2(K2.T @ F @ K1)
            
            # 3. Insert Geometry (Use keyword arguments!)
            insert_two_view_geometry(
                conn, 
                image_id1=image_id1, 
                image_id2=image_id2, 
                inlier_matches=good_matches, 
                config=3,
                F_mat=F,
                E_mat=E,
                H_mat=None,
                qvec=None,
                tvec=None
            )
            
            good_pairs += 1
            if verbose and (idx % 25 == 0 or idx == len(pairs)):
                print(f"[aliked] Pair {idx}/{len(pairs)}: {name_a} <-> {name_b} ({len(good_matches)} inliers)", flush=True)

        print(f"[done] Database built: {db_path} (Images: {len(image_paths)}, Pairs: {len(pairs)}, Valid: {good_pairs})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a COLMAP database using ALIKED + LightGlue")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    
    # Kept arguments generic so your CLI calls still work
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda, cuda:0, cpu)")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--camera-model", default="SIMPLE_RADIAL")
    parser.add_argument("--single-camera", action=argparse.BooleanOptionalAction, default=True)

    # ALIKED Params
    parser.add_argument("--max-keypoints", type=int, default=2048)
    
    # Matching Params
    parser.add_argument("--min-inliers", type=int, default=15)
    
    # Pairing Params
    parser.add_argument("--pairing", default="auto", choices=["auto", "exhaustive", "topk"])
    parser.add_argument("--topk", type=int, default=10)
    
    args = parser.parse_args()

    pairing_cfg = PairingConfig(mode=str(args.pairing), topk=int(args.topk))

    build_colmap_db_aliked(
        images_dir=args.images_dir,
        db_path=args.db_path,
        overwrite=bool(args.overwrite),
        camera_model=str(args.camera_model),
        single_camera=bool(args.single_camera),
        max_keypoints=int(args.max_keypoints),
        min_inliers=int(args.min_inliers),
        pairing_cfg=pairing_cfg,
        device_str=str(args.device) if args.device else None,
        verbose=bool(args.verbose),
    )

if __name__ == "__main__":
    main()