from __future__ import annotations

import argparse
import hashlib
import itertools
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from imc25.matching.colmap_db import CameraSpec, create_empty_colmap_db, guess_simple_radial, insert_camera
from imc25.matching.colmap_db import insert_dummy_descriptors, insert_image, insert_keypoints, insert_matches, insert_two_view_geometry
from imc25.matching.match_utils import ransac_fundamental


def _require_torch():  # pragma: no cover - optional dependency
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("This feature requires PyTorch. Install torch.") from e
    return torch


def _require_lightglue():  # pragma: no cover - optional dependency
    try:
        from lightglue import ALIKED, LightGlue  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("This feature requires LightGlue + ALIKED. Install `lightglue`.") from e
    return ALIKED, LightGlue


@dataclass(frozen=True)
class FeatureConfig:
    max_keypoints: int = 2048


@dataclass(frozen=True)
class MatchConfig:
    ransac_thresh_px: float = 1.0
    ransac_confidence: float = 0.999
    ransac_max_iters: int = 10000
    min_inliers: int = 15


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
    raise ValueError(f"Unsupported camera model for LightGlue DB builder: {model}")


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


def _cache_path(*, cache_dir: Path, image_path: Path, feat_cfg: FeatureConfig) -> Path:
    tag = f"aliked_k{int(feat_cfg.max_keypoints)}"
    h = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return cache_dir / tag / f"{image_path.stem}_{h}.npz"


def _torch_image_from_path(path: Path, *, device) -> Any:
    torch = _require_torch()
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    x = torch.from_numpy(arr).to(device)
    x = x.permute(2, 0, 1).contiguous().float() / 255.0
    return x[None]


def _to_device(d: dict[str, Any], device) -> dict[str, Any]:
    torch = _require_torch()
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _load_cached_features(path: Path, *, device) -> dict[str, Any]:
    torch = _require_torch()
    data = np.load(str(path), allow_pickle=False)
    out: dict[str, Any] = {}
    for k in data.files:
        out[k] = torch.from_numpy(np.asarray(data[k])).to(device)
    return out


def _save_cached_features(path: Path, feats: dict[str, Any]) -> None:
    torch = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for k, v in feats.items():
        if isinstance(v, torch.Tensor):
            arrays[k] = v.detach().cpu().numpy()
    if not arrays:
        return
    np.savez_compressed(str(path), **arrays)


def extract_aliked_features(
    image_path: Path,
    *,
    extractor,
    feat_cfg: FeatureConfig,
    cache_dir: Path | None,
    device,
) -> dict[str, Any]:
    if cache_dir is not None:
        cp = _cache_path(cache_dir=cache_dir, image_path=image_path, feat_cfg=feat_cfg)
        if cp.exists():
            return _load_cached_features(cp, device=device)

    x = _torch_image_from_path(image_path, device=device)
    torch = _require_torch()
    with torch.inference_mode():
        feats = extractor.extract(x)

    if cache_dir is not None:
        cp = _cache_path(cache_dir=cache_dir, image_path=image_path, feat_cfg=feat_cfg)
        _save_cached_features(cp, feats)

    return feats


def build_colmap_db_from_lightglue(
    *,
    images_dir: Path,
    db_path: Path,
    overwrite: bool,
    camera_model: str,
    single_camera: bool,
    feat_cfg: FeatureConfig,
    match_cfg: MatchConfig,
    pairing_cfg: PairingConfig,
    cache_dir: Path | None,
    device: str | None = None,
    extractor=None,
    matcher=None,
    verbose: bool = True,
) -> None:
    torch = _require_torch()
    ALIKED, LightGlue = _require_lightglue()

    image_paths = _iter_images(images_dir)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    torch_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(
            f"[lightglue] build db={db_path} images={len(image_paths)} device={torch_device} "
            f"pairing={pairing_cfg.mode} topk={pairing_cfg.topk} max_kpts={feat_cfg.max_keypoints}",
            flush=True,
        )
        print(
            f"[lightglue] RANSAC: thresh_px={match_cfg.ransac_thresh_px} min_inliers={match_cfg.min_inliers}",
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

        image_id_by_name: dict[str, int] = {}
        for path, cam_id in zip(image_paths, cameras_by_index, strict=False):
            image_id = insert_image(conn, name=path.name, camera_id=cam_id)
            image_id_by_name[path.name] = image_id

        if extractor is None or matcher is None:
            if verbose:
                print("[lightglue] loading ALIKED + LightGlue models...", flush=True)
            extractor = ALIKED(max_num_keypoints=int(feat_cfg.max_keypoints)).eval().to(torch_device)
            matcher = LightGlue(features="aliked").eval().to(torch_device)
        else:
            extractor = extractor.to(torch_device).eval()
            matcher = matcher.to(torch_device).eval()

        cache_root = Path(cache_dir) if cache_dir is not None else None

        feats_by_name: dict[str, dict[str, Any]] = {}
        keypoints_by_name: dict[str, np.ndarray] = {}
        for i, p in enumerate(image_paths, start=1):
            if verbose:
                print(f"[lightglue] image {i}/{len(image_paths)}: {p.name}", flush=True)
            feats = extract_aliked_features(
                p,
                extractor=extractor,
                feat_cfg=feat_cfg,
                cache_dir=cache_root,
                device=torch_device,
            )
            feats_by_name[p.name] = feats
            kpts = feats.get("keypoints")
            if kpts is None:
                raise RuntimeError("ALIKED features missing 'keypoints'")
            kpts_xy = np.asarray(kpts[0].detach().cpu().numpy(), dtype=np.float32)
            keypoints_by_name[p.name] = kpts_xy

            image_id = image_id_by_name[p.name]
            insert_keypoints(conn, image_id=image_id, keypoints_xy=kpts_xy)
            insert_dummy_descriptors(conn, image_id=image_id, num_keypoints=int(kpts_xy.shape[0]))

        pairs = list(_iter_pairs(image_paths, pairing=pairing_cfg))
        if verbose:
            print(f"[lightglue] matching pairs={len(pairs)} (RANSAC)...", flush=True)

        good_pairs = 0
        for idx, (ia, ib) in enumerate(pairs, start=1):
            name_a = image_paths[ia].name
            name_b = image_paths[ib].name
            fa = feats_by_name[name_a]
            fb = feats_by_name[name_b]

            with torch.inference_mode():
                out = matcher({"image0": fa, "image1": fb})

            matches: np.ndarray
            if isinstance(out, dict) and "matches" in out:
                m = out["matches"][0].detach().cpu().numpy()
                matches = np.asarray(m, dtype=np.uint32)
            elif isinstance(out, dict) and "matches0" in out:
                m0 = out["matches0"][0].detach().cpu().numpy()
                valid = m0 > -1
                idx0 = np.where(valid)[0].astype(np.uint32)
                idx1 = m0[valid].astype(np.uint32)
                matches = np.stack([idx0, idx1], axis=1).astype(np.uint32, copy=False) if idx0.size else np.zeros((0, 2), dtype=np.uint32)
            else:
                matches = np.zeros((0, 2), dtype=np.uint32)

            if matches.shape[0] < 8:
                continue

            kpts_a = keypoints_by_name[name_a]
            kpts_b = keypoints_by_name[name_b]
            r = ransac_fundamental(
                kpts_a,
                kpts_b,
                matches,
                thresh_px=float(match_cfg.ransac_thresh_px),
                confidence=float(match_cfg.ransac_confidence),
                max_iters=int(match_cfg.ransac_max_iters),
            )
            if r.F is None:
                if verbose and (len(pairs) <= 20 or idx % 25 == 0):
                    print(f"[lightglue] pair {idx}/{len(pairs)}: {name_a} ↔ {name_b} (no F)", flush=True)
                continue
            if r.inlier_matches.shape[0] < int(match_cfg.min_inliers):
                if verbose and (len(pairs) <= 20 or idx % 25 == 0):
                    print(
                        f"[lightglue] pair {idx}/{len(pairs)}: {name_a} ↔ {name_b} "
                        f"(inliers={r.inlier_matches.shape[0]} < min_inliers={match_cfg.min_inliers})",
                        flush=True,
                    )
                continue

            if verbose and (len(pairs) <= 20 or idx % 25 == 0):
                print(f"[lightglue] pair {idx}/{len(pairs)}: {name_a} ↔ {name_b} (inliers={r.inlier_matches.shape[0]})", flush=True)

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
            f"[ok] wrote {db_path} (images={len(image_paths)} pairs={len(pairs)} "
            f"good_pairs={good_pairs}, min_inliers={match_cfg.min_inliers})",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a COLMAP database using ALIKED keypoints + LightGlue matches + RANSAC inliers")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/lightglue_features"))
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda, cuda:0, cpu)")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--camera-model", default="SIMPLE_RADIAL")
    parser.add_argument("--single-camera", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--max-keypoints", type=int, default=FeatureConfig.max_keypoints)
    parser.add_argument("--ransac-thresh-px", type=float, default=MatchConfig.ransac_thresh_px)
    parser.add_argument("--min-inliers", type=int, default=MatchConfig.min_inliers)

    parser.add_argument("--pairing", default="auto", choices=["auto", "exhaustive", "topk"])
    parser.add_argument("--topk", type=int, default=PairingConfig.topk)
    args = parser.parse_args()

    feat_cfg = FeatureConfig(max_keypoints=int(args.max_keypoints))
    match_cfg = MatchConfig(ransac_thresh_px=float(args.ransac_thresh_px), min_inliers=int(args.min_inliers))
    pairing_cfg = PairingConfig(mode=str(args.pairing), topk=int(args.topk))
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    build_colmap_db_from_lightglue(
        images_dir=args.images_dir,
        db_path=args.db_path,
        overwrite=bool(args.overwrite),
        camera_model=str(args.camera_model),
        single_camera=bool(args.single_camera),
        feat_cfg=feat_cfg,
        match_cfg=match_cfg,
        pairing_cfg=pairing_cfg,
        cache_dir=cache_dir,
        device=str(args.device) if args.device else None,
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main()

