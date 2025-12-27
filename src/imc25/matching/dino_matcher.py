from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from imc25.matching.dino_features import DinoFeatureConfig, DinoPatchFeatureExtractor
from imc25.matching.match_utils import RansacResult, mutual_nn_matches, ransac_fundamental


@dataclass(frozen=True)
class KeypointConfig:
    method: str = "patch"  # patch | gftt
    max_keypoints: int = 2048
    border: int = 8
    gftt_quality: float = 0.01
    gftt_min_distance: float = 12.0


@dataclass(frozen=True)
class MatchConfig:
    min_similarity: float = 0.75
    max_matches: int = 4096
    ransac_thresh_px: float = 1.5
    ransac_confidence: float = 0.999
    ransac_max_iters: int = 10000
    min_inliers: int = 15


@dataclass(frozen=True)
class DinoImageFeatures:
    image_name: str
    width: int
    height: int
    keypoints_xy: np.ndarray  # (N,2) float32 in original pixel coords
    descriptors: np.ndarray  # (N,C) float32 L2-normalized
    meta: dict

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            image_name=self.image_name,
            width=np.int32(self.width),
            height=np.int32(self.height),
            keypoints_xy=self.keypoints_xy.astype(np.float32, copy=False),
            descriptors=self.descriptors.astype(np.float16, copy=False),
            meta=json.dumps(self.meta),
        )

    @staticmethod
    def load_npz(path: Path) -> "DinoImageFeatures":
        data = np.load(str(path), allow_pickle=False)
        image_name = str(data["image_name"])
        width = int(np.asarray(data["width"]).item())
        height = int(np.asarray(data["height"]).item())
        keypoints_xy = np.asarray(data["keypoints_xy"], dtype=np.float32)
        descriptors = np.asarray(data["descriptors"], dtype=np.float32)
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-6
        meta = json.loads(str(data["meta"]))
        return DinoImageFeatures(
            image_name=image_name,
            width=width,
            height=height,
            keypoints_xy=keypoints_xy,
            descriptors=descriptors,
            meta=meta,
        )


def _cache_path(
    *,
    cache_dir: Path,
    image_path: Path,
    feat_cfg: DinoFeatureConfig,
    kp_cfg: KeypointConfig,
    extractor: DinoPatchFeatureExtractor,
) -> Path:
    cfg_tag = extractor.cache_tag()
    kp_tag = f"{kp_cfg.method}_k{int(kp_cfg.max_keypoints)}"
    h = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return cache_dir / cfg_tag / kp_tag / f"{image_path.stem}_{h}.npz"


def _detect_keypoints_gftt(image: Image.Image, cfg: KeypointConfig) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV is required for gftt keypoints (pip install opencv-python).") from e

    w, h = image.size
    border = int(max(0, cfg.border))
    gray = np.asarray(image.convert("L"))
    max_corners = int(max(0, cfg.max_keypoints))
    if max_corners == 0:
        return np.zeros((0, 2), dtype=np.float32)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=float(cfg.gftt_quality),
        minDistance=float(cfg.gftt_min_distance),
        blockSize=7,
        useHarrisDetector=False,
    )
    if corners is None or len(corners) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = corners.reshape(-1, 2).astype(np.float32, copy=False)
    if border > 0:
        keep = (pts[:, 0] >= border) & (pts[:, 0] < (w - border)) & (pts[:, 1] >= border) & (pts[:, 1] < (h - border))
        pts = pts[keep]
    if pts.shape[0] > max_corners:
        pts = pts[:max_corners]
    return pts


def _patch_grid_keypoints(
    *,
    desc_grid: np.ndarray,
    meta: dict,
    max_keypoints: int,
    border: int,
) -> tuple[np.ndarray, np.ndarray]:
    gh, gw, dim = desc_grid.shape
    patch = int(meta["patch_size"])
    resized_w = int(meta["resized_w"])
    resized_h = int(meta["resized_h"])
    scale_x = float(meta["scale_x"])
    scale_y = float(meta["scale_y"])
    orig_w = int(meta["orig_w"])
    orig_h = int(meta["orig_h"])

    # Compute stride in patch-grid to respect max_keypoints.
    valid_gw = int(np.ceil(float(resized_w) / float(patch)))
    valid_gh = int(np.ceil(float(resized_h) / float(patch)))
    approx = int(max(1, valid_gw * valid_gh))
    if max_keypoints and max_keypoints > 0 and approx > max_keypoints:
        stride = int(np.ceil(np.sqrt(float(approx) / float(max_keypoints))))
    else:
        stride = 1

    ys_idx = np.arange(0, gh, stride, dtype=np.int64)
    xs_idx = np.arange(0, gw, stride, dtype=np.int64)
    desc_sub = desc_grid[np.ix_(ys_idx, xs_idx)].reshape(-1, dim)

    centers_y = (ys_idx.astype(np.float32) + 0.5) * float(patch)
    centers_x = (xs_idx.astype(np.float32) + 0.5) * float(patch)
    grid_x, grid_y = np.meshgrid(centers_x, centers_y)

    # Drop patches whose centers fall into the right/bottom padding.
    mask = (grid_x <= float(resized_w)) & (grid_y <= float(resized_h))
    mask_flat = mask.reshape(-1)
    desc_keep = desc_sub[mask_flat]

    # Map to original pixel coordinates.
    kx = (grid_x.reshape(-1)[mask_flat] / float(scale_x)).astype(np.float32)
    ky = (grid_y.reshape(-1)[mask_flat] / float(scale_y)).astype(np.float32)
    kpts = np.stack([kx, ky], axis=1)

    b = int(max(0, border))
    if b > 0:
        keep = (kpts[:, 0] >= b) & (kpts[:, 0] < (orig_w - b)) & (kpts[:, 1] >= b) & (kpts[:, 1] < (orig_h - b))
        kpts = kpts[keep]
        desc_keep = desc_keep[keep]

    if max_keypoints and max_keypoints > 0 and kpts.shape[0] > max_keypoints:
        kpts = kpts[: int(max_keypoints)]
        desc_keep = desc_keep[: int(max_keypoints)]
    return kpts.astype(np.float32, copy=False), desc_keep.astype(np.float32, copy=False)


def extract_image_features(
    image_path: Path,
    *,
    extractor: DinoPatchFeatureExtractor,
    keypoints: KeypointConfig,
    cache_dir: Path | None = None,
    verbose: bool = False,
) -> DinoImageFeatures:
    cache_path = None
    if cache_dir is not None:
        cache_path = _cache_path(
            cache_dir=cache_dir,
            image_path=image_path,
            feat_cfg=extractor.config,
            kp_cfg=keypoints,
            extractor=extractor,
        )
        if cache_path.exists():
            if verbose:
                print(f"[dino] cache hit: {image_path.name}", flush=True)
            return DinoImageFeatures.load_npz(cache_path)

    if verbose:
        print(f"[dino] extracting: {image_path.name}", flush=True)
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    desc_grid, meta = extractor.extract_patch_descriptors(img)

    method = str(keypoints.method).lower().strip()
    if method == "patch":
        kpts_xy, desc = _patch_grid_keypoints(
            desc_grid=desc_grid,
            meta=meta,
            max_keypoints=int(keypoints.max_keypoints),
            border=int(keypoints.border),
        )
    elif method == "gftt":
        kpts_xy = _detect_keypoints_gftt(img, keypoints)
        if kpts_xy.shape[0] == 0:
            kpts_xy, desc = _patch_grid_keypoints(
                desc_grid=desc_grid,
                meta=meta,
                max_keypoints=int(keypoints.max_keypoints),
                border=int(keypoints.border),
            )
        else:
            # Snap each keypoint to nearest patch center and reuse that descriptor.
            patch = int(meta["patch_size"])
            scale_x = float(meta["scale_x"])
            scale_y = float(meta["scale_y"])
            resized_w = int(meta["resized_w"])
            resized_h = int(meta["resized_h"])
            gh, gw, dim = desc_grid.shape

            rx = np.clip(kpts_xy[:, 0] * scale_x, 0.0, float(resized_w - 1))
            ry = np.clip(kpts_xy[:, 1] * scale_y, 0.0, float(resized_h - 1))
            ix = np.clip(np.floor(rx / float(patch)).astype(np.int64), 0, gw - 1)
            iy = np.clip(np.floor(ry / float(patch)).astype(np.int64), 0, gh - 1)
            desc = desc_grid[iy, ix].reshape(-1, dim).astype(np.float32, copy=False)
            desc /= np.linalg.norm(desc, axis=1, keepdims=True) + 1e-6
    else:
        raise ValueError(f"Unknown keypoint method: {keypoints.method}")

    desc /= np.linalg.norm(desc, axis=1, keepdims=True) + 1e-6

    feats = DinoImageFeatures(
        image_name=image_path.name,
        width=int(w),
        height=int(h),
        keypoints_xy=kpts_xy.astype(np.float32, copy=False),
        descriptors=desc.astype(np.float32, copy=False),
        meta=meta,
    )
    if cache_path is not None:
        feats.save_npz(cache_path)
        if verbose:
            print(f"[dino] cached: {image_path.name}", flush=True)
    return feats


def match_pair(
    a: DinoImageFeatures,
    b: DinoImageFeatures,
    *,
    match_cfg: MatchConfig,
    device=None,
) -> RansacResult:
    matches = mutual_nn_matches(
        a.descriptors,
        b.descriptors,
        min_similarity=float(match_cfg.min_similarity),
        max_matches=int(match_cfg.max_matches),
        device=device,
    )
    return ransac_fundamental(
        a.keypoints_xy,
        b.keypoints_xy,
        matches,
        thresh_px=float(match_cfg.ransac_thresh_px),
        confidence=float(match_cfg.ransac_confidence),
        max_iters=int(match_cfg.ransac_max_iters),
    )

