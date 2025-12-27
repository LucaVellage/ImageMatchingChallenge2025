from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from imc25.matching.diffusion_features import DiffusionFeatureConfig, DiffusionUNetFeatureExtractor


@dataclass(frozen=True)
class KeypointConfig:
    method: str = "gftt"  # gftt | grid
    max_keypoints: int = 1024
    gftt_quality: float = 0.01
    gftt_min_distance: float = 8.0
    grid_step: int = 16
    border: int = 8


@dataclass(frozen=True)
class MatchConfig:
    min_similarity: float = 0.7
    max_matches: int = 4096
    ransac_thresh_px: float = 1.0
    ransac_confidence: float = 0.999
    ransac_max_iters: int = 10000
    min_inliers: int = 15


@dataclass(frozen=True)
class DiffusionImageFeatures:
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
    def load_npz(path: Path) -> "DiffusionImageFeatures":
        data = np.load(str(path), allow_pickle=False)
        image_name = str(data["image_name"])
        width = int(np.asarray(data["width"]).item())
        height = int(np.asarray(data["height"]).item())
        keypoints_xy = np.asarray(data["keypoints_xy"], dtype=np.float32)
        descriptors = np.asarray(data["descriptors"], dtype=np.float32)
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-6
        meta = json.loads(str(data["meta"]))
        return DiffusionImageFeatures(
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
    feat_cfg: DiffusionFeatureConfig,
    kp_cfg: KeypointConfig,
) -> Path:
    model_tag = feat_cfg.model_id.replace("/", "_").replace(":", "_")
    layers = "-".join(feat_cfg.layers)
    cfg_tag = f"{model_tag}_t{feat_cfg.timestep}_s{feat_cfg.max_side}_seed{feat_cfg.noise_seed}_{layers}"
    kp_tag = f"{kp_cfg.method}_k{kp_cfg.max_keypoints}"
    h = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return cache_dir / cfg_tag / kp_tag / f"{image_path.stem}_{h}.npz"


def _detect_keypoints(image: Image.Image, cfg: KeypointConfig) -> np.ndarray:
    w, h = image.size
    border = int(max(0, cfg.border))

    if cfg.method.lower() == "grid":
        step = int(max(1, cfg.grid_step))
        xs = np.arange(border, max(border + 1, w - border), step, dtype=np.float32)
        ys = np.arange(border, max(border + 1, h - border), step, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        pts = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
        if pts.shape[0] > cfg.max_keypoints:
            pts = pts[: int(cfg.max_keypoints)]
        return pts.astype(np.float32, copy=False)

    if cfg.method.lower() != "gftt":
        raise ValueError(f"Unknown keypoint method: {cfg.method}")

    try:
        import cv2
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV is required for keypoint detection (pip install opencv-python).") from e

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
        # Fallback to a small grid.
        fallback = KeypointConfig(method="grid", max_keypoints=cfg.max_keypoints, grid_step=max(8, cfg.grid_step))
        return _detect_keypoints(image, fallback)

    pts = corners.reshape(-1, 2).astype(np.float32, copy=False)
    if border > 0:
        keep = (pts[:, 0] >= border) & (pts[:, 0] < (w - border)) & (pts[:, 1] >= border) & (pts[:, 1] < (h - border))
        pts = pts[keep]
    if pts.shape[0] > cfg.max_keypoints:
        pts = pts[: int(cfg.max_keypoints)]
    return pts


def extract_image_features(
    image_path: Path,
    *,
    extractor: DiffusionUNetFeatureExtractor,
    keypoints: KeypointConfig,
    cache_dir: Path | None = None,
    verbose: bool = False,
) -> DiffusionImageFeatures:
    cache_path = None
    if cache_dir is not None:
        cache_path = _cache_path(cache_dir=cache_dir, image_path=image_path, feat_cfg=extractor.config, kp_cfg=keypoints)
        if cache_path.exists():
            if verbose:
                print(f"[diffusion] cache hit: {image_path.name}", flush=True)
            return DiffusionImageFeatures.load_npz(cache_path)

    if verbose:
        print(f"[diffusion] extracting: {image_path.name}", flush=True)

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    kpts_xy = _detect_keypoints(img, keypoints)

    desc_map, meta = extractor.extract_descriptor_map(img)
    sx = float(meta.get("scale_x", 1.0))
    sy = float(meta.get("scale_y", 1.0))
    kpts_resized = kpts_xy.copy()
    kpts_resized[:, 0] *= sx
    kpts_resized[:, 1] *= sy

    desc = extractor.sample_descriptors(desc_map, keypoints_xy_resized=kpts_resized)
    feats = DiffusionImageFeatures(
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
            print(f"[diffusion] cached: {image_path.name}", flush=True)
    return feats


def mutual_nn_matches(
    desc1: np.ndarray,
    desc2: np.ndarray,
    *,
    min_similarity: float,
    max_matches: int,
    device: torch.device | None = None,
) -> np.ndarray:
    if desc1.ndim != 2 or desc2.ndim != 2:
        raise ValueError("Descriptors must be 2D arrays")
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.uint32)
    if desc1.shape[1] != desc2.shape[1]:
        raise ValueError(f"Descriptor dims mismatch: {desc1.shape} vs {desc2.shape}")

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d1 = torch.from_numpy(desc1).to(dev)
    d2 = torch.from_numpy(desc2).to(dev)

    # Similarity matrix (cosine, since descriptors are L2-normalized).
    sim = d1 @ d2.T
    idx12 = torch.argmax(sim, dim=1)
    score12 = torch.max(sim, dim=1).values
    idx21 = torch.argmax(sim, dim=0)

    i = torch.arange(sim.shape[0], device=dev, dtype=idx12.dtype)
    mutual = idx21[idx12] == i
    good = mutual & (score12 >= float(min_similarity))
    ii = i[good]
    jj = idx12[good]
    if ii.numel() == 0:
        return np.zeros((0, 2), dtype=np.uint32)

    # Keep strongest matches first (use similarity score).
    scores = score12[good]
    order = torch.argsort(scores, descending=True)
    if max_matches > 0:
        order = order[: int(max_matches)]

    matches = torch.stack([ii[order], jj[order]], dim=1).to(torch.int64).cpu().numpy()
    return matches.astype(np.uint32, copy=False)


@dataclass(frozen=True)
class RansacResult:
    inlier_matches: np.ndarray  # (K,2) uint32
    F: np.ndarray | None  # (3,3) float64


def ransac_fundamental(
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    matches: np.ndarray,
    *,
    thresh_px: float,
    confidence: float,
    max_iters: int,
) -> RansacResult:
    if matches.shape[0] < 8:
        return RansacResult(inlier_matches=np.zeros((0, 2), dtype=np.uint32), F=None)

    try:
        import cv2
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV is required for RANSAC geometry verification (pip install opencv-python).") from e

    pts1 = kpts1[matches[:, 0].astype(np.int64)]
    pts2 = kpts2[matches[:, 1].astype(np.int64)]
    pts1 = pts1.astype(np.float32, copy=False)
    pts2 = pts2.astype(np.float32, copy=False)

    method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        method,
        ransacReprojThreshold=float(thresh_px),
        confidence=float(confidence),
        maxIters=int(max_iters),
    )
    if F is None or mask is None:
        return RansacResult(inlier_matches=np.zeros((0, 2), dtype=np.uint32), F=None)

    mask = mask.reshape(-1).astype(bool)
    inliers = matches[mask]
    return RansacResult(inlier_matches=inliers.astype(np.uint32, copy=False), F=np.asarray(F, dtype=np.float64).reshape(3, 3))


def match_pair(
    a: DiffusionImageFeatures,
    b: DiffusionImageFeatures,
    *,
    match_cfg: MatchConfig,
    device: torch.device | None = None,
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
