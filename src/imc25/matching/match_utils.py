from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _require_torch():  # pragma: no cover - imported lazily for optional deps
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("This feature requires PyTorch. Install torch.") from e
    return torch


@dataclass(frozen=True)
class RansacResult:
    inlier_matches: np.ndarray  # (K,2) uint32
    F: np.ndarray | None  # (3,3) float64


def mutual_nn_matches(
    desc1: np.ndarray,
    desc2: np.ndarray,
    *,
    min_similarity: float,
    max_matches: int,
    device=None,
) -> np.ndarray:
    """
    Mutual nearest-neighbor matching on L2-normalized descriptors.

    Returns (N,2) uint32 indices into desc1/desc2.
    """
    if desc1.ndim != 2 or desc2.ndim != 2:
        raise ValueError("Descriptors must be 2D arrays")
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.uint32)
    if desc1.shape[1] != desc2.shape[1]:
        raise ValueError(f"Descriptor dims mismatch: {desc1.shape} vs {desc2.shape}")

    torch = _require_torch()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d1 = torch.from_numpy(desc1).to(dev)
    d2 = torch.from_numpy(desc2).to(dev)

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

    scores = score12[good]
    order = torch.argsort(scores, descending=True)
    if max_matches > 0:
        order = order[: int(max_matches)]

    matches = torch.stack([ii[order], jj[order]], dim=1).to(torch.int64).cpu().numpy()
    return matches.astype(np.uint32, copy=False)


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
        import cv2  # type: ignore
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

