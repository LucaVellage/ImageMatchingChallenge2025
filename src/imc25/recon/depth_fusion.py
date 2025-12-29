from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def _require_torch():  # pragma: no cover
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("This feature requires PyTorch. Install torch.") from e
    return torch


def _require_transformers():  # pragma: no cover
    try:
        import transformers  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("This feature requires transformers. Install transformers.") from e
    return transformers


@dataclass(frozen=True)
class ColmapCamera:
    model: str
    width: int
    height: int
    params: np.ndarray  # float64


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    name: str
    camera_id: int
    qvec: np.ndarray  # (4,) float64 (qw,qx,qy,qz)
    tvec: np.ndarray  # (3,) float64
    xys: np.ndarray  # (N,2) float64
    point3d_ids: np.ndarray  # (N,) int64


def _read_c_string(f) -> str:
    buf = bytearray()
    while True:
        b = f.read(1)
        if b == b"":
            raise EOFError("Unexpected EOF while reading string")
        if b == b"\x00":
            break
        buf.extend(b)
    return buf.decode("utf-8", errors="replace")


_CAMERA_MODEL_IDS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}


def read_cameras_bin(path: Path) -> dict[int, ColmapCamera]:
    cams: dict[int, ColmapCamera] = {}
    with path.open("rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(int(num)):
            camera_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            model = _CAMERA_MODEL_IDS.get(int(model_id), str(int(model_id)))

            # Param count depends on model. Use common defaults; fall back to reading remaining by looking up.
            # COLMAP stores parameters as doubles; for known models, param counts are fixed.
            param_counts = {
                "SIMPLE_PINHOLE": 3,
                "PINHOLE": 4,
                "SIMPLE_RADIAL": 4,
                "RADIAL": 5,
                "OPENCV": 8,
                "FULL_OPENCV": 12,
                "OPENCV_FISHEYE": 8,
                "FOV": 5,
                "SIMPLE_RADIAL_FISHEYE": 4,
                "RADIAL_FISHEYE": 5,
                "THIN_PRISM_FISHEYE": 12,
            }
            n_params = int(param_counts.get(model, 0))
            if n_params <= 0:
                raise ValueError(f"Unsupported/unknown camera model in cameras.bin: id={model_id} ({model})")
            params = struct.unpack("<" + "d" * n_params, f.read(8 * n_params))
            cams[int(camera_id)] = ColmapCamera(
                model=str(model),
                width=int(width),
                height=int(height),
                params=np.asarray(params, dtype=np.float64),
            )
    return cams


def read_images_bin(path: Path) -> dict[int, ColmapImage]:
    imgs: dict[int, ColmapImage] = {}
    with path.open("rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(int(num_images)):
            image_id = struct.unpack("<i", f.read(4))[0]
            qvec = np.asarray(struct.unpack("<dddd", f.read(32)), dtype=np.float64)
            tvec = np.asarray(struct.unpack("<ddd", f.read(24)), dtype=np.float64)
            camera_id = struct.unpack("<i", f.read(4))[0]
            name = _read_c_string(f)
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            xys = np.zeros((int(num_points2d), 2), dtype=np.float64)
            pids = np.full((int(num_points2d),), -1, dtype=np.int64)
            for j in range(int(num_points2d)):
                x, y, pid = struct.unpack("<ddq", f.read(24))
                xys[j, 0] = float(x)
                xys[j, 1] = float(y)
                pids[j] = int(pid)
            imgs[int(image_id)] = ColmapImage(
                image_id=int(image_id),
                name=str(name),
                camera_id=int(camera_id),
                qvec=qvec,
                tvec=tvec,
                xys=xys,
                point3d_ids=pids,
            )
    return imgs


def read_points3d_bin(path: Path) -> dict[int, np.ndarray]:
    pts: dict[int, np.ndarray] = {}
    with path.open("rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(int(num)):
            point_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            _rgb = f.read(3)  # uint8[3]
            _error = f.read(8)  # double
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.seek(int(track_len) * 8, 1)  # image_id(int32), point2d_idx(int32)
            pts[int(point_id)] = np.asarray(xyz, dtype=np.float64)
    return pts


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = [float(x) for x in qvec.reshape(-1).tolist()]
    return np.array(
        [
            [1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qw * qz, 2.0 * qx * qz + 2.0 * qw * qy],
            [2.0 * qx * qy + 2.0 * qw * qz, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qw * qx],
            [2.0 * qx * qz - 2.0 * qw * qy, 2.0 * qy * qz + 2.0 * qw * qx, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy],
        ],
        dtype=np.float64,
    )


def K_from_camera(cam: ColmapCamera) -> np.ndarray:
    p = cam.params.reshape(-1).astype(np.float64)
    m = cam.model.upper()
    if m in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        f, cx, cy = float(p[0]), float(p[1]), float(p[2])
        fx, fy = f, f
    elif m in {"PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"}:
        fx, fy, cx, cy = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    else:
        f = 1.2 * float(max(cam.width, cam.height))
        fx, fy = f, f
        cx, cy = float(cam.width) / 2.0, float(cam.height) / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


class DepthEstimator:
    def __init__(
        self,
        *,
        model_id: str,
        device: str | None,
        fp16: bool,
        hf_endpoint: str | None,
    ) -> None:
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = str(hf_endpoint)
        _require_transformers()
        torch = _require_torch()
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # type: ignore

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if fp16 and self.device.type == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

    @property
    def torch_device(self):
        return self.device

    def predict(self, img: Image.Image) -> np.ndarray:
        torch = _require_torch()
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            out = self.model(**inputs)
            depth = out.predicted_depth
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        depth = torch.nn.functional.interpolate(
            depth,
            size=img.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        return depth.squeeze().detach().float().cpu().numpy()


def _sample_depth(depth: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h, w = depth.shape[:2]
    x = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, h - 1)
    return depth[y, x].astype(np.float64, copy=False)


def _estimate_scale_from_sparse(
    *,
    depth: np.ndarray,
    img: ColmapImage,
    cam: ColmapCamera,
    pts3d: dict[int, np.ndarray],
) -> float | None:
    valid = img.point3d_ids >= 0
    if not bool(valid.any()):
        return None
    ids = img.point3d_ids[valid]
    xy = img.xys[valid]
    xyz = []
    xy_keep = []
    for pid, pxy in zip(ids.tolist(), xy, strict=False):
        p = pts3d.get(int(pid))
        if p is None:
            continue
        xyz.append(p)
        xy_keep.append(pxy)
    if len(xyz) < 20:
        return None
    xyz_w = np.stack(xyz, axis=0).astype(np.float64)
    xy_keep = np.stack(xy_keep, axis=0).astype(np.float64)

    R_cw = qvec_to_rotmat(img.qvec)
    t = img.tvec.reshape(1, 3).astype(np.float64)
    z_gt = (R_cw @ xyz_w.T).T[:, 2] + float(t[0, 2])
    d_pred = _sample_depth(depth, xy_keep)

    good = (z_gt > 1e-6) & np.isfinite(z_gt) & (d_pred > 1e-6) & np.isfinite(d_pred)
    if int(good.sum()) < 20:
        return None

    ratios = z_gt[good] / d_pred[good]
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size < 20:
        return None

    # Robust scaling (median).
    s = float(np.median(ratios))
    if not np.isfinite(s) or s <= 0:
        return None
    return s


def _backproject(
    *,
    depth: np.ndarray,
    rgb: np.ndarray | None,
    K: np.ndarray,
    R_cw: np.ndarray | None,
    t_cw: np.ndarray | None,
    stride: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    h, w = depth.shape[:2]
    s = int(max(1, stride))
    ys, xs = np.mgrid[0:h:s, 0:w:s]
    z = depth[ys, xs].astype(np.float64)
    valid = (z > 1e-6) & np.isfinite(z)
    if not bool(valid.any()):
        return np.zeros((0, 3), dtype=np.float32), None

    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    z = z[valid].astype(np.float64)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    pts_c = np.stack([x, y, z], axis=1)  # (N,3)

    if R_cw is not None and t_cw is not None:
        t = t_cw.reshape(3, 1).astype(np.float64)
        pts_w = (R_cw.T @ (pts_c.T - t)).T
    else:
        pts_w = pts_c

    cols = None
    if rgb is not None:
        cols = rgb[ys.astype(np.int64), xs.astype(np.int64)].astype(np.uint8, copy=False)
    return pts_w.astype(np.float32, copy=False), cols


def write_ply_binary(path: Path, *, points: np.ndarray, colors: np.ndarray | None) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    cols = None
    if colors is not None:
        cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if cols.shape[0] != pts.shape[0]:
            raise ValueError("colors must match points")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n".encode("ascii"))
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        if cols is not None:
            f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(b"end_header\n")

        if cols is None:
            f.write(pts.astype("<f4", copy=False).tobytes(order="C"))
        else:
            data = np.empty((pts.shape[0],), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
            data["x"] = pts[:, 0]
            data["y"] = pts[:, 1]
            data["z"] = pts[:, 2]
            data["r"] = cols[:, 0]
            data["g"] = cols[:, 1]
            data["b"] = cols[:, 2]
            f.write(data.tobytes(order="C"))


def build_dense_points_from_depth(
    *,
    images_dir: Path,
    out_ply: Path,
    model_dir: Path | None,
    depth_model_id: str,
    hf_endpoint: str | None,
    device: str | None,
    fp16: bool,
    stride: int,
    max_points: int,
    align_scale: bool,
    with_color: bool,
    verbose: bool = True,
) -> int:
    """
    Create a dense pointcloud using monocular depth estimation.

    If `model_dir` is provided and contains COLMAP binaries, uses camera intrinsics/extrinsics to place points into the
    SfM coordinate system and optionally scale-align depths using sparse points.

    Returns the number of vertices written.
    """
    images = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
    if not images:
        raise ValueError(f"No images found in {images_dir}")

    cams = None
    imgs = None
    pts3d = None
    if model_dir is not None:
        cam_path = model_dir / "cameras.bin"
        img_path = model_dir / "images.bin"
        pts_path = model_dir / "points3D.bin"
        if cam_path.exists() and img_path.exists():
            cams = read_cameras_bin(cam_path)
            imgs = read_images_bin(img_path)
            if align_scale and pts_path.exists():
                pts3d = read_points3d_bin(pts_path)
        else:
            model_dir = None

    estimator = DepthEstimator(model_id=depth_model_id, device=device, fp16=fp16, hf_endpoint=hf_endpoint)

    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []

    for i, p in enumerate(images, start=1):
        if verbose:
            print(f"[depth] {i}/{len(images)}: {p.name}", flush=True)
        img_pil = Image.open(p).convert("RGB")
        depth = estimator.predict(img_pil)

        R_cw = None
        t_cw = None
        K = None
        if model_dir is not None and cams is not None and imgs is not None:
            # Match by basename
            entry = None
            for im in imgs.values():
                if Path(im.name).name == p.name:
                    entry = im
                    break
            if entry is not None:
                cam = cams.get(int(entry.camera_id))
                if cam is not None:
                    K = K_from_camera(cam)
                    R_cw = qvec_to_rotmat(entry.qvec)
                    t_cw = entry.tvec.astype(np.float64)
                    if align_scale and pts3d is not None:
                        s = _estimate_scale_from_sparse(depth=depth, img=entry, cam=cam, pts3d=pts3d)
                        if s is not None:
                            depth = depth.astype(np.float64) * float(s)

        if K is None:
            # Guess intrinsics
            w, h = img_pil.size
            f = 1.2 * float(max(w, h))
            K = np.array([[f, 0.0, float(w) / 2.0], [0.0, f, float(h) / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        rgb = None
        if with_color:
            rgb = np.asarray(img_pil, dtype=np.uint8)

        pts, cols = _backproject(depth=depth, rgb=rgb, K=K, R_cw=R_cw, t_cw=t_cw, stride=int(stride))
        if pts.shape[0] == 0:
            continue
        all_pts.append(pts)
        if cols is not None:
            all_cols.append(cols)

    if not all_pts:
        write_ply_binary(out_ply, points=np.zeros((0, 3), dtype=np.float32), colors=None)
        return 0

    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0) if all_cols else None

    if max_points and pts.shape[0] > int(max_points):
        rng = np.random.default_rng(0)
        idx = rng.choice(pts.shape[0], size=int(max_points), replace=False)
        pts = pts[idx]
        cols = cols[idx] if cols is not None else None

    write_ply_binary(out_ply, points=pts, colors=cols)
    return int(pts.shape[0])

