from __future__ import annotations

import argparse
import csv
import os
import shlex
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class TestImage:
    dataset: str
    stem: str
    name: str
    path: Path


def _run(cmd: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    printable = shlex.join(cmd)
    print(f"+ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [{k: (v if v is not None else "") for k, v in row.items()} for row in reader]


def _write_csv(path: Path, rows: Iterable[dict[str, str]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _list_test_images(data_root: Path) -> list[TestImage]:
    out: list[TestImage] = []
    for ds_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        dataset = ds_dir.name
        for p in sorted(ds_dir.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() not in _IMG_EXTS:
                continue
            out.append(TestImage(dataset=dataset, stem=p.stem, name=p.name, path=p))
    return out


def _u64_head(path: Path) -> int:
    with path.open("rb") as f:
        head = f.read(8)
    if len(head) != 8:
        return 0
    (n,) = struct.unpack("<Q", head)
    return int(n)


def _pick_best_sparse_model(sparse_root: Path) -> Path | None:
    if not sparse_root.exists():
        return None
    candidates = [p for p in sparse_root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not candidates:
        return None

    best_score: tuple[int, int, int] | None = None
    best_path: Path | None = None
    for p in sorted(candidates, key=lambda x: int(x.name)):
        n_images = _u64_head(p / "images.bin") if (p / "images.bin").exists() else 0
        n_points = _u64_head(p / "points3D.bin") if (p / "points3D.bin").exists() else 0
        score = (int(n_images), int(n_points), int(p.name))
        if best_score is None or score > best_score:
            best_score = score
            best_path = p
    return best_path


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


def _read_colmap_images_bin(path: Path) -> dict[str, tuple[tuple[float, float, float, float], tuple[float, float, float]]]:
    """
    Returns mapping: image_name -> (qvec, tvec)
    qvec is (qw, qx, qy, qz) in COLMAP convention.
    """
    out: dict[str, tuple[tuple[float, float, float, float], tuple[float, float, float]]] = {}
    with path.open("rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(int(num_images)):
            _image_id = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<dddd", f.read(32))
            tvec = struct.unpack("<ddd", f.read(24))
            _camera_id = struct.unpack("<i", f.read(4))[0]
            name = _read_c_string(f)
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(int(num_points2d) * 24, 1)
            out[str(name)] = ((float(qvec[0]), float(qvec[1]), float(qvec[2]), float(qvec[3])), (float(tvec[0]), float(tvec[1]), float(tvec[2])))
    return out


def _qvec_to_rotmat(q: tuple[float, float, float, float]) -> list[list[float]]:
    qw, qx, qy, qz = q
    return [
        [1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qw * qz, 2.0 * qx * qz + 2.0 * qw * qy],
        [2.0 * qx * qy + 2.0 * qw * qz, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qw * qx],
        [2.0 * qx * qz - 2.0 * qw * qy, 2.0 * qy * qz + 2.0 * qw * qx, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy],
    ]


def _fmt_floats(vals: Iterable[float]) -> str:
    return ";".join(["nan" if (v != v) else f"{float(v):.12g}" for v in vals])


def _nan_pose() -> tuple[str, str]:
    return (";".join(["nan"] * 9), ";".join(["nan"] * 3))


def _build_pose_index(
    clusters: list[dict[str, str]],
    *,
    recon_root: Path,
    outliers_label: str,
) -> dict[tuple[str, str, str], tuple[str, str]]:
    """
    Returns mapping: (dataset, scene, image_stem) -> (rotation_matrix_str, translation_vector_str)
    """
    scenes = {(r["dataset"], r["scene"]) for r in clusters if r.get("scene", "").lower() != outliers_label.lower()}
    out: dict[tuple[str, str, str], tuple[str, str]] = {}
    for dataset, scene in sorted(scenes):
        sparse_root = recon_root / f"{dataset}_{scene}" / "sparse"
        best = _pick_best_sparse_model(sparse_root)
        if best is None:
            continue
        images_bin = best / "images.bin"
        if not images_bin.exists():
            continue
        try:
            imgs = _read_colmap_images_bin(images_bin)
        except Exception:
            continue
        for name, (qvec, tvec) in imgs.items():
            base = PurePosixPath(str(name)).name
            stem = Path(base).stem
            R = _qvec_to_rotmat(qvec)
            R_flat = [R[0][0], R[0][1], R[0][2], R[1][0], R[1][1], R[1][2], R[2][0], R[2][1], R[2][2]]
            r_str = _fmt_floats(R_flat)
            t_str = _fmt_floats(tvec)
            out[(dataset, scene, stem)] = (r_str, t_str)
    return out


def _export_submission(
    *,
    clusters_csv: Path,
    data_test_root: Path,
    recon_root: Path,
    out_csv: Path,
    outliers_label: str,
    image_id_suffix: str,
) -> None:
    clusters = _read_csv_rows(clusters_csv)
    required = {"dataset", "scene", "image_id"}
    if not clusters:
        raise SystemExit(f"clusters CSV is empty: {clusters_csv}")
    missing = required - set(clusters[0].keys())
    if missing:
        raise SystemExit(f"{clusters_csv} missing columns: {sorted(missing)}")

    test_images = _list_test_images(data_test_root)
    name_of: dict[tuple[str, str], str] = {(ti.dataset, ti.stem): ti.name for ti in test_images}
    pose_idx = _build_pose_index(clusters, recon_root=recon_root, outliers_label=outliers_label)
    nan_R, nan_t = _nan_pose()

    def make_row(r: dict[str, str]) -> dict[str, str]:
        dataset = str(r["dataset"])
        scene = str(r["scene"])
        raw_image_id = str(r["image_id"])
        stem = Path(raw_image_id).stem if Path(raw_image_id).suffix.lower() in _IMG_EXTS else raw_image_id
        image_name = name_of.get((dataset, stem), f"{stem}.png")
        image_id = f"{dataset}_{image_name}_{image_id_suffix}" if image_id_suffix else f"{dataset}_{image_name}"

        if scene.lower() == outliers_label.lower():
            rot, trans = nan_R, nan_t
        else:
            rot, trans = pose_idx.get((dataset, scene, stem), (nan_R, nan_t))

        return {
            "image_id": image_id,
            "dataset": dataset,
            "scene": scene,
            "image": image_name,
            "rotation_matrix": rot,
            "translation_vector": trans,
        }

    rows = [make_row(r) for r in clusters]
    rows.sort(key=lambda x: (x["dataset"], x["image"]))
    _write_csv(
        out_csv,
        rows,
        fieldnames=["image_id", "dataset", "scene", "image", "rotation_matrix", "translation_vector"],
    )
    print(f"[ok] wrote submission: {out_csv} rows={len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IMC25 pipeline end-to-end (train → cache → cluster → COLMAP → submission.csv)")
    parser.add_argument("--hf-endpoint", default=None, help="Hugging Face Hub endpoint (mirror), e.g. https://hf-mirror.com")
    parser.add_argument(
        "--preset",
        default=None,
        choices=["baseline", "sota"],
        help="Convenience preset (sota enables matcher=auto + dense + depth fallback + demo export).",
    )

    parser.add_argument("--data-train-root", type=Path, default=Path("data/train"))
    parser.add_argument("--train-labels-csv", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument("--data-test-root", type=Path, default=Path("data/test"))

    parser.add_argument("--train-output-dir", type=Path, default=Path("outputs/retrieval_finetune"))
    parser.add_argument("--model-id", default="facebook/dinov2-small")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--train",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force enable/disable training (default: train only if checkpoint missing, or --overwrite).",
    )

    parser.add_argument("--checkpoint", type=Path, default=None, help="Override retrieval checkpoint (default: <train-output-dir>/best.pt)")
    parser.add_argument("--cache-root", type=Path, default=Path("cache_retrieval"))
    parser.add_argument("--clusters-csv", type=Path, default=Path("cache/clusters_retrieval.csv"))

    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--mutual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-sim", type=float, default=0.35)
    parser.add_argument("--cluster-method", default="louvain", choices=["louvain", "greedy", "components"])
    parser.add_argument("--min-cluster-size", type=int, default=3)

    parser.add_argument("--recon-output-root", type=Path, default=Path("outputs_retrieval_test"))
    parser.add_argument("--runner", default="auto", choices=["auto", "local", "docker"])
    parser.add_argument("--matcher", default="colmap", choices=["colmap", "dino", "diffusion", "auto"])
    parser.add_argument("--dense", action="store_true", help="Also run dense MVS (slower); default is SfM only.")
    parser.add_argument(
        "--dino-model-id",
        default="facebook/dinov2-small",
        help="DINO matcher model id used when --matcher dino/auto (e.g. facebook/dinov2-base).",
    )
    parser.add_argument("--dino-max-side", type=int, default=512, help="Max image side for DINO matching (lower uses less VRAM).")
    parser.add_argument("--dino-max-keypoints", type=int, default=2048, help="Max DINO patch keypoints per image (lower is faster).")
    parser.add_argument("--dino-min-inliers", type=int, default=15, help="Min RANSAC inliers to accept a DINO pair.")
    parser.add_argument("--dino-min-similarity", type=float, default=0.75, help="Min cosine similarity for DINO mutual NN matches.")
    parser.add_argument("--dino-topk", type=int, default=10, help="Top-k pairing when DINO pairing=auto selects topk mode.")
    parser.add_argument(
        "--dense-min-vertices",
        type=int,
        default=1,
        help="Treat dense_points.ply as failed if it has fewer points; used with --depth-fallback.",
    )
    parser.add_argument("--depth-fallback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--depth-model-id", default="Intel/dpt-hybrid-midas")
    parser.add_argument("--demo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--demo-out-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--demo-max-points", type=int, default=150_000)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite caches and recon outputs when supported.")

    parser.add_argument("--submission-csv", type=Path, default=Path("submission.csv"))
    parser.add_argument("--outliers-label", default="outliers")
    parser.add_argument("--image-id-suffix", default="public", help="Suffix used to form image_id as <dataset>_<image>_<suffix>")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = str(args.hf_endpoint)

    if args.preset == "sota":
        args.matcher = "auto"
        args.dense = True
        args.depth_fallback = True
        args.demo = True
        args.dense_min_vertices = max(int(args.dense_min_vertices), 5000)
    elif args.preset == "baseline":
        pass

    ckpt = args.checkpoint if args.checkpoint is not None else (args.train_output_dir / "best.pt")
    if args.train is None:
        do_train = bool(args.overwrite) or not ckpt.exists()
    else:
        do_train = bool(args.train)
    if do_train:
        cmd = [
            sys.executable,
            "scripts/train_retrieval.py",
            "--data-root",
            str(args.data_train_root),
            "--labels-csv",
            str(args.train_labels_csv),
            "--output-dir",
            str(args.train_output_dir),
            "--model-id",
            str(args.model_id),
            "--embed-dim",
            str(int(args.embed_dim)),
        ]
        if args.hf_endpoint:
            cmd += ["--hf-endpoint", str(args.hf_endpoint)]
        if args.freeze_backbone:
            cmd += ["--freeze-backbone"]
        else:
            cmd += ["--no-freeze-backbone"]
        _run(cmd, dry_run=bool(args.dry_run), env=env)
    else:
        print(f"[skip] training: checkpoint exists at {ckpt}", flush=True)

    if not do_train and not ckpt.exists() and not args.dry_run:
        raise SystemExit(f"--no-train specified but checkpoint does not exist: {ckpt}")

    if not ckpt.exists() and not args.dry_run:
        raise SystemExit(f"checkpoint not found after training: {ckpt}")

    cmd = [
        sys.executable,
        "scripts/build_retrieval_cache.py",
        "--data-root",
        str(args.data_test_root),
        "--cache-root",
        str(args.cache_root),
        "--checkpoint",
        str(ckpt),
        "--topk",
        str(int(args.topk)),
    ]
    if args.hf_endpoint:
        cmd += ["--hf-endpoint", str(args.hf_endpoint)]
    if args.mutual:
        cmd += ["--mutual"]
    else:
        cmd += ["--no-mutual"]
    if args.overwrite:
        cmd += ["--overwrite"]
    _run(cmd, dry_run=bool(args.dry_run), env=env)

    if args.clusters_csv.exists() and not args.overwrite:
        print(f"[skip] clustering: {args.clusters_csv} exists (use --overwrite to rebuild)", flush=True)
    else:
        cmd = [
            sys.executable,
            "scripts/cluster_retrieval.py",
            "--cache-root",
            str(args.cache_root),
            "--out-csv",
            str(args.clusters_csv),
            "--topk",
            str(int(args.topk)),
            "--min-sim",
            str(float(args.min_sim)),
            "--method",
            str(args.cluster_method),
            "--min-cluster-size",
            str(int(args.min_cluster_size)),
        ]
        if args.mutual:
            cmd += ["--mutual"]
        else:
            cmd += ["--no-mutual"]
        _run(cmd, dry_run=bool(args.dry_run), env=env)

    if not args.clusters_csv.exists() and not args.dry_run:
        raise SystemExit(f"clusters CSV not found: {args.clusters_csv}")

    cmd = [
        sys.executable,
        "scripts/colmap_dense_clusters.py",
        "--clusters-csv",
        str(args.clusters_csv),
        "--cache-root",
        str(args.cache_root),
        "--data-root",
        str(args.data_test_root),
        "--output-root",
        str(args.recon_output_root),
        "--runner",
        str(args.runner),
        "--matcher",
        str(args.matcher),
        "--dense-min-vertices",
        str(int(args.dense_min_vertices)),
    ]
    if not args.dense:
        cmd += ["--sparse-only"]
    if args.overwrite:
        cmd += ["--overwrite"]
    if bool(args.depth_fallback):
        cmd += ["--depth-fallback"]
        cmd += ["--depth-model-id", str(args.depth_model_id)]
        if args.hf_endpoint:
            cmd += ["--depth-hf-endpoint", str(args.hf_endpoint)]
    if args.hf_endpoint and args.matcher in {"diffusion", "auto"}:
        cmd += ["--diffusion-hf-endpoint", str(args.hf_endpoint)]
    if args.hf_endpoint and args.matcher in {"dino", "auto"}:
        cmd += ["--dino-hf-endpoint", str(args.hf_endpoint)]
    if args.matcher in {"dino", "auto"}:
        cmd += [
            "--dino-model-id",
            str(args.dino_model_id),
            "--dino-max-side",
            str(int(args.dino_max_side)),
            "--dino-max-keypoints",
            str(int(args.dino_max_keypoints)),
            "--dino-min-inliers",
            str(int(args.dino_min_inliers)),
            "--dino-min-similarity",
            str(float(args.dino_min_similarity)),
            "--dino-topk",
            str(int(args.dino_topk)),
        ]
    _run(cmd, dry_run=bool(args.dry_run), env=env)

    if args.dry_run:
        print("[dry-run] skipping submission export", flush=True)
        return

    _export_submission(
        clusters_csv=args.clusters_csv,
        data_test_root=args.data_test_root,
        recon_root=args.recon_output_root,
        out_csv=args.submission_csv,
        outliers_label=str(args.outliers_label),
        image_id_suffix=str(args.image_id_suffix),
    )

    if bool(args.demo):
        cmd = [
            sys.executable,
            "scripts/export_demo.py",
            "--submission-csv",
            str(args.submission_csv),
            "--recon-root",
            str(args.recon_output_root),
            "--out-dir",
            str(args.demo_out_dir),
            "--max-points",
            str(int(args.demo_max_points)),
        ]
        _run(cmd, dry_run=bool(args.dry_run), env=env)


if __name__ == "__main__":
    main()
