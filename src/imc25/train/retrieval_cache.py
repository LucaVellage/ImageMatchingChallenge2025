from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from imc25.train.retrieval_model import RetrievalEmbedder, RetrievalModelConfig
from imc25.train.retrieval_transforms import TransformSpec, make_eval_transform


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class _PathDataset(Dataset[Path]):
    def __init__(self, paths: list[Path], *, transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image

        p = self.paths[int(idx)]
        with Image.open(p) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return x, str(p)


def _list_images(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in _IMG_EXTS:
            out.append(p)
    return out


def _load_checkpoint(path: Path) -> tuple[RetrievalModelConfig, TransformSpec, dict, dict]:
    ckpt = torch.load(path, map_location="cpu")
    model_cfg = RetrievalModelConfig(**ckpt["model_cfg"])
    spec = TransformSpec(**ckpt["transform_spec"])
    state = ckpt["state_dict"]
    meta = ckpt.get("meta", {})
    return model_cfg, spec, state, meta


def _make_pairs(emb: np.ndarray, *, k: int, mutual: bool) -> tuple[np.ndarray, np.ndarray]:
    emb = emb.astype(np.float32, copy=False)
    n = int(emb.shape[0])
    if n <= 1:
        return np.zeros((0,), np.int32), np.zeros((0,), np.int32)
    kk = int(min(k + 1, n))
    nn = NearestNeighbors(n_neighbors=kk, metric="cosine", algorithm="auto")
    nn.fit(emb)
    _dist, ind = nn.kneighbors(emb, return_distance=True)
    neigh = ind[:, 1:]

    directed: set[tuple[int, int]] = set()
    for i in range(n):
        for j in neigh[i].tolist():
            if i == int(j):
                continue
            directed.add((int(i), int(j)))

    if mutual:
        undirected: list[tuple[int, int]] = []
        for i, j in directed:
            if (j, i) in directed and i < j:
                undirected.append((i, j))
        undirected.sort()
        a = np.array([i for i, _ in undirected], dtype=np.int32)
        b = np.array([j for _, j in undirected], dtype=np.int32)
        return a, b

    edges = sorted(directed)
    a = np.array([i for i, _ in edges], dtype=np.int32)
    b = np.array([j for _, j in edges], dtype=np.int32)
    return a, b


@torch.inference_mode()
def _embed_paths(
    model: RetrievalEmbedder,
    paths: list[Path],
    *,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    ds = _PathDataset(paths, transform=transform)
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    zs: list[np.ndarray] = []
    for xb, _ in tqdm(loader, desc="embed", leave=False):
        xb = xb.to(device, non_blocking=True)
        z = model(xb).float().cpu().numpy()
        zs.append(z)
    return np.concatenate(zs, axis=0) if zs else np.zeros((0, model.embed_dim), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cache/<dataset>/{meta.json,embeddings.npy,pairs_topk_K*.npz} using a trained retrieval model")
    parser.add_argument("--data-root", type=Path, default=Path("data/test"), help="Folder with per-dataset subfolders")
    parser.add_argument("--cache-root", type=Path, default=Path("cache_retrieval"), help="Where to write per-dataset cache dirs")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt / last.pt from retrieval training")
    parser.add_argument("--hf-endpoint", default=None, help="Hugging Face Hub endpoint (mirror), e.g. https://hf-mirror.com")

    parser.add_argument("--dataset", action="append", default=None, help="Only process this dataset (repeatable)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)

    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--mutual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = str(args.hf_endpoint)

    model_cfg, spec, state, meta = _load_checkpoint(args.checkpoint)
    if args.hf_endpoint and "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = str(args.hf_endpoint)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetrievalEmbedder(model_cfg)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    tf = make_eval_transform(spec)

    datasets_filter = set(args.dataset) if args.dataset else None
    data_root = args.data_root
    cache_root = args.cache_root
    cache_root.mkdir(parents=True, exist_ok=True)

    ds_dirs = [p for p in sorted(data_root.iterdir()) if p.is_dir()]
    for ds_dir in ds_dirs:
        dataset = ds_dir.name
        if datasets_filter and dataset not in datasets_filter:
            continue

        out_dir = cache_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_path = out_dir / "embeddings.npy"
        pairs_path = out_dir / f"pairs_topk_K{int(args.topk)}.npz"
        meta_path = out_dir / "meta.json"

        if emb_path.exists() and pairs_path.exists() and meta_path.exists() and not args.overwrite:
            print(f"[skip] {dataset}: cache exists at {out_dir}")
            continue

        paths = _list_images(ds_dir)
        if not paths:
            print(f"[skip] {dataset}: no images in {ds_dir}")
            continue

        image_ids = [p.stem for p in paths]
        repo_root = Path.cwd().resolve()
        rel_paths: list[str] = []
        for p in paths:
            pr = p.resolve()
            try:
                rel_paths.append(pr.relative_to(repo_root).as_posix())
            except Exception:
                rel_paths.append(pr.as_posix())
        emb = _embed_paths(
            model,
            paths,
            transform=tf,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )

        a, b = _make_pairs(emb, k=int(args.topk), mutual=bool(args.mutual))
        np.save(emb_path, emb.astype(np.float32, copy=False))
        np.savez_compressed(pairs_path, a=a, b=b)
        meta_out = {
            "dataset": dataset,
            "backend": "retrieval_finetune",
            "model": model_cfg.model_id,
            "embed_dim": int(model_cfg.embed_dim),
            "checkpoint": str(args.checkpoint),
            "image_ids": image_ids,
            "paths": rel_paths,
            "topk": int(args.topk),
            "mutual": bool(args.mutual),
            "checkpoint_meta": meta,
        }
        meta_path.write_text(json.dumps(meta_out, indent=2))
        print(f"[ok] {dataset}: embeddings={emb.shape} pairs={len(a)} -> {out_dir}")


if __name__ == "__main__":
    main()
