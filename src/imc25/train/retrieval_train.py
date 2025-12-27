from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from imc25.train.retrieval_data import RetrievalImageDataset, build_train_samples, split_train_val
from imc25.train.retrieval_loss import supervised_contrastive_loss
from imc25.train.retrieval_model import RetrievalEmbedder, RetrievalModelConfig
from imc25.train.retrieval_transforms import load_transform_spec, make_eval_transform, make_train_transform


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))


@torch.inference_mode()
def _embed_loader(model: RetrievalEmbedder, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    zs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    model.eval()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        z = model(xb).float()
        zs.append(z.detach().cpu())
        ys.append(yb.detach().cpu())
    return torch.cat(zs, dim=0), torch.cat(ys, dim=0)


def _recall_at_k(z: torch.Tensor, y: torch.Tensor, ks: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    z = F.normalize(z.float(), p=2, dim=1)
    y = y.to(dtype=torch.int64)
    n = int(z.shape[0])
    if n <= 1:
        return {f"r@{k}": 0.0 for k in ks}

    uniq, counts = torch.unique(y, return_counts=True)
    count_map = dict(zip(uniq.tolist(), counts.tolist(), strict=False))
    valid = torch.tensor([count_map[int(lbl)] > 1 for lbl in y.tolist()], dtype=torch.bool)
    if not bool(valid.any()):
        return {f"r@{k}": 0.0 for k in ks}

    z = z[valid]
    y = y[valid]
    n = int(z.shape[0])
    sim = z @ z.T
    sim.fill_diagonal_(-1e9)
    kmax = int(max(ks))
    idx = sim.topk(k=min(kmax, n - 1), dim=1).indices
    hits = y[idx] == y.view(-1, 1)
    out: dict[str, float] = {}
    for k in ks:
        kk = int(min(k, hits.shape[1]))
        out[f"r@{k}"] = float(hits[:, :kk].any(dim=1).float().mean().item())
    return out


def _save_checkpoint(path: Path, *, model: RetrievalEmbedder, cfg: RetrievalModelConfig, transform_spec, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_cfg": cfg.__dict__,
            "state_dict": model.state_dict(),
            "transform_spec": transform_spec.__dict__,
            "meta": meta,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a retrieval embedding model using IMC25 train scene labels")
    parser.add_argument("--data-root", type=Path, default=Path("data/train"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/retrieval_finetune"))
    parser.add_argument("--hf-endpoint", default=None, help="Hugging Face Hub endpoint (mirror), e.g. https://hf-mirror.com")

    parser.add_argument("--model-id", default="facebook/dinov2-small")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--include-outliers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset", action="append", default=None, help="Only train on this dataset (repeatable)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on total samples (debug)")

    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda, cuda:0, cpu)")
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = str(args.hf_endpoint)

    _set_seed(int(args.seed))

    ds_filter = set(args.dataset) if args.dataset else None
    samples, id_to_name = build_train_samples(
        data_root=args.data_root,
        labels_csv=args.labels_csv,
        datasets=ds_filter,
        include_outliers=bool(args.include_outliers),
    )
    if args.max_samples is not None:
        samples = samples[: int(args.max_samples)]

    train_samples, val_samples = split_train_val(samples, val_ratio=float(args.val_ratio), seed=int(args.seed))

    spec = load_transform_spec(model_id=str(args.model_id), hf_endpoint=str(args.hf_endpoint) if args.hf_endpoint else None)
    train_tf = make_train_transform(spec)
    eval_tf = make_eval_transform(spec)

    train_ds = RetrievalImageDataset(train_samples, transform=train_tf)
    val_ds = RetrievalImageDataset(val_samples, transform=eval_tf)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} model={args.model_id} samples={len(samples)} train={len(train_ds)} val={len(val_ds)}")

    cfg = RetrievalModelConfig(model_id=str(args.model_id), embed_dim=int(args.embed_dim), use_fp16=bool(args.fp16))
    model = RetrievalEmbedder(cfg).to(device)

    if bool(args.freeze_backbone):
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.fp16) and device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=len(train_ds) >= int(args.batch_size),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "label_map.json").write_text(json.dumps(id_to_name, indent=2, default=str))

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": str(args.model_id),
        "embed_dim": int(args.embed_dim),
        "freeze_backbone": bool(args.freeze_backbone),
        "train_size": int(len(train_ds)),
        "val_size": int(len(val_ds)),
        "args": vars(args),
    }
    (out_dir / "train_config.json").write_text(json.dumps(meta, indent=2, default=str))

    best = -1.0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        losses: list[float] = []
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(args.fp16) and device.type == "cuda"):
                z = model(xb)
                loss = supervised_contrastive_loss(z, yb, temperature=float(args.temperature))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))
            if losses:
                pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

        metrics = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0}
        if len(val_ds) > 1:
            z_val, y_val = _embed_loader(model, val_loader, device)
            rec = _recall_at_k(z_val, y_val, ks=(1, 5))
            metrics.update(rec)
        print(f"[train] epoch={epoch} " + " ".join([f"{k}={v:.4f}" for k, v in metrics.items() if k != "epoch"]))

        ckpt_path = out_dir / "last.pt"
        _save_checkpoint(ckpt_path, model=model, cfg=cfg, transform_spec=spec, meta=metrics)

        score = float(metrics.get("r@1", 0.0))
        if score >= best:
            best = score
            _save_checkpoint(out_dir / "best.pt", model=model, cfg=cfg, transform_spec=spec, meta=metrics)

    print(f"[train] done. best_r@1={best:.4f} checkpoint={out_dir/'best.pt'}")


if __name__ == "__main__":
    main()
