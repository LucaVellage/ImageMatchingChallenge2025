from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class RetrievalSample:
    path: Path
    dataset: str
    scene: str
    image_id: str
    label: int


class RetrievalImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, samples: list[RetrievalSample], *, transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        s = self.samples[int(idx)]
        with Image.open(s.path) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return x, int(s.label)


def _list_images(dataset_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(dataset_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in _IMG_EXTS:
            out.append(p)
    return out


def build_train_samples(
    *,
    data_root: Path,
    labels_csv: Path,
    datasets: set[str] | None = None,
    include_outliers: bool = True,
) -> tuple[list[RetrievalSample], dict[int, str]]:
    df = pd.read_csv(labels_csv)
    required = {"dataset", "scene", "image"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{labels_csv} missing columns: {sorted(missing)}")

    df = df.copy()
    df["dataset"] = df["dataset"].astype(str)
    df["scene"] = df["scene"].astype(str)
    df["image"] = df["image"].astype(str)

    if datasets is not None:
        df = df[df["dataset"].isin(sorted(datasets))].reset_index(drop=True)

    label_names: list[str] = []
    for row in df.itertuples(index=False):
        label_names.append(f"{row.dataset}/{row.scene}")

    label_name_to_id: dict[str, int] = {}
    id_to_name: dict[int, str] = {}

    def get_label_id(name: str) -> int:
        if name in label_name_to_id:
            return label_name_to_id[name]
        idx = len(label_name_to_id)
        label_name_to_id[name] = idx
        id_to_name[idx] = name
        return idx

    samples: list[RetrievalSample] = []
    labeled_by_dataset: dict[str, set[str]] = {}
    for row in df.itertuples(index=False):
        dataset = str(row.dataset)
        scene = str(row.scene)
        img_name = str(row.image)
        path = (data_root / dataset / img_name).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        image_id = Path(img_name).stem
        labeled_by_dataset.setdefault(dataset, set()).add(image_id)
        label = get_label_id(f"{dataset}/{scene}")
        samples.append(
            RetrievalSample(
                path=path,
                dataset=dataset,
                scene=scene,
                image_id=image_id,
                label=label,
            )
        )

    if include_outliers:
        for dataset_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
            dataset = dataset_dir.name
            if datasets is not None and dataset not in datasets:
                continue
            labeled = labeled_by_dataset.get(dataset, set())
            for p in _list_images(dataset_dir):
                image_id = p.stem
                if image_id in labeled:
                    continue
                label = get_label_id(f"{dataset}/outlier/{image_id}")
                samples.append(
                    RetrievalSample(
                        path=p.resolve(),
                        dataset=dataset,
                        scene="outlier",
                        image_id=image_id,
                        label=label,
                    )
                )

    return samples, id_to_name


def split_train_val(
    samples: list[RetrievalSample],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[RetrievalSample], list[RetrievalSample]]:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0,1)")

    by_label: dict[int, list[int]] = {}
    for i, s in enumerate(samples):
        by_label.setdefault(int(s.label), []).append(i)

    rng = random.Random(int(seed))
    train_idx: set[int] = set()
    val_idx: set[int] = set()

    for _, idxs in by_label.items():
        if len(idxs) <= 1:
            train_idx.update(idxs)
            continue
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_val = int(round(len(idxs) * float(val_ratio)))
        n_val = max(1, min(n_val, len(idxs) - 1))
        val_idx.update(idxs[:n_val])
        train_idx.update(idxs[n_val:])

    train = [samples[i] for i in sorted(train_idx)]
    val = [samples[i] for i in sorted(val_idx)]
    return train, val

