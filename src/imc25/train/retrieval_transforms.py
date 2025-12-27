from __future__ import annotations

import os
from dataclasses import dataclass

from torchvision import transforms as T
from transformers import AutoImageProcessor


@dataclass(frozen=True)
class TransformSpec:
    image_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def _infer_size(size) -> tuple[int, int]:
    if isinstance(size, int):
        return int(size), int(size)
    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        if "shortest_edge" in size:
            s = int(size["shortest_edge"])
            return s, s
    raise ValueError(f"Unsupported processor.size: {size}")


def load_transform_spec(*, model_id: str, hf_endpoint: str | None) -> TransformSpec:
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = str(hf_endpoint)
    proc = AutoImageProcessor.from_pretrained(model_id)
    size = _infer_size(getattr(proc, "size", {"shortest_edge": 224}))
    mean = tuple(float(x) for x in getattr(proc, "image_mean", (0.485, 0.456, 0.406)))
    std = tuple(float(x) for x in getattr(proc, "image_std", (0.229, 0.224, 0.225)))
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(f"Unexpected mean/std from processor: mean={mean} std={std}")
    return TransformSpec(image_size=size, mean=mean, std=std)


def make_train_transform(spec: TransformSpec):
    h, w = spec.image_size
    return T.Compose(
        [
            T.RandomResizedCrop((h, w), scale=(0.6, 1.0), ratio=(0.75, 1.3333)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.08),
            T.RandomGrayscale(p=0.15),
            T.ToTensor(),
            T.Normalize(mean=spec.mean, std=spec.std),
        ]
    )


def make_eval_transform(spec: TransformSpec):
    h, w = spec.image_size
    resize_to = int(round(max(h, w) / 0.875))
    return T.Compose(
        [
            T.Resize(resize_to, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop((h, w)),
            T.ToTensor(),
            T.Normalize(mean=spec.mean, std=spec.std),
        ]
    )

