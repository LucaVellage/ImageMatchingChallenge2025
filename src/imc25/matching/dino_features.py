from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class DinoFeatureConfig:
    model_id: str = "facebook/dinov2-small"
    max_side: int = 512
    layer: int | None = None  # None => use last_hidden_state
    use_fp16: bool = True


def _get_patch_size(model) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("patch_size", "patch_sizes"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return int(v)
        if isinstance(v, (list, tuple)) and v:
            vv = v[0]
            if isinstance(vv, int) and vv > 0:
                return int(vv)
    raise ValueError(f"Could not infer patch size from model config: {type(cfg)}")


def _load_mean_std(model_id: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    try:
        from transformers import AutoImageProcessor  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers is required for DINO features (pip install transformers).") from e

    proc = AutoImageProcessor.from_pretrained(model_id)
    mean = tuple(float(x) for x in getattr(proc, "image_mean", (0.485, 0.456, 0.406)))
    std = tuple(float(x) for x in getattr(proc, "image_std", (0.229, 0.224, 0.225)))
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(f"Unexpected mean/std from processor: mean={mean} std={std}")
    return mean, std


def _resize_and_pad(img: Image.Image, *, max_side: int, patch_size: int) -> tuple[Image.Image, dict]:
    w, h = img.size
    if max_side and max_side > 0:
        scale = min(float(max_side) / float(max(w, h)), 1.0)
    else:
        scale = 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if (new_w, new_h) != (w, h):
        img_r = img.resize((new_w, new_h), resample=Image.BICUBIC)
    else:
        img_r = img
    pad_w = (-new_w) % int(patch_size)
    pad_h = (-new_h) % int(patch_size)
    if pad_w or pad_h:
        canvas = Image.new("RGB", (new_w + pad_w, new_h + pad_h), (0, 0, 0))
        canvas.paste(img_r, (0, 0))
        img_r = canvas
    meta = {
        "orig_w": int(w),
        "orig_h": int(h),
        "resized_w": int(new_w),
        "resized_h": int(new_h),
        "pad_w": int(pad_w),
        "pad_h": int(pad_h),
        "scale_x": float(new_w) / float(w) if w else 1.0,
        "scale_y": float(new_h) / float(h) if h else 1.0,
        "patch_size": int(patch_size),
        "padded_w": int(new_w + pad_w),
        "padded_h": int(new_h + pad_h),
    }
    return img_r, meta


class DinoPatchFeatureExtractor:
    def __init__(
        self,
        cfg: DinoFeatureConfig,
        *,
        device: str | None = None,
        hf_endpoint: str | None = None,
    ) -> None:
        self.config = cfg
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = str(hf_endpoint)

        try:
            from transformers import AutoModel  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("transformers is required for DINO features (pip install transformers).") from e

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if cfg.use_fp16 and self.device.type == "cuda" else torch.float32

        self.model = AutoModel.from_pretrained(cfg.model_id)
        self.model.eval()
        self.model.to(self.device, dtype=self.dtype)

        self.patch_size = _get_patch_size(self.model)
        mean, std = _load_mean_std(cfg.model_id)
        self._mean = torch.tensor(mean, dtype=self.dtype, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(std, dtype=self.dtype, device=self.device).view(1, 3, 1, 1)

    def preprocess(self, img: Image.Image) -> tuple[torch.Tensor, dict]:
        img = img.convert("RGB")
        img_p, meta = _resize_and_pad(img, max_side=int(self.config.max_side), patch_size=int(self.patch_size))
        arr = np.asarray(img_p, dtype=np.float32)
        x = torch.from_numpy(arr).to(self.device, dtype=self.dtype)
        x = x.permute(2, 0, 1).unsqueeze(0) / 255.0
        x = (x - self._mean) / self._std
        return x, meta

    @torch.inference_mode()
    def extract_patch_descriptors(self, img: Image.Image) -> tuple[np.ndarray, dict]:
        x, meta = self.preprocess(img)
        out = self.model(pixel_values=x, output_hidden_states=bool(self.config.layer is not None))
        if self.config.layer is None:
            tokens = out.last_hidden_state
        else:
            hs = getattr(out, "hidden_states", None)
            if hs is None:
                raise ValueError("Model output does not include hidden_states; cannot select layer")
            tokens = hs[int(self.config.layer)]

        if tokens.ndim != 3 or tokens.shape[0] != 1:
            raise ValueError(f"Unexpected token shape: {tuple(tokens.shape)}")
        tokens = tokens[:, 1:, :]  # drop CLS token

        gh = int(meta["padded_h"]) // int(self.patch_size)
        gw = int(meta["padded_w"]) // int(self.patch_size)
        if tokens.shape[1] != gh * gw:
            raise ValueError(f"Token count mismatch: got {tokens.shape[1]} expected {gh*gw} (gh={gh} gw={gw})")
        desc = tokens.reshape(1, gh, gw, -1)
        desc = F.normalize(desc, p=2, dim=-1)
        meta = dict(meta)
        meta["grid_h"] = gh
        meta["grid_w"] = gw
        return desc.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False), meta

    def cache_tag(self) -> str:
        model_tag = self.config.model_id.replace("/", "_").replace(":", "_")
        layer = "last" if self.config.layer is None else f"l{self.config.layer}"
        return f"{model_tag}_{layer}_s{int(self.config.max_side)}_p{int(self.patch_size)}"

