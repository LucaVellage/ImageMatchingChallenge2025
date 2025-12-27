from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class DiffusionFeatureConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    timestep: int = 200
    max_side: int = 512
    noise_seed: int = 0
    layers: tuple[str, ...] = ("d0", "d1", "d2", "mid")
    use_fp16: bool = True


class _FeatureHook:
    def __init__(self, modules: dict[str, torch.nn.Module]) -> None:
        self._modules = modules
        self.activations: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "_FeatureHook":
        self.activations = {}

        def make_hook(name: str):
            def _hook(_module, _inputs, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output

            return _hook

        for name, module in self._modules.items():
            self._handles.append(module.register_forward_hook(make_hook(name)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def _resize_keep_aspect_multiple_of_8(img: Image.Image, max_side: int) -> tuple[Image.Image, float, float]:
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Bad image size: {img.size}")

    scale = min(1.0, float(max_side) / float(max(w, h)))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    new_w = max(8, new_w - (new_w % 8))
    new_h = max(8, new_h - (new_h % 8))
    if new_w == w and new_h == h:
        return img, 1.0, 1.0

    resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
    return resized, float(new_w) / float(w), float(new_h) / float(h)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t.unsqueeze(0)


def _require_diffusers():
    try:
        from diffusers import StableDiffusionPipeline  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing diffusion dependencies. Install with: `pip install -e .[diffusion]`"
        ) from e


class DiffusionUNetFeatureExtractor:
    """
    Extracts multi-scale dense descriptors from a Stable Diffusion U-Net.

    This is intended for research/demo matching (not for fast production matching).
    """

    def __init__(self, config: DiffusionFeatureConfig, *, device: str | None = None) -> None:
        _require_diffusers()
        from diffusers import StableDiffusionPipeline

        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        dtype = torch.float16 if (self.device.type == "cuda" and config.use_fp16) else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
        )
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)

        self.vae = pipe.vae.eval()
        self.unet = pipe.unet.eval()
        self.scheduler = pipe.scheduler
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.eval()

        with torch.inference_mode():
            tokens = self.tokenizer(
                [""],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            self._text_emb = self.text_encoder(tokens.input_ids.to(self.device))[0]

        # Select hook modules.
        modules: dict[str, torch.nn.Module] = {}
        layers = set(config.layers)
        if "d0" in layers:
            modules["d0"] = self.unet.down_blocks[0].resnets[0]
        if "d1" in layers and len(self.unet.down_blocks) > 1:
            modules["d1"] = self.unet.down_blocks[1].resnets[0]
        if "d2" in layers and len(self.unet.down_blocks) > 2:
            modules["d2"] = self.unet.down_blocks[2].resnets[0]
        if "d3" in layers and len(self.unet.down_blocks) > 3:
            modules["d3"] = self.unet.down_blocks[3].resnets[0]
        if "mid" in layers:
            modules["mid"] = self.unet.mid_block

        if not modules:
            raise ValueError(f"No valid layers selected: {config.layers}")
        self._hook_modules = modules

    def extract_descriptor_map(self, image: Image.Image) -> tuple[torch.Tensor, dict]:
        """
        Returns:
          desc_map: (C, H_lat, W_lat) float32 on CPU
          meta: dict with scaling info from original->resized
        """
        img = image.convert("RGB")
        resized, sx, sy = _resize_keep_aspect_multiple_of_8(img, self.config.max_side)
        x = _pil_to_tensor(resized).to(self.device, dtype=self.unet.dtype)

        with torch.inference_mode():
            latent_dist = self.vae.encode(x).latent_dist
            if hasattr(latent_dist, "mode"):
                latents = latent_dist.mode()
            elif hasattr(latent_dist, "mean"):
                latents = latent_dist.mean
            else:  # pragma: no cover
                latents = latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            t = torch.tensor([int(self.config.timestep)], device=self.device, dtype=torch.long)
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(self.config.noise_seed))
            noise = torch.randn(latents.shape, generator=gen, device=self.device, dtype=latents.dtype)
            noised = self.scheduler.add_noise(latents, noise, t)

            with _FeatureHook(self._hook_modules) as hook:
                _ = self.unet(noised, t, encoder_hidden_states=self._text_emb).sample

            h_lat, w_lat = int(noised.shape[-2]), int(noised.shape[-1])
            feats: list[torch.Tensor] = []
            for name, feat in hook.activations.items():
                if feat.ndim != 4:
                    continue
                if feat.shape[-2:] != (h_lat, w_lat):
                    feat = F.interpolate(feat, size=(h_lat, w_lat), mode="bilinear", align_corners=False)
                feats.append(feat)

            if not feats:
                raise RuntimeError("No features captured from the U-Net (check layer names/model)")

            desc = torch.cat(feats, dim=1).float()
            desc = F.normalize(desc, dim=1, eps=1e-6)
            desc_map = desc.squeeze(0).detach().cpu()  # (C,H,W)

        meta = {
            "orig_size": tuple(img.size),
            "resized_size": tuple(resized.size),
            "scale_x": float(sx),
            "scale_y": float(sy),
        }
        return desc_map, meta

    @staticmethod
    def sample_descriptors(desc_map: torch.Tensor, *, keypoints_xy_resized: np.ndarray) -> np.ndarray:
        """
        desc_map: (C, H_lat, W_lat) on CPU
        keypoints_xy_resized: (N,2) in resized image pixel coords
        Returns: (N,C) float32
        """
        if keypoints_xy_resized.ndim != 2 or keypoints_xy_resized.shape[1] != 2:
            raise ValueError(f"Expected keypoints (N,2), got {keypoints_xy_resized.shape}")

        c, h_lat, w_lat = desc_map.shape
        # Stable Diffusion latents are 1/8 resolution of the input image.
        x_lat = keypoints_xy_resized[:, 0] / 8.0
        y_lat = keypoints_xy_resized[:, 1] / 8.0

        # Normalize to [-1, 1] with align_corners=True semantics.
        x_norm = (x_lat / max(1.0, (w_lat - 1))) * 2.0 - 1.0
        y_norm = (y_lat / max(1.0, (h_lat - 1))) * 2.0 - 1.0

        grid = np.stack([x_norm, y_norm], axis=-1).astype(np.float32)
        grid_t = torch.from_numpy(grid).view(1, -1, 1, 2)

        feat_t = desc_map.unsqueeze(0)  # (1,C,H,W)
        sampled = F.grid_sample(feat_t, grid_t, mode="bilinear", padding_mode="border", align_corners=True)
        sampled = sampled.squeeze(0).squeeze(-1).T.contiguous()  # (N,C)
        sampled = F.normalize(sampled, dim=1, eps=1e-6)
        return sampled.numpy().astype(np.float32, copy=False)
