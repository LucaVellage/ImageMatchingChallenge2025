from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

try:
    from transformers import CLIPVisionModel
except Exception:  # pragma: no cover
    CLIPVisionModel = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RetrievalModelConfig:
    model_id: str
    embed_dim: int = 256
    use_fp16: bool = True


def _load_backbone(model_id: str):
    cfg = AutoConfig.from_pretrained(model_id)
    if getattr(cfg, "model_type", "") == "clip" and CLIPVisionModel is not None:
        return CLIPVisionModel.from_pretrained(model_id)
    return AutoModel.from_pretrained(model_id)


def _backbone_dim(backbone) -> int:
    for attr in ("hidden_size", "vision_embed_dim", "projection_dim", "embedding_size"):
        v = getattr(backbone.config, attr, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    raise ValueError(f"Could not determine embedding dimension for backbone: {type(backbone)}")


def _pool_output(out) -> torch.Tensor:
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state[:, 0]
    raise ValueError("Backbone output does not expose pooler_output or last_hidden_state")


class RetrievalEmbedder(nn.Module):
    def __init__(self, cfg: RetrievalModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = _load_backbone(cfg.model_id)
        dim = _backbone_dim(self.backbone)
        self.proj = nn.Sequential(
            nn.Linear(dim, int(cfg.embed_dim)),
            nn.GELU(),
            nn.Linear(int(cfg.embed_dim), int(cfg.embed_dim)),
        )

    @property
    def embed_dim(self) -> int:
        return int(self.cfg.embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        x = _pool_output(out)
        z = self.proj(x)
        z = F.normalize(z, p=2, dim=1)
        return z

