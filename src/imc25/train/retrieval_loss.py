from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError(f"Expected z (B,D), got {tuple(z.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"Expected labels (B,), got {tuple(labels.shape)}")
    if z.shape[0] != labels.shape[0]:
        raise ValueError("Batch size mismatch between z and labels")

    bsz = int(z.shape[0])
    if bsz <= 1:
        return z.new_tensor(0.0)

    z = F.normalize(z, p=2, dim=1)
    labels = labels.to(device=z.device)

    logits = (z @ z.T) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    mask = labels.view(-1, 1).eq(labels.view(1, -1)).to(dtype=torch.float32)
    mask.fill_diagonal_(0.0)

    logits_mask = torch.ones_like(mask)
    logits_mask.fill_diagonal_(0.0)

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos = mask.sum(dim=1)
    valid = pos > 0
    if not bool(valid.any()):
        return z.new_tensor(0.0)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (pos + 1e-12)
    loss = -mean_log_prob_pos[valid].mean()
    return loss

