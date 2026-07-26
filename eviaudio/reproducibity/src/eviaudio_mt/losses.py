from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_evidence_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    balance_positive: bool = True,
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError("evidence logits and targets must have identical shapes")
    if mask is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    if mask.shape != logits.shape:
        raise ValueError("evidence mask must have the same shape as logits")
    selected_logits = logits[mask]
    selected_targets = targets[mask].to(logits.dtype)
    if selected_logits.numel() == 0:
        return logits.sum() * 0.0
    pos_weight = None
    if balance_positive:
        positives = selected_targets.sum().clamp_min(1.0)
        negatives = (1.0 - selected_targets).sum().clamp_min(1.0)
        pos_weight = (negatives / positives).clamp(1.0, 20.0)
    return F.binary_cross_entropy_with_logits(
        selected_logits,
        selected_targets,
        pos_weight=pos_weight,
    )
