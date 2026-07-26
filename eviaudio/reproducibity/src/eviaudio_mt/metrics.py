from __future__ import annotations

from typing import Iterable

import torch


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    mask = labels.ne(-100)
    if not mask.any():
        return 0.0
    return float((predictions[mask] == labels[mask]).float().mean().item())


def sequence_exact_match(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    mask = labels.ne(-100)
    matches = ((predictions == labels) | ~mask).all(dim=1)
    return float(matches.float().mean().item())


def evidence_recall_at_k(
    logits: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int,
) -> float:
    k = min(k, logits.shape[1])
    masked_logits = logits.masked_fill(~attention_mask.bool(), -1e4)
    indices = torch.topk(masked_logits, k=k, dim=1).indices
    gathered = torch.gather(targets, 1, indices)
    has_gold = targets.sum(dim=1) > 0
    hit = gathered.sum(dim=1) > 0
    if not has_gold.any():
        return 0.0
    return float(hit[has_gold].float().mean().item())


def chunk_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    predictions = torch.sigmoid(logits) >= threshold
    predictions &= attention_mask.bool()
    gold = targets.bool() & attention_mask.bool()
    intersection = (predictions & gold).sum(dim=1).float()
    union = (predictions | gold).sum(dim=1).float()
    valid = union > 0
    if not valid.any():
        return 0.0
    return float((intersection[valid] / union[valid]).mean().item())


def logits_to_intervals(
    logits: torch.Tensor,
    start_sec: torch.Tensor,
    end_sec: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    threshold: float = 0.5,
    top_k: int | None = None,
    merge_tolerance_sec: float = 1e-6,
) -> list[list[tuple[float, float, float]]]:
    """Convert chunk logits to merged ``(start, end, max_probability)`` intervals."""
    probabilities = torch.sigmoid(logits)
    all_intervals: list[list[tuple[float, float, float]]] = []
    for batch_index in range(logits.shape[0]):
        valid_indices = torch.nonzero(attention_mask[batch_index].bool(), as_tuple=False).flatten()
        if top_k is not None:
            count = min(top_k, valid_indices.numel())
            values = probabilities[batch_index, valid_indices]
            chosen = valid_indices[torch.topk(values, k=count).indices] if count > 0 else valid_indices
        else:
            chosen = valid_indices[probabilities[batch_index, valid_indices] >= threshold]
        chosen = chosen.sort().values
        intervals: list[tuple[float, float, float]] = []
        for index_tensor in chosen:
            index = int(index_tensor.item())
            start = float(start_sec[batch_index, index].item())
            end = float(end_sec[batch_index, index].item())
            score = float(probabilities[batch_index, index].item())
            if intervals and start <= intervals[-1][1] + merge_tolerance_sec:
                previous = intervals[-1]
                intervals[-1] = (previous[0], max(previous[1], end), max(previous[2], score))
            else:
                intervals.append((start, end, score))
        all_intervals.append(intervals)
    return all_intervals
