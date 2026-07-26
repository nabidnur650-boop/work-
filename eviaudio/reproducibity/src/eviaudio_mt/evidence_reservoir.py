from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class _CausalDilatedBranch(nn.Module):
    def __init__(self, d_model: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=3,
            dilation=self.dilation,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        left_padding = 2 * self.dilation
        values = F.pad(tokens.transpose(1, 2), (left_padding, 0))
        values = self.depthwise(values)
        return self.dropout(self.pointwise(F.silu(values)).transpose(1, 2))


class QueryConditionedTemporalPyramid(nn.Module):
    """Causal multi-scale audio adapter with prompt-conditioned scale fusion."""

    def __init__(
        self,
        d_model: int,
        *,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model < 1 or not dilations or any(value < 1 for value in dilations):
            raise ValueError("invalid temporal-pyramid dimensions")
        self.d_model = int(d_model)
        self.dilations = tuple(int(value) for value in dilations)
        self.norm = nn.LayerNorm(d_model)
        self.prompt_norm = nn.LayerNorm(d_model)
        self.branches = nn.ModuleList(
            [
                _CausalDilatedBranch(d_model, dilation, dropout)
                for dilation in self.dilations
            ]
        )
        self.prompt_gate = nn.Linear(d_model, len(self.dilations))
        self.token_gate = nn.Linear(d_model, len(self.dilations))
        self.output = nn.Linear(d_model, d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        prompt: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B,N,D]")
        if prompt.shape != (tokens.shape[0], tokens.shape[2]):
            raise ValueError("prompt shape mismatch")
        normalized = self.norm(tokens)
        branch_values = torch.stack(
            [branch(normalized) for branch in self.branches],
            dim=2,
        )
        prompt_logits = self.prompt_gate(self.prompt_norm(prompt)).unsqueeze(1)
        token_logits = self.token_gate(normalized)
        scale_weights = torch.softmax(prompt_logits + token_logits, dim=-1)
        fused = (branch_values * scale_weights.unsqueeze(-1)).sum(dim=2)
        output = tokens + self.output(fused)
        if attention_mask is not None:
            if attention_mask.shape != tokens.shape[:2]:
                raise ValueError("attention_mask shape mismatch")
            output = output * attention_mask.to(output.dtype).unsqueeze(-1)
        return output


@dataclass(frozen=True)
class ReservoirSelection:
    indices: torch.Tensor
    valid: torch.Tensor
    marginal_utility: torch.Tensor


class DiverseEvidenceReservoir(nn.Module):
    """Greedy bounded evidence selection with relevance and coverage utility.

    Selection is intentionally deterministic. Relevance is normalized within
    each example, while redundancy, temporal proximity, and repeated source
    membership are penalized. The selected token values remain differentiable;
    only the discrete index choice is non-differentiable, like ordinary top-K.
    """

    def __init__(
        self,
        budget: int,
        *,
        relevance_weight: float = 1.0,
        semantic_diversity_weight: float = 0.30,
        temporal_coverage_weight: float = 0.15,
        new_group_bonus: float = 0.20,
        uncertainty_weight: float = 0.15,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if budget < 1:
            raise ValueError("budget must be positive")
        nonnegative = {
            "relevance_weight": relevance_weight,
            "semantic_diversity_weight": semantic_diversity_weight,
            "temporal_coverage_weight": temporal_coverage_weight,
            "new_group_bonus": new_group_bonus,
            "uncertainty_weight": uncertainty_weight,
        }
        if any(value < 0.0 for value in nonnegative.values()):
            raise ValueError("reservoir utility weights must be non-negative")
        self.budget = int(budget)
        self.relevance_weight = float(relevance_weight)
        self.semantic_diversity_weight = float(semantic_diversity_weight)
        self.temporal_coverage_weight = float(temporal_coverage_weight)
        self.new_group_bonus = float(new_group_bonus)
        self.uncertainty_weight = float(uncertainty_weight)
        self.eps = float(eps)

    def forward(
        self,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        positions: torch.Tensor | None = None,
        group_indices: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
    ) -> ReservoirSelection:
        if tokens.ndim != 3 or scores.shape != tokens.shape[:2]:
            raise ValueError("tokens and scores must have shapes [B,N,D] and [B,N]")
        batch, length, _ = tokens.shape
        device = tokens.device
        if valid_mask is None:
            valid_mask = torch.ones(batch, length, dtype=torch.bool, device=device)
        if valid_mask.shape != (batch, length):
            raise ValueError("valid_mask shape mismatch")
        if positions is None:
            positions = torch.arange(
                length, device=device, dtype=tokens.dtype
            ).expand(batch, -1)
        if positions.shape != (batch, length):
            raise ValueError("positions shape mismatch")
        if group_indices is not None and group_indices.shape != (batch, length):
            raise ValueError("group_indices shape mismatch")
        if uncertainty is None:
            uncertainty = torch.zeros_like(scores)
        if uncertainty.shape != scores.shape:
            raise ValueError("uncertainty shape mismatch")

        normalized_tokens = F.normalize(tokens.float(), dim=-1)
        chosen = torch.full(
            (batch, self.budget), -1, dtype=torch.long, device=device
        )
        chosen_valid = torch.zeros(
            (batch, self.budget), dtype=torch.bool, device=device
        )
        utilities = torch.full(
            (batch, self.budget),
            float("nan"),
            dtype=scores.dtype,
            device=device,
        )

        for row in range(batch):
            candidates = torch.nonzero(valid_mask[row], as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            row_scores = scores[row, candidates].float()
            center = row_scores.median()
            scale = (row_scores - center).abs().median()
            if float(scale.item()) <= self.eps:
                scale = row_scores.std(unbiased=False).clamp_min(self.eps)
            relevance = torch.sigmoid((scores[row].float() - center) / scale)
            row_positions = positions[row].float()
            position_span = (
                row_positions[candidates].max() - row_positions[candidates].min()
            ).clamp_min(1.0)
            selected: list[int] = []
            selected_groups: set[int] = set()
            for slot in range(min(self.budget, int(candidates.numel()))):
                best_index = -1
                best_utility = -float("inf")
                for candidate_tensor in candidates:
                    candidate = int(candidate_tensor.item())
                    if candidate in selected:
                        continue
                    redundancy = 0.0
                    temporal_coverage = 1.0
                    if selected:
                        selected_tensor = torch.tensor(selected, device=device)
                        redundancy = float(
                            (
                                normalized_tokens[row, candidate]
                                @ normalized_tokens[row, selected_tensor].T
                            )
                            .clamp(min=0.0)
                            .max()
                            .item()
                        )
                        temporal_coverage = float(
                            (
                                (
                                    row_positions[candidate]
                                    - row_positions[selected_tensor]
                                )
                                .abs()
                                .min()
                                / position_span
                            ).item()
                        )
                    group_bonus = 0.0
                    if group_indices is not None:
                        group = int(group_indices[row, candidate].item())
                        if group >= 0 and group not in selected_groups:
                            group_bonus = self.new_group_bonus
                    utility = (
                        self.relevance_weight
                        * float(relevance[candidate].item())
                        - self.semantic_diversity_weight * redundancy
                        + self.temporal_coverage_weight * temporal_coverage
                        + group_bonus
                        - self.uncertainty_weight
                        * float(uncertainty[row, candidate].item())
                    )
                    if utility > best_utility + self.eps or (
                        abs(utility - best_utility) <= self.eps
                        and (best_index < 0 or candidate < best_index)
                    ):
                        best_index = candidate
                        best_utility = utility
                selected.append(best_index)
                if group_indices is not None:
                    group = int(group_indices[row, best_index].item())
                    if group >= 0:
                        selected_groups.add(group)
                chosen[row, slot] = best_index
                chosen_valid[row, slot] = True
                utilities[row, slot] = best_utility

        return ReservoirSelection(
            indices=chosen,
            valid=chosen_valid,
            marginal_utility=utilities,
        )


def hierarchical_source_select(
    scores: torch.Tensor,
    source_indices: torch.Tensor,
    budget: int,
    *,
    pooling: str = "logmeanexp",
    temperature: float = 0.10,
) -> ReservoirSelection:
    """Allocate evidence first across sources, then to chunks within each source."""

    if scores.ndim != 2 or source_indices.shape != scores.shape:
        raise ValueError("scores and source_indices must be matching [B,N] tensors")
    if budget < 1 or temperature <= 0.0:
        raise ValueError("budget and temperature must be positive")
    if pooling not in {"max", "mean", "logmeanexp"}:
        raise ValueError("pooling must be max, mean, or logmeanexp")
    batch, _ = scores.shape
    selected = torch.full(
        (batch, budget), -1, dtype=torch.long, device=scores.device
    )
    valid = torch.zeros((batch, budget), dtype=torch.bool, device=scores.device)
    utilities = torch.full(
        (batch, budget),
        float("nan"),
        dtype=scores.dtype,
        device=scores.device,
    )
    for row in range(batch):
        row_sources = source_indices[row]
        unique_sources = torch.unique(row_sources[row_sources >= 0], sorted=True)
        source_records: list[tuple[float, int, int]] = []
        for source_tensor in unique_sources:
            source = int(source_tensor.item())
            chunk_indices = torch.nonzero(
                row_sources == source, as_tuple=False
            ).flatten()
            chunk_scores = scores[row, chunk_indices]
            if pooling == "max":
                pooled = chunk_scores.max()
            elif pooling == "mean":
                pooled = chunk_scores.mean()
            else:
                pooled = temperature * (
                    torch.logsumexp(chunk_scores / temperature, dim=0)
                    - math.log(len(chunk_indices))
                )
            best_local = int(torch.argmax(chunk_scores).item())
            source_records.append(
                (
                    float(pooled.item()),
                    source,
                    int(chunk_indices[best_local].item()),
                )
            )
        source_records.sort(key=lambda item: (-item[0], item[1], item[2]))
        chosen: list[int] = []
        for pooled, _, chunk in source_records[:budget]:
            slot = len(chosen)
            selected[row, slot] = chunk
            valid[row, slot] = True
            utilities[row, slot] = pooled
            chosen.append(chunk)
        if len(chosen) < budget:
            remaining = [
                int(index.item())
                for index in torch.argsort(scores[row], descending=True)
                if source_indices[row, index] >= 0
                and int(index.item()) not in chosen
            ]
            for chunk in remaining[: budget - len(chosen)]:
                slot = len(chosen)
                selected[row, slot] = chunk
                valid[row, slot] = True
                utilities[row, slot] = scores[row, chunk]
                chosen.append(chunk)
    return ReservoirSelection(
        indices=selected,
        valid=valid,
        marginal_utility=utilities,
    )
