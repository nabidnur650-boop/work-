from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class QCRRankerOutput:
    scores: torch.Tensor
    prior_scores: torch.Tensor
    residual_scores: torch.Tensor


class CLAPPriorResidualRanker(nn.Module):
    """Low-rank query-conditioned residual on a frozen CLAP cosine prior."""

    def __init__(
        self,
        embedding_dim: int = 512,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        maximum_residual: float = 0.25,
        use_prior: bool = True,
        shared_projection: bool = False,
        interaction_only: bool = False,
        include_position: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim < 2 or hidden_dim < 2:
            raise ValueError("embedding dimensions must be at least two")
        if maximum_residual <= 0.0:
            raise ValueError("maximum_residual must be positive")
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.maximum_residual = float(maximum_residual)
        self.use_prior = bool(use_prior)
        self.shared_projection = bool(shared_projection)
        self.interaction_only = bool(interaction_only)
        self.include_position = bool(include_position)
        self.audio_projection = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.query_projection = (
            None
            if self.shared_projection
            else nn.Linear(embedding_dim, hidden_dim, bias=False)
        )
        feature_dim = (2 if self.interaction_only else 4) * hidden_dim + 1
        if self.include_position:
            feature_dim += 2
        self.residual_network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # The untrained model is exactly the auditable CLAP-only baseline.
        nn.init.zeros_(self.residual_network[-1].weight)
        nn.init.zeros_(self.residual_network[-1].bias)

    def forward(
        self,
        audio_embeddings: torch.Tensor,
        question_embeddings: torch.Tensor,
        audio_attention_mask: torch.Tensor | None = None,
    ) -> QCRRankerOutput:
        if audio_embeddings.ndim != 3:
            raise ValueError("audio_embeddings must have shape (batch, chunks, dim)")
        if question_embeddings.ndim != 2:
            raise ValueError("question_embeddings must have shape (batch, dim)")
        if audio_embeddings.shape[0] != question_embeddings.shape[0]:
            raise ValueError("audio and question batch sizes differ")
        if (
            audio_embeddings.shape[-1] != self.embedding_dim
            or question_embeddings.shape[-1] != self.embedding_dim
        ):
            raise ValueError("embedding dimension mismatch")
        batch, chunks, _ = audio_embeddings.shape
        if audio_attention_mask is None:
            audio_attention_mask = torch.ones(
                batch, chunks, dtype=torch.bool, device=audio_embeddings.device
            )
        if audio_attention_mask.shape != (batch, chunks):
            raise ValueError("audio_attention_mask shape mismatch")

        audio = nn.functional.normalize(audio_embeddings.float(), dim=-1)
        question = nn.functional.normalize(question_embeddings.float(), dim=-1)
        prior = torch.einsum("bnd,bd->bn", audio, question)
        audio_hidden = self.audio_projection(audio)
        query_projection = (
            self.audio_projection if self.query_projection is None else self.query_projection
        )
        question_hidden = query_projection(question).unsqueeze(1).expand(-1, chunks, -1)
        if chunks > 1:
            position = torch.linspace(
                0.0, 1.0, chunks, device=audio.device, dtype=audio.dtype
            )
        else:
            position = torch.zeros(1, device=audio.device, dtype=audio.dtype)
        position = position.view(1, chunks, 1).expand(batch, -1, -1)
        feature_parts = []
        if not self.interaction_only:
            feature_parts.extend([audio_hidden, question_hidden])
        feature_parts.extend(
            [
                audio_hidden * question_hidden,
                torch.abs(audio_hidden - question_hidden),
                prior.unsqueeze(-1),
            ]
        )
        if self.include_position:
            feature_parts.extend([position, torch.sin(torch.pi * position)])
        features = torch.cat(feature_parts, dim=-1)
        residual = self.maximum_residual * torch.tanh(
            self.residual_network(features).squeeze(-1)
        )
        scores = (prior if self.use_prior else torch.zeros_like(prior)) + residual
        invalid = ~audio_attention_mask.bool()
        return QCRRankerOutput(
            scores=scores.masked_fill(invalid, -1e4),
            prior_scores=prior.masked_fill(invalid, -1e4),
            residual_scores=residual.masked_fill(invalid, 0.0),
        )


class CLAPDiagonalMetricRanker(nn.Module):
    """Data-efficient diagonal metric adaptation initialized exactly to CLAP."""

    def __init__(
        self,
        embedding_dim: int = 512,
        *,
        maximum_residual: float = 0.25,
        use_prior: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim < 2 or maximum_residual <= 0.0:
            raise ValueError("invalid diagonal ranker dimensions")
        self.embedding_dim = int(embedding_dim)
        self.maximum_residual = float(maximum_residual)
        self.use_prior = bool(use_prior)
        self.log_dimension_weights = nn.Parameter(torch.zeros(embedding_dim))

    def forward(
        self,
        audio_embeddings: torch.Tensor,
        question_embeddings: torch.Tensor,
        audio_attention_mask: torch.Tensor | None = None,
    ) -> QCRRankerOutput:
        if audio_embeddings.ndim != 3 or question_embeddings.ndim != 2:
            raise ValueError("invalid embedding ranks")
        if audio_embeddings.shape[0] != question_embeddings.shape[0]:
            raise ValueError("audio and question batch sizes differ")
        if audio_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError("audio embedding dimension mismatch")
        if question_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError("question embedding dimension mismatch")
        batch, chunks, _ = audio_embeddings.shape
        if audio_attention_mask is None:
            audio_attention_mask = torch.ones(
                batch, chunks, dtype=torch.bool, device=audio_embeddings.device
            )
        audio = nn.functional.normalize(audio_embeddings.float(), dim=-1)
        question = nn.functional.normalize(question_embeddings.float(), dim=-1)
        prior = torch.einsum("bnd,bd->bn", audio, question)
        weights = self.log_dimension_weights.clamp(-4.0, 4.0).exp()
        weights = weights / weights.mean()
        numerator = torch.einsum("bnd,bd,d->bn", audio, question, weights)
        audio_norm = torch.sqrt(
            torch.einsum("bnd,bnd,d->bn", audio, audio, weights).clamp_min(1e-8)
        )
        question_norm = torch.sqrt(
            torch.einsum("bd,bd,d->b", question, question, weights).clamp_min(1e-8)
        )
        adapted = numerator / (audio_norm * question_norm[:, None]).clamp_min(1e-8)
        residual = self.maximum_residual * torch.tanh(
            (adapted - prior) / self.maximum_residual
        )
        scores = (prior if self.use_prior else torch.zeros_like(prior)) + residual
        invalid = ~audio_attention_mask.bool()
        return QCRRankerOutput(
            scores=scores.masked_fill(invalid, -1e4),
            prior_scores=prior.masked_fill(invalid, -1e4),
            residual_scores=residual.masked_fill(invalid, 0.0),
        )


def source_logmeanexp(
    chunk_scores: torch.Tensor,
    source_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool chunks into length-normalized source scores."""

    if chunk_scores.ndim != 2 or source_indices.shape != chunk_scores.shape:
        raise ValueError("scores and source_indices must be matching rank-two tensors")
    valid = source_indices >= 0
    maximum_sources = int(source_indices[valid].max().item()) + 1 if valid.any() else 0
    if maximum_sources == 0:
        return (
            chunk_scores.new_empty((chunk_scores.shape[0], 0)),
            torch.empty(
                chunk_scores.shape[0], 0, dtype=torch.bool, device=chunk_scores.device
            ),
        )
    source_numbers = torch.arange(maximum_sources, device=source_indices.device)
    selected = source_indices.unsqueeze(-1) == source_numbers.view(1, 1, -1)
    counts = selected.sum(dim=1)
    source_mask = counts > 0
    expanded_scores = chunk_scores.unsqueeze(-1).expand(-1, -1, maximum_sources)
    pooled = torch.logsumexp(
        expanded_scores.masked_fill(~selected, -torch.inf), dim=1
    ) - counts.clamp_min(1).to(chunk_scores.dtype).log()
    pooled = pooled.masked_fill(~source_mask, -1e4)
    return pooled, source_mask


def qcr_source_loss(
    output: QCRRankerOutput,
    source_indices: torch.Tensor,
    target_source_indices: torch.Tensor,
    *,
    residual_penalty: float = 0.01,
    source_temperature: float = 1.0,
    evidence_targets: torch.Tensor | None = None,
    evidence_weight: float = 0.0,
    evidence_temperature: float = 0.1,
) -> torch.Tensor:
    if source_temperature <= 0.0 or evidence_temperature <= 0.0:
        raise ValueError("loss temperatures must be positive")
    if evidence_weight < 0.0:
        raise ValueError("evidence_weight must be non-negative")
    source_scores, source_mask = source_logmeanexp(output.scores, source_indices)
    if source_scores.shape[1] == 0:
        raise ValueError("batch contains no valid source chunks")
    source_scores = source_scores.masked_fill(~source_mask, -1e4)
    classification = nn.functional.cross_entropy(
        source_scores / float(source_temperature), target_source_indices
    )
    valid_chunks = source_indices >= 0
    regularization = output.residual_scores[valid_chunks].square().mean()
    loss = classification + float(residual_penalty) * regularization
    if evidence_weight > 0.0:
        if evidence_targets is None or evidence_targets.shape != output.scores.shape:
            raise ValueError("evidence_targets must match score shape")
        positive = evidence_targets.to(output.scores.dtype) * valid_chunks
        positive_count = positive.sum(dim=1)
        if torch.any(positive_count <= 0):
            raise ValueError("every example must contain positive evidence")
        log_probabilities = nn.functional.log_softmax(
            output.scores.masked_fill(~valid_chunks, -1e4)
            / float(evidence_temperature),
            dim=1,
        )
        # The development labels identify the relevant source rather than a
        # single privileged chunk. A uniform target over its chunks therefore
        # matches the declared evidence-AP estimand without inventing timing
        # supervision that the dataset does not contain.
        evidence = -(
            log_probabilities * positive / positive_count[:, None]
        ).sum(dim=1).mean()
        loss = loss + float(evidence_weight) * evidence
    return loss


def qcr_evidence_loss(
    output: QCRRankerOutput,
    evidence_targets: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    evidence_temperature: float = 0.1,
    hard_negative_weight: float = 0.25,
    margin: float = 0.0,
    residual_penalty: float = 0.01,
) -> torch.Tensor:
    """Listwise evidence loss plus a within-recording hardest-negative term."""

    if evidence_targets.shape != output.scores.shape:
        raise ValueError("evidence_targets must match scores")
    if attention_mask.shape != output.scores.shape:
        raise ValueError("attention_mask must match scores")
    if evidence_temperature <= 0.0 or hard_negative_weight < 0.0:
        raise ValueError("invalid evidence-loss hyperparameters")
    valid = attention_mask.bool()
    positive = evidence_targets.to(output.scores.dtype) * valid
    positive_count = positive.sum(dim=1)
    if torch.any(positive_count <= 0):
        raise ValueError("every recording must contain positive evidence")
    if torch.any((valid & ~positive.bool()).sum(dim=1) <= 0):
        raise ValueError("every recording must contain negative chunks")

    scaled = output.scores.masked_fill(~valid, -1e4) / float(evidence_temperature)
    log_probabilities = nn.functional.log_softmax(scaled, dim=1)
    listwise = -(
        log_probabilities * positive / positive_count[:, None]
    ).sum(dim=1).mean()

    positive_mask = positive.bool()
    negative_mask = valid & ~positive_mask
    positive_score = torch.logsumexp(
        output.scores.masked_fill(~positive_mask, -torch.inf)
        / float(evidence_temperature),
        dim=1,
    ) * float(evidence_temperature)
    positive_score = positive_score - positive_count.log() * float(evidence_temperature)
    hardest_negative = output.scores.masked_fill(~negative_mask, -torch.inf).max(dim=1).values
    pairwise = nn.functional.softplus(
        (hardest_negative - positive_score + float(margin))
        / float(evidence_temperature)
    ).mean()
    residual_regularization = output.residual_scores[valid].square().mean()
    return (
        listwise
        + float(hard_negative_weight) * pairwise
        + float(residual_penalty) * residual_regularization
    )
