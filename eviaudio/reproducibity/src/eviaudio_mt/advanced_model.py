from __future__ import annotations

from typing import Any

import torch

from .evidence_reservoir import (
    DiverseEvidenceReservoir,
    QueryConditionedTemporalPyramid,
)
from .memory import AudioMemoryDiagnostics
from .model import EviAudioMT


class AdvancedEviAudioMT(EviAudioMT):
    """EviAudio extension with causal multi-scale fusion and diverse selection.

    The historical :class:`EviAudioMT` implementation remains untouched for
    exact reproduction. This subclass overrides only token refinement and
    evidence allocation, while retaining the original decoder, loss, memory,
    and public forward/generation contracts.
    """

    def __init__(
        self,
        *args: Any,
        temporal_pyramid_dilations: tuple[int, ...] = (1, 2, 4, 8),
        reservoir_semantic_diversity_weight: float = 0.30,
        reservoir_temporal_coverage_weight: float = 0.15,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.temporal_pyramid = QueryConditionedTemporalPyramid(
            self.d_model,
            dilations=tuple(temporal_pyramid_dilations),
        )
        self.evidence_reservoir = DiverseEvidenceReservoir(
            self.evidence_top_k,
            semantic_diversity_weight=reservoir_semantic_diversity_weight,
            temporal_coverage_weight=reservoir_temporal_coverage_weight,
        )

    def _encode_audio(
        self,
        audio_embeddings: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        prompt_audio: torch.Tensor,
        memory_state: torch.Tensor | None,
        *,
        update_memory: bool,
        memory_update_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        AudioMemoryDiagnostics | None,
        torch.Tensor,
    ]:
        tokens, state, diagnostics, valid = super()._encode_audio(
            audio_embeddings,
            audio_attention_mask,
            prompt_audio,
            memory_state,
            update_memory=update_memory,
            memory_update_mask=memory_update_mask,
        )
        tokens = self.temporal_pyramid(
            tokens,
            prompt_audio,
            audio_attention_mask,
        )
        return tokens * valid, state, diagnostics, valid

    def _select(
        self,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        valid: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selection = self.evidence_reservoir(
            tokens,
            logits,
            valid,
            positions=positions,
        )
        return selection.indices.clamp_min(0), selection.valid

    def _evidence_and_prefix(
        self,
        tokens: torch.Tensor,
        prompt_audio: torch.Tensor,
        audio_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        evidence_logits = self._evidence_logits(
            tokens, prompt_audio, audio_attention_mask
        )
        weights = torch.softmax(evidence_logits, dim=1)
        soft_summary = torch.bmm(weights.unsqueeze(1), tokens)
        k = min(self.evidence_top_k, tokens.shape[1])
        top_indices, selected_valid = self._select(
            tokens,
            evidence_logits,
            audio_attention_mask.bool(),
        )
        top_indices = top_indices[:, :k]
        selected_valid = selected_valid[:, :k]
        gather_index = top_indices.unsqueeze(-1).expand(
            -1, -1, tokens.shape[-1]
        )
        top_tokens = torch.gather(tokens, 1, gather_index)
        top_tokens = top_tokens * selected_valid.to(top_tokens.dtype).unsqueeze(-1)
        prefix = torch.cat([soft_summary, top_tokens], dim=1)
        return evidence_logits, top_indices, self.prefix_projection(prefix)

    def _prepare_bounded_decoder_inputs(
        self,
        audio_embeddings: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        memory_state: torch.Tensor | None,
        *,
        update_memory: bool,
        memory_update_mask: torch.Tensor | None,
    ) -> dict[str, Any]:
        if self.stream_window_size is None:
            raise RuntimeError("Bounded preparation requires stream_window_size")
        prompt_embeddings, prompt_audio = self._prompt_summary(
            prompt_input_ids, prompt_attention_mask
        )
        batch, total_chunks, _ = audio_embeddings.shape
        state = memory_state
        reservoir_tokens = audio_embeddings.new_zeros((batch, 0, self.d_model))
        reservoir_logits = audio_embeddings.new_zeros((batch, 0))
        reservoir_indices = torch.empty(
            batch, 0, dtype=torch.long, device=audio_embeddings.device
        )
        reservoir_valid = torch.empty(
            batch, 0, dtype=torch.bool, device=audio_embeddings.device
        )
        evidence_parts: list[torch.Tensor] = []
        gate_parts: list[torch.Tensor] = []
        surprise_parts: list[torch.Tensor] = []
        relevance_parts: list[torch.Tensor] = []
        norm_parts: list[torch.Tensor] = []

        for start in range(0, total_chunks, self.stream_window_size):
            stop = min(total_chunks, start + self.stream_window_size)
            window_mask = audio_attention_mask[:, start:stop]
            update_slice = (
                None
                if memory_update_mask is None
                else memory_update_mask[:, start:stop]
            )
            tokens, state, diagnostics, _ = self._encode_audio(
                audio_embeddings[:, start:stop],
                window_mask,
                prompt_audio,
                state,
                update_memory=update_memory,
                memory_update_mask=update_slice,
            )
            logits = self._evidence_logits(tokens, prompt_audio, window_mask)
            evidence_parts.append(logits)
            indices = torch.arange(
                start,
                stop,
                dtype=torch.long,
                device=audio_embeddings.device,
            ).expand(batch, -1)
            candidate_tokens = torch.cat([reservoir_tokens, tokens], dim=1)
            candidate_logits = torch.cat([reservoir_logits, logits], dim=1)
            candidate_indices = torch.cat([reservoir_indices, indices], dim=1)
            candidate_valid = torch.cat(
                [reservoir_valid, window_mask.bool()], dim=1
            )
            keep = min(self.evidence_top_k, candidate_logits.shape[1])
            selected, selected_valid = self._select(
                candidate_tokens,
                candidate_logits,
                candidate_valid,
                candidate_indices.to(candidate_tokens.dtype),
            )
            selected = selected[:, :keep]
            selected_valid = selected_valid[:, :keep]
            token_index = selected.unsqueeze(-1).expand(
                -1, -1, self.d_model
            )
            reservoir_tokens = torch.gather(candidate_tokens, 1, token_index)
            reservoir_logits = torch.gather(candidate_logits, 1, selected)
            reservoir_indices = torch.gather(candidate_indices, 1, selected)
            reservoir_valid = selected_valid & torch.gather(
                candidate_valid, 1, selected
            )
            if diagnostics is not None:
                gate_parts.append(diagnostics.gates)
                surprise_parts.append(diagnostics.surprises)
                relevance_parts.append(diagnostics.relevance)
                norm_parts.append(diagnostics.state_norms)

        evidence_logits = torch.cat(evidence_parts, dim=1)
        normalized_scores = reservoir_logits.masked_fill(
            ~reservoir_valid, -1e4
        )
        weights = torch.softmax(normalized_scores, dim=1)
        soft_summary = torch.bmm(weights.unsqueeze(1), reservoir_tokens)
        prefix = self.prefix_projection(
            torch.cat([soft_summary, reservoir_tokens], dim=1)
        )
        prefix_mask = torch.cat(
            [
                torch.ones(
                    batch,
                    1,
                    dtype=prompt_attention_mask.dtype,
                    device=prompt_attention_mask.device,
                ),
                reservoir_valid.to(prompt_attention_mask.dtype),
            ],
            dim=1,
        )
        decoder_inputs = torch.cat([prefix, prompt_embeddings], dim=1)
        decoder_mask = torch.cat([prefix_mask, prompt_attention_mask], dim=1)
        diagnostics = None
        if gate_parts:
            diagnostics = AudioMemoryDiagnostics(
                gates=torch.cat(gate_parts, dim=1),
                surprises=torch.cat(surprise_parts, dim=1),
                relevance=torch.cat(relevance_parts, dim=1),
                state_norms=torch.cat(norm_parts, dim=1),
            )
        return {
            "decoder_inputs": decoder_inputs,
            "decoder_mask": decoder_mask,
            "evidence_logits": evidence_logits,
            "top_indices": reservoir_indices,
            "memory_state": state,
            "memory_diagnostics": diagnostics,
            "audio_attention_mask": audio_attention_mask,
            "bounded_stream": True,
            "selection_strategy": "diverse_evidence_reservoir",
            "working_set_bound_chunks": (
                self.stream_window_size + self.evidence_top_k
            ),
            "decoder_prefix_audio_tokens": 1 + reservoir_tokens.shape[1],
        }
