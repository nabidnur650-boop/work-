from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .blocks import build_temporal_block
from .decoders import HFSeq2SeqDecoder, ToySeq2SeqDecoder
from .losses import masked_evidence_bce
from .memory import AudioMemoryDiagnostics, PromptConditionedFastWeightMemory


class EviAudioMT(nn.Module):
    """Evidence-grounded audio-text model operating on chunk embeddings."""

    def __init__(
        self,
        audio_dim: int,
        *,
        d_model: int = 256,
        n_blocks: int = 4,
        backend: str = "fallback",
        d_state: int = 64,
        dropout: float = 0.1,
        evidence_top_k: int = 8,
        decoder_type: str = "toy",
        decoder_name: str | None = None,
        decoder_revision: str | None = None,
        decoder_frozen: bool = False,
        decoder_vocab_size: int = 128,
        decoder_d_model: int = 128,
        decoder_max_target_length: int = 32,
        memory_enabled: bool = True,
        memory_decay: float = 0.97,
        memory_update_rate: float = 0.05,
        memory_max_state_norm: float = 50.0,
        memory_detach_returned_state: bool = True,
        memory_gate_mode: str = "learned",
        memory_constant_gate: float = 0.5,
        control_adapter: str = "none",
        stream_window_size: int | None = None,
        evidence_weight: float = 1.0,
        gate_regularization_weight: float = 0.0,
        gate_target_density: float = 0.15,
    ) -> None:
        super().__init__()
        if audio_dim < 1 or d_model < 1:
            raise ValueError("audio_dim and d_model must be positive")
        if evidence_top_k < 1:
            raise ValueError("evidence_top_k must be positive")
        self.audio_dim = audio_dim
        self.d_model = d_model
        self.evidence_top_k = evidence_top_k
        self.memory_enabled = memory_enabled
        self.stream_window_size = (
            int(stream_window_size) if stream_window_size is not None else None
        )
        if self.stream_window_size is not None and self.stream_window_size < 1:
            raise ValueError("stream_window_size must be positive when provided")
        control_adapter = control_adapter.lower()
        if control_adapter not in {"none", "matched_mlp"}:
            raise ValueError("control_adapter must be 'none' or 'matched_mlp'")
        self.evidence_weight = float(evidence_weight)
        self.gate_regularization_weight = float(gate_regularization_weight)
        self.gate_target_density = float(gate_target_density)

        if decoder_type.lower() == "hf":
            if not decoder_name:
                raise ValueError("decoder_name is required when decoder_type='hf'")
            self.decoder = HFSeq2SeqDecoder(
                decoder_name,
                revision=decoder_revision,
                frozen=decoder_frozen,
            )
            decoder_width = self.decoder.d_model
        elif decoder_type.lower() == "toy":
            self.decoder = ToySeq2SeqDecoder(
                vocab_size=decoder_vocab_size,
                d_model=decoder_d_model,
                max_target_length=decoder_max_target_length,
            )
            decoder_width = decoder_d_model
        else:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")

        self.audio_projection = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_blocks = nn.ModuleList(
            [
                build_temporal_block(
                    backend=backend,
                    d_model=d_model,
                    d_state=d_state,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        prompt_embedding_width = int(self.decoder.get_input_embeddings().weight.shape[1])
        self.prompt_to_audio = nn.Linear(prompt_embedding_width, d_model)

        if memory_enabled:
            self.memory = PromptConditionedFastWeightMemory(
                d_model,
                decay=memory_decay,
                update_rate=memory_update_rate,
                max_state_norm=memory_max_state_norm,
                detach_returned_state=memory_detach_returned_state,
                gate_mode=memory_gate_mode,
                constant_gate=memory_constant_gate,
            )
            self.memory_fusion_gate = nn.Linear(3 * d_model, d_model)
            self.memory_projection = nn.Linear(d_model, d_model)
        else:
            self.memory = None

        if not memory_enabled and control_adapter == "matched_mlp":
            self.capacity_adapter: nn.Module | None = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 6 * d_model),
                nn.GELU(),
                nn.Linear(6 * d_model, d_model),
            )
        else:
            self.capacity_adapter = None

        self.evidence_audio = nn.Linear(d_model, d_model)
        self.evidence_prompt = nn.Linear(d_model, d_model)
        self.evidence_head = nn.Linear(d_model, 1)
        self.prefix_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, decoder_width),
        )

    def _prompt_summary(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeddings = self.decoder.get_input_embeddings()(prompt_input_ids)
        mask = prompt_attention_mask.to(prompt_embeddings.dtype).unsqueeze(-1)
        summary = (prompt_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return prompt_embeddings, self.prompt_to_audio(summary)

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
        tokens = self.audio_projection(audio_embeddings)
        valid = audio_attention_mask.to(tokens.dtype).unsqueeze(-1)
        tokens = tokens * valid
        for block in self.temporal_blocks:
            tokens = block(tokens)
            tokens = tokens * valid

        diagnostics = None
        if self.memory is not None:
            combined_update_mask = audio_attention_mask.bool()
            if memory_update_mask is not None:
                combined_update_mask = combined_update_mask & memory_update_mask.bool()
            reads, next_state, diagnostics = self.memory(
                tokens,
                prompt_audio,
                memory_state,
                update=update_memory,
                update_mask=combined_update_mask,
            )
            prompt_expanded = prompt_audio.unsqueeze(1).expand_as(tokens)
            gate = torch.sigmoid(
                self.memory_fusion_gate(torch.cat([tokens, reads, prompt_expanded], dim=-1))
            )
            tokens = tokens + gate * self.memory_projection(reads)
            tokens = tokens * valid
        else:
            next_state = memory_state
            if self.capacity_adapter is not None:
                tokens = tokens + self.capacity_adapter(tokens)
        return tokens, next_state, diagnostics, valid

    def _evidence_logits(
        self,
        tokens: torch.Tensor,
        prompt_audio: torch.Tensor,
        audio_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        prompt_term = self.evidence_prompt(prompt_audio).unsqueeze(1)
        hidden = torch.tanh(self.evidence_audio(tokens) + prompt_term)
        logits = self.evidence_head(hidden).squeeze(-1)
        return logits.masked_fill(~audio_attention_mask.bool(), -1e4)

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
        top_indices = torch.topk(evidence_logits, k=k, dim=1).indices
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        top_tokens = torch.gather(tokens, 1, gather_index)
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
        """Process fixed-size windows and retain only a fixed top-K reservoir.

        Scalar evidence scores may be returned for audit, but decoder-side audio
        state is bounded by ``stream_window_size + evidence_top_k`` chunks.
        """
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
                start, stop, dtype=torch.long, device=audio_embeddings.device
            ).expand(batch, -1)
            candidate_tokens = torch.cat([reservoir_tokens, tokens], dim=1)
            candidate_logits = torch.cat([reservoir_logits, logits], dim=1)
            candidate_indices = torch.cat([reservoir_indices, indices], dim=1)
            candidate_valid = torch.cat([reservoir_valid, window_mask.bool()], dim=1)
            keep = min(self.evidence_top_k, candidate_logits.shape[1])
            selected = torch.topk(candidate_logits, k=keep, dim=1).indices
            token_index = selected.unsqueeze(-1).expand(-1, -1, self.d_model)
            reservoir_tokens = torch.gather(candidate_tokens, 1, token_index)
            reservoir_logits = torch.gather(candidate_logits, 1, selected)
            reservoir_indices = torch.gather(candidate_indices, 1, selected)
            reservoir_valid = torch.gather(candidate_valid, 1, selected)

            if diagnostics is not None:
                gate_parts.append(diagnostics.gates)
                surprise_parts.append(diagnostics.surprises)
                relevance_parts.append(diagnostics.relevance)
                norm_parts.append(diagnostics.state_norms)

        evidence_logits = torch.cat(evidence_parts, dim=1)
        normalized_scores = reservoir_logits.masked_fill(~reservoir_valid, -1e4)
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
            "working_set_bound_chunks": self.stream_window_size + self.evidence_top_k,
            "decoder_prefix_audio_tokens": 1 + reservoir_tokens.shape[1],
        }

    def _prepare_decoder_inputs(
        self,
        audio_embeddings: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        audio_attention_mask: torch.Tensor | None,
        memory_state: torch.Tensor | None,
        *,
        update_memory: bool,
        memory_update_mask: torch.Tensor | None,
    ) -> dict[str, Any]:
        if audio_embeddings.ndim != 3:
            raise ValueError("audio_embeddings must have shape [B,N,D_audio]")
        if audio_embeddings.shape[-1] != self.audio_dim:
            raise ValueError(
                f"Expected audio dimension {self.audio_dim}, received {audio_embeddings.shape[-1]}"
            )
        if audio_attention_mask is None:
            audio_attention_mask = torch.ones(
                audio_embeddings.shape[:2], dtype=torch.bool, device=audio_embeddings.device
            )
        if self.stream_window_size is not None:
            return self._prepare_bounded_decoder_inputs(
                audio_embeddings,
                prompt_input_ids,
                prompt_attention_mask,
                audio_attention_mask,
                memory_state,
                update_memory=update_memory,
                memory_update_mask=memory_update_mask,
            )
        prompt_embeddings, prompt_audio = self._prompt_summary(
            prompt_input_ids, prompt_attention_mask
        )
        tokens, next_state, diagnostics, _ = self._encode_audio(
            audio_embeddings,
            audio_attention_mask,
            prompt_audio,
            memory_state,
            update_memory=update_memory,
            memory_update_mask=memory_update_mask,
        )
        evidence_logits, top_indices, prefix = self._evidence_and_prefix(
            tokens,
            prompt_audio,
            audio_attention_mask,
        )
        decoder_inputs = torch.cat([prefix, prompt_embeddings], dim=1)
        prefix_mask = torch.ones(
            prefix.shape[:2],
            dtype=prompt_attention_mask.dtype,
            device=prompt_attention_mask.device,
        )
        decoder_mask = torch.cat([prefix_mask, prompt_attention_mask], dim=1)
        return {
            "decoder_inputs": decoder_inputs,
            "decoder_mask": decoder_mask,
            "evidence_logits": evidence_logits,
            "top_indices": top_indices,
            "memory_state": next_state,
            "memory_diagnostics": diagnostics,
            "audio_attention_mask": audio_attention_mask,
        }

    def forward(
        self,
        audio_embeddings: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        evidence_targets: torch.Tensor | None = None,
        audio_attention_mask: torch.Tensor | None = None,
        memory_state: torch.Tensor | None = None,
        update_memory: bool = True,
        memory_update_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        prepared = self._prepare_decoder_inputs(
            audio_embeddings,
            prompt_input_ids,
            prompt_attention_mask,
            audio_attention_mask,
            memory_state,
            update_memory=update_memory,
            memory_update_mask=memory_update_mask,
        )
        decoder_output = self.decoder(
            inputs_embeds=prepared["decoder_inputs"],
            attention_mask=prepared["decoder_mask"],
            labels=labels,
        )
        evidence_loss = None
        if evidence_targets is not None:
            evidence_loss = masked_evidence_bce(
                prepared["evidence_logits"],
                evidence_targets,
                prepared["audio_attention_mask"].bool(),
            )
        gate_regularization = None
        diagnostics = prepared["memory_diagnostics"]
        if diagnostics is not None:
            gate_regularization = (
                diagnostics.gates.mean() - self.gate_target_density
            ).square()

        total_loss = decoder_output.loss
        if evidence_loss is not None:
            total_loss = (
                self.evidence_weight * evidence_loss
                if total_loss is None
                else total_loss + self.evidence_weight * evidence_loss
            )
        if gate_regularization is not None and self.gate_regularization_weight > 0:
            weighted_gate = self.gate_regularization_weight * gate_regularization
            total_loss = weighted_gate if total_loss is None else total_loss + weighted_gate
        return {
            **prepared,
            "decoder_logits": decoder_output.logits,
            "text_loss": decoder_output.loss,
            "evidence_loss": evidence_loss,
            "gate_regularization": gate_regularization,
            "loss": total_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        audio_embeddings: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        *,
        audio_attention_mask: torch.Tensor | None = None,
        memory_state: torch.Tensor | None = None,
        update_memory: bool = True,
        memory_update_mask: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        num_beams: int = 1,
    ) -> dict[str, Any]:
        prepared = self._prepare_decoder_inputs(
            audio_embeddings,
            prompt_input_ids,
            prompt_attention_mask,
            audio_attention_mask,
            memory_state,
            update_memory=update_memory,
            memory_update_mask=memory_update_mask,
        )
        token_ids = self.decoder.generate_from_embeddings(
            inputs_embeds=prepared["decoder_inputs"],
            attention_mask=prepared["decoder_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        return {**prepared, "generated_ids": token_ids}
