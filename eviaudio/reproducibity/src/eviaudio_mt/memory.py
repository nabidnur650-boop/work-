from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AudioMemoryDiagnostics:
    gates: torch.Tensor
    surprises: torch.Tensor
    relevance: torch.Tensor
    state_norms: torch.Tensor


class PromptConditionedFastWeightMemory(nn.Module):
    """Titans-inspired fast memory conditioned on a known text prompt."""

    def __init__(
        self,
        d_model: int,
        decay: float = 0.97,
        update_rate: float = 0.05,
        max_state_norm: float = 50.0,
        detach_returned_state: bool = True,
        gate_mode: str = "learned",
        constant_gate: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0,1]")
        if update_rate <= 0.0:
            raise ValueError("update_rate must be positive")
        gate_mode = gate_mode.lower()
        if gate_mode not in {"learned", "surprise", "relevance", "always", "constant"}:
            raise ValueError(
                "gate_mode must be one of: learned, surprise, relevance, always, constant"
            )
        if not 0.0 <= constant_gate <= 1.0:
            raise ValueError("constant_gate must be in [0, 1]")
        self.d_model = d_model
        self.decay = float(decay)
        self.update_rate = float(update_rate)
        self.max_state_norm = float(max_state_norm)
        self.detach_returned_state = bool(detach_returned_state)
        self.gate_mode = gate_mode
        self.constant_gate = float(constant_gate)
        self.eps = eps

        self.token_norm = nn.LayerNorm(d_model)
        self.prompt_norm = nn.LayerNorm(d_model)
        self.prompt_condition = nn.Linear(d_model, d_model, bias=False)
        self.to_query = nn.Linear(d_model, d_model, bias=False)
        self.to_key = nn.Linear(d_model, d_model, bias=False)
        self.to_value = nn.Linear(d_model, d_model, bias=False)
        self.relevance_audio = nn.Linear(d_model, d_model, bias=False)
        self.relevance_prompt = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(d_model + 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.surprise_gate = nn.Linear(1, 1)
        self.relevance_gate = nn.Linear(1, 1)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.d_model, self.d_model, device=device, dtype=dtype)

    def _bound(self, state: torch.Tensor) -> torch.Tensor:
        if self.max_state_norm <= 0:
            return state
        norms = state.flatten(1).norm(dim=1).clamp_min(self.eps)
        scales = (self.max_state_norm / norms).clamp(max=1.0)
        return state * scales[:, None, None]

    def forward(
        self,
        tokens: torch.Tensor,
        prompt: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        update: bool = True,
        update_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, AudioMemoryDiagnostics]:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B,N,D]")
        if prompt.shape != (tokens.shape[0], tokens.shape[2]):
            raise ValueError(
                f"prompt must have shape {(tokens.shape[0], tokens.shape[2])}, received {prompt.shape}"
            )
        batch, length, width = tokens.shape
        if state is None:
            state = self.initial_state(batch, device=tokens.device, dtype=tokens.dtype)
        if state.shape != (batch, width, width):
            raise ValueError("Invalid memory-state shape")
        state = state.detach()

        if update_mask is None:
            update_mask = torch.ones(batch, length, dtype=torch.bool, device=tokens.device)
        if update_mask.shape != (batch, length):
            raise ValueError("update_mask must have shape [B,N]")

        normalized_tokens = self.token_norm(tokens)
        normalized_prompt = self.prompt_norm(prompt)
        conditioned = normalized_tokens + self.prompt_condition(normalized_prompt).unsqueeze(1)
        queries = self.to_query(conditioned)
        keys = self.to_key(conditioned)
        values = self.to_value(normalized_tokens)
        audio_rel = F.normalize(self.relevance_audio(normalized_tokens), dim=-1)
        prompt_rel = F.normalize(self.relevance_prompt(normalized_prompt), dim=-1)
        relevance_all = (audio_rel * prompt_rel.unsqueeze(1)).sum(dim=-1)

        reads: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        surprises: list[torch.Tensor] = []
        relevances: list[torch.Tensor] = []
        state_norms: list[torch.Tensor] = []

        for index in range(length):
            query = queries[:, index]
            key = keys[:, index]
            value = values[:, index]
            read = torch.bmm(state, query.unsqueeze(-1)).squeeze(-1)
            predicted = torch.bmm(state, key.unsqueeze(-1)).squeeze(-1)
            error = value - predicted
            surprise = error.square().mean(dim=-1, keepdim=True).sqrt()
            relevance = relevance_all[:, index : index + 1]
            if self.gate_mode == "learned":
                gate_input = torch.cat(
                    [conditioned[:, index], surprise, relevance], dim=-1
                )
                gate = torch.sigmoid(self.gate(gate_input))
            elif self.gate_mode == "surprise":
                gate = torch.sigmoid(self.surprise_gate(surprise))
            elif self.gate_mode == "relevance":
                gate = torch.sigmoid(self.relevance_gate(relevance))
            elif self.gate_mode == "always":
                gate = torch.ones_like(surprise)
            else:
                gate = torch.full_like(surprise, self.constant_gate)

            reads.append(read)
            gates.append(gate.squeeze(-1))
            surprises.append(surprise.squeeze(-1))
            relevances.append(relevance.squeeze(-1))

            if update:
                allowed = update_mask[:, index].view(batch, 1, 1)
                denominator = key.square().sum(dim=-1, keepdim=True).clamp_min(self.eps)
                delta = error.unsqueeze(-1) * key.unsqueeze(-2) / denominator.unsqueeze(-1)
                candidate = self.decay * state + self.update_rate * gate.unsqueeze(-1) * delta
                state = torch.where(allowed, candidate, state)
                state = self._bound(state)
            state_norms.append(state.flatten(1).norm(dim=1))

        returned = state.detach() if self.detach_returned_state else state
        diagnostics = AudioMemoryDiagnostics(
            gates=torch.stack(gates, dim=1),
            surprises=torch.stack(surprises, dim=1),
            relevance=torch.stack(relevances, dim=1),
            state_norms=torch.stack(state_norms, dim=1),
        )
        return torch.stack(reads, dim=1), returned, diagnostics
