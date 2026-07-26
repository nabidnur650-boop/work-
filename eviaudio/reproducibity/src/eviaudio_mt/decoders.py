from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DecoderOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    raw_output: Any = None


class ToySeq2SeqDecoder(nn.Module):
    """Tiny non-autoregressive decoder used only for offline smoke tests."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_target_length: int = 8,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_target_length = max_target_length
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.encoder = nn.GRU(d_model, d_model, batch_first=True)
        self.output_positions = nn.Embedding(max_target_length, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def _context(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.encoder(inputs_embeds)
        lengths = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        batch_index = torch.arange(outputs.shape[0], device=outputs.device)
        return outputs[batch_index, lengths]

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> DecoderOutput:
        context = self._context(inputs_embeds, attention_mask)
        target_length = labels.shape[1] if labels is not None else self.max_target_length
        if target_length > self.max_target_length:
            raise ValueError("labels exceed max_target_length for the toy decoder")
        positions = torch.arange(target_length, device=inputs_embeds.device)
        hidden = torch.tanh(context.unsqueeze(1) + self.output_positions(positions).unsqueeze(0))
        logits = self.lm_head(self.output_norm(hidden))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return DecoderOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate_from_embeddings(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int | None = None,
        **_: Any,
    ) -> torch.Tensor:
        length = min(max_new_tokens or self.max_target_length, self.max_target_length)
        context = self._context(inputs_embeds, attention_mask)
        positions = torch.arange(length, device=inputs_embeds.device)
        hidden = torch.tanh(context.unsqueeze(1) + self.output_positions(positions).unsqueeze(0))
        logits = self.lm_head(self.output_norm(hidden))
        return logits.argmax(dim=-1)


class HFSeq2SeqDecoder(nn.Module):
    """Hugging Face encoder-decoder wrapper supporting an audio embedding prefix."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        frozen: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional
            raise ImportError("Install requirements-full.txt for Hugging Face decoders") from exc
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.d_model = int(self.model.get_input_embeddings().weight.shape[1])
        self.frozen = bool(frozen)
        if self.frozen:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> DecoderOutput:  # pragma: no cover - optional
        output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return DecoderOutput(loss=output.loss, logits=output.logits, raw_output=output)

    @torch.no_grad()
    def generate_from_embeddings(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        num_beams: int = 1,
        **kwargs: Any,
    ) -> torch.Tensor:  # pragma: no cover - optional
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            **kwargs,
        )

    def batch_decode(self, token_ids: torch.Tensor) -> list[str]:  # pragma: no cover - optional
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
