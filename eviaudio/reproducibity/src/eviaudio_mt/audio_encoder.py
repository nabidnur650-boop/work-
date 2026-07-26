from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class ChunkedAudioFeatures:
    embeddings: torch.Tensor
    start_sec: torch.Tensor
    end_sec: torch.Tensor
    attention_mask: torch.Tensor


def chunk_waveform(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    chunk_duration_sec: float,
    hop_duration_sec: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split [B,T] or [T] waveform into zero-padded chunks."""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [T] or [B,T]")
    chunk_samples = max(1, int(round(chunk_duration_sec * sample_rate)))
    hop_samples = max(1, int(round(hop_duration_sec * sample_rate)))
    total = waveform.shape[1]
    starts = list(range(0, max(total - chunk_samples + 1, 1), hop_samples))
    final_start = max(0, total - chunk_samples)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    starts = sorted(set(starts))

    chunks: list[torch.Tensor] = []
    end_samples: list[int] = []
    for start in starts:
        end = min(start + chunk_samples, total)
        chunk = waveform[:, start:end]
        if chunk.shape[1] < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.shape[1]))
        chunks.append(chunk)
        end_samples.append(end)
    stacked = torch.stack(chunks, dim=1)  # [B,N,S]
    start_sec = torch.tensor(starts, dtype=torch.float32, device=waveform.device) / sample_rate
    end_sec = torch.tensor(end_samples, dtype=torch.float32, device=waveform.device) / sample_rate
    return stacked, start_sec, end_sec


class ChunkedCLAPEncoder(nn.Module):
    """Optional Hugging Face CLAP wrapper for timestamped chunk embeddings."""

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        *,
        revision: str | None = None,
        sample_rate: int = 48_000,
        chunk_duration_sec: float = 4.0,
        hop_duration_sec: float = 2.0,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoProcessor, ClapModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional
            raise ImportError("Install requirements-full.txt for CLAP support") from exc
        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        self.model = ClapModel.from_pretrained(model_name, revision=revision)
        self.sample_rate = sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.hop_duration_sec = hop_duration_sec
        self.frozen = frozen
        if frozen:
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def forward(self, waveform: torch.Tensor) -> ChunkedAudioFeatures:  # pragma: no cover - optional
        chunks, start_sec, end_sec = chunk_waveform(
            waveform,
            sample_rate=self.sample_rate,
            chunk_duration_sec=self.chunk_duration_sec,
            hop_duration_sec=self.hop_duration_sec,
        )
        batch, n_chunks, _ = chunks.shape
        audio_list = [
            chunks[b, i].detach().cpu().numpy().astype(np.float32)
            for b in range(batch)
            for i in range(n_chunks)
        ]
        inputs: dict[str, Any] = self.processor(
            audios=audio_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        context = torch.no_grad() if self.frozen else nullcontext()
        with context:
            features = self.model.get_audio_features(**inputs)
        features = features.view(batch, n_chunks, -1)
        mask = torch.ones(batch, n_chunks, dtype=torch.bool, device=features.device)
        return ChunkedAudioFeatures(
            embeddings=features,
            start_sec=start_sec.to(features.device).expand(batch, -1),
            end_sec=end_sec.to(features.device).expand(batch, -1),
            attention_mask=mask,
        )
