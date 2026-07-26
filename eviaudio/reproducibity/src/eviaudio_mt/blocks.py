from __future__ import annotations

import warnings

import torch
from torch import nn
from torch.nn import functional as F


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,D], received {tuple(x.shape)}")
        y = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        return self.conv(y).transpose(1, 2)


class PortableSelectiveSSM(nn.Module):
    """Small Mamba-style causal recurrent block for CPU validation."""

    def __init__(self, d_model: int, dropout: float = 0.0, conv_kernel: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 4 * d_model)
        self.conv = CausalDepthwiseConv1d(d_model, conv_kernel)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.retention_bias = nn.Parameter(torch.full((d_model,), 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.norm(x)
        content, gate, retain_raw, inject_raw = self.in_proj(z).chunk(4, dim=-1)
        content = F.silu(content + self.conv(content))
        retain = torch.sigmoid(retain_raw + self.retention_bias)
        inject = torch.sigmoid(inject_raw)
        state = torch.zeros_like(content[:, 0])
        outputs: list[torch.Tensor] = []
        for index in range(content.shape[1]):
            state = retain[:, index] * state + (1.0 - retain[:, index]) * inject[:, index] * content[:, index]
            outputs.append(state * torch.sigmoid(gate[:, index]))
        return residual + self.dropout(self.out_proj(torch.stack(outputs, dim=1)))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, expansion: int = 4) -> None:
        super().__init__()
        hidden = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class PortableMambaBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ssm = PortableSelectiveSSM(d_model, dropout=dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.ssm(x))


class ExactMamba2Block(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba2  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional
            raise ImportError(
                "backend='mamba2' requires mamba-ssm in a compatible CUDA/PyTorch environment"
            ) from exc
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - optional
        x = x + self.dropout(self.mamba(self.norm(x)))
        return self.ffn(x)


def build_temporal_block(
    backend: str,
    d_model: int,
    d_state: int = 64,
    dropout: float = 0.0,
) -> nn.Module:
    backend = backend.lower()
    if backend == "mamba2":
        return ExactMamba2Block(d_model=d_model, d_state=d_state, dropout=dropout)
    if backend not in {"fallback", "portable", "mamba_style"}:
        warnings.warn(f"Unknown backend {backend!r}; using portable fallback", stacklevel=2)
    return PortableMambaBlock(d_model=d_model, dropout=dropout)
