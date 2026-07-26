from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {path!s} must contain a mapping")
    return config


def save_json(data: Any, path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)


def resolve_device(value: str = "auto") -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
