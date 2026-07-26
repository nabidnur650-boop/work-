from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly


@dataclass(frozen=True)
class MaterializedNeedle:
    waveform: np.ndarray
    sample_rate: int
    target_start_sec: float
    target_end_sec: float


@lru_cache(maxsize=8192)
def _load_resampled(path_text: str, sample_rate: int) -> np.ndarray:
    import soundfile as sf

    waveform, native_rate = sf.read(path_text, always_2d=True, dtype="float32")
    mono = waveform.mean(axis=1, dtype=np.float32)
    if native_rate != sample_rate:
        divisor = math.gcd(int(native_rate), int(sample_rate))
        mono = resample_poly(
            mono,
            sample_rate // divisor,
            native_rate // divisor,
        ).astype(np.float32)
    mono.setflags(write=False)
    return mono


def load_resampled(path: str | Path, sample_rate: int = 48_000) -> np.ndarray:
    """Return an immutable cached waveform for deterministic read-only mixing."""

    return _load_resampled(str(Path(path).resolve()), int(sample_rate))


def _rms(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(waveform.astype(np.float64))) + 1e-12))


def _insert_at_snr(
    mixture: np.ndarray,
    event: np.ndarray,
    start_sample: int,
    snr_db: float,
    fallback_rms: float,
) -> None:
    if start_sample < 0 or start_sample >= len(mixture):
        raise ValueError("event start is outside mixture")
    usable = min(len(event), len(mixture) - start_sample)
    if usable <= 0:
        raise ValueError("event has no samples inside mixture")
    event = event[:usable].astype(np.float32)
    event = event - float(event.mean())
    background = mixture[start_sample : start_sample + usable]
    background_rms = _rms(background)
    if background_rms < 1e-5:
        background_rms = fallback_rms
    event_rms = _rms(event)
    desired_rms = background_rms * (10.0 ** (float(snr_db) / 20.0))
    gain = desired_rms / max(event_rms, 1e-8)
    mixture[start_sample : start_sample + usable] += event * gain


def materialize_recipe(
    recipe: dict[str, Any],
    *,
    project_root: str | Path,
    sample_rate: int = 48_000,
) -> MaterializedNeedle:
    """Deterministically materialize a recipe in memory; no long WAV is written."""

    project_root = Path(project_root)
    total_samples = int(round(float(recipe["duration_sec"]) * sample_rate))
    background_parts = [
        load_resampled(project_root / item["path"], sample_rate)
        for item in recipe["background_files"]
    ]
    if not background_parts:
        raise ValueError("recipe has no background audio")
    background = np.concatenate(background_parts)
    if len(background) < total_samples:
        raise ValueError("background recipe is shorter than declared duration")
    mixture = background[:total_samples].astype(np.float32, copy=True)
    mixture -= float(mixture.mean())
    fallback_rms = max(_rms(mixture), 1e-4)

    events = [*recipe["distractor_events"], recipe["target_event"]]
    # Target is inserted last so its declared SNR is measured against the final
    # local background if a numerical boundary happens to be nearby.
    for item in events:
        event = load_resampled(project_root / item["path"], sample_rate)
        start_sample = int(round(float(item["start_sec"]) * sample_rate))
        _insert_at_snr(
            mixture,
            event,
            start_sample,
            float(item["snr_db"]),
            fallback_rms,
        )
    peak = float(np.max(np.abs(mixture)))
    if peak > 0.99:
        mixture *= np.float32(0.99 / peak)
    return MaterializedNeedle(
        waveform=mixture,
        sample_rate=sample_rate,
        target_start_sec=float(recipe["target_event"]["start_sec"]),
        target_end_sec=float(recipe["target_event"]["end_sec"]),
    )


def temporal_overlap_fraction(
    starts: np.ndarray,
    ends: np.ndarray,
    target_start: float,
    target_end: float,
) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    overlap = np.maximum(
        0.0, np.minimum(ends, float(target_end)) - np.maximum(starts, float(target_start))
    )
    duration = np.maximum(ends - starts, 1e-8)
    return (overlap / duration).astype(np.float32)
