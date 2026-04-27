from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedChunk:
    samples: np.ndarray
    sample_rate: int
    start_seconds: float
    duration_seconds: float

    def as_model_input(self) -> np.ndarray:
        return np.asarray(self.samples, dtype=np.float32).reshape(1, 1, -1)


def extract_fixed_chunk(
    waveform: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    duration_seconds: float,
) -> FixedChunk:
    if waveform.ndim != 1:
        raise ValueError("expected mono waveform")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if start_seconds < 0:
        raise ValueError("start_seconds must be non-negative")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    start_sample = int(round(start_seconds * sample_rate))
    sample_count = int(round(duration_seconds * sample_rate))
    end_sample = start_sample + sample_count

    samples = np.asarray(waveform[start_sample:end_sample], dtype=np.float32)
    if samples.shape[0] < sample_count:
        samples = np.pad(samples, (0, sample_count - samples.shape[0]))

    return FixedChunk(
        samples=samples.astype(np.float32, copy=False),
        sample_rate=sample_rate,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )
