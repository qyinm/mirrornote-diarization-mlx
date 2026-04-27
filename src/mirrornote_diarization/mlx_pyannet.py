"""Shape-correct MLX PyanNet segmentation candidate runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mirrornote_diarization.pyannet_contract import (
    PYANNET_CHUNK_DURATION_SECONDS,
    PYANNET_EXPECTED_OUTPUT_SHAPE,
    PYANNET_SAMPLE_RATE,
)
from mirrornote_diarization.weight_conversion import (
    build_pyannet_mapping_rules,
    validate_weight_mapping,
)

PYANNET_EXPECTED_WAVEFORM_SHAPE = (1, 1, 160000)


def _dense(x: Any, weight: np.ndarray, bias: np.ndarray) -> Any:
    import mlx.core as mx

    return mx.matmul(x, mx.array(weight.T, dtype=mx.float32)) + mx.array(
        bias,
        dtype=mx.float32,
    )


@dataclass(frozen=True)
class MlxPyanNetSegmentation:
    """MLX PyanNet candidate scaffold for segmentation parity plumbing."""

    reference_weights: dict[str, np.ndarray]
    output_classes: int = PYANNET_EXPECTED_OUTPUT_SHAPE[2]
    sample_rate: int = PYANNET_SAMPLE_RATE
    chunk_duration_seconds: float = PYANNET_CHUNK_DURATION_SECONDS

    @classmethod
    def from_reference_weights(
        cls,
        reference_weights: Mapping[str, np.ndarray],
    ) -> MlxPyanNetSegmentation:
        result = validate_weight_mapping(
            reference_weights,
            build_pyannet_mapping_rules(),
        )
        if not result.passed:
            raise ValueError(str(result.to_dict()))

        return cls(
            reference_weights={
                name: np.asarray(weight, dtype=np.float32)
                for name, weight in reference_weights.items()
            }
        )

    def __call__(self, waveform: Any) -> Any:
        import mlx.core as mx

        if tuple(waveform.shape) != PYANNET_EXPECTED_WAVEFORM_SHAPE:
            raise ValueError(
                "PyanNet MLX waveform must have shape "
                f"{PYANNET_EXPECTED_WAVEFORM_SHAPE}; got {tuple(waveform.shape)}"
            )

        return mx.zeros(PYANNET_EXPECTED_OUTPUT_SHAPE, dtype=mx.float32)

    def linear_head(self, features: Any) -> Any:
        import mlx.core as mx

        x = _dense(
            features,
            self.reference_weights["linear.0.weight"],
            self.reference_weights["linear.0.bias"],
        )
        x = mx.maximum(x, 0)
        x = _dense(
            x,
            self.reference_weights["linear.1.weight"],
            self.reference_weights["linear.1.bias"],
        )
        x = mx.maximum(x, 0)
        return _dense(
            x,
            self.reference_weights["classifier.weight"],
            self.reference_weights["classifier.bias"],
        )

    def write_candidate_npz(self, waveform: np.ndarray, path: str | Path) -> None:
        import mlx.core as mx

        output = self(mx.array(waveform, dtype=mx.float32))
        np.savez(path, output=np.asarray(output, dtype=np.float32))
