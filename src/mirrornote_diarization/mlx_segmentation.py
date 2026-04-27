"""MLX segmentation runtime skeleton for parity work."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mirrornote_diarization.pyannet_contract import (
    PYANNET_ARCHITECTURE_NAME,
    PYANNET_CHUNK_DURATION_SECONDS,
    PYANNET_EXPECTED_OUTPUT_SHAPE,
    PYANNET_SAMPLE_RATE,
)


class UnsupportedArchitectureError(RuntimeError):
    """Raised when an MLX segmentation architecture is not implemented yet."""

    def __init__(self, architecture_name: str) -> None:
        super().__init__(
            f"MLX segmentation architecture is not supported yet: {architecture_name}"
        )


@dataclass(frozen=True)
class MlxSegmentationConfig:
    sample_rate: int
    chunk_duration_seconds: float
    output_classes: int
    architecture_name: str

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "sampleRate": self.sample_rate,
            "chunkDurationSeconds": self.chunk_duration_seconds,
            "outputClasses": self.output_classes,
            "architectureName": self.architecture_name,
        }


def build_mlx_segmentation(
    config: MlxSegmentationConfig,
    reference_weights: dict[str, np.ndarray] | None = None,
) -> object:
    if config.architecture_name == PYANNET_ARCHITECTURE_NAME:
        _validate_pyannet_config(config)

        if reference_weights is None:
            raise ValueError("reference_weights are required for PyanNet MLX runtime")

        from mirrornote_diarization.mlx_pyannet import MlxPyanNetSegmentation

        return MlxPyanNetSegmentation.from_reference_weights(reference_weights)

    raise UnsupportedArchitectureError(config.architecture_name)


def _validate_pyannet_config(config: MlxSegmentationConfig) -> None:
    expected_fields: dict[str, int | float] = {
        "sample_rate": PYANNET_SAMPLE_RATE,
        "chunk_duration_seconds": PYANNET_CHUNK_DURATION_SECONDS,
        "output_classes": PYANNET_EXPECTED_OUTPUT_SHAPE[2],
    }

    for field_name, expected_value in expected_fields.items():
        actual_value = getattr(config, field_name)
        if actual_value != expected_value:
            raise ValueError(
                "PyanNet MLX config mismatch for "
                f"{field_name}: expected {expected_value}, got {actual_value}"
            )
