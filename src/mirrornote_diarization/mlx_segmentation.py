"""MLX segmentation runtime skeleton for parity work."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mirrornote_diarization.pyannet_contract import PYANNET_ARCHITECTURE_NAME


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
        if reference_weights is None:
            raise ValueError("reference_weights are required for PyanNet MLX runtime")

        from mirrornote_diarization.mlx_pyannet import MlxPyanNetSegmentation

        return MlxPyanNetSegmentation.from_reference_weights(reference_weights)

    raise UnsupportedArchitectureError(config.architecture_name)
