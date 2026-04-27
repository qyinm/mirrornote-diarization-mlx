"""MLX segmentation runtime skeleton for parity work."""

from __future__ import annotations

from dataclasses import dataclass


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


def build_mlx_segmentation(config: MlxSegmentationConfig) -> None:
    raise UnsupportedArchitectureError(config.architecture_name)
