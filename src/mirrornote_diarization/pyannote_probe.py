"""Pyannote reference probe metadata contract and gated runtime hooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PyannoteProbeMetadata:
    """Metadata describing a pyannote segmentation reference probe run."""

    model_class: str
    sample_rate: int
    chunk_duration_seconds: float
    frame_resolution_seconds: float
    module_tree: Sequence[str]
    weight_shapes: Mapping[str, Sequence[int]]
    output_shape: Sequence[int]

    def to_dict(self) -> dict[str, Any]:
        """Return the public JSON contract using camelCase keys."""
        return {
            "modelClass": self.model_class,
            "sampleRate": self.sample_rate,
            "chunkDurationSeconds": self.chunk_duration_seconds,
            "frameResolutionSeconds": self.frame_resolution_seconds,
            "moduleTree": list(self.module_tree),
            "weightShapes": {
                name: list(shape) for name, shape in self.weight_shapes.items()
            },
            "outputShape": list(self.output_shape),
        }


def require_pyannote_enabled(env: Mapping[str, str]) -> None:
    """Require explicit opt-in and Hugging Face credentials for pyannote runtime use."""
    if env.get("MIRRORNOTE_RUN_PYANNOTE_PROBE") != "1":
        raise RuntimeError(
            "Pyannote runtime probe is disabled. Set "
            "MIRRORNOTE_RUN_PYANNOTE_PROBE=1 to run it."
        )
    if env.get("HUGGINGFACE_ACCESS_TOKEN", "").strip() == "":
        raise RuntimeError(
            "Pyannote runtime probe requires a non-empty "
            "HUGGINGFACE_ACCESS_TOKEN."
        )


def write_probe_artifacts(
    metadata: PyannoteProbeMetadata,
    reference_output: np.ndarray,
    out_dir: str | Path,
) -> None:
    """Write probe metadata and float32 reference output artifacts."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n"
    )

    output = np.asarray(reference_output, dtype=np.float32)
    np.savez(output_dir / "reference-output.npz", output=output)


def run_pyannote_probe(audio_chunk: Any, out_dir: str | Path) -> None:
    """Placeholder for the real pyannote runtime probe."""
    raise RuntimeError(
        "Real pyannote runtime probe is implemented later after metadata "
        "contract tests pass."
    )
