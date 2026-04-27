"""Pyannote reference probe metadata contract and gated runtime hooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
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


def run_pyannote_probe(audio_chunk: Any, out_dir: str | Path) -> PyannoteProbeMetadata:
    """Run the real pyannote segmentation model and write reference artifacts."""
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "install pyannote dependencies with: uv sync --extra pyannote"
        ) from exc

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.environ["HUGGINGFACE_ACCESS_TOKEN"],
    )
    segmentation = getattr(pipeline, "_segmentation", None) or getattr(
        pipeline, "segmentation", None
    )
    if segmentation is None:
        raise RuntimeError("pyannote pipeline does not expose a segmentation model")

    model = getattr(segmentation, "model", segmentation)
    module_tree = _module_tree(model)
    weight_shapes = _weight_shapes(model)

    chunk_array = np.asarray(audio_chunk, dtype=np.float32)
    tensor = torch.from_numpy(chunk_array.astype(np.float32, copy=False))

    model.eval()
    with torch.no_grad():
        output = model(tensor)

    if hasattr(output, "data") and not isinstance(output, torch.Tensor):
        output = output.data
    output_array = _to_float32_numpy(output, torch)

    sample_rate = int(getattr(model, "sample_rate", 16000))
    specifications = getattr(model, "specifications", None)
    chunk_duration = _duration_seconds(
        getattr(specifications, "duration", None),
        fallback=float(chunk_array.shape[-1]) / float(sample_rate),
    )
    frame_resolution = _duration_seconds(
        getattr(getattr(specifications, "resolution", None), "duration", None),
        fallback=0.0,
    )

    metadata = PyannoteProbeMetadata(
        model_class=f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        sample_rate=sample_rate,
        chunk_duration_seconds=chunk_duration,
        frame_resolution_seconds=frame_resolution,
        module_tree=module_tree,
        weight_shapes=weight_shapes,
        output_shape=list(output_array.shape),
    )
    write_probe_artifacts(metadata, output_array, out_dir)
    return metadata


def _module_tree(model: Any) -> list[str]:
    named_modules = getattr(model, "named_modules", None)
    if named_modules is None:
        return []
    return ["model" if name == "" else f"model.{name}" for name, _ in named_modules()]


def _weight_shapes(model: Any) -> dict[str, list[int]]:
    state_dict = getattr(model, "state_dict", None)
    if state_dict is None:
        return {}
    return {name: list(value.shape) for name, value in state_dict().items()}


def _to_float32_numpy(output: Any, torch: Any) -> np.ndarray:
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    return np.asarray(output, dtype=np.float32)


def _duration_seconds(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    total_seconds = getattr(value, "total_seconds", None)
    if callable(total_seconds):
        return float(total_seconds())
    if total_seconds is not None:
        return float(total_seconds)
    return float(value)
