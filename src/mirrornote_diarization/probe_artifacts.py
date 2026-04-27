from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProbeArtifacts:
    metadata: dict[str, Any]
    reference_output: np.ndarray

    @property
    def module_count(self) -> int:
        return len(self.metadata.get("moduleTree", []))

    @property
    def parameter_count(self) -> int:
        total = 0
        for shape in self.metadata.get("weightShapes", {}).values():
            product = 1
            for dimension in shape:
                product *= int(dimension)
            total += product
        return total


def load_probe_artifacts(probe_dir: Path) -> ProbeArtifacts:
    metadata_path = probe_dir / "metadata.json"
    output_path = probe_dir / "reference-output.npz"
    if not metadata_path.exists():
        raise ValueError(f"missing metadata.json in probe directory: {probe_dir}")
    if not output_path.exists():
        raise ValueError(f"missing reference-output.npz in probe directory: {probe_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(output_path) as payload:
        if "output" not in payload:
            raise ValueError(f"reference-output.npz missing output array: {output_path}")
        reference_output = np.asarray(payload["output"], dtype=np.float32)

    return ProbeArtifacts(metadata=metadata, reference_output=reference_output)
