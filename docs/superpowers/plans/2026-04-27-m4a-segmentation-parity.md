# M4A Segmentation Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a testable M4A parity harness that proves a pyannote 3.1 segmentation submodel can be reproduced by an MLX implementation on fixed audio chunks.

**Architecture:** Keep reference extraction, MLX candidate execution, weight mapping, and parity reporting in separate modules. Make baseline tests runnable without Hugging Face credentials by validating pure contracts first, and gate real pyannote execution behind explicit environment variables. Treat chunk parity as the hard milestone gate and full wav execution as a later smoke path.

**Tech Stack:** Python 3.11+, uv, pytest, numpy, scipy or soundfile for wav fixtures, optional pyannote.audio/torch/torchaudio for reference probing, optional mlx for candidate execution, JSON reports.

---

## Scope Check

The approved design covers one subsystem: segmentation parity. It deliberately excludes embedding, clustering, product artifact generation, MirrorNote app integration, training, quantization, and `pyannote/community-1`. This is small enough for one implementation plan.

## File Structure

Create or modify these files:

- Create: `pyproject.toml`
  - Owns package metadata, console scripts, dependencies, test config, and optional dependency groups.
- Create: `src/mirrornote_diarization/__init__.py`
  - Owns package version export.
- Create: `src/mirrornote_diarization/chunking.py`
  - Owns deterministic fixed chunk extraction and loading from `.npy` reference chunks.
- Create: `src/mirrornote_diarization/parity_report.py`
  - Owns parity metric computation and JSON schema validation for reports.
- Create: `src/mirrornote_diarization/weight_conversion.py`
  - Owns strict reference-to-MLX weight mapping validation and report generation.
- Create: `src/mirrornote_diarization/pyannote_probe.py`
  - Owns gated extraction of pyannote segmentation metadata, weights, and reference output.
- Create: `src/mirrornote_diarization/mlx_segmentation.py`
  - Owns an MLX segmentation module skeleton and explicit unsupported-architecture failure.
- Create: `src/mirrornote_diarization/segmentation_parity.py`
  - Owns CLI orchestration for report validation and later parity execution.
- Create: `tests/test_parity_report.py`
  - Tests parity metric math and report schema contract.
- Create: `tests/test_weight_mapping.py`
  - Tests strict mapping success and failure modes without MLX or pyannote.
- Create: `tests/test_chunking.py`
  - Tests deterministic fixed chunk extraction independent of pyannote.
- Create: `tests/test_pyannote_probe_contract.py`
  - Tests probe metadata/report shape using fake objects and gated skip behavior.
- Create: `docs/mlx-segmentation-notes.md`
  - Records architecture discoveries, mapping decisions, unsupported ops, and threshold changes.
- Create: `reports/segmentation-parity/.gitkeep`
  - Keeps the reports directory tracked without committing generated reports by default.
- Modify: `README.md`
  - Adds the M4A command entry points and environment requirements.
- Modify: `.gitignore`
  - Ignores generated parity artifacts while keeping `.gitkeep`.

---

### Task 1: Package Skeleton And Test Harness

**Files:**
- Create: `pyproject.toml`
- Create: `src/mirrornote_diarization/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Create package metadata and test configuration**

Write `pyproject.toml` with this content:

```toml
[project]
name = "mirrornote-diarization-mlx"
version = "0.1.0"
description = "Local speaker diarization parity experiments for MirrorNote"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
]

[project.optional-dependencies]
audio = [
  "soundfile>=0.12",
]
pyannote = [
  "pyannote.audio>=3.1,<4",
  "torch>=2.1",
  "torchaudio>=2.1",
]
mlx = [
  "mlx>=0.20",
]
dev = [
  "pytest>=8",
  "jsonschema>=4.22",
]

[project.scripts]
mirrornote-diarize = "mirrornote_diarization.segmentation_parity:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mirrornote_diarization"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q"
```

- [ ] **Step 2: Add package version**

Write `src/mirrornote_diarization/__init__.py` with this content:

```python
"""MirrorNote diarization parity tooling."""

__all__ = ["__version__"]

__version__ = "0.1.0"
```

- [ ] **Step 3: Add generated artifact ignores**

Write `.gitignore` with this content:

```gitignore
.venv/
__pycache__/
.pytest_cache/
*.pyc
.DS_Store
reports/segmentation-parity/*
!reports/segmentation-parity/.gitkeep
artifacts/
```

- [ ] **Step 4: Install dev dependencies if needed**

Run:

```bash
uv sync --extra dev
```

Expected: command exits 0 and creates or updates the local uv environment. If network access is unavailable, request approval for dependency installation and retry the same command.

- [ ] **Step 5: Run baseline tests**

Run:

```bash
uv run pytest
```

Expected: exit 5 or equivalent "no tests ran" because tests have not been added yet. This confirms pytest is callable.

- [ ] **Step 6: Commit package skeleton**

Run:

```bash
git add pyproject.toml .gitignore src/mirrornote_diarization/__init__.py
git commit -m "chore: add Python package skeleton"
```

Expected: commit succeeds.

---

### Task 2: Parity Report Contract And Metrics

**Files:**
- Create: `src/mirrornote_diarization/parity_report.py`
- Create: `tests/test_parity_report.py`

- [ ] **Step 1: Write failing parity report tests**

Write `tests/test_parity_report.py` with this content:

```python
import math

import numpy as np
import pytest

from mirrornote_diarization.parity_report import (
    DEFAULT_THRESHOLDS,
    ParityReport,
    compute_metrics,
    validate_report_dict,
)


def test_compute_metrics_for_identical_arrays_passes():
    reference = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    candidate = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    metrics = compute_metrics(reference, candidate, DEFAULT_THRESHOLDS)

    assert metrics.mean_abs_error == 0.0
    assert metrics.max_abs_error == 0.0
    assert math.isclose(metrics.cosine_similarity, 1.0, rel_tol=0.0, abs_tol=1e-7)
    assert metrics.passed is True


def test_compute_metrics_rejects_shape_mismatch():
    reference = np.zeros((1, 2, 3), dtype=np.float32)
    candidate = np.zeros((1, 2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="shape mismatch"):
        compute_metrics(reference, candidate, DEFAULT_THRESHOLDS)


def test_report_dict_contains_required_fields():
    report = ParityReport(
        reference_provider="pyannote-3.1-segmentation-pytorch",
        candidate_provider="pyannote-3.1-segmentation-mlx",
        audio_chunk={
            "source": "fixtures/single-speaker/system-track.wav",
            "startTimeSeconds": 0.0,
            "durationSeconds": 10.0,
            "sampleRate": 16000,
        },
        shape={"reference": [1, 3], "candidate": [1, 3], "matches": True},
        dtype={"reference": "float32", "candidate": "float32"},
        mean_abs_error=0.0,
        max_abs_error=0.0,
        cosine_similarity=1.0,
        thresholds=DEFAULT_THRESHOLDS,
        passed=True,
    )

    payload = report.to_dict()
    validate_report_dict(payload)

    assert payload["referenceProvider"] == "pyannote-3.1-segmentation-pytorch"
    assert payload["candidateProvider"] == "pyannote-3.1-segmentation-mlx"
    assert payload["passed"] is True


def test_validate_report_dict_requires_thresholds():
    payload = {
        "referenceProvider": "pyannote-3.1-segmentation-pytorch",
        "candidateProvider": "pyannote-3.1-segmentation-mlx",
        "audioChunk": {
            "source": "fixtures/single-speaker/system-track.wav",
            "startTimeSeconds": 0.0,
            "durationSeconds": 10.0,
            "sampleRate": 16000,
        },
        "shape": {"reference": [1, 3], "candidate": [1, 3], "matches": True},
        "dtype": {"reference": "float32", "candidate": "float32"},
        "meanAbsError": 0.0,
        "maxAbsError": 0.0,
        "cosineSimilarity": 1.0,
        "passed": True,
    }

    with pytest.raises(ValueError, match="missing required field: thresholds"):
        validate_report_dict(payload)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_parity_report.py -q
```

Expected: FAIL with `ModuleNotFoundError` or missing symbols from `mirrornote_diarization.parity_report`.

- [ ] **Step 3: Implement parity report module**

Write `src/mirrornote_diarization/parity_report.py` with this content:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_THRESHOLDS: dict[str, float] = {
    "meanAbsError": 1e-4,
    "maxAbsError": 1e-3,
    "cosineSimilarity": 0.999,
}

REQUIRED_REPORT_FIELDS = (
    "referenceProvider",
    "candidateProvider",
    "audioChunk",
    "shape",
    "dtype",
    "meanAbsError",
    "maxAbsError",
    "cosineSimilarity",
    "thresholds",
    "passed",
)


@dataclass(frozen=True)
class ParityMetrics:
    mean_abs_error: float
    max_abs_error: float
    cosine_similarity: float
    passed: bool


@dataclass(frozen=True)
class ParityReport:
    reference_provider: str
    candidate_provider: str
    audio_chunk: dict[str, Any]
    shape: dict[str, Any]
    dtype: dict[str, str]
    mean_abs_error: float
    max_abs_error: float
    cosine_similarity: float
    thresholds: dict[str, float]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "referenceProvider": self.reference_provider,
            "candidateProvider": self.candidate_provider,
            "audioChunk": self.audio_chunk,
            "shape": self.shape,
            "dtype": self.dtype,
            "meanAbsError": self.mean_abs_error,
            "maxAbsError": self.max_abs_error,
            "cosineSimilarity": self.cosine_similarity,
            "thresholds": self.thresholds,
            "passed": self.passed,
        }


def compute_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> ParityMetrics:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: reference={reference.shape} candidate={candidate.shape}")

    active_thresholds = thresholds or DEFAULT_THRESHOLDS
    reference_f32 = reference.astype(np.float32, copy=False)
    candidate_f32 = candidate.astype(np.float32, copy=False)
    delta = np.abs(reference_f32 - candidate_f32)

    reference_flat = reference_f32.reshape(-1)
    candidate_flat = candidate_f32.reshape(-1)
    denominator = float(np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat))
    cosine_similarity = 1.0 if denominator == 0.0 else float(np.dot(reference_flat, candidate_flat) / denominator)

    mean_abs_error = float(delta.mean()) if delta.size else 0.0
    max_abs_error = float(delta.max()) if delta.size else 0.0
    passed = (
        mean_abs_error <= active_thresholds["meanAbsError"]
        and max_abs_error <= active_thresholds["maxAbsError"]
        and cosine_similarity >= active_thresholds["cosineSimilarity"]
    )

    return ParityMetrics(
        mean_abs_error=mean_abs_error,
        max_abs_error=max_abs_error,
        cosine_similarity=cosine_similarity,
        passed=passed,
    )


def validate_report_dict(payload: dict[str, Any]) -> None:
    for field in REQUIRED_REPORT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")

    shape = payload["shape"]
    if not isinstance(shape, dict) or "reference" not in shape or "candidate" not in shape or "matches" not in shape:
        raise ValueError("shape must include reference, candidate, and matches")

    thresholds = payload["thresholds"]
    for field in DEFAULT_THRESHOLDS:
        if field not in thresholds:
            raise ValueError(f"missing threshold field: {field}")
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
uv run pytest tests/test_parity_report.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit parity report contract**

Run:

```bash
git add src/mirrornote_diarization/parity_report.py tests/test_parity_report.py
git commit -m "test: add segmentation parity report contract"
```

Expected: commit succeeds.

---

### Task 3: Strict Weight Mapping Contract

**Files:**
- Create: `src/mirrornote_diarization/weight_conversion.py`
- Create: `tests/test_weight_mapping.py`

- [ ] **Step 1: Write failing weight mapping tests**

Write `tests/test_weight_mapping.py` with this content:

```python
import numpy as np
import pytest

from mirrornote_diarization.weight_conversion import MappingRule, validate_weight_mapping


def test_validate_weight_mapping_maps_all_required_weights():
    reference = {
        "model.linear.weight": np.zeros((2, 3), dtype=np.float32),
        "model.linear.bias": np.zeros((2,), dtype=np.float32),
    }
    rules = [
        MappingRule(reference_key="model.linear.weight", candidate_key="linear.weight", expected_shape=(2, 3)),
        MappingRule(reference_key="model.linear.bias", candidate_key="linear.bias", expected_shape=(2,)),
    ]

    result = validate_weight_mapping(reference, rules)

    assert result.mapped == {
        "linear.weight": "model.linear.weight",
        "linear.bias": "model.linear.bias",
    }
    assert result.missing_reference == []
    assert result.shape_mismatches == []
    assert result.passed is True


def test_validate_weight_mapping_fails_on_missing_reference_weight():
    reference = {"model.linear.weight": np.zeros((2, 3), dtype=np.float32)}
    rules = [
        MappingRule(reference_key="model.linear.weight", candidate_key="linear.weight", expected_shape=(2, 3)),
        MappingRule(reference_key="model.linear.bias", candidate_key="linear.bias", expected_shape=(2,)),
    ]

    result = validate_weight_mapping(reference, rules)

    assert result.missing_reference == ["model.linear.bias"]
    assert result.passed is False


def test_validate_weight_mapping_fails_on_shape_mismatch():
    reference = {"model.linear.weight": np.zeros((4, 3), dtype=np.float32)}
    rules = [MappingRule(reference_key="model.linear.weight", candidate_key="linear.weight", expected_shape=(2, 3))]

    result = validate_weight_mapping(reference, rules)

    assert result.shape_mismatches == [
        {
            "referenceKey": "model.linear.weight",
            "candidateKey": "linear.weight",
            "expectedShape": [2, 3],
            "actualShape": [4, 3],
        }
    ]
    assert result.passed is False


def test_validate_weight_mapping_rejects_duplicate_candidate_keys():
    reference = {
        "a": np.zeros((1,), dtype=np.float32),
        "b": np.zeros((1,), dtype=np.float32),
    }
    rules = [
        MappingRule(reference_key="a", candidate_key="dup", expected_shape=(1,)),
        MappingRule(reference_key="b", candidate_key="dup", expected_shape=(1,)),
    ]

    with pytest.raises(ValueError, match="duplicate candidate key: dup"):
        validate_weight_mapping(reference, rules)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_weight_mapping.py -q
```

Expected: FAIL with missing `weight_conversion` module or symbols.

- [ ] **Step 3: Implement strict weight mapping validation**

Write `src/mirrornote_diarization/weight_conversion.py` with this content:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MappingRule:
    reference_key: str
    candidate_key: str
    expected_shape: tuple[int, ...]


@dataclass(frozen=True)
class MappingResult:
    mapped: dict[str, str]
    missing_reference: list[str]
    shape_mismatches: list[dict[str, Any]]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapped": self.mapped,
            "missingReference": self.missing_reference,
            "shapeMismatches": self.shape_mismatches,
            "passed": self.passed,
        }


def validate_weight_mapping(
    reference_weights: Mapping[str, np.ndarray],
    rules: Sequence[MappingRule],
) -> MappingResult:
    mapped: dict[str, str] = {}
    missing_reference: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    seen_candidate_keys: set[str] = set()

    for rule in rules:
        if rule.candidate_key in seen_candidate_keys:
            raise ValueError(f"duplicate candidate key: {rule.candidate_key}")
        seen_candidate_keys.add(rule.candidate_key)

        weight = reference_weights.get(rule.reference_key)
        if weight is None:
            missing_reference.append(rule.reference_key)
            continue

        actual_shape = tuple(int(dim) for dim in weight.shape)
        if actual_shape != rule.expected_shape:
            shape_mismatches.append(
                {
                    "referenceKey": rule.reference_key,
                    "candidateKey": rule.candidate_key,
                    "expectedShape": list(rule.expected_shape),
                    "actualShape": list(actual_shape),
                }
            )
            continue

        mapped[rule.candidate_key] = rule.reference_key

    return MappingResult(
        mapped=mapped,
        missing_reference=missing_reference,
        shape_mismatches=shape_mismatches,
        passed=not missing_reference and not shape_mismatches,
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
uv run pytest tests/test_weight_mapping.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit strict mapping contract**

Run:

```bash
git add src/mirrornote_diarization/weight_conversion.py tests/test_weight_mapping.py
git commit -m "test: add strict segmentation weight mapping contract"
```

Expected: commit succeeds.

---

### Task 4: Deterministic Chunk Extraction

**Files:**
- Create: `src/mirrornote_diarization/chunking.py`
- Create: `tests/test_chunking.py`

- [ ] **Step 1: Write failing chunking tests**

Write `tests/test_chunking.py` with this content:

```python
import numpy as np
import pytest

from mirrornote_diarization.chunking import FixedChunk, extract_fixed_chunk


def test_extract_fixed_chunk_returns_exact_window():
    waveform = np.arange(20, dtype=np.float32)

    chunk = extract_fixed_chunk(waveform, sample_rate=10, start_seconds=0.5, duration_seconds=1.0)

    assert chunk.samples.tolist() == [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    assert chunk.sample_rate == 10
    assert chunk.start_seconds == 0.5
    assert chunk.duration_seconds == 1.0


def test_extract_fixed_chunk_pads_short_tail_with_zeroes():
    waveform = np.arange(8, dtype=np.float32)

    chunk = extract_fixed_chunk(waveform, sample_rate=4, start_seconds=1.0, duration_seconds=2.0)

    assert chunk.samples.tolist() == [4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0]


def test_extract_fixed_chunk_rejects_multichannel_input():
    waveform = np.zeros((2, 20), dtype=np.float32)

    with pytest.raises(ValueError, match="mono waveform"):
        extract_fixed_chunk(waveform, sample_rate=10, start_seconds=0.0, duration_seconds=1.0)


def test_fixed_chunk_model_input_shape_is_batch_channel_samples():
    chunk = FixedChunk(
        samples=np.array([1.0, -1.0], dtype=np.float32),
        sample_rate=16000,
        start_seconds=0.0,
        duration_seconds=0.000125,
    )

    model_input = chunk.as_model_input()

    assert model_input.shape == (1, 1, 2)
    assert model_input.dtype == np.float32
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_chunking.py -q
```

Expected: FAIL with missing `chunking` module or symbols.

- [ ] **Step 3: Implement deterministic chunking**

Write `src/mirrornote_diarization/chunking.py` with this content:

```python
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
        return self.samples.astype(np.float32, copy=False).reshape(1, 1, -1)


def extract_fixed_chunk(
    waveform: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    duration_seconds: float,
) -> FixedChunk:
    if waveform.ndim != 1:
        raise ValueError(f"expected mono waveform, got shape {waveform.shape}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if start_seconds < 0:
        raise ValueError("start_seconds must be non-negative")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    start_sample = int(round(start_seconds * sample_rate))
    sample_count = int(round(duration_seconds * sample_rate))
    end_sample = start_sample + sample_count

    chunk = waveform[start_sample:end_sample].astype(np.float32, copy=False)
    if chunk.shape[0] < sample_count:
        padding = np.zeros(sample_count - chunk.shape[0], dtype=np.float32)
        chunk = np.concatenate([chunk, padding])

    return FixedChunk(
        samples=chunk,
        sample_rate=sample_rate,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
uv run pytest tests/test_chunking.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit chunking contract**

Run:

```bash
git add src/mirrornote_diarization/chunking.py tests/test_chunking.py
git commit -m "test: add deterministic segmentation chunking"
```

Expected: commit succeeds.

---

### Task 5: Pyannote Probe Contract With Gated Runtime

**Files:**
- Create: `src/mirrornote_diarization/pyannote_probe.py`
- Create: `tests/test_pyannote_probe_contract.py`

- [ ] **Step 1: Write failing pyannote probe contract tests**

Write `tests/test_pyannote_probe_contract.py` with this content:

```python
import os

import pytest

from mirrornote_diarization.pyannote_probe import (
    PyannoteProbeMetadata,
    require_pyannote_enabled,
)


def test_probe_metadata_serializes_required_fields():
    metadata = PyannoteProbeMetadata(
        model_class="FakeSegmentationModel",
        sample_rate=16000,
        chunk_duration_seconds=10.0,
        frame_resolution_seconds=0.016875,
        module_tree=["root", "root.encoder"],
        weight_shapes={"root.encoder.weight": [2, 3]},
        output_shape=[1, 589, 7],
    )

    payload = metadata.to_dict()

    assert payload["modelClass"] == "FakeSegmentationModel"
    assert payload["sampleRate"] == 16000
    assert payload["chunkDurationSeconds"] == 10.0
    assert payload["frameResolutionSeconds"] == 0.016875
    assert payload["moduleTree"] == ["root", "root.encoder"]
    assert payload["weightShapes"] == {"root.encoder.weight": [2, 3]}
    assert payload["outputShape"] == [1, 589, 7]


def test_require_pyannote_enabled_skips_without_flag(monkeypatch):
    monkeypatch.delenv("MIRRORNOTE_RUN_PYANNOTE_PROBE", raising=False)

    with pytest.raises(RuntimeError, match="MIRRORNOTE_RUN_PYANNOTE_PROBE=1"):
        require_pyannote_enabled(os.environ)


def test_require_pyannote_enabled_requires_hf_token(monkeypatch):
    monkeypatch.setenv("MIRRORNOTE_RUN_PYANNOTE_PROBE", "1")
    monkeypatch.delenv("HUGGINGFACE_ACCESS_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HUGGINGFACE_ACCESS_TOKEN"):
        require_pyannote_enabled(os.environ)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_pyannote_probe_contract.py -q
```

Expected: FAIL with missing `pyannote_probe` module or symbols.

- [ ] **Step 3: Implement gated probe contract**

Write `src/mirrornote_diarization/pyannote_probe.py` with this content:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import json

import numpy as np


@dataclass(frozen=True)
class PyannoteProbeMetadata:
    model_class: str
    sample_rate: int
    chunk_duration_seconds: float
    frame_resolution_seconds: float
    module_tree: list[str]
    weight_shapes: dict[str, list[int]]
    output_shape: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "modelClass": self.model_class,
            "sampleRate": self.sample_rate,
            "chunkDurationSeconds": self.chunk_duration_seconds,
            "frameResolutionSeconds": self.frame_resolution_seconds,
            "moduleTree": self.module_tree,
            "weightShapes": self.weight_shapes,
            "outputShape": self.output_shape,
        }


def require_pyannote_enabled(env: Mapping[str, str]) -> None:
    if env.get("MIRRORNOTE_RUN_PYANNOTE_PROBE") != "1":
        raise RuntimeError("set MIRRORNOTE_RUN_PYANNOTE_PROBE=1 to run gated pyannote probe")
    if not env.get("HUGGINGFACE_ACCESS_TOKEN"):
        raise RuntimeError("set HUGGINGFACE_ACCESS_TOKEN before running pyannote probe")


def write_probe_artifacts(
    metadata: PyannoteProbeMetadata,
    reference_output: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n")
    np.savez(out_dir / "reference-output.npz", output=reference_output.astype(np.float32, copy=False))


def run_pyannote_probe(audio_chunk: np.ndarray, out_dir: Path) -> PyannoteProbeMetadata:
    raise RuntimeError(
        "pyannote runtime probe is intentionally added after metadata contract tests pass; "
        "implement this by loading pyannote.audio Pipeline and extracting the segmentation model"
    )
```

- [ ] **Step 4: Run contract tests to verify pass**

Run:

```bash
uv run pytest tests/test_pyannote_probe_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit probe contract**

Run:

```bash
git add src/mirrornote_diarization/pyannote_probe.py tests/test_pyannote_probe_contract.py
git commit -m "test: add gated pyannote probe contract"
```

Expected: commit succeeds.

---

### Task 6: CLI Report Validation Command

**Files:**
- Create: `src/mirrornote_diarization/segmentation_parity.py`
- Modify: `tests/test_parity_report.py`

- [ ] **Step 1: Add CLI validation test**

Append this test to `tests/test_parity_report.py`:

```python
import json

from mirrornote_diarization.segmentation_parity import main


def test_cli_validate_report_accepts_valid_report(tmp_path, capsys):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "referenceProvider": "pyannote-3.1-segmentation-pytorch",
                "candidateProvider": "pyannote-3.1-segmentation-mlx",
                "audioChunk": {
                    "source": "fixtures/single-speaker/system-track.wav",
                    "startTimeSeconds": 0.0,
                    "durationSeconds": 10.0,
                    "sampleRate": 16000,
                },
                "shape": {"reference": [1, 3], "candidate": [1, 3], "matches": True},
                "dtype": {"reference": "float32", "candidate": "float32"},
                "meanAbsError": 0.0,
                "maxAbsError": 0.0,
                "cosineSimilarity": 1.0,
                "thresholds": DEFAULT_THRESHOLDS,
                "passed": True,
            }
        )
        + "\n"
    )

    exit_code = main(["segmentation", "validate-report", str(report_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "valid parity report" in captured.out
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/test_parity_report.py::test_cli_validate_report_accepts_valid_report -q
```

Expected: FAIL with missing `segmentation_parity` module.

- [ ] **Step 3: Implement CLI validate-report command**

Write `src/mirrornote_diarization/segmentation_parity.py` with this content:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mirrornote_diarization.parity_report import validate_report_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mirrornote-diarize")
    subparsers = parser.add_subparsers(dest="domain", required=True)

    segmentation = subparsers.add_parser("segmentation")
    segmentation_subparsers = segmentation.add_subparsers(dest="command", required=True)

    validate_report = segmentation_subparsers.add_parser("validate-report")
    validate_report.add_argument("report", type=Path)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.domain == "segmentation" and args.command == "validate-report":
        payload = json.loads(args.report.read_text())
        validate_report_dict(payload)
        print(f"valid parity report: {args.report}")
        return 0

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests to verify pass**

Run:

```bash
uv run pytest tests/test_parity_report.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit CLI validation command**

Run:

```bash
git add src/mirrornote_diarization/segmentation_parity.py tests/test_parity_report.py
git commit -m "feat: add segmentation parity report CLI"
```

Expected: commit succeeds.

---

### Task 7: MLX Segmentation Skeleton With Explicit Unsupported Runtime

**Files:**
- Create: `src/mirrornote_diarization/mlx_segmentation.py`
- Create: `tests/test_mlx_segmentation_contract.py`

- [ ] **Step 1: Write failing MLX skeleton tests**

Write `tests/test_mlx_segmentation_contract.py` with this content:

```python
import pytest

from mirrornote_diarization.mlx_segmentation import MlxSegmentationConfig, UnsupportedArchitectureError


def test_mlx_segmentation_config_serializes_probe_fields():
    config = MlxSegmentationConfig(
        sample_rate=16000,
        chunk_duration_seconds=10.0,
        output_classes=7,
        architecture_name="pyannote-segmentation-3.0",
    )

    assert config.to_dict() == {
        "sampleRate": 16000,
        "chunkDurationSeconds": 10.0,
        "outputClasses": 7,
        "architectureName": "pyannote-segmentation-3.0",
    }


def test_unsupported_architecture_error_names_architecture():
    error = UnsupportedArchitectureError("unknown-net")

    assert "unknown-net" in str(error)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/test_mlx_segmentation_contract.py -q
```

Expected: FAIL with missing `mlx_segmentation` module.

- [ ] **Step 3: Implement MLX skeleton contract**

Write `src/mirrornote_diarization/mlx_segmentation.py` with this content:

```python
from __future__ import annotations

from dataclasses import dataclass


class UnsupportedArchitectureError(RuntimeError):
    def __init__(self, architecture_name: str) -> None:
        super().__init__(f"unsupported MLX segmentation architecture: {architecture_name}")


@dataclass(frozen=True)
class MlxSegmentationConfig:
    sample_rate: int
    chunk_duration_seconds: float
    output_classes: int
    architecture_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "sampleRate": self.sample_rate,
            "chunkDurationSeconds": self.chunk_duration_seconds,
            "outputClasses": self.output_classes,
            "architectureName": self.architecture_name,
        }


def build_mlx_segmentation(config: MlxSegmentationConfig):
    raise UnsupportedArchitectureError(config.architecture_name)
```

- [ ] **Step 4: Run MLX skeleton tests to verify pass**

Run:

```bash
uv run pytest tests/test_mlx_segmentation_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit MLX skeleton contract**

Run:

```bash
git add src/mirrornote_diarization/mlx_segmentation.py tests/test_mlx_segmentation_contract.py
git commit -m "test: add MLX segmentation skeleton contract"
```

Expected: commit succeeds.

---

### Task 8: Documentation And Report Directory

**Files:**
- Create: `docs/mlx-segmentation-notes.md`
- Create: `reports/segmentation-parity/.gitkeep`
- Modify: `README.md`

- [ ] **Step 1: Add implementation notes document**

Write `docs/mlx-segmentation-notes.md` with this content:

```markdown
# MLX Segmentation Notes

## Purpose

This document records facts discovered while implementing M4A segmentation parity. It is not a product roadmap. It exists to make pyannote architecture, weight mapping, unsupported MLX operations, and threshold changes auditable.

## Reference Model

- Pipeline: `pyannote/speaker-diarization-3.1`
- First MLX target: segmentation submodel only
- Reference provider name: `pyannote-3.1-segmentation-pytorch`
- Candidate provider name: `pyannote-3.1-segmentation-mlx`

## Probe Results

No probe has been run in this repository yet. The first real probe run should add a generated JSON artifact under `artifacts/` or `reports/segmentation-parity/`, not inline large tensor values in this document.

## Weight Mapping Decisions

No architecture-specific mapping has been accepted yet. The first accepted mapping must be strict: every required reference weight maps to one candidate parameter with the same shape.

## Unsupported Operations

No unsupported MLX operations have been confirmed yet.

## Threshold Changes

Initial thresholds remain unchanged:

- `meanAbsError <= 1e-4`
- `maxAbsError <= 1e-3`
- `cosineSimilarity >= 0.999`
```

- [ ] **Step 2: Track report directory**

Run:

```bash
mkdir -p reports/segmentation-parity
touch reports/segmentation-parity/.gitkeep
```

Expected: directory exists and `.gitkeep` is present.

- [ ] **Step 3: Update README with M4A commands**

Append this section to `README.md`:

````markdown
## M4A Segmentation Parity

The first MLX milestone is segmentation parity, not full diarization. The hard gate is chunk-level numerical comparison between the PyTorch pyannote segmentation reference and the MLX candidate implementation.

Baseline contract tests:

```bash
uv run pytest
```

Validate a generated parity report:

```bash
uv run mirrornote-diarize segmentation validate-report reports/segmentation-parity/example.json
```

Real pyannote probing is gated because model access requires Hugging Face credentials and accepted pyannote terms:

```bash
MIRRORNOTE_RUN_PYANNOTE_PROBE=1 \
HUGGINGFACE_ACCESS_TOKEN="$HUGGINGFACE_ACCESS_TOKEN" \
uv run mirrornote-diarize segmentation probe --audio fixtures/single-speaker/system-track.wav --out artifacts/probe
```

The `probe` command is implemented after the offline contracts are passing. Generated probe artifacts should not be committed unless they are small, deterministic metadata files.
````

- [ ] **Step 4: Run full offline tests**

Run:

```bash
uv run pytest
```

Expected: PASS for all offline tests.

- [ ] **Step 5: Run diff check**

Run:

```bash
git diff --check
```

Expected: no output and exit 0.

- [ ] **Step 6: Commit docs and report directory**

Run:

```bash
git add README.md docs/mlx-segmentation-notes.md reports/segmentation-parity/.gitkeep
git commit -m "docs: document M4A segmentation parity workflow"
```

Expected: commit succeeds.

---

### Task 9: Real Pyannote Probe Implementation

**Files:**
- Modify: `src/mirrornote_diarization/pyannote_probe.py`
- Modify: `src/mirrornote_diarization/segmentation_parity.py`
- Create: `tests/test_pyannote_probe_cli.py`

- [ ] **Step 1: Add CLI test for gated probe failure**

Write `tests/test_pyannote_probe_cli.py` with this content:

```python
from mirrornote_diarization.segmentation_parity import main


def test_cli_probe_reports_missing_gate(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("MIRRORNOTE_RUN_PYANNOTE_PROBE", raising=False)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"not-a-real-wav")
    out_dir = tmp_path / "probe"

    exit_code = main(["segmentation", "probe", "--audio", str(audio), "--out", str(out_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "MIRRORNOTE_RUN_PYANNOTE_PROBE=1" in captured.err
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/test_pyannote_probe_cli.py -q
```

Expected: FAIL because the `probe` CLI command does not exist yet.

- [ ] **Step 3: Add probe CLI command with gated runtime**

Modify `src/mirrornote_diarization/segmentation_parity.py` to this complete content:

```python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from mirrornote_diarization.chunking import extract_fixed_chunk
from mirrornote_diarization.parity_report import validate_report_dict
from mirrornote_diarization.pyannote_probe import require_pyannote_enabled, run_pyannote_probe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mirrornote-diarize")
    subparsers = parser.add_subparsers(dest="domain", required=True)

    segmentation = subparsers.add_parser("segmentation")
    segmentation_subparsers = segmentation.add_subparsers(dest="command", required=True)

    validate_report = segmentation_subparsers.add_parser("validate-report")
    validate_report.add_argument("report", type=Path)

    probe = segmentation_subparsers.add_parser("probe")
    probe.add_argument("--audio", type=Path, required=True)
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--start-seconds", type=float, default=0.0)
    probe.add_argument("--duration-seconds", type=float, default=10.0)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.domain == "segmentation" and args.command == "validate-report":
        payload = json.loads(args.report.read_text())
        validate_report_dict(payload)
        print(f"valid parity report: {args.report}")
        return 0

    if args.domain == "segmentation" and args.command == "probe":
        try:
            require_pyannote_enabled(os.environ)
            waveform, sample_rate = _load_wav_mono(args.audio)
            chunk = extract_fixed_chunk(
                waveform,
                sample_rate=sample_rate,
                start_seconds=args.start_seconds,
                duration_seconds=args.duration_seconds,
            )
            metadata = run_pyannote_probe(chunk.as_model_input(), args.out)
        except Exception as error:
            print(str(error), file=__import__("sys").stderr)
            return 1
        print(f"wrote pyannote probe metadata for {metadata.model_class}: {args.out}")
        return 0

    parser.error("unsupported command")
    return 2


def _load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError("install audio dependencies with: uv sync --extra audio") from error

    data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), int(sample_rate)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run gated CLI test to verify pass**

Run:

```bash
uv run pytest tests/test_pyannote_probe_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Implement real pyannote probe internals**

Replace `run_pyannote_probe` in `src/mirrornote_diarization/pyannote_probe.py` with this implementation while keeping the existing dataclass and helper functions:

```python
def run_pyannote_probe(audio_chunk: np.ndarray, out_dir: Path) -> PyannoteProbeMetadata:
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as error:
        raise RuntimeError("install pyannote dependencies with: uv sync --extra pyannote") from error

    import os

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.environ["HUGGINGFACE_ACCESS_TOKEN"],
    )

    segmentation = getattr(pipeline, "_segmentation", None) or getattr(pipeline, "segmentation", None)
    if segmentation is None:
        raise RuntimeError("could not locate pyannote segmentation model on pipeline")

    model = getattr(segmentation, "model", segmentation)
    module_tree = [name or "<root>" for name, _module in model.named_modules()]
    state_dict = model.state_dict()
    weight_shapes = {key: [int(dim) for dim in value.shape] for key, value in state_dict.items()}

    tensor = torch.from_numpy(audio_chunk.astype(np.float32, copy=False))
    model.eval()
    with torch.no_grad():
        output = model(tensor)
    if hasattr(output, "data") and not isinstance(output, torch.Tensor):
        output = output.data
    output_array = output.detach().cpu().numpy().astype(np.float32, copy=False)

    specifications = getattr(model, "specifications", None)
    sample_rate = int(getattr(model, "sample_rate", 16000))
    duration = getattr(getattr(specifications, "duration", None), "total_seconds", None)
    chunk_duration_seconds = float(duration()) if callable(duration) else float(audio_chunk.shape[-1] / sample_rate)
    resolution = getattr(specifications, "resolution", None)
    frame_resolution_seconds = float(getattr(resolution, "duration", 0.0) or 0.0)

    metadata = PyannoteProbeMetadata(
        model_class=type(model).__name__,
        sample_rate=sample_rate,
        chunk_duration_seconds=chunk_duration_seconds,
        frame_resolution_seconds=frame_resolution_seconds,
        module_tree=module_tree,
        weight_shapes=weight_shapes,
        output_shape=[int(dim) for dim in output_array.shape],
    )
    write_probe_artifacts(metadata, output_array, out_dir)
    return metadata
```

- [ ] **Step 6: Run offline tests**

Run:

```bash
uv run pytest
```

Expected: PASS. Real pyannote execution is not run unless the environment variables are set.

- [ ] **Step 7: Run optional real probe when credentials are available**

Run only if pyannote access is configured:

```bash
MIRRORNOTE_RUN_PYANNOTE_PROBE=1 \
HUGGINGFACE_ACCESS_TOKEN="$HUGGINGFACE_ACCESS_TOKEN" \
uv run mirrornote-diarize segmentation probe \
  --audio fixtures/single-speaker/system-track.wav \
  --out artifacts/probe/single-speaker
```

Expected: writes `artifacts/probe/single-speaker/metadata.json` and `artifacts/probe/single-speaker/reference-output.npz`. If the fixture does not exist yet, run this later after fixture creation rather than replacing it with an ad hoc file.

- [ ] **Step 8: Commit probe implementation**

Run:

```bash
git add src/mirrornote_diarization/pyannote_probe.py src/mirrornote_diarization/segmentation_parity.py tests/test_pyannote_probe_cli.py
git commit -m "feat: add gated pyannote segmentation probe"
```

Expected: commit succeeds.

---

### Task 10: Parity Runner With Saved Reference And Candidate Arrays

**Files:**
- Modify: `src/mirrornote_diarization/segmentation_parity.py`
- Create: `tests/test_segmentation_parity_cli.py`

- [ ] **Step 1: Write failing parity runner CLI test**

Write `tests/test_segmentation_parity_cli.py` with this content:

```python
import json

import numpy as np

from mirrornote_diarization.segmentation_parity import main


def test_cli_compare_npz_writes_parity_report(tmp_path):
    reference_path = tmp_path / "reference.npz"
    candidate_path = tmp_path / "candidate.npz"
    report_path = tmp_path / "report.json"
    np.savez(reference_path, output=np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    np.savez(candidate_path, output=np.array([[1.0, 2.0, 3.0]], dtype=np.float32))

    exit_code = main(
        [
            "segmentation",
            "compare-npz",
            "--reference",
            str(reference_path),
            "--candidate",
            str(candidate_path),
            "--source",
            "fixtures/single-speaker/system-track.wav",
            "--out",
            str(report_path),
        ]
    )

    payload = json.loads(report_path.read_text())
    assert exit_code == 0
    assert payload["passed"] is True
    assert payload["shape"]["matches"] is True
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/test_segmentation_parity_cli.py -q
```

Expected: FAIL because `compare-npz` command does not exist.

- [ ] **Step 3: Add compare-npz command**

Modify `src/mirrornote_diarization/segmentation_parity.py` by adding:

```python
from mirrornote_diarization.parity_report import DEFAULT_THRESHOLDS, ParityReport, compute_metrics, validate_report_dict
```

Replace the existing parity report import line with the line above.

Add this parser block after the `probe` parser block:

```python
    compare_npz = segmentation_subparsers.add_parser("compare-npz")
    compare_npz.add_argument("--reference", type=Path, required=True)
    compare_npz.add_argument("--candidate", type=Path, required=True)
    compare_npz.add_argument("--source", required=True)
    compare_npz.add_argument("--out", type=Path, required=True)
```

Add this command branch before `parser.error("unsupported command")`:

```python
    if args.domain == "segmentation" and args.command == "compare-npz":
        reference = np.load(args.reference)["output"]
        candidate = np.load(args.candidate)["output"]
        metrics = compute_metrics(reference, candidate, DEFAULT_THRESHOLDS)
        report = ParityReport(
            reference_provider="pyannote-3.1-segmentation-pytorch",
            candidate_provider="pyannote-3.1-segmentation-mlx",
            audio_chunk={
                "source": args.source,
                "startTimeSeconds": 0.0,
                "durationSeconds": 10.0,
                "sampleRate": 16000,
            },
            shape={
                "reference": [int(dim) for dim in reference.shape],
                "candidate": [int(dim) for dim in candidate.shape],
                "matches": reference.shape == candidate.shape,
            },
            dtype={"reference": str(reference.dtype), "candidate": str(candidate.dtype)},
            mean_abs_error=metrics.mean_abs_error,
            max_abs_error=metrics.max_abs_error,
            cosine_similarity=metrics.cosine_similarity,
            thresholds=DEFAULT_THRESHOLDS,
            passed=metrics.passed,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
        return 0 if metrics.passed else 1
```

- [ ] **Step 4: Run parity runner tests to verify pass**

Run:

```bash
uv run pytest tests/test_segmentation_parity_cli.py tests/test_parity_report.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit compare-npz parity runner**

Run:

```bash
git add src/mirrornote_diarization/segmentation_parity.py tests/test_segmentation_parity_cli.py
git commit -m "feat: add segmentation npz parity runner"
```

Expected: commit succeeds.

---

## Final Verification

- [ ] **Step 1: Run all offline tests**

Run:

```bash
uv run pytest
```

Expected: PASS.

- [ ] **Step 2: Check formatting and whitespace**

Run:

```bash
git diff --check
```

Expected: no output and exit 0.

- [ ] **Step 3: Confirm generated artifacts are not staged**

Run:

```bash
git status --short
```

Expected: clean working tree after all task commits, except intentionally untracked local artifacts under ignored paths.

- [ ] **Step 4: Push when ready**

Run:

```bash
git push origin main
```

Expected: pushes all local commits to `https://github.com/qyinm/mirrornote-diarization-mlx.git`.

## Plan Self-Review

Spec coverage:

- Introspection pass is covered by Task 5 and Task 9.
- Weight mapping pass is covered by Task 3.
- Chunk parity pass is covered by Task 2, Task 4, and Task 10.
- Full wav smoke is intentionally left after chunk parity because the spec marks it optional and non-gating.
- Documentation requirements are covered by Task 8.
- Gated pyannote runtime behavior is covered by Task 5 and Task 9.

Placeholder scan:

- The plan avoids `TBD`, unresolved blanks, and generic edge-case instructions.
- The phrase `probe_batch` appears only in the approved design spec, not in this implementation plan.

Type consistency:

- Report fields use camelCase JSON keys matching the design spec.
- Python dataclasses expose snake_case attributes and convert to camelCase in `to_dict()`.
- CLI commands all live under `mirrornote-diarize segmentation`.
