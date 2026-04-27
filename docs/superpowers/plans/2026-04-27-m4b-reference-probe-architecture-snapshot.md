# M4B Reference Probe Architecture Snapshot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the gated pyannote segmentation probe into an auditable reference snapshot that records architecture metadata, validates generated artifacts, and prepares the exact target for later MLX implementation.

**Architecture:** Keep real pyannote execution optional and gated, but make everything around its output testable offline. Add a probe artifact loader/validator, a deterministic architecture summary command, and docs that tell an engineer exactly how to run the real probe when credentials and fixtures exist. Do not implement MLX layers or weight conversion yet.

**Tech Stack:** Python 3.11+, uv, pytest, numpy, optional soundfile/pyannote.audio/torch for gated probe execution, JSON/NPZ artifacts, Markdown docs.

---

## Scope Check

This plan covers one subsystem: reference probe artifact quality and architecture snapshotting. It does not implement MLX segmentation layers, actual weight conversion, layer-by-layer parity, full wav smoke, embedding, clustering, or MirrorNote integration.

## File Structure

Create or modify these files:

- Create: `src/mirrornote_diarization/probe_artifacts.py`
  - Owns loading and validating generated probe artifact directories.
- Create: `tests/test_probe_artifacts.py`
  - Tests metadata/reference-output loading and failure cases without pyannote.
- Modify: `src/mirrornote_diarization/segmentation_parity.py`
  - Adds `segmentation inspect-probe` to summarize a probe artifact directory.
- Create: `tests/test_probe_inspect_cli.py`
  - Tests inspect CLI output and error handling offline.
- Modify: `src/mirrornote_diarization/pyannote_probe.py`
  - Hardens metadata extraction with deterministic fields needed by the inspector.
- Modify: `tests/test_pyannote_probe_contract.py`
  - Covers new metadata fields and artifact writing behavior.
- Modify: `docs/mlx-segmentation-notes.md`
  - Adds a section for the first reference snapshot procedure and what to paste after running it.
- Modify: `README.md`
  - Documents `inspect-probe` and the exact gated real-probe workflow.

---

### Task 1: Probe Artifact Loader Contract

**Files:**
- Create: `src/mirrornote_diarization/probe_artifacts.py`
- Create: `tests/test_probe_artifacts.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/test_probe_artifacts.py` with this content:

```python
import json

import numpy as np
import pytest

from mirrornote_diarization.probe_artifacts import load_probe_artifacts


def test_load_probe_artifacts_reads_metadata_and_reference_output(tmp_path):
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    metadata = {
        "modelClass": "FakeSegmentationModel",
        "sampleRate": 16000,
        "chunkDurationSeconds": 10.0,
        "frameResolutionSeconds": 0.016875,
        "moduleTree": ["<root>", "encoder", "classifier"],
        "weightShapes": {"encoder.weight": [2, 3], "classifier.bias": [7]},
        "outputShape": [1, 589, 7],
    }
    (probe_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.savez(probe_dir / "reference-output.npz", output=np.zeros((1, 589, 7), dtype=np.float32))

    artifacts = load_probe_artifacts(probe_dir)

    assert artifacts.metadata["modelClass"] == "FakeSegmentationModel"
    assert artifacts.reference_output.shape == (1, 589, 7)
    assert artifacts.reference_output.dtype == np.float32
    assert artifacts.parameter_count == 13
    assert artifacts.module_count == 3


def test_load_probe_artifacts_rejects_missing_metadata(tmp_path):
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    np.savez(probe_dir / "reference-output.npz", output=np.zeros((1, 1, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="missing metadata.json"):
        load_probe_artifacts(probe_dir)


def test_load_probe_artifacts_rejects_missing_output_key(tmp_path):
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "metadata.json").write_text("{}", encoding="utf-8")
    np.savez(probe_dir / "reference-output.npz", logits=np.zeros((1, 1, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="missing output array"):
        load_probe_artifacts(probe_dir)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_probe_artifacts.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mirrornote_diarization.probe_artifacts'`.

- [ ] **Step 3: Implement the probe artifact loader**

Write `src/mirrornote_diarization/probe_artifacts.py` with this content:

```python
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
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run --extra dev pytest tests/test_probe_artifacts.py -q
```

Expected: PASS.

- [ ] **Step 5: Run full tests and commit**

Run:

```bash
uv run --extra dev pytest
git diff --check
git add src/mirrornote_diarization/probe_artifacts.py tests/test_probe_artifacts.py
git commit -m "test: add probe artifact loader"
```

Expected: `pytest` passes and commit succeeds.

---

### Task 2: Probe Inspection CLI

**Files:**
- Modify: `src/mirrornote_diarization/segmentation_parity.py`
- Create: `tests/test_probe_inspect_cli.py`

- [ ] **Step 1: Write the failing CLI tests**

Write `tests/test_probe_inspect_cli.py` with this content:

```python
import json

import numpy as np

from mirrornote_diarization.segmentation_parity import main


def _write_probe(probe_dir):
    probe_dir.mkdir()
    metadata = {
        "modelClass": "FakeSegmentationModel",
        "sampleRate": 16000,
        "chunkDurationSeconds": 10.0,
        "frameResolutionSeconds": 0.016875,
        "moduleTree": ["<root>", "encoder", "classifier"],
        "weightShapes": {"encoder.weight": [2, 3], "classifier.bias": [7]},
        "outputShape": [1, 589, 7],
    }
    (probe_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.savez(probe_dir / "reference-output.npz", output=np.zeros((1, 589, 7), dtype=np.float32))


def test_inspect_probe_prints_human_summary(tmp_path, capsys):
    probe_dir = tmp_path / "probe"
    _write_probe(probe_dir)

    exit_code = main(["segmentation", "inspect-probe", str(probe_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "modelClass: FakeSegmentationModel" in captured.out
    assert "outputShape: [1, 589, 7]" in captured.out
    assert "moduleCount: 3" in captured.out
    assert "parameterCount: 13" in captured.out


def test_inspect_probe_writes_json_summary(tmp_path):
    probe_dir = tmp_path / "probe"
    out_path = tmp_path / "summary.json"
    _write_probe(probe_dir)

    exit_code = main(["segmentation", "inspect-probe", str(probe_dir), "--json-out", str(out_path)])

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["modelClass"] == "FakeSegmentationModel"
    assert payload["moduleCount"] == 3
    assert payload["parameterCount"] == 13


def test_inspect_probe_reports_invalid_directory(tmp_path, capsys):
    exit_code = main(["segmentation", "inspect-probe", str(tmp_path / "missing")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "probe inspection failed" in captured.err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_probe_inspect_cli.py -q
```

Expected: FAIL because `inspect-probe` is not registered.

- [ ] **Step 3: Implement `inspect-probe`**

Modify `src/mirrornote_diarization/segmentation_parity.py`:

- Import `load_probe_artifacts`.
- Add parser:

```python
inspect_probe_parser = segmentation_subparsers.add_parser(
    "inspect-probe", help="Inspect a generated pyannote probe artifact directory"
)
inspect_probe_parser.add_argument("probe_dir", type=Path)
inspect_probe_parser.add_argument("--json-out", type=Path)
```

- Add dispatch:

```python
if args.segmentation_command == "inspect-probe":
    return _inspect_probe(args)
```

- Add helpers:

```python
def _inspect_probe(args: argparse.Namespace) -> int:
    try:
        artifacts = load_probe_artifacts(args.probe_dir)
        summary = _build_probe_summary(artifacts)
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"probe inspection failed: {exc}", file=sys.stderr)
        return 1

    print(f"modelClass: {summary['modelClass']}")
    print(f"sampleRate: {summary['sampleRate']}")
    print(f"chunkDurationSeconds: {summary['chunkDurationSeconds']}")
    print(f"frameResolutionSeconds: {summary['frameResolutionSeconds']}")
    print(f"outputShape: {summary['outputShape']}")
    print(f"moduleCount: {summary['moduleCount']}")
    print(f"parameterCount: {summary['parameterCount']}")
    return 0


def _build_probe_summary(artifacts) -> dict[str, object]:
    metadata = artifacts.metadata
    return {
        "modelClass": metadata.get("modelClass"),
        "sampleRate": metadata.get("sampleRate"),
        "chunkDurationSeconds": metadata.get("chunkDurationSeconds"),
        "frameResolutionSeconds": metadata.get("frameResolutionSeconds"),
        "outputShape": [int(dimension) for dimension in artifacts.reference_output.shape],
        "moduleCount": artifacts.module_count,
        "parameterCount": artifacts.parameter_count,
    }
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run --extra dev pytest tests/test_probe_inspect_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Run full tests and commit**

Run:

```bash
uv run --extra dev pytest
git diff --check
git add src/mirrornote_diarization/segmentation_parity.py tests/test_probe_inspect_cli.py
git commit -m "feat: add probe artifact inspection CLI"
```

Expected: `pytest` passes and commit succeeds.

---

### Task 3: Metadata Contract Hardening

**Files:**
- Modify: `src/mirrornote_diarization/pyannote_probe.py`
- Modify: `tests/test_pyannote_probe_contract.py`
- Modify: `src/mirrornote_diarization/probe_artifacts.py`
- Modify: `tests/test_probe_artifacts.py`

- [ ] **Step 1: Add tests for richer deterministic metadata**

Append to `tests/test_pyannote_probe_contract.py`:

```python

def test_probe_metadata_includes_weight_dtype_and_parameter_count():
    metadata = PyannoteProbeMetadata(
        model_class="FakeSegmentationModel",
        sample_rate=16000,
        chunk_duration_seconds=10.0,
        frame_resolution_seconds=0.016875,
        module_tree=["<root>", "encoder"],
        weight_shapes={"encoder.weight": [2, 3]},
        output_shape=[1, 589, 7],
        weight_dtypes={"encoder.weight": "float32"},
        parameter_count=6,
    )

    payload = metadata.to_dict()

    assert payload["weightDtypes"] == {"encoder.weight": "float32"}
    assert payload["parameterCount"] == 6
```

Append to `tests/test_probe_artifacts.py`:

```python

def test_load_probe_artifacts_prefers_metadata_parameter_count(tmp_path):
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    metadata = {
        "modelClass": "FakeSegmentationModel",
        "sampleRate": 16000,
        "chunkDurationSeconds": 10.0,
        "frameResolutionSeconds": 0.016875,
        "moduleTree": ["<root>"],
        "weightShapes": {"encoder.weight": [2, 3]},
        "weightDtypes": {"encoder.weight": "float32"},
        "parameterCount": 123,
        "outputShape": [1, 589, 7],
    }
    (probe_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.savez(probe_dir / "reference-output.npz", output=np.zeros((1, 589, 7), dtype=np.float32))

    artifacts = load_probe_artifacts(probe_dir)

    assert artifacts.parameter_count == 123
```

- [ ] **Step 2: Run focused tests to verify failure**

Run:

```bash
uv run --extra dev pytest tests/test_pyannote_probe_contract.py tests/test_probe_artifacts.py -q
```

Expected: FAIL because `PyannoteProbeMetadata` lacks the new fields and loader computes parameter count only from shapes.

- [ ] **Step 3: Update metadata dataclass and serialization**

Modify `src/mirrornote_diarization/pyannote_probe.py`:

- Add fields to `PyannoteProbeMetadata`:

```python
weight_dtypes: dict[str, str]
parameter_count: int
```

- Add keys in `to_dict()`:

```python
"weightDtypes": self.weight_dtypes,
"parameterCount": self.parameter_count,
```

- In `run_pyannote_probe`, compute:

```python
weight_dtypes = {key: str(value.dtype).replace("torch.", "") for key, value in state_dict.items()}
parameter_count = sum(int(value.numel()) for value in state_dict.values())
```

- Pass `weight_dtypes=weight_dtypes` and `parameter_count=parameter_count` into `PyannoteProbeMetadata`.

- Update existing tests that instantiate `PyannoteProbeMetadata` to include the two new fields.

- [ ] **Step 4: Update probe artifact loader parameter count**

Modify `ProbeArtifacts.parameter_count` in `src/mirrornote_diarization/probe_artifacts.py`:

```python
metadata_count = self.metadata.get("parameterCount")
if isinstance(metadata_count, int) and metadata_count >= 0:
    return metadata_count
```

Then keep the existing shape-derived fallback.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
uv run --extra dev pytest tests/test_pyannote_probe_contract.py tests/test_probe_artifacts.py -q
uv run --extra dev pytest
git diff --check
git add src/mirrornote_diarization/pyannote_probe.py src/mirrornote_diarization/probe_artifacts.py tests/test_pyannote_probe_contract.py tests/test_probe_artifacts.py
git commit -m "feat: enrich pyannote probe metadata"
```

Expected: tests pass and commit succeeds.

---

### Task 4: Reference Snapshot Documentation Workflow

**Files:**
- Modify: `docs/mlx-segmentation-notes.md`
- Modify: `README.md`

- [ ] **Step 1: Update notes with the snapshot workflow**

Add this section to `docs/mlx-segmentation-notes.md`:

```markdown
## Reference Snapshot Procedure

The first real snapshot should be generated from a real MirrorNote-style `system-track.wav` fixture after pyannote dependencies and Hugging Face access are configured.

Run the gated probe:

```bash
MIRRORNOTE_RUN_PYANNOTE_PROBE=1 \
HUGGINGFACE_ACCESS_TOKEN="$HUGGINGFACE_ACCESS_TOKEN" \
uv run --extra audio --extra pyannote mirrornote-diarize segmentation probe \
  --audio fixtures/single-speaker/system-track.wav \
  --out artifacts/probe/single-speaker
```

Inspect the generated snapshot:

```bash
uv run --extra dev mirrornote-diarize segmentation inspect-probe \
  artifacts/probe/single-speaker \
  --json-out reports/segmentation-parity/single-speaker-probe-summary.json
```

Do not commit large tensor artifacts from `artifacts/`. A small JSON summary under `reports/segmentation-parity/` may be committed only when it is deterministic and useful for later MLX implementation.
```

- [ ] **Step 2: Update README command list**

In `README.md`, add an `inspect-probe` example after the gated probe command:

```markdown
Inspect a generated probe directory:

```bash
uv run mirrornote-diarize segmentation inspect-probe artifacts/probe --json-out reports/segmentation-parity/probe-summary.json
```
```

- [ ] **Step 3: Run tests and commit docs**

Run:

```bash
uv run --extra dev pytest
git diff --check
git add README.md docs/mlx-segmentation-notes.md
git commit -m "docs: add reference snapshot workflow"
```

Expected: tests pass and commit succeeds.

---

### Task 5: Final Verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Run the full offline test suite**

Run:

```bash
uv run --extra dev pytest
```

Expected: all tests pass.

- [ ] **Step 2: Verify CLI commands manually with synthetic artifacts**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path('/tmp/mirrornote-m4b-probe')
root.mkdir(parents=True, exist_ok=True)
metadata = {
    'modelClass': 'FakeSegmentationModel',
    'sampleRate': 16000,
    'chunkDurationSeconds': 10.0,
    'frameResolutionSeconds': 0.016875,
    'moduleTree': ['<root>', 'encoder'],
    'weightShapes': {'encoder.weight': [2, 3]},
    'weightDtypes': {'encoder.weight': 'float32'},
    'parameterCount': 6,
    'outputShape': [1, 2, 3],
}
(root / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
np.savez(root / 'reference-output.npz', output=np.zeros((1, 2, 3), dtype=np.float32))
PY
uv run --extra dev mirrornote-diarize segmentation inspect-probe /tmp/mirrornote-m4b-probe --json-out /tmp/mirrornote-m4b-probe-summary.json
uv run --extra dev mirrornote-diarize segmentation validate-report reports/segmentation-parity/example.json
```

Expected: `inspect-probe` succeeds. The `validate-report` example may fail if `reports/segmentation-parity/example.json` does not exist; do not create that file unless a real report exists.

- [ ] **Step 3: Check git hygiene**

Run:

```bash
git diff --check
git status --short
```

Expected: clean working tree, except intentionally ignored local generated artifacts under `artifacts/` or `/tmp`.

## Plan Self-Review

Spec coverage:

- Probe artifact validation is covered by Task 1.
- Human/JSON architecture snapshot summary is covered by Task 2.
- Deterministic metadata needed by MLX planning is covered by Task 3.
- Real-probe operating procedure is covered by Task 4.
- Final offline verification is covered by Task 5.

Placeholder scan:

- No unresolved placeholders are included.
- Token-dependent real pyannote execution remains documented as gated and optional, not required for offline tests.

Type consistency:

- Probe metadata JSON uses camelCase keys consistent with existing `PyannoteProbeMetadata.to_dict()`.
- CLI commands remain under `mirrornote-diarize segmentation`.
- `ProbeArtifacts.parameter_count` returns an `int` from metadata when available, otherwise derives from `weightShapes`.
