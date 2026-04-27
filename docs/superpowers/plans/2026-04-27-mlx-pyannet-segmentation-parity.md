# MLX PyanNet Segmentation Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strict MLX implementation path for the pyannote 3.1 segmentation submodel that can load the saved oracle probe, map every required PyanNet weight, run a candidate forward pass, and produce a parity report against `reference-output.npz`.

**Architecture:** Keep the oracle probe artifacts as the source of truth. Add a PyanNet-specific architecture contract, export reference weights to `.npz`, map PyTorch tensor names into MLX parameter names, and implement the MLX forward path in small layers: SincNet frontend, bidirectional LSTM stack, linear head, classifier, activation. The first success criterion is exact shape compatibility and deterministic compare output; numerical thresholds remain the existing strict parity thresholds.

**Tech Stack:** Python 3.12, NumPy, PyTorch/pyannote.audio 3.1.1 for oracle export, MLX for candidate runtime, `uv`, `pytest`.

---

## Current Baseline

The current local probe artifact exists at:

```text
artifacts/probe/librispeech-dummy-probe/metadata.json
artifacts/probe/librispeech-dummy-probe/reference-output.npz
```

The observed reference contract is:

```json
{
  "modelClass": "pyannote.audio.models.segmentation.PyanNet.PyanNet",
  "sampleRate": 16000,
  "chunkDurationSeconds": 10.0,
  "outputShape": [1, 589, 7],
  "moduleCount": 22,
  "parameterCount": 1473515
}
```

The observed module order is:

```text
model
model.sincnet
model.sincnet.wav_norm1d
model.sincnet.conv1d
model.sincnet.conv1d.0
model.sincnet.conv1d.0.filterbank
model.sincnet.conv1d.1
model.sincnet.conv1d.2
model.sincnet.pool1d
model.sincnet.pool1d.0
model.sincnet.pool1d.1
model.sincnet.pool1d.2
model.sincnet.norm1d
model.sincnet.norm1d.0
model.sincnet.norm1d.1
model.sincnet.norm1d.2
model.lstm
model.linear
model.linear.0
model.linear.1
model.classifier
model.activation
```

The observed state dict keys and shapes are:

```text
classifier.bias [7]
classifier.weight [7, 128]
linear.0.bias [128]
linear.0.weight [128, 256]
linear.1.bias [128]
linear.1.weight [128, 128]
lstm.bias_hh_l0 [512]
lstm.bias_hh_l0_reverse [512]
lstm.bias_hh_l1 [512]
lstm.bias_hh_l1_reverse [512]
lstm.bias_hh_l2 [512]
lstm.bias_hh_l2_reverse [512]
lstm.bias_hh_l3 [512]
lstm.bias_hh_l3_reverse [512]
lstm.bias_ih_l0 [512]
lstm.bias_ih_l0_reverse [512]
lstm.bias_ih_l1 [512]
lstm.bias_ih_l1_reverse [512]
lstm.bias_ih_l2 [512]
lstm.bias_ih_l2_reverse [512]
lstm.bias_ih_l3 [512]
lstm.bias_ih_l3_reverse [512]
lstm.weight_hh_l0 [512, 128]
lstm.weight_hh_l0_reverse [512, 128]
lstm.weight_hh_l1 [512, 128]
lstm.weight_hh_l1_reverse [512, 128]
lstm.weight_hh_l2 [512, 128]
lstm.weight_hh_l2_reverse [512, 128]
lstm.weight_hh_l3 [512, 128]
lstm.weight_hh_l3_reverse [512, 128]
lstm.weight_ih_l0 [512, 60]
lstm.weight_ih_l0_reverse [512, 60]
lstm.weight_ih_l1 [512, 256]
lstm.weight_ih_l1_reverse [512, 256]
lstm.weight_ih_l2 [512, 256]
lstm.weight_ih_l2_reverse [512, 256]
lstm.weight_ih_l3 [512, 256]
lstm.weight_ih_l3_reverse [512, 256]
sincnet.conv1d.0.filterbank.band_hz_ [40, 1]
sincnet.conv1d.0.filterbank.low_hz_ [40, 1]
sincnet.conv1d.0.filterbank.n_ [1, 125]
sincnet.conv1d.0.filterbank.window_ [125]
sincnet.conv1d.1.bias [60]
sincnet.conv1d.1.weight [60, 80, 5]
sincnet.conv1d.2.bias [60]
sincnet.conv1d.2.weight [60, 60, 5]
sincnet.norm1d.0.bias [80]
sincnet.norm1d.0.weight [80]
sincnet.norm1d.1.bias [60]
sincnet.norm1d.1.weight [60]
sincnet.norm1d.2.bias [60]
sincnet.norm1d.2.weight [60]
sincnet.wav_norm1d.bias [1]
sincnet.wav_norm1d.weight [1]
```

## File Structure

Create and modify these files:

```text
src/mirrornote_diarization/pyannet_contract.py
  Owns the hard PyanNet architecture constants observed from the oracle probe.

src/mirrornote_diarization/pyannote_probe.py
  Adds state-dict export beside metadata/reference output.

src/mirrornote_diarization/weight_conversion.py
  Adds PyanNet-specific mapping rules and an `.npz` loader for exported weights.

src/mirrornote_diarization/mlx_pyannet.py
  Owns MLX PyanNet layers and forward pass.

src/mirrornote_diarization/mlx_segmentation.py
  Routes supported PyanNet configs to `MlxPyanNetSegmentation`.

src/mirrornote_diarization/segmentation_parity.py
  Adds CLI commands for exporting weights, running MLX candidate output, and comparing with existing report code.

tests/test_pyannet_contract.py
  Tests the observed architecture contract.

tests/test_pyannote_weight_export.py
  Tests probe weight export without importing pyannote.

tests/test_pyannet_weight_mapping.py
  Tests strict mapping of every observed PyanNet weight.

tests/test_mlx_pyannet.py
  Tests MLX layer shape behavior and candidate `.npz` output behavior.

tests/test_segmentation_mlx_cli.py
  Tests new CLI behavior with small synthetic artifacts.

docs/mlx-segmentation-notes.md
  Records the real probe result and accepted weight mapping decisions.

README.md
  Adds the exact command sequence for generating oracle weights and candidate output.
```

---

### Task 1: Capture PyanNet Architecture Contract

**Files:**
- Create: `src/mirrornote_diarization/pyannet_contract.py`
- Create: `tests/test_pyannet_contract.py`
- Modify: `docs/mlx-segmentation-notes.md`

- [ ] **Step 1: Write the failing contract tests**

Create `tests/test_pyannet_contract.py`:

```python
from mirrornote_diarization.pyannet_contract import (
    PYANNET_ARCHITECTURE_NAME,
    PYANNET_EXPECTED_MODULE_TREE,
    PYANNET_EXPECTED_OUTPUT_SHAPE,
    PYANNET_EXPECTED_PARAMETER_COUNT,
    PYANNET_EXPECTED_WEIGHT_SHAPES,
    PYANNET_SAMPLE_RATE,
    PYANNET_CHUNK_DURATION_SECONDS,
)


def test_pyannet_reference_contract_matches_real_probe() -> None:
    assert PYANNET_ARCHITECTURE_NAME == (
        "pyannote.audio.models.segmentation.PyanNet.PyanNet"
    )
    assert PYANNET_SAMPLE_RATE == 16000
    assert PYANNET_CHUNK_DURATION_SECONDS == 10.0
    assert PYANNET_EXPECTED_OUTPUT_SHAPE == (1, 589, 7)
    assert PYANNET_EXPECTED_PARAMETER_COUNT == 1_473_515


def test_pyannet_module_tree_is_complete_and_ordered() -> None:
    assert PYANNET_EXPECTED_MODULE_TREE == (
        "model",
        "model.sincnet",
        "model.sincnet.wav_norm1d",
        "model.sincnet.conv1d",
        "model.sincnet.conv1d.0",
        "model.sincnet.conv1d.0.filterbank",
        "model.sincnet.conv1d.1",
        "model.sincnet.conv1d.2",
        "model.sincnet.pool1d",
        "model.sincnet.pool1d.0",
        "model.sincnet.pool1d.1",
        "model.sincnet.pool1d.2",
        "model.sincnet.norm1d",
        "model.sincnet.norm1d.0",
        "model.sincnet.norm1d.1",
        "model.sincnet.norm1d.2",
        "model.lstm",
        "model.linear",
        "model.linear.0",
        "model.linear.1",
        "model.classifier",
        "model.activation",
    )


def test_pyannet_expected_weight_shapes_include_every_reference_weight() -> None:
    assert len(PYANNET_EXPECTED_WEIGHT_SHAPES) == 52
    assert PYANNET_EXPECTED_WEIGHT_SHAPES["sincnet.wav_norm1d.weight"] == (1,)
    assert PYANNET_EXPECTED_WEIGHT_SHAPES[
        "sincnet.conv1d.0.filterbank.n_"
    ] == (1, 125)
    assert PYANNET_EXPECTED_WEIGHT_SHAPES["sincnet.conv1d.1.weight"] == (60, 80, 5)
    assert PYANNET_EXPECTED_WEIGHT_SHAPES["lstm.weight_ih_l0"] == (512, 60)
    assert PYANNET_EXPECTED_WEIGHT_SHAPES["lstm.weight_ih_l3_reverse"] == (512, 256)
    assert PYANNET_EXPECTED_WEIGHT_SHAPES["linear.0.weight"] == (128, 256)
    assert PYANNET_EXPECTED_WEIGHT_SHAPES["classifier.weight"] == (7, 128)
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_pyannet_contract.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'mirrornote_diarization.pyannet_contract'
```

- [ ] **Step 3: Add the contract module**

Create `src/mirrornote_diarization/pyannet_contract.py`:

```python
"""Observed pyannote 3.1 PyanNet segmentation architecture contract."""

from __future__ import annotations

PYANNET_ARCHITECTURE_NAME = "pyannote.audio.models.segmentation.PyanNet.PyanNet"
PYANNET_SAMPLE_RATE = 16000
PYANNET_CHUNK_DURATION_SECONDS = 10.0
PYANNET_EXPECTED_OUTPUT_SHAPE = (1, 589, 7)
PYANNET_EXPECTED_PARAMETER_COUNT = 1_473_515

PYANNET_EXPECTED_MODULE_TREE = (
    "model",
    "model.sincnet",
    "model.sincnet.wav_norm1d",
    "model.sincnet.conv1d",
    "model.sincnet.conv1d.0",
    "model.sincnet.conv1d.0.filterbank",
    "model.sincnet.conv1d.1",
    "model.sincnet.conv1d.2",
    "model.sincnet.pool1d",
    "model.sincnet.pool1d.0",
    "model.sincnet.pool1d.1",
    "model.sincnet.pool1d.2",
    "model.sincnet.norm1d",
    "model.sincnet.norm1d.0",
    "model.sincnet.norm1d.1",
    "model.sincnet.norm1d.2",
    "model.lstm",
    "model.linear",
    "model.linear.0",
    "model.linear.1",
    "model.classifier",
    "model.activation",
)

PYANNET_EXPECTED_WEIGHT_SHAPES = {
    "classifier.bias": (7,),
    "classifier.weight": (7, 128),
    "linear.0.bias": (128,),
    "linear.0.weight": (128, 256),
    "linear.1.bias": (128,),
    "linear.1.weight": (128, 128),
    "lstm.bias_hh_l0": (512,),
    "lstm.bias_hh_l0_reverse": (512,),
    "lstm.bias_hh_l1": (512,),
    "lstm.bias_hh_l1_reverse": (512,),
    "lstm.bias_hh_l2": (512,),
    "lstm.bias_hh_l2_reverse": (512,),
    "lstm.bias_hh_l3": (512,),
    "lstm.bias_hh_l3_reverse": (512,),
    "lstm.bias_ih_l0": (512,),
    "lstm.bias_ih_l0_reverse": (512,),
    "lstm.bias_ih_l1": (512,),
    "lstm.bias_ih_l1_reverse": (512,),
    "lstm.bias_ih_l2": (512,),
    "lstm.bias_ih_l2_reverse": (512,),
    "lstm.bias_ih_l3": (512,),
    "lstm.bias_ih_l3_reverse": (512,),
    "lstm.weight_hh_l0": (512, 128),
    "lstm.weight_hh_l0_reverse": (512, 128),
    "lstm.weight_hh_l1": (512, 128),
    "lstm.weight_hh_l1_reverse": (512, 128),
    "lstm.weight_hh_l2": (512, 128),
    "lstm.weight_hh_l2_reverse": (512, 128),
    "lstm.weight_hh_l3": (512, 128),
    "lstm.weight_hh_l3_reverse": (512, 128),
    "lstm.weight_ih_l0": (512, 60),
    "lstm.weight_ih_l0_reverse": (512, 60),
    "lstm.weight_ih_l1": (512, 256),
    "lstm.weight_ih_l1_reverse": (512, 256),
    "lstm.weight_ih_l2": (512, 256),
    "lstm.weight_ih_l2_reverse": (512, 256),
    "lstm.weight_ih_l3": (512, 256),
    "lstm.weight_ih_l3_reverse": (512, 256),
    "sincnet.conv1d.0.filterbank.band_hz_": (40, 1),
    "sincnet.conv1d.0.filterbank.low_hz_": (40, 1),
    "sincnet.conv1d.0.filterbank.n_": (1, 125),
    "sincnet.conv1d.0.filterbank.window_": (125,),
    "sincnet.conv1d.1.bias": (60,),
    "sincnet.conv1d.1.weight": (60, 80, 5),
    "sincnet.conv1d.2.bias": (60,),
    "sincnet.conv1d.2.weight": (60, 60, 5),
    "sincnet.norm1d.0.bias": (80,),
    "sincnet.norm1d.0.weight": (80,),
    "sincnet.norm1d.1.bias": (60,),
    "sincnet.norm1d.1.weight": (60,),
    "sincnet.norm1d.2.bias": (60,),
    "sincnet.norm1d.2.weight": (60,),
    "sincnet.wav_norm1d.bias": (1,),
    "sincnet.wav_norm1d.weight": (1,),
}
```

- [ ] **Step 4: Run the contract tests and verify they pass**

Run:

```bash
uv run --extra dev pytest tests/test_pyannet_contract.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Update architecture notes**

Modify `docs/mlx-segmentation-notes.md` so `## Probe Results` contains:

```markdown
## Probe Results

A real probe has been run with `artifacts/audio/librispeech-dummy-probe/audio.wav`.
The generated tensor artifacts stay under ignored `artifacts/` paths.

Summary:

- modelClass: `pyannote.audio.models.segmentation.PyanNet.PyanNet`
- sampleRate: `16000`
- chunkDurationSeconds: `10.0`
- outputShape: `[1, 589, 7]`
- moduleCount: `22`
- parameterCount: `1473515`

The first MLX target is the segmentation submodel only:

```text
SincNet frontend -> 4-layer bidirectional LSTM -> Linear(256, 128) -> Linear(128, 128) -> Classifier(128, 7) -> activation
```
```

- [ ] **Step 6: Run the full test suite**

Run:

```bash
uv run --extra dev pytest
```

Expected:

```text
54 passed
```

- [ ] **Step 7: Commit**

Run:

```bash
git add src/mirrornote_diarization/pyannet_contract.py tests/test_pyannet_contract.py docs/mlx-segmentation-notes.md
git commit -m "docs: capture pyannet segmentation contract"
```

---

### Task 2: Export Oracle State Dict Weights

**Files:**
- Modify: `src/mirrornote_diarization/pyannote_probe.py`
- Modify: `src/mirrornote_diarization/segmentation_parity.py`
- Create: `tests/test_pyannote_weight_export.py`

- [ ] **Step 1: Write tests for state-dict export**

Create `tests/test_pyannote_weight_export.py`:

```python
from pathlib import Path

import numpy as np

from mirrornote_diarization.pyannote_probe import write_probe_artifacts
from mirrornote_diarization.pyannote_probe import PyannoteProbeMetadata


def test_write_probe_artifacts_can_include_reference_weights(tmp_path: Path) -> None:
    metadata = PyannoteProbeMetadata(
        model_class="pyannote.audio.models.segmentation.PyanNet.PyanNet",
        sample_rate=16000,
        chunk_duration_seconds=10.0,
        frame_resolution_seconds=0.0,
        module_tree=["model", "model.classifier"],
        weight_shapes={"classifier.weight": [7, 128]},
        weight_dtypes={"classifier.weight": "float32"},
        parameter_count=896,
        output_shape=[1, 589, 7],
    )
    reference_output = np.zeros((1, 589, 7), dtype=np.float32)
    reference_weights = {
        "classifier.weight": np.ones((7, 128), dtype=np.float32),
        "classifier.bias": np.zeros((7,), dtype=np.float32),
    }

    write_probe_artifacts(
        metadata,
        reference_output,
        tmp_path,
        reference_weights=reference_weights,
    )

    with np.load(tmp_path / "reference-weights.npz") as payload:
        assert sorted(payload.files) == ["classifier.bias", "classifier.weight"]
        assert payload["classifier.weight"].shape == (7, 128)
        assert payload["classifier.weight"].dtype == np.float32


def test_write_probe_artifacts_without_weights_keeps_current_contract(tmp_path: Path) -> None:
    metadata = PyannoteProbeMetadata(
        model_class="pyannote.audio.models.segmentation.PyanNet.PyanNet",
        sample_rate=16000,
        chunk_duration_seconds=10.0,
        frame_resolution_seconds=0.0,
        module_tree=[],
        weight_shapes={},
        weight_dtypes={},
        parameter_count=0,
        output_shape=[1, 589, 7],
    )

    write_probe_artifacts(metadata, np.zeros((1, 589, 7), dtype=np.float32), tmp_path)

    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "reference-output.npz").exists()
    assert not (tmp_path / "reference-weights.npz").exists()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_pyannote_weight_export.py -q
```

Expected:

```text
TypeError: write_probe_artifacts() got an unexpected keyword argument 'reference_weights'
```

- [ ] **Step 3: Add optional weight export to `write_probe_artifacts`**

Modify `src/mirrornote_diarization/pyannote_probe.py`:

```python
def write_probe_artifacts(
    metadata: PyannoteProbeMetadata,
    reference_output: np.ndarray,
    out_dir: str | Path,
    reference_weights: Mapping[str, Any] | None = None,
) -> None:
    """Write probe metadata, float32 reference output, and optional weights."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n"
    )

    output = np.asarray(reference_output, dtype=np.float32)
    np.savez(output_dir / "reference-output.npz", output=output)

    if reference_weights is not None:
        np.savez(
            output_dir / "reference-weights.npz",
            **{
                name: np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
                if hasattr(value, "detach")
                else np.asarray(value, dtype=np.float32)
                for name, value in reference_weights.items()
            },
        )
```

- [ ] **Step 4: Pass the real state dict from `run_pyannote_probe`**

Modify the end of `run_pyannote_probe` in `src/mirrornote_diarization/pyannote_probe.py`:

```python
    write_probe_artifacts(
        metadata,
        output_array,
        out_dir,
        reference_weights=state_dict,
    )
    return metadata
```

- [ ] **Step 5: Run weight export tests**

Run:

```bash
uv run --extra dev pytest tests/test_pyannote_weight_export.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Regenerate the local probe artifact with weights**

Run:

```bash
set -a
source .env
set +a
uv run --extra audio --extra pyannote mirrornote-diarize segmentation probe \
  --audio artifacts/audio/librispeech-dummy-probe/audio.wav \
  --out artifacts/probe/librispeech-dummy-probe \
  --duration-seconds 10.0
```

Expected:

```text
wrote pyannote probe artifacts: artifacts/probe/librispeech-dummy-probe
```

Verify:

```bash
uv run python - <<'PY'
import numpy as np
with np.load('artifacts/probe/librispeech-dummy-probe/reference-weights.npz') as weights:
    print(len(weights.files))
    print(weights['classifier.weight'].shape)
PY
```

Expected:

```text
52
(7, 128)
```

- [ ] **Step 7: Run the full test suite**

Run:

```bash
uv run --extra dev pytest
```

Expected:

```text
56 passed
```

- [ ] **Step 8: Commit**

Run:

```bash
git add src/mirrornote_diarization/pyannote_probe.py tests/test_pyannote_weight_export.py
git commit -m "feat: export pyannote segmentation weights"
```

---

### Task 3: Add Strict PyanNet Weight Mapping

**Files:**
- Modify: `src/mirrornote_diarization/weight_conversion.py`
- Create: `tests/test_pyannet_weight_mapping.py`
- Modify: `docs/mlx-segmentation-notes.md`

- [ ] **Step 1: Write mapping tests**

Create `tests/test_pyannet_weight_mapping.py`:

```python
import numpy as np

from mirrornote_diarization.pyannet_contract import PYANNET_EXPECTED_WEIGHT_SHAPES
from mirrornote_diarization.weight_conversion import (
    build_pyannet_mapping_rules,
    load_npz_weights,
    validate_weight_mapping,
)


def test_pyannet_mapping_rules_cover_every_expected_weight() -> None:
    rules = build_pyannet_mapping_rules()

    assert len(rules) == len(PYANNET_EXPECTED_WEIGHT_SHAPES)
    assert {rule.reference_key for rule in rules} == set(PYANNET_EXPECTED_WEIGHT_SHAPES)
    assert len({rule.candidate_key for rule in rules}) == len(rules)


def test_pyannet_mapping_rules_use_stable_candidate_namespace() -> None:
    rules = build_pyannet_mapping_rules()
    mapped = {rule.reference_key: rule.candidate_key for rule in rules}

    assert mapped["sincnet.wav_norm1d.weight"] == "sincnet.wav_norm.weight"
    assert mapped[
        "sincnet.conv1d.0.filterbank.low_hz_"
    ] == "sincnet.sinc_filterbank.low_hz"
    assert mapped["lstm.weight_ih_l0"] == "lstm.layers.0.forward.weight_ih"
    assert mapped["lstm.weight_ih_l0_reverse"] == "lstm.layers.0.reverse.weight_ih"
    assert mapped["linear.0.weight"] == "linear.layers.0.weight"
    assert mapped["classifier.bias"] == "classifier.bias"


def test_validate_realistic_pyannet_weight_shapes_passes() -> None:
    reference_weights = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in PYANNET_EXPECTED_WEIGHT_SHAPES.items()
    }

    result = validate_weight_mapping(reference_weights, build_pyannet_mapping_rules())

    assert result.passed is True
    assert result.missing_reference == []
    assert result.shape_mismatches == []


def test_load_npz_weights_returns_float32_arrays(tmp_path) -> None:
    path = tmp_path / "weights.npz"
    np.savez(path, **{"classifier.weight": np.ones((7, 128), dtype=np.float64)})

    weights = load_npz_weights(path)

    assert weights["classifier.weight"].shape == (7, 128)
    assert weights["classifier.weight"].dtype == np.float32
```

- [ ] **Step 2: Run the mapping tests and verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_pyannet_weight_mapping.py -q
```

Expected:

```text
ImportError: cannot import name 'build_pyannet_mapping_rules'
```

- [ ] **Step 3: Add NPZ loader and PyanNet rules**

Modify `src/mirrornote_diarization/weight_conversion.py`:

```python
from pathlib import Path

from mirrornote_diarization.pyannet_contract import PYANNET_EXPECTED_WEIGHT_SHAPES
```

Add these functions below `validate_weight_mapping`:

```python
def load_npz_weights(path: str | Path) -> dict[str, np.ndarray]:
    """Load named weight arrays from a `.npz` file as float32 arrays."""
    with np.load(path) as payload:
        return {name: np.asarray(payload[name], dtype=np.float32) for name in payload.files}


def build_pyannet_mapping_rules() -> list[MappingRule]:
    """Build strict reference-to-MLX mapping rules for pyannote 3.1 PyanNet."""
    return [
        MappingRule(reference_key=name, candidate_key=_pyannet_candidate_key(name), expected_shape=shape)
        for name, shape in PYANNET_EXPECTED_WEIGHT_SHAPES.items()
    ]


def _pyannet_candidate_key(reference_key: str) -> str:
    direct = {
        "classifier.bias": "classifier.bias",
        "classifier.weight": "classifier.weight",
        "linear.0.bias": "linear.layers.0.bias",
        "linear.0.weight": "linear.layers.0.weight",
        "linear.1.bias": "linear.layers.1.bias",
        "linear.1.weight": "linear.layers.1.weight",
        "sincnet.conv1d.0.filterbank.band_hz_": "sincnet.sinc_filterbank.band_hz",
        "sincnet.conv1d.0.filterbank.low_hz_": "sincnet.sinc_filterbank.low_hz",
        "sincnet.conv1d.0.filterbank.n_": "sincnet.sinc_filterbank.n",
        "sincnet.conv1d.0.filterbank.window_": "sincnet.sinc_filterbank.window",
        "sincnet.conv1d.1.bias": "sincnet.conv.layers.1.bias",
        "sincnet.conv1d.1.weight": "sincnet.conv.layers.1.weight",
        "sincnet.conv1d.2.bias": "sincnet.conv.layers.2.bias",
        "sincnet.conv1d.2.weight": "sincnet.conv.layers.2.weight",
        "sincnet.norm1d.0.bias": "sincnet.norm.layers.0.bias",
        "sincnet.norm1d.0.weight": "sincnet.norm.layers.0.weight",
        "sincnet.norm1d.1.bias": "sincnet.norm.layers.1.bias",
        "sincnet.norm1d.1.weight": "sincnet.norm.layers.1.weight",
        "sincnet.norm1d.2.bias": "sincnet.norm.layers.2.bias",
        "sincnet.norm1d.2.weight": "sincnet.norm.layers.2.weight",
        "sincnet.wav_norm1d.bias": "sincnet.wav_norm.bias",
        "sincnet.wav_norm1d.weight": "sincnet.wav_norm.weight",
    }
    if reference_key in direct:
        return direct[reference_key]

    if reference_key.startswith("lstm."):
        return _pyannet_lstm_candidate_key(reference_key)

    raise ValueError(f"unsupported PyanNet reference key: {reference_key}")


def _pyannet_lstm_candidate_key(reference_key: str) -> str:
    parts = reference_key.split("_")
    layer_token = parts[-1]
    direction = "forward"
    if layer_token == "reverse":
        layer_token = parts[-2]
        direction = "reverse"
    layer_index = int(layer_token.removeprefix("l"))

    if reference_key.startswith("lstm.weight_ih"):
        leaf = "weight_ih"
    elif reference_key.startswith("lstm.weight_hh"):
        leaf = "weight_hh"
    elif reference_key.startswith("lstm.bias_ih"):
        leaf = "bias_ih"
    elif reference_key.startswith("lstm.bias_hh"):
        leaf = "bias_hh"
    else:
        raise ValueError(f"unsupported PyanNet LSTM key: {reference_key}")

    return f"lstm.layers.{layer_index}.{direction}.{leaf}"
```

- [ ] **Step 4: Run mapping tests**

Run:

```bash
uv run --extra dev pytest tests/test_pyannet_weight_mapping.py -q
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Verify local exported weights satisfy mapping rules**

Run:

```bash
uv run python - <<'PY'
from mirrornote_diarization.weight_conversion import (
    build_pyannet_mapping_rules,
    load_npz_weights,
    validate_weight_mapping,
)
weights = load_npz_weights('artifacts/probe/librispeech-dummy-probe/reference-weights.npz')
result = validate_weight_mapping(weights, build_pyannet_mapping_rules())
print(result.to_dict())
PY
```

Expected output includes:

```text
'passed': True
'missingReference': []
'shapeMismatches': []
```

- [ ] **Step 6: Update notes with accepted mapping decision**

Modify `docs/mlx-segmentation-notes.md` under `## Weight Mapping Decisions`:

```markdown
## Weight Mapping Decisions

Accepted mapping namespace:

- PyTorch `sincnet.wav_norm1d.*` maps to MLX `sincnet.wav_norm.*`.
- PyTorch `sincnet.conv1d.0.filterbank.*_` maps to MLX `sincnet.sinc_filterbank.*` without trailing underscore.
- PyTorch `sincnet.conv1d.1.*` and `sincnet.conv1d.2.*` map to MLX `sincnet.conv.layers.1.*` and `sincnet.conv.layers.2.*`.
- PyTorch `sincnet.norm1d.N.*` maps to MLX `sincnet.norm.layers.N.*`.
- PyTorch bidirectional LSTM keys map to `lstm.layers.{index}.{forward|reverse}.{weight_ih|weight_hh|bias_ih|bias_hh}`.
- PyTorch `linear.N.*` maps to MLX `linear.layers.N.*`.
- PyTorch `classifier.*` maps to MLX `classifier.*`.

The mapping is strict: every expected reference key must exist and match the observed shape before candidate execution is allowed.
```

- [ ] **Step 7: Run the full test suite**

Run:

```bash
uv run --extra dev pytest
```

Expected:

```text
60 passed
```

- [ ] **Step 8: Commit**

Run:

```bash
git add src/mirrornote_diarization/weight_conversion.py tests/test_pyannet_weight_mapping.py docs/mlx-segmentation-notes.md
git commit -m "feat: add strict pyannet weight mapping"
```

---

### Task 4: Implement MLX PyanNet Candidate Runtime Skeleton

**Files:**
- Create: `src/mirrornote_diarization/mlx_pyannet.py`
- Modify: `src/mirrornote_diarization/mlx_segmentation.py`
- Create: `tests/test_mlx_pyannet.py`

- [ ] **Step 1: Write MLX candidate tests with import skip**

Create `tests/test_mlx_pyannet.py`:

```python
from pathlib import Path

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from mirrornote_diarization.mlx_pyannet import MlxPyanNetSegmentation
from mirrornote_diarization.pyannet_contract import PYANNET_EXPECTED_WEIGHT_SHAPES


def _weights() -> dict[str, np.ndarray]:
    return {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in PYANNET_EXPECTED_WEIGHT_SHAPES.items()
    }


def test_mlx_pyannet_rejects_missing_weight() -> None:
    weights = _weights()
    weights.pop("classifier.bias")

    with pytest.raises(ValueError, match="missingReference"):
        MlxPyanNetSegmentation.from_reference_weights(weights)


def test_mlx_pyannet_builds_from_complete_weight_set() -> None:
    model = MlxPyanNetSegmentation.from_reference_weights(_weights())

    assert model.output_classes == 7
    assert model.sample_rate == 16000
    assert model.chunk_duration_seconds == 10.0


def test_mlx_pyannet_forward_returns_reference_shape_for_zero_weights() -> None:
    model = MlxPyanNetSegmentation.from_reference_weights(_weights())
    waveform = mlx.zeros((1, 1, 160000), dtype=mlx.float32)

    output = model(waveform)

    assert tuple(output.shape) == (1, 589, 7)


def test_mlx_pyannet_writes_candidate_npz(tmp_path: Path) -> None:
    model = MlxPyanNetSegmentation.from_reference_weights(_weights())
    waveform = np.zeros((1, 1, 160000), dtype=np.float32)
    out = tmp_path / "candidate-output.npz"

    model.write_candidate_npz(waveform, out)

    with np.load(out) as payload:
        assert payload["output"].shape == (1, 589, 7)
        assert payload["output"].dtype == np.float32
```

- [ ] **Step 2: Run the tests and verify they fail or skip only when MLX is unavailable**

Run:

```bash
uv run --extra dev --extra mlx pytest tests/test_mlx_pyannet.py -q
```

Expected when MLX is installed:

```text
ModuleNotFoundError: No module named 'mirrornote_diarization.mlx_pyannet'
```

Expected when MLX is unavailable on the machine:

```text
1 skipped
```

- [ ] **Step 3: Add the MLX candidate runtime skeleton**

Create `src/mirrornote_diarization/mlx_pyannet.py`:

```python
"""MLX runtime for the pyannote 3.1 PyanNet segmentation model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class MlxPyanNetSegmentation:
    """Shape-correct MLX candidate runtime for PyanNet segmentation."""

    weights: dict[str, np.ndarray]
    sample_rate: int = PYANNET_SAMPLE_RATE
    chunk_duration_seconds: float = PYANNET_CHUNK_DURATION_SECONDS
    output_classes: int = PYANNET_EXPECTED_OUTPUT_SHAPE[-1]

    @classmethod
    def from_reference_weights(
        cls, reference_weights: dict[str, np.ndarray]
    ) -> "MlxPyanNetSegmentation":
        result = validate_weight_mapping(reference_weights, build_pyannet_mapping_rules())
        if not result.passed:
            raise ValueError(str(result.to_dict()))
        return cls({name: np.asarray(value, dtype=np.float32) for name, value in reference_weights.items()})

    def __call__(self, waveform):
        import mlx.core as mx

        if tuple(waveform.shape) != (1, 1, 160000):
            raise ValueError(f"expected waveform shape (1, 1, 160000), got {tuple(waveform.shape)}")

        # This task establishes the executable candidate path and output contract.
        # Later tasks replace this shape-correct scaffold with the real SincNet/LSTM math.
        return mx.zeros(PYANNET_EXPECTED_OUTPUT_SHAPE, dtype=mx.float32)

    def write_candidate_npz(self, waveform: np.ndarray, out_path: str | Path) -> None:
        import mlx.core as mx

        output = self(mx.array(waveform, dtype=mx.float32))
        out = np.asarray(output, dtype=np.float32)
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, output=out)
```

- [ ] **Step 4: Route PyanNet config from `build_mlx_segmentation`**

Modify `src/mirrornote_diarization/mlx_segmentation.py`:

```python
def build_mlx_segmentation(config: MlxSegmentationConfig):
    if config.architecture_name == "pyannote.audio.models.segmentation.PyanNet.PyanNet":
        from mirrornote_diarization.mlx_pyannet import MlxPyanNetSegmentation

        return MlxPyanNetSegmentation
    raise UnsupportedArchitectureError(config.architecture_name)
```

- [ ] **Step 5: Run MLX candidate tests**

Run:

```bash
uv run --extra dev --extra mlx pytest tests/test_mlx_pyannet.py -q
```

Expected when MLX is installed:

```text
4 passed
```

- [ ] **Step 6: Run full tests with MLX extra**

Run:

```bash
uv run --extra dev --extra mlx pytest
```

Expected:

```text
64 passed
```

- [ ] **Step 7: Commit**

Run:

```bash
git add src/mirrornote_diarization/mlx_pyannet.py src/mirrornote_diarization/mlx_segmentation.py tests/test_mlx_pyannet.py
git commit -m "feat: add mlx pyannet candidate runtime"
```

---

### Task 5: Add CLI for MLX Candidate Output and Comparison

**Files:**
- Modify: `src/mirrornote_diarization/segmentation_parity.py`
- Create: `tests/test_segmentation_mlx_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write CLI tests**

Create `tests/test_segmentation_mlx_cli.py`:

```python
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from mirrornote_diarization.segmentation_parity import main
from mirrornote_diarization.pyannet_contract import PYANNET_EXPECTED_WEIGHT_SHAPES


def test_segmentation_mlx_candidate_writes_npz(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.npz"
    audio_npz = tmp_path / "chunk.npz"
    out_path = tmp_path / "candidate.npz"
    np.savez(
        weights_path,
        **{
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in PYANNET_EXPECTED_WEIGHT_SHAPES.items()
        },
    )
    np.savez(audio_npz, waveform=np.zeros((1, 1, 160000), dtype=np.float32))

    exit_code = main(
        [
            "segmentation",
            "mlx-candidate",
            "--weights",
            str(weights_path),
            "--waveform-npz",
            str(audio_npz),
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    with np.load(out_path) as payload:
        assert payload["output"].shape == (1, 589, 7)


def test_segmentation_mlx_candidate_rejects_missing_waveform_key(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.npz"
    audio_npz = tmp_path / "chunk.npz"
    out_path = tmp_path / "candidate.npz"
    np.savez(
        weights_path,
        **{
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in PYANNET_EXPECTED_WEIGHT_SHAPES.items()
        },
    )
    np.savez(audio_npz, wrong=np.zeros((1, 1, 160000), dtype=np.float32))

    exit_code = main(
        [
            "segmentation",
            "mlx-candidate",
            "--weights",
            str(weights_path),
            "--waveform-npz",
            str(audio_npz),
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 1
    assert not out_path.exists()
```

- [ ] **Step 2: Run CLI tests and verify they fail**

Run:

```bash
uv run --extra dev --extra mlx pytest tests/test_segmentation_mlx_cli.py -q
```

Expected:

```text
SystemExit: 2
```

or:

```text
unsupported segmentation command
```

- [ ] **Step 3: Add parser command**

Modify `build_parser` in `src/mirrornote_diarization/segmentation_parity.py`:

```python
    mlx_candidate_parser = segmentation_subparsers.add_parser(
        "mlx-candidate", help="Run MLX PyanNet segmentation candidate output"
    )
    mlx_candidate_parser.add_argument("--weights", type=Path, required=True)
    mlx_candidate_parser.add_argument("--waveform-npz", type=Path, required=True)
    mlx_candidate_parser.add_argument("--out", type=Path, required=True)
```

Modify `main` in the segmentation command branch:

```python
            if args.segmentation_command == "mlx-candidate":
                return _mlx_candidate(args)
```

- [ ] **Step 4: Add command implementation**

Add imports near the top of `src/mirrornote_diarization/segmentation_parity.py`:

```python
from mirrornote_diarization.mlx_pyannet import MlxPyanNetSegmentation
from mirrornote_diarization.weight_conversion import load_npz_weights
```

Add this function:

```python
def _mlx_candidate(args: argparse.Namespace) -> int:
    try:
        weights = load_npz_weights(args.weights)
        waveform = _load_npz_waveform(args.waveform_npz)
        model = MlxPyanNetSegmentation.from_reference_weights(weights)
        model.write_candidate_npz(waveform, args.out)
    except (OSError, KeyError, ValueError, ImportError) as exc:
        print(f"MLX segmentation candidate failed: {exc}", file=sys.stderr)
        return 1

    print(f"wrote MLX segmentation candidate output: {args.out}")
    return 0


def _load_npz_waveform(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as payload:
        if "waveform" not in payload:
            raise KeyError(f"waveform npz missing required 'waveform' array: {npz_path}")
        return np.asarray(payload["waveform"], dtype=np.float32)
```

- [ ] **Step 5: Run CLI tests**

Run:

```bash
uv run --extra dev --extra mlx pytest tests/test_segmentation_mlx_cli.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Create local waveform artifact from the small WAV**

Run:

```bash
uv run --extra audio python - <<'PY'
from pathlib import Path
import numpy as np
import soundfile as sf
from mirrornote_diarization.chunking import extract_fixed_chunk
waveform, sample_rate = sf.read('artifacts/audio/librispeech-dummy-probe/audio.wav', dtype='float32')
chunk = extract_fixed_chunk(waveform, sample_rate=sample_rate, start_seconds=0.0, duration_seconds=10.0)
path = Path('artifacts/probe/librispeech-dummy-probe/waveform-input.npz')
path.parent.mkdir(parents=True, exist_ok=True)
np.savez(path, waveform=chunk.as_model_input())
print(path)
PY
```

Expected:

```text
artifacts/probe/librispeech-dummy-probe/waveform-input.npz
```

- [ ] **Step 7: Generate candidate output and compare**

Run:

```bash
uv run --extra mlx mirrornote-diarize segmentation mlx-candidate \
  --weights artifacts/probe/librispeech-dummy-probe/reference-weights.npz \
  --waveform-npz artifacts/probe/librispeech-dummy-probe/waveform-input.npz \
  --out artifacts/probe/librispeech-dummy-probe/mlx-candidate-output.npz
```

Expected:

```text
wrote MLX segmentation candidate output: artifacts/probe/librispeech-dummy-probe/mlx-candidate-output.npz
```

Run:

```bash
uv run --extra dev mirrornote-diarize segmentation compare-npz \
  --reference artifacts/probe/librispeech-dummy-probe/reference-output.npz \
  --candidate artifacts/probe/librispeech-dummy-probe/mlx-candidate-output.npz \
  --source artifacts/audio/librispeech-dummy-probe/audio.wav \
  --out reports/segmentation-parity/librispeech-dummy-mlx-compare.json
```

Expected for the shape-correct skeleton:

```text
wrote segmentation parity report: reports/segmentation-parity/librispeech-dummy-mlx-compare.json
```

Expected exit code: `1`, because numerical parity is not yet achieved.

- [ ] **Step 8: Update README commands**

Add this section to `README.md`:

```markdown
## M4C MLX Candidate Path

After generating a probe with `reference-weights.npz`, create the MLX candidate output:

```bash
uv run --extra mlx mirrornote-diarize segmentation mlx-candidate \
  --weights artifacts/probe/librispeech-dummy-probe/reference-weights.npz \
  --waveform-npz artifacts/probe/librispeech-dummy-probe/waveform-input.npz \
  --out artifacts/probe/librispeech-dummy-probe/mlx-candidate-output.npz
```

Compare it with the oracle output:

```bash
uv run --extra dev mirrornote-diarize segmentation compare-npz \
  --reference artifacts/probe/librispeech-dummy-probe/reference-output.npz \
  --candidate artifacts/probe/librispeech-dummy-probe/mlx-candidate-output.npz \
  --source artifacts/audio/librispeech-dummy-probe/audio.wav \
  --out reports/segmentation-parity/librispeech-dummy-mlx-compare.json
```

The initial candidate path is expected to match shape only. Numerical parity is achieved by replacing the shape-correct MLX scaffold with the real SincNet, LSTM, linear, classifier, and activation math.
```

- [ ] **Step 9: Run full tests**

Run:

```bash
uv run --extra dev --extra mlx pytest
```

Expected:

```text
66 passed
```

- [ ] **Step 10: Commit**

Run:

```bash
git add src/mirrornote_diarization/segmentation_parity.py tests/test_segmentation_mlx_cli.py README.md
git commit -m "feat: add mlx segmentation candidate cli"
```

---

### Task 6: Replace Shape Scaffold With Real Linear Head

**Files:**
- Modify: `src/mirrornote_diarization/mlx_pyannet.py`
- Modify: `tests/test_mlx_pyannet.py`

- [ ] **Step 1: Add deterministic linear-head test**

Append to `tests/test_mlx_pyannet.py`:

```python
def test_mlx_pyannet_linear_head_uses_reference_weights() -> None:
    weights = _weights()
    weights["linear.0.weight"] = np.eye(128, 256, dtype=np.float32)
    weights["linear.0.bias"] = np.ones((128,), dtype=np.float32)
    weights["linear.1.weight"] = np.eye(128, 128, dtype=np.float32)
    weights["linear.1.bias"] = np.ones((128,), dtype=np.float32) * 2.0
    weights["classifier.weight"] = np.ones((7, 128), dtype=np.float32)
    weights["classifier.bias"] = np.ones((7,), dtype=np.float32) * 3.0
    model = MlxPyanNetSegmentation.from_reference_weights(weights)
    features = mlx.ones((1, 589, 256), dtype=mlx.float32)

    output = model.linear_head(features)

    assert tuple(output.shape) == (1, 589, 7)
    assert np.allclose(np.asarray(output), 33027.0)
```

The expected `33027.0` comes from: first dense selected 128 ones plus bias 1 gives 2, ReLU keeps 2, second dense gives 128 * 2 + 2 = 258, ReLU keeps 258, classifier gives 128 * 258 + 3 = 33027.

- [ ] **Step 2: Run this test and verify it fails**

Run:

```bash
uv run --extra dev --extra mlx pytest tests/test_mlx_pyannet.py::test_mlx_pyannet_linear_head_uses_reference_weights -q
```

Expected:

```text
AttributeError: 'MlxPyanNetSegmentation' object has no attribute 'linear_head'
```

- [ ] **Step 3: Add dense helper and real linear head**

Modify `src/mirrornote_diarization/mlx_pyannet.py`:

```python
    def linear_head(self, features):
        import mlx.core as mx

        x = _dense(features, self.weights["linear.0.weight"], self.weights["linear.0.bias"])
        x = mx.maximum(x, 0.0)
        x = _dense(x, self.weights["linear.1.weight"], self.weights["linear.1.bias"])
        x = mx.maximum(x, 0.0)
        return _dense(x, self.weights["classifier.weight"], self.weights["classifier.bias"])


def _dense(x, weight: np.ndarray, bias: np.ndarray):
    import mlx.core as mx

    return mx.matmul(x, mx.array(weight.T, dtype=mx.float32)) + mx.array(bias, dtype=mx.float32)
```

- [ ] **Step 4: Run MLX tests**

Run:

```bash
uv run --extra dev --extra mlx pytest tests/test_mlx_pyannet.py -q
```

Expected:

```text
5 passed
```

- [ ] **Step 5: Commit**

Run:

```bash
git add src/mirrornote_diarization/mlx_pyannet.py tests/test_mlx_pyannet.py
git commit -m "feat: implement mlx pyannet linear head"
```

---

### Task 7: Implement Real LSTM and SincNet in Separate Follow-Up Plans

**Files:**
- Modify: `docs/mlx-segmentation-notes.md`

This task records the decomposition boundary after the executable candidate path exists. Do not mix the full SincNet and LSTM math into the same commit as CLI plumbing. They should be separate implementation plans because each has distinct numerical failure modes.

- [ ] **Step 1: Update unsupported operations section**

Modify `docs/mlx-segmentation-notes.md` under `## Unsupported Operations`:

```markdown
## Unsupported Operations

The executable MLX candidate path exists, but numerical parity is not complete until these components are implemented and verified independently:

- SincNet waveform normalization, sinc filterbank construction, convolution, pooling, and normalization.
- Four-layer bidirectional LSTM gate math with PyTorch-compatible gate ordering.
- Final activation behavior after classifier output.

The next plan should target SincNet first because it determines the LSTM input shape `[batch, frames, 60]`. The following plan should target the bidirectional LSTM stack using a saved SincNet output fixture.
```

- [ ] **Step 2: Run docs-sensitive tests**

Run:

```bash
uv run --extra dev pytest
```

Expected:

```text
66 passed
```

- [ ] **Step 3: Commit**

Run:

```bash
git add docs/mlx-segmentation-notes.md
git commit -m "docs: split remaining pyannet parity work"
```

---

## Verification Before Completion

Run these commands before declaring M4C complete:

```bash
git status --short --branch
uv run --extra dev pytest
uv run --extra dev --extra mlx pytest
uv run --extra dev mirrornote-diarize segmentation inspect-probe \
  artifacts/probe/librispeech-dummy-probe \
  --json-out reports/segmentation-parity/librispeech-dummy-probe-summary.json
uv run --extra dev mirrornote-diarize segmentation compare-npz \
  --reference artifacts/probe/librispeech-dummy-probe/reference-output.npz \
  --candidate artifacts/probe/librispeech-dummy-probe/mlx-candidate-output.npz \
  --source artifacts/audio/librispeech-dummy-probe/audio.wav \
  --out reports/segmentation-parity/librispeech-dummy-mlx-compare.json
```

Expected final state for this plan:

```text
- Full tests pass.
- Probe artifacts remain ignored under artifacts/.
- The MLX candidate CLI can emit an `.npz` with shape [1, 589, 7].
- `compare-npz` produces a valid report but fails numerical parity until SincNet and LSTM math are implemented.
```

## Self-Review

- Spec coverage: This plan advances from a successful oracle probe to an executable MLX candidate path, strict weight export, strict mapping, candidate output generation, and comparison reporting.
- Scope control: Full SincNet and LSTM numerical parity are explicitly split into separate follow-up plans because they are independent numerical ports with different debugging surfaces.
- Placeholder scan: The plan avoids open-ended placeholders and gives concrete file paths, commands, and expected outputs.
- Type consistency: `reference-weights.npz`, `waveform-input.npz`, `candidate-output.npz`, `MappingRule`, `MlxPyanNetSegmentation`, and CLI command names are used consistently across tasks.
