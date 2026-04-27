# M4A Segmentation Parity Design

## Status

Approved for planning on 2026-04-27.

## Goal

Prove that the pyannote 3.1 segmentation submodel can be reproduced in MLX for Apple Silicon with measurable chunk-level parity.

This milestone does not attempt to port the full diarization pipeline. It isolates segmentation because pyannote diarization is a composition of preprocessing, segmentation, embedding, clustering, and post-processing. A full rewrite would make failures hard to diagnose. The first useful proof is that the MLX segmentation implementation can compute the same output as the PyTorch reference for the same fixed audio chunk.

## Non-Goals

- Porting embedding to MLX.
- Porting clustering to MLX.
- Improving full diarization quality.
- Integrating with the MirrorNote app.
- Generating production `speaker-segments.jsonl` from MLX output.
- Implementing speaker naming or UX.
- Supporting `pyannote/community-1`.
- Training, fine-tuning, float16 inference, or quantization.

## Success Criteria

M4A passes when all required gates pass:

1. Introspection pass: the project can extract segmentation metadata from the pyannote 3.1 reference path.
2. Weight mapping pass: all required segmentation weights map into the MLX module with matching shapes.
3. Chunk parity pass: the same fixed chunk input produces matching PyTorch and MLX segmentation outputs within the configured thresholds.

Full wav execution is a smoke test only. It records runtime and output characteristics, but it is not the exit gate for this milestone.

## Components

### `pyannote_probe`

Extracts the reference segmentation model information from pyannote 3.1.

Responsibilities:

- Load the pyannote 3.1 diarization pipeline through the reference Python path.
- Locate the segmentation model and record its module tree.
- Dump model hyperparameters that affect inference.
- Dump weight names and tensor shapes.
- Record expected sample rate, chunk duration, and output frame resolution.
- Run the PyTorch segmentation model on a deterministic fixed chunk.
- Save reference raw output for parity comparison.

The probe is intentionally separate from MLX implementation code so reference extraction and candidate execution cannot accidentally share implementation logic.

### `mlx_segmentation`

Implements the pyannote segmentation network as an MLX inference module.

Responsibilities:

- Recreate the segmentation architecture closely enough to load the reference weights.
- Expose a forward method that accepts the same normalized fixed chunk tensor used by the reference path.
- Run inference in float32 for the first parity milestone.
- Avoid training, fine-tuning, quantization, and product-level post-processing.

### `weight_conversion`

Converts and validates reference weights for MLX.

Responsibilities:

- Read PyTorch or safetensors weights from the reference segmentation model.
- Apply an explicit name mapping from reference keys to MLX module keys.
- Fail on missing weights, unexpected required weights, or shape mismatches.
- Produce an MLX-loadable artifact such as safetensors or npz.
- Write a mapping report that can be inspected when parity fails.

Strict mapping is the default. Partial loading is not acceptable for the pass criteria.

### `parity_runner`

Compares reference and candidate outputs for the same fixed input.

Responsibilities:

- Load the deterministic audio chunk fixture.
- Run or read PyTorch reference output.
- Run MLX candidate output.
- Compare output shape, dtype, and numerical metrics.
- Write a JSON parity report under `reports/segmentation-parity/`.
- Optionally run full wav smoke execution after chunk parity is available.

## Data Flow

```text
fixture wav
  -> deterministic chunk extractor
  -> PyTorch pyannote segmentation
  -> reference output .npz / .json
  -> MLX segmentation
  -> candidate output .npz / .json
  -> parity metrics report
```

The deterministic chunk extractor should remove waveform loading and resampling ambiguity from the first parity pass. The initial MLX implementation should consume the same normalized array that the reference model consumes, not independently load audio from disk.

## Proposed File Layout

```text
src/mirrornote_diarization/
  pyannote_probe.py
  mlx_segmentation.py
  weight_conversion.py
  segmentation_parity.py

tests/
  test_segmentation_parity_contract.py
  test_weight_mapping.py

docs/
  mlx-segmentation-notes.md
  superpowers/specs/2026-04-27-m4a-segmentation-parity-design.md

reports/
  segmentation-parity/
```

## Parity Report Contract

A parity report must include these fields:

```json
{
  "referenceProvider": "pyannote-3.1-segmentation-pytorch",
  "candidateProvider": "pyannote-3.1-segmentation-mlx",
  "audioChunk": {
    "source": "fixtures/.../system-track.wav",
    "startTimeSeconds": 0.0,
    "durationSeconds": 10.0,
    "sampleRate": 16000
  },
  "shape": {
    "reference": ["probe_batch", "probe_frames", "probe_classes"],
    "candidate": ["probe_batch", "probe_frames", "probe_classes"],
    "matches": true
  },
  "dtype": {
    "reference": "float32",
    "candidate": "float32"
  },
  "meanAbsError": 0.0,
  "maxAbsError": 0.0,
  "cosineSimilarity": 1.0,
  "thresholds": {
    "meanAbsError": 0.0001,
    "maxAbsError": 0.001,
    "cosineSimilarity": 0.999
  },
  "passed": true
}
```

The exact output shape must come from the pyannote probe. The shape shown above is illustrative and must not be hard-coded from this document.

## Thresholds

Initial float32 thresholds:

- `meanAbsError <= 1e-4`
- `maxAbsError <= 1e-3`
- `cosineSimilarity >= 0.999`

If these fail, the first response is layer-by-layer diagnosis, not loosening thresholds. Threshold changes require an explicit note in `docs/mlx-segmentation-notes.md` explaining the observed numerical difference and why it is acceptable.

## Testing

### `test_weight_mapping.py`

Verifies that MLX module parameter names and converted weight names match exactly.

Failure cases:

- Missing required weight.
- Unexpected required weight.
- Shape mismatch.
- Silent partial load.

### `test_segmentation_parity_contract.py`

Validates the parity report schema and required fields.

Required fields:

- `referenceProvider`
- `candidateProvider`
- `audioChunk`
- `shape`
- `dtype`
- `meanAbsError`
- `maxAbsError`
- `cosineSimilarity`
- `thresholds`
- `passed`

### Optional Local Pyannote Test

A local integration test may run the real pyannote reference path when the environment has:

- Hugging Face token configured through environment variable.
- Accepted access terms for `pyannote/segmentation-3.0` and `pyannote/speaker-diarization-3.1`.
- Installed `pyannote.audio` and compatible runtime dependencies.

This test should not be required for baseline CI because model access is gated and environment-dependent.

## Risk Handling

### Pyannote internals may change

The segmentation model is accessed through pyannote internals rather than a stable public conversion API. The probe must save metadata artifacts so changes in module tree, weight names, or output shapes are visible.

### Preprocessing can hide model errors

Audio loading, resampling, channel mixing, and normalization can create parity failures unrelated to MLX. The first pass feeds MLX the same normalized chunk tensor used by the PyTorch reference path.

### MLX may lack an operation

Unsupported operations should be listed in `docs/mlx-segmentation-notes.md` with the chosen replacement. Replacements must be covered by focused tests or layer-level comparisons when practical.

### Thresholds may be too strict

The project should diagnose layer-level differences before changing thresholds. Threshold relaxation is allowed only with a written rationale and an updated report.

## Implementation Order

1. Add the pyannote segmentation probe.
2. Add deterministic fixed chunk extraction and reference output dump.
3. Add the MLX segmentation module skeleton.
4. Add strict weight conversion and mapping validation.
5. Add the chunk parity runner.
6. Add JSON parity metrics reports.
7. Add optional full wav smoke execution.

## Documentation Requirements

- Keep this design document as the milestone source of truth.
- Add `docs/mlx-segmentation-notes.md` during implementation to capture architecture discoveries, weight mapping details, unsupported operations, and threshold changes.
- Store generated parity outputs under `reports/segmentation-parity/`.

## Transition To Implementation Planning

After user review of this spec, the next step is to invoke the `superpowers:writing-plans` skill and produce a detailed implementation plan. No implementation work should happen before that plan exists.
