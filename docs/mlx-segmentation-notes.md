# MLX Segmentation Notes

## Purpose

This document records facts discovered while implementing M4A segmentation parity. It is not a product roadmap. It exists to make pyannote architecture, weight mapping, unsupported MLX operations, and threshold changes auditable.

## Reference Model

- Pipeline: `pyannote/speaker-diarization-3.1`
- First MLX target: segmentation submodel only
- Reference provider name: `pyannote-3.1-segmentation-pytorch`
- Candidate provider name: `pyannote-3.1-segmentation-mlx`

## Probe Results

No probe has been run yet.

The first real probe should add a generated JSON artifact under `artifacts/` or `reports/segmentation-parity/`. Do not inline large tensor values in this document.

## Reference Snapshot Procedure

The first real snapshot should be generated from a real MirrorNote-style `system-track.wav` fixture after pyannote dependencies and Hugging Face access are configured.

Generate the gated reference probe snapshot with:

```bash
MIRRORNOTE_RUN_PYANNOTE_PROBE=1 \
HUGGINGFACE_ACCESS_TOKEN="$HUGGINGFACE_ACCESS_TOKEN" \
uv run --extra audio --extra pyannote mirrornote-diarize segmentation probe \
  --audio fixtures/single-speaker/system-track.wav \
  --out artifacts/probe/single-speaker
```

Inspect the generated probe snapshot with:

```bash
uv run --extra dev mirrornote-diarize segmentation inspect-probe \
  artifacts/probe/single-speaker \
  --json-out reports/segmentation-parity/single-speaker-probe-summary.json
```

Do not commit large tensor artifacts from `artifacts/`. A small JSON summary under `reports/segmentation-parity/` may be committed only when it is deterministic and useful for later MLX implementation.

## Weight Mapping Decisions

No architecture-specific mapping has been accepted yet.

The first accepted mapping must be strict: every required reference weight maps to one candidate parameter with the same shape.

## Unsupported Operations

No unsupported MLX operations confirmed yet.

## Threshold Changes

Initial thresholds are unchanged:

- `meanAbsError <= 1e-4`
- `maxAbsError <= 1e-3`
- `cosineSimilarity >= 0.999`
