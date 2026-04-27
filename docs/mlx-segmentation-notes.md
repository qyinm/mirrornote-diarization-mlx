# MLX Segmentation Notes

## Purpose

This document records facts discovered while implementing M4A segmentation parity. It is not a product roadmap.

## Reference Model

The reference model is `pyannote/speaker-diarization-3.1`.

The first MLX target is segmentation only, not full diarization. Provider names should remain explicit when comparing behavior between the reference provider and the MLX provider.

## Probe Results

No probe has been run yet.

Future generated artifacts should go under `artifacts/` or `reports/`. Do not inline large tensors in this document.

## Weight Mapping Decisions

No architecture-specific mapping has been accepted yet.

The first accepted mapping must be strict.

## Unsupported Operations

None confirmed yet.

## Threshold Changes

Initial thresholds are unchanged:

- `meanAbsError <= 1e-4`
- `maxAbsError <= 1e-3`
- `cosineSimilarity >= 0.999`
