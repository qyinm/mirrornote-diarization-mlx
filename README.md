# MirrorNote Diarization MLX

Local speaker diarization experiments for MirrorNote.

This repository starts with an oracle-first path: run `pyannote/speaker-diarization-3.1` as the reference implementation, normalize its output into MirrorNote's artifact contract, and only then port the proven path to MLX.

## Status

Planning scaffold plus offline segmentation parity contracts and the `segmentation validate-report` CLI exist. No pyannote or MLX runtime implementation exists yet.

## First Principle

MirrorNote does not need a generic diarization playground. It needs a local, measurable, app-compatible way to turn `system-track.wav` into stable speaker labels.

## Target Output

The default CLI output is MirrorNote-native:

```text
out/
  speaker-segments.jsonl
  speaker-map.json
  metrics.json
```

RTTM export can exist later as an optional debug/export format. It is not the product contract.

## Source Model

Initial reference model:

- `pyannote/speaker-diarization-3.1`
- License: MIT, per the Hugging Face model card
- Runtime: `pyannote.audio` 3.1+ / PyTorch
- Input: mono 16 kHz audio, with pyannote able to downmix/resample when loading

Links:

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://github.com/pyannote/pyannote-audio/blob/main/LICENSE
- https://opensource.apple.com/projects/mlx/

## Documents

- [Identity](docs/IDENTITY.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)

## M4A Segmentation Parity

The first MLX milestone is segmentation parity, not full diarization. The initial goal is to match the segmentation component against the reference path before expanding into the rest of the diarization pipeline.

Run the baseline test suite with:

```bash
uv run pytest
```

Validate a segmentation parity report with:

```bash
uv run mirrornote-diarize segmentation validate-report reports/segmentation-parity/example.json
```

Planned gated pyannote probe command, not currently runnable until the probe parser/runtime is implemented in a later task:

```bash
MIRRORNOTE_RUN_PYANNOTE_PROBE=1 HUGGINGFACE_ACCESS_TOKEN="$HUGGINGFACE_ACCESS_TOKEN" uv run mirrornote-diarize segmentation probe --audio fixtures/single-speaker/system-track.wav --out artifacts/probe
```

The probe command is implemented after the offline contracts are passing. Generated probe artifacts should not be committed unless they are small, deterministic metadata files.
