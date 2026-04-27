# MLX Segmentation Notes

## Purpose

This document records facts discovered while implementing M4A segmentation parity. It is not a product roadmap. It exists to make pyannote architecture, weight mapping, unsupported MLX operations, and threshold changes auditable.

## Reference Model

- Pipeline: `pyannote/speaker-diarization-3.1`
- First MLX target: segmentation submodel only
- Reference provider name: `pyannote-3.1-segmentation-pytorch`
- Candidate provider name: `pyannote-3.1-segmentation-mlx`

## Probe Results

The committed PyanNet contract comes from the dummy-audio oracle probe at `artifacts/audio/librispeech-dummy-probe/audio.wav`.
The generated tensor artifacts stay under ignored `artifacts/` paths. A MirrorNote-style fixture snapshot remains future work.

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

Accepted mapping namespace:

- PyTorch `sincnet.wav_norm1d.*` maps to MLX `sincnet.wav_norm.*`.
- PyTorch `sincnet.conv1d.0.filterbank.*_` maps to MLX `sincnet.sinc_filterbank.*` without trailing underscore.
- PyTorch `sincnet.conv1d.1.*` and `sincnet.conv1d.2.*` map to MLX `sincnet.conv.layers.1.*` and `sincnet.conv.layers.2.*`.
- PyTorch `sincnet.norm1d.N.*` maps to MLX `sincnet.norm.layers.N.*`.
- PyTorch bidirectional LSTM keys map to `lstm.layers.{index}.{forward|reverse}.{weight_ih|weight_hh|bias_ih|bias_hh}`.
- PyTorch `linear.N.*` maps to MLX `linear.layers.N.*`.
- PyTorch `classifier.*` maps to MLX `classifier.*`.

The mapping is strict: every expected reference key must exist and match the observed shape before candidate execution is allowed.

## Unsupported Operations

No unsupported MLX operations confirmed yet.

## Threshold Changes

Initial thresholds are unchanged:

- `meanAbsError <= 1e-4`
- `maxAbsError <= 1e-3`
- `cosineSimilarity >= 0.999`
