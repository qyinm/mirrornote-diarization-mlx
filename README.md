# MirrorNote Diarization MLX

> 🧠 **AI-Agent Developed** — The performance optimizations in this project
> (Metal threadgroup-parallel LSTM kernel, max-pool vectorization) were
> iteratively designed, implemented, and benchmarked by Agent. See [Optimization history](#optimization-history)
> for the full progression from 189ms → 31.6ms.

Local speaker diarization experiments for MirrorNote.

This repository starts with an oracle-first path: run `pyannote/speaker-diarization-3.1` as the reference implementation, normalize its output into MirrorNote's artifact contract, and only then port the proven path to MLX.

## Status

Planning scaffold plus offline segmentation parity contracts, the `segmentation validate-report` CLI, and a gated pyannote segmentation probe CLI exist. The pyannote runtime path remains opt-in because it requires extra dependencies and Hugging Face credentials.

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

Compare saved segmentation reference and candidate `.npz` artifacts with:

```bash
uv run mirrornote-diarize segmentation compare-npz --reference artifacts/reference.npz --candidate artifacts/candidate.npz --source fixtures/single-speaker/system-track.wav --out reports/segmentation-parity/compare.json
```

Run the gated pyannote segmentation probe with:

```bash
MIRRORNOTE_RUN_PYANNOTE_PROBE=1 HUGGINGFACE_ACCESS_TOKEN="$HUGGINGFACE_ACCESS_TOKEN" uv run mirrornote-diarize segmentation probe --audio fixtures/single-speaker/system-track.wav --out artifacts/probe
```

Inspect a saved segmentation probe with:

```bash
uv run mirrornote-diarize segmentation inspect-probe artifacts/probe --json-out reports/segmentation-parity/probe-summary.json
```

The probe command is registered, but it is intentionally gated by `MIRRORNOTE_RUN_PYANNOTE_PROBE=1` and `HUGGINGFACE_ACCESS_TOKEN`. Generated probe artifacts should not be committed unless they are small, deterministic metadata files.

## M4C MLX Candidate Path

Generate a local waveform input artifact from the dummy probe audio when needed:

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

Run the current MLX segmentation candidate:

```bash
uv run --extra mlx mirrornote-diarize segmentation mlx-candidate \
  --weights artifacts/probe/librispeech-dummy-probe/reference-weights.npz \
  --waveform-npz artifacts/probe/librispeech-dummy-probe/waveform-input.npz \
  --out artifacts/probe/librispeech-dummy-probe/mlx-candidate-output.npz
```

Compare the candidate against the reference output:

```bash
uv run --extra dev mirrornote-diarize segmentation compare-npz \
  --reference artifacts/probe/librispeech-dummy-probe/reference-output.npz \
  --candidate artifacts/probe/librispeech-dummy-probe/mlx-candidate-output.npz \
  --source artifacts/audio/librispeech-dummy-probe/audio.wav \
  --out reports/segmentation-parity/librispeech-dummy-mlx-compare.json
```

The current candidate is shape-correct only. `compare-npz` is expected to write a report and return nonzero until numerical parity work lands.

## Runtime Benchmark (10-second chunk)

Run the runtime comparison between the MLX candidate and the reference pyannote segmentation model:

```bash
uv run python scripts/benchmark_segmentation_runtime.py --runs 12 --warmup 3
```

For stage-level profiling (sincnet / lstm / linear):

```bash
uv run python scripts/benchmark_segmentation_runtime.py --runs 12 --warmup 3 --profile-stages
```

To profile stage timing while keeping model compilation enabled (usually faster but less isolated):

```bash
uv run python scripts/benchmark_segmentation_runtime.py --runs 12 --warmup 3 --profile-stages --profile-with-compile
```

To compare LSTM backends (`nn` or `legacy`) without editing environment variables:

```bash
uv run python scripts/benchmark_segmentation_runtime.py --runs 12 --warmup 3 --profile-stages --profile-with-compile --lstm-backend=legacy --no-pyannote
```

Current environment and settings:

- Input: `artifacts/probe/librispeech-dummy-probe/waveform-input.npz` (10.0 s, 16,000 Hz mono)
- Warm-up: 8 runs
- Measurement runs: 30-50
- LSTM backend: `metal` (threadgroup-parallel Metal kernel, 128 threads)
- Max-pool: vectorized (eliminated fallback Python loops)
- `mx.compile` wrapping full forward pass (SincNet + LSTM + Linear)
- Device/platform: `macOS-26.3-arm64-arm-64bit`

Result files:

- `reports/segmentation-benchmark/runtime-benchmark.json`
- `reports/segmentation-benchmark/runtime-benchmark.png`

### Full model (compiled forward pass, 50 runs)

| Provider | Mean (ms) | Real-time factor |
|---|---:|---:|
| `pyannote-3.1-segmentation-mlx` (metal) | **31.6** | **317x** |

### Stage breakdown (profiling mode, 30 runs)

| Stage | Mean (ms) | % of total |
|---|---:|---:|
| SincNet (frontend) | `2.29` | `6.9%` |
| LSTM (4-layer bidir, Metal kernel) | `30.04` | `90.7%` |
| Linear (head) | `0.79` | `2.4%` |
| **Total** | **33.11** | |

### Optimization history

| Milestone | LSTM backend | SincNet (ms) | LSTM (ms) | Total (ms) | vs PyTorch MPS |
|---|---:|---:|---:|---:|---:|
| Original baseline | legacy (Python loop) | ~16 | ~154 | ~189 | 3.4x slower |
| Pass 1 | nn.LSTM + compiled | 3.3 | ~97 | ~107 | 1.9x slower |
| **Pass 2** | **Metal kernel** | **2.3** | **30.0** | **31.6** | **1.77x faster ✓** |

The breakthrough came from writing a custom **threadgroup-parallel Metal kernel** (`src/mirrornote_diarization/lstm_metal.py`) that processes the LSTM recurrence with 128 GPU threads over the hidden dimension, synchronized via threadgroup barriers between timesteps. This eliminates the Python-level loop overhead that limited both `nn.LSTM` and the manual implementation.

PyTorch MPS (56ms) relies on Apple's MPSGraph native LSTM kernel. MLX 0.31.2 does not ship an equivalent, but `mx.fast.metal_kernel` allowed us to write one — and the custom kernel turned out faster than MPSGraph for this model size (batch=1, hidden=128, 589 timesteps).

---

## Built With

- **Agent** — performance optimization, Metal kernel authoring, benchmarking
- **MLX 0.31.2** — Apple Silicon array framework, `mx.fast.metal_kernel`
- **PyTorch + pyannote.audio 3.1** — reference oracle implementation
- **Python 3.12** — uv-managed project
