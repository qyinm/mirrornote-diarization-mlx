# Implementation Plan

## Goal

Build an oracle-first speaker diarization CLI for MirrorNote.

The first milestone is not MLX. The first milestone is a reliable reference implementation that uses `pyannote/speaker-diarization-3.1`, emits MirrorNote-native artifacts, and records metrics. Once that is stable, MLX can be built and compared against the oracle.

## Current Decision Log

- Use `pyannote/speaker-diarization-3.1` before `community-1`.
- Start oracle-first, not MLX-first.
- Do not add this repo as a MirrorNote submodule yet.
- Default output is MirrorNote-native JSON, not RTTM.
- Add schemas and golden fixtures before MLX work.
- Add performance metrics from v0.

## Target CLI

Initial command shape:

```bash
uv run mirrornote-diarize oracle \
  --audio fixtures/single-speaker/system-track.wav \
  --out /tmp/diarization-out \
  --metrics /tmp/diarization-metrics.json
```

Expected output:

```text
/tmp/diarization-out/
  speaker-segments.jsonl
  speaker-map.json
/tmp/diarization-metrics.json
```

Optional later:

```bash
uv run mirrornote-diarize oracle \
  --audio system-track.wav \
  --out out \
  --export-rttm out/oracle.rttm
```

## Proposed Repo Structure

```text
mirrornote-diarization-mlx/
  README.md
  pyproject.toml
  docs/
    IDENTITY.md
    IMPLEMENTATION_PLAN.md
  contracts/
    speaker-segments.schema.json
    speaker-map.schema.json
    metrics.schema.json
  fixtures/
    silence/
      system-track.wav
      expected.speaker-segments.jsonl
      expected.speaker-map.json
    single-speaker/
      system-track.wav
      expected.speaker-segments.jsonl
      expected.speaker-map.json
    two-speaker-overlap-lite/
      system-track.wav
      expected.speaker-segments.jsonl
      expected.speaker-map.json
  src/
    mirrornote_diarization/
      __init__.py
      cli.py
      contract.py
      metrics.py
      oracle_pyannote31.py
      rttm.py
      compare.py
  tests/
    test_contract.py
    test_metrics.py
    test_oracle_cli.py
    test_rttm_normalization.py
```

## MirrorNote Artifact Contract

### `speaker-segments.jsonl`

One JSON object per line.

Required fields:

```json
{
  "speakerId": "speaker-1",
  "source": "system",
  "startTimeSeconds": 1.23,
  "endTimeSeconds": 4.56,
  "confidence": 0.94,
  "provider": "pyannote-3.1-oracle"
}
```

Rules:

- `source` is `system` for this project's v0 output.
- `speakerId` should be deterministic within a file, for example `speaker-1`, `speaker-2`.
- `startTimeSeconds` and `endTimeSeconds` are seconds from the start of `system-track.wav`.
- `confidence` may be `1.0` if pyannote does not expose a useful confidence for the segment.
- `provider` should identify the runtime path.

### `speaker-map.json`

```json
{
  "speakers": [
    {
      "speakerId": "speaker-1",
      "label": "Speaker 1",
      "name": null,
      "locked": false
    }
  ]
}
```

Rules:

- Do not emit `you`; MirrorNote reserves `you` for the local microphone track.
- Remote/system speakers should be unlocked.
- Names are user-editable later, so default `name` is null.

### `metrics.json`

```json
{
  "audioSeconds": 1680.0,
  "wallTimeMs": 420000,
  "peakRSSBytes": 1234567890,
  "device": "cpu",
  "provider": "pyannote-3.1-oracle",
  "ok": true
}
```

Rules:

- Always write metrics for successful and failed runs when possible.
- Include failure reason on failed runs.
- Record device as `cpu`, `mps`, `cuda`, or `unknown`.

## Pipeline

```text
system-track.wav
  |
  v
validate input
  |
  v
load pyannote/speaker-diarization-3.1
  |
  v
run oracle diarization
  |
  v
normalize speakers and timestamps
  |
  +--> speaker-segments.jsonl
  |
  +--> speaker-map.json
  |
  +--> metrics.json
```

## Milestones

### M0, Planning Scaffold

Deliverables:

- `README.md`
- `docs/IDENTITY.md`
- `docs/IMPLEMENTATION_PLAN.md`

Exit criteria:

- The repo states what it is, what it is not, and why oracle-first exists.

### M1, Contract and Fixtures

Deliverables:

- JSON schemas for `speaker-segments.jsonl`, `speaker-map.json`, and `metrics.json`.
- Golden fixture directories for silence, single speaker, and two-speaker overlap-lite.
- Contract tests that validate all expected outputs.

Exit criteria:

- `uv run pytest` validates fixture outputs against schemas.
- Fixture output uses MirrorNote-native fields exactly.

### M2, pyannote 3.1 Oracle CLI

Deliverables:

- `mirrornote-diarize oracle` command.
- Hugging Face token handling via environment variable, not CLI args.
- Clear errors for missing token, gated model access, missing audio, unsupported audio.
- Optional RTTM export.

Exit criteria:

- Oracle CLI emits valid MirrorNote artifacts for all fixtures.
- CLI exits nonzero with clear messages on failure.
- Metrics JSON is written for successful runs and best-effort failures.

### M3, Comparator

Deliverables:

- Compare oracle output to another provider output.
- Basic timing/overlap tolerance report.
- Deterministic speaker-id check.

Exit criteria:

- Future MLX output can be compared against oracle without MirrorNote.

### M4, MLX Spike

Deliverables:

- Pick the smallest pyannote subcomponent to port first.
- Document unsupported pieces explicitly.
- Produce provider output in the same MirrorNote artifact contract.

Exit criteria:

- MLX spike can run on at least one fixture and produce comparable JSON.

### M5, MirrorNote Integration Decision

Deliverables:

- Quality/performance report on at least three dogfood meeting system tracks.
- Decision: keep external CLI, vendor/submodule, or defer.

Exit criteria:

- Submodule integration is allowed only after measured value beats source-only labeling.

## Test Plan

```text
CODE PATHS                                      USER FLOWS
[+] oracle CLI                                  [+] developer runs oracle on fixture
  ├── missing audio -> clear error                ├── silence -> empty valid artifacts
  ├── missing token -> clear error                ├── single speaker -> one speaker
  ├── pyannote load failure -> clear error         ├── two speakers -> stable ids
  ├── successful diarization -> artifacts          └── rerun -> deterministic output
  ├── metrics success -> metrics.json
  └── metrics failure -> best-effort metrics
```

Required tests:

- Contract schema accepts valid fixture outputs.
- Contract schema rejects missing required fields.
- JSONL parser rejects malformed lines with file/line context.
- Metrics records success and failure shape.
- CLI returns nonzero on missing audio.
- CLI returns nonzero on missing HF token when the model requires it.
- Speaker ids are stable across repeated normalization of the same oracle annotation.

## Performance Plan

From v0, every run records:

- `audioSeconds`
- `wallTimeMs`
- `peakRSSBytes`
- `device`
- `provider`
- `ok`

Do not gate the oracle on speed. The oracle is allowed to be slow.

Do gate future MLX work against the oracle metrics. The MLX runtime exists to make local app use viable, not just to be interesting.

Initial benchmark target to measure:

```text
30 minute system-track.wav
  oracle: measured baseline
  MLX target: aim for <= 0.5x audio duration on Apple Silicon if feasible
  memory: record peak RSS before setting a hard cap
```

## Failure Modes

| Failure | Expected behavior |
|---|---|
| Missing audio file | Nonzero exit, clear message, metrics best effort |
| Empty/silence audio | Valid empty artifacts, success |
| HF token missing | Nonzero exit, clear setup hint |
| Gated model not accepted | Nonzero exit, link to model access requirement |
| pyannote output has no tracks | Valid empty artifacts, success if audio processed |
| Speaker ids reorder between runs | Comparator/test failure |
| Output JSON invalid | Contract test failure |
| Runtime too slow | Metrics show it, no app integration until solved |

## Open Questions

- Should fixture audio be generated synthetically or stored as small checked-in WAV files?
- What tolerance should comparator use for timestamp drift?
- Should the oracle support `--num-speakers`, `--min-speakers`, and `--max-speakers` in M2 or defer to M3?
- Should MPS be attempted for pyannote oracle on Apple Silicon, or keep oracle CPU-only for repeatability?

## Not In Scope

- MirrorNote app UI.
- MirrorNote submodule integration.
- `community-1` model port.
- GGML runtime.
- Real-time diarization.
- Cloud diarization.
