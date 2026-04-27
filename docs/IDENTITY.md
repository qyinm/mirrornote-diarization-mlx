# Identity

## Name

MirrorNote Diarization MLX

## One Line

A local speaker diarization runtime for MirrorNote, proven first against a pyannote 3.1 oracle and later ported to MLX for Apple Silicon.

## What This Is

This project turns meeting audio into MirrorNote speaker artifacts:

```text
system-track.wav
  -> diarization
  -> speaker-segments.jsonl
  -> speaker-map.json
```

The first implementation is not the final runtime. The first implementation is the reference path. It exists to define the contract, fixtures, metrics, and acceptance tests before MLX work starts.

## What This Is Not

- Not a general-purpose diarization product.
- Not a replacement for MirrorNote's STT engine.
- Not a cloud diarization service.
- Not a submodule dependency yet.
- Not a full pyannote rewrite on day one.
- Not a `community-1` port yet.

## Product Job

MirrorNote currently has source-only speaker labeling:

```text
microphone -> You
system     -> Speaker
```

That is useful, but it collapses all remote speakers into one label. This project exists to split the system track into stable remote speaker identities while preserving MirrorNote's local-first artifact model.

## Architecture Bias

Boring first.

The first working version should be a CLI that can run outside MirrorNote, write deterministic files, and fail loudly. The app should not depend on this project until the CLI has proven quality and performance on real meeting fixtures.

## Model Choice

Start with `pyannote/speaker-diarization-3.1`.

Reasons:

- MIT model license.
- Pure PyTorch pipeline in pyannote 3.1, with the model card noting the removal of problematic `onnxruntime` usage from the older 3.0 path.
- Good baseline for decomposing segmentation, embedding, clustering, and output normalization.
- Safer first target than `community-1`, which is CC BY 4.0 and a better later quality target.

## MLX Role

MLX is the later runtime target, not the first milestone.

MLX matters because MirrorNote is macOS-first and Apple Silicon-first. Apple describes MLX as an array framework for efficient and flexible machine learning on Apple Silicon, with Python, Swift, C++, and C bindings. That makes it a plausible long-term local runtime for MirrorNote.

## Contract

The contract is MirrorNote-native output.

Default output:

```text
speaker-segments.jsonl
speaker-map.json
metrics.json
```

Optional debug output later:

```text
oracle.rttm
```

RTTM is an interoperability format. It is not the main product API.

## Quality Bar

A result is useful only if it is:

- Deterministic enough that speaker ids do not shuffle on every run.
- Compatible with MirrorNote without an adapter rewrite.
- Measured with wall time, audio duration, peak memory, device, and provider.
- Better than source-only labeling on real system-track audio.
- Fast enough that post-meeting finalization does not feel broken.

## Repo Boundary

This repo should stay outside MirrorNote until it passes the oracle milestone and at least one MLX parity milestone.

Submodule integration is explicitly deferred.
