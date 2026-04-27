"""CLI commands for MirrorNote diarization segmentation parity."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import sys

import numpy as np

from mirrornote_diarization.chunking import extract_fixed_chunk
from mirrornote_diarization.parity_report import (
    DEFAULT_THRESHOLDS,
    ParityReport,
    compute_metrics,
    validate_report_dict,
)
from mirrornote_diarization.pyannote_probe import (
    require_pyannote_enabled,
    run_pyannote_probe,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the mirrornote-diarize command parser."""
    parser = argparse.ArgumentParser(prog="mirrornote-diarize")
    subparsers = parser.add_subparsers(dest="command")

    segmentation_parser = subparsers.add_parser(
        "segmentation", help="Segmentation parity utilities"
    )
    segmentation_subparsers = segmentation_parser.add_subparsers(
        dest="segmentation_command"
    )

    validate_parser = segmentation_subparsers.add_parser(
        "validate-report", help="Validate a segmentation parity report JSON file"
    )
    validate_parser.add_argument("report", type=Path)

    probe_parser = segmentation_subparsers.add_parser(
        "probe", help="Run the gated pyannote segmentation reference probe"
    )
    probe_parser.add_argument("--audio", type=Path, required=True)
    probe_parser.add_argument("--out", type=Path, required=True)
    probe_parser.add_argument("--start-seconds", type=float, default=0.0)
    probe_parser.add_argument("--duration-seconds", type=float, default=10.0)

    compare_npz_parser = segmentation_subparsers.add_parser(
        "compare-npz", help="Compare saved segmentation reference and candidate arrays"
    )
    compare_npz_parser.add_argument("--reference", type=Path, required=True)
    compare_npz_parser.add_argument("--candidate", type=Path, required=True)
    compare_npz_parser.add_argument("--source", required=True)
    compare_npz_parser.add_argument("--out", type=Path, required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the segmentation parity CLI."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)

        if args.command is None:
            parser.print_help()
            return 0

        if args.command == "segmentation":
            if args.segmentation_command == "validate-report":
                return _validate_report(args.report)
            if args.segmentation_command == "probe":
                return _probe(args)
            if args.segmentation_command == "compare-npz":
                return _compare_npz(args)
            parser.error("unsupported segmentation command")

        parser.error("unsupported command")
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    return 2


def _validate_report(report_path: Path) -> int:
    try:
        with report_path.open(encoding="utf-8") as report_file:
            payload = json.load(report_file)
    except OSError as exc:
        print(
            f"could not read parity report: {report_path}: {exc}",
            file=sys.stderr,
        )
        return 1
    except json.JSONDecodeError as exc:
        print(
            f"invalid JSON parity report: {report_path}: {exc.msg}",
            file=sys.stderr,
        )
        return 1

    try:
        validate_report_dict(payload)
    except ValueError as exc:
        print(f"invalid parity report: {report_path}: {exc}", file=sys.stderr)
        return 1

    print(f"valid parity report: {report_path}")
    return 0


def _probe(args: argparse.Namespace) -> int:
    try:
        require_pyannote_enabled(os.environ)
        waveform, sample_rate = _load_wav_mono(args.audio)
        chunk = extract_fixed_chunk(
            waveform,
            sample_rate=sample_rate,
            start_seconds=args.start_seconds,
            duration_seconds=args.duration_seconds,
        )
        run_pyannote_probe(chunk.as_model_input(), args.out)
    except Exception as exc:
        print(f"pyannote probe failed: {exc}", file=sys.stderr)
        return 1

    print(f"wrote pyannote probe artifacts: {args.out}")
    return 0


def _compare_npz(args: argparse.Namespace) -> int:
    try:
        reference = _load_npz_output(args.reference, "reference")
        candidate = _load_npz_output(args.candidate, "candidate")
        report = _build_npz_parity_report(reference, candidate, args.source)
        payload = report.to_dict()
        validate_report_dict(payload)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, KeyError, ValueError) as exc:
        print(f"segmentation npz comparison failed: {exc}", file=sys.stderr)
        return 1

    print(f"wrote segmentation parity report: {args.out}")
    return 0 if report.passed else 1


def _build_npz_parity_report(
    reference: np.ndarray,
    candidate: np.ndarray,
    source: str,
) -> ParityReport:
    shape_matches = reference.shape == candidate.shape
    dtype_matches = reference.dtype == candidate.dtype
    if shape_matches:
        metrics = compute_metrics(reference, candidate, DEFAULT_THRESHOLDS)
        mean_abs_error = metrics.mean_abs_error
        max_abs_error = metrics.max_abs_error
        cosine_similarity = _clamp_cosine_similarity(metrics.cosine_similarity)
    else:
        mean_abs_error = DEFAULT_THRESHOLDS["meanAbsError"] + 1.0
        max_abs_error = DEFAULT_THRESHOLDS["maxAbsError"] + 1.0
        cosine_similarity = 0.0

    passed = (
        shape_matches
        and dtype_matches
        and mean_abs_error <= DEFAULT_THRESHOLDS["meanAbsError"]
        and max_abs_error <= DEFAULT_THRESHOLDS["maxAbsError"]
        and cosine_similarity >= DEFAULT_THRESHOLDS["cosineSimilarity"]
    )

    return ParityReport(
        reference_provider="pyannote-3.1-segmentation-pytorch",
        candidate_provider="pyannote-3.1-segmentation-mlx",
        audio_chunk={
            "source": source,
            "startTimeSeconds": 0.0,
            "durationSeconds": 10.0,
            "sampleRate": 16000,
        },
        shape={
            "reference": [int(dimension) for dimension in reference.shape],
            "candidate": [int(dimension) for dimension in candidate.shape],
            "matches": shape_matches,
        },
        dtype={
            "reference": str(reference.dtype),
            "candidate": str(candidate.dtype),
        },
        mean_abs_error=mean_abs_error,
        max_abs_error=max_abs_error,
        cosine_similarity=cosine_similarity,
        thresholds=DEFAULT_THRESHOLDS,
        passed=passed,
    )


def _load_npz_output(npz_path: Path, label: str) -> np.ndarray:
    try:
        with np.load(npz_path) as payload:
            if "output" not in payload:
                raise KeyError(f"{label} npz missing required 'output' array: {npz_path}")
            return np.asarray(payload["output"])
    except OSError as exc:
        raise OSError(f"could not read {label} npz: {npz_path}: {exc}") from exc


def _clamp_cosine_similarity(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _load_wav_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("install audio dependencies with: uv sync --extra audio") from exc

    waveform, sample_rate = sf.read(audio_path, dtype="float32")
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1:
        raise RuntimeError("expected mono or stereo WAV audio")
    return waveform.astype(np.float32, copy=False), int(sample_rate)


if __name__ == "__main__":
    raise SystemExit(main())
