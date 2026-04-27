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
from mirrornote_diarization.parity_report import validate_report_dict
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
