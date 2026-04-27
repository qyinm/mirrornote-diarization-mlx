"""CLI commands for MirrorNote diarization segmentation parity."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from mirrornote_diarization.parity_report import validate_report_dict


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
                with args.report.open(encoding="utf-8") as report_file:
                    payload = json.load(report_file)
                validate_report_dict(payload)
                print(f"valid parity report: {args.report}")
                return 0
            parser.error("unsupported segmentation command")

        parser.error("unsupported command")
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
