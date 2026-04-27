"""Placeholder CLI for MirrorNote diarization segmentation parity."""

from collections.abc import Sequence


_PLACEHOLDER_MESSAGE = (
    "mirrornote-diarize: segmentation parity CLI placeholder; "
    "real implementation will be added in Task 6."
)


def main(argv: Sequence[str] | None = None) -> int:
    """Print a placeholder message until the real CLI is implemented."""
    _ = argv
    print(_PLACEHOLDER_MESSAGE)
    return 0
