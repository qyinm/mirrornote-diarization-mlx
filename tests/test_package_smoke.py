from mirrornote_diarization import __version__
from mirrornote_diarization.segmentation_parity import main


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_placeholder_cli_returns_success() -> None:
    assert main([]) == 0
