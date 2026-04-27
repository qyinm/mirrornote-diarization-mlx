import numpy as np
import pytest

pytest.importorskip("mlx.core")

from mirrornote_diarization.pyannet_contract import PYANNET_EXPECTED_WEIGHT_SHAPES
from mirrornote_diarization.segmentation_parity import main


def _write_complete_weights(path) -> None:
    weights = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in PYANNET_EXPECTED_WEIGHT_SHAPES.items()
    }
    np.savez(path, **weights)


def _run_mlx_candidate(weights_path, waveform_path, output_path) -> int:
    return main(
        [
            "segmentation",
            "mlx-candidate",
            "--weights",
            str(weights_path),
            "--waveform-npz",
            str(waveform_path),
            "--out",
            str(output_path),
        ]
    )


def test_mlx_candidate_cli_writes_shape_correct_candidate_npz(tmp_path) -> None:
    weights_path = tmp_path / "weights.npz"
    waveform_path = tmp_path / "waveform.npz"
    output_path = tmp_path / "nested" / "candidate.npz"
    _write_complete_weights(weights_path)
    np.savez(waveform_path, waveform=np.zeros((1, 1, 160000), dtype=np.float32))

    exit_code = _run_mlx_candidate(weights_path, waveform_path, output_path)

    assert exit_code == 0
    with np.load(output_path) as payload:
        output = payload["output"]

    assert output.shape == (1, 589, 7)
    assert output.dtype == np.float32


def test_mlx_candidate_cli_missing_waveform_key_returns_1_without_output(
    tmp_path,
) -> None:
    weights_path = tmp_path / "weights.npz"
    waveform_path = tmp_path / "waveform.npz"
    output_path = tmp_path / "candidate.npz"
    _write_complete_weights(weights_path)
    np.savez(waveform_path, audio=np.zeros((1, 1, 160000), dtype=np.float32))

    exit_code = _run_mlx_candidate(weights_path, waveform_path, output_path)

    assert exit_code == 1
    assert not output_path.exists()


def test_mlx_candidate_cli_corrupt_weights_returns_1_without_traceback(
    tmp_path,
    capsys,
) -> None:
    weights_path = tmp_path / "weights.npz"
    waveform_path = tmp_path / "waveform.npz"
    output_path = tmp_path / "candidate.npz"
    weights_path.write_bytes(b"not a valid npz")
    np.savez(waveform_path, waveform=np.zeros((1, 1, 160000), dtype=np.float32))

    exit_code = _run_mlx_candidate(weights_path, waveform_path, output_path)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert not output_path.exists()
    assert "MLX segmentation candidate failed" in captured.err
    assert "Traceback" not in captured.err


def test_mlx_candidate_cli_corrupt_waveform_returns_1_without_traceback(
    tmp_path,
    capsys,
) -> None:
    weights_path = tmp_path / "weights.npz"
    waveform_path = tmp_path / "waveform.npz"
    output_path = tmp_path / "candidate.npz"
    _write_complete_weights(weights_path)
    waveform_path.write_bytes(b"not a valid npz")

    exit_code = _run_mlx_candidate(weights_path, waveform_path, output_path)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert not output_path.exists()
    assert "MLX segmentation candidate failed" in captured.err
    assert "Traceback" not in captured.err
