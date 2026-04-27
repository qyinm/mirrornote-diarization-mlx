import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from mirrornote_diarization.mlx_pyannet import MlxPyanNetSegmentation
from mirrornote_diarization.mlx_segmentation import (
    MlxSegmentationConfig,
    build_mlx_segmentation,
)
from mirrornote_diarization.pyannet_contract import (
    PYANNET_ARCHITECTURE_NAME,
    PYANNET_EXPECTED_OUTPUT_SHAPE,
    PYANNET_EXPECTED_WEIGHT_SHAPES,
)


def _zero_reference_weights() -> dict[str, np.ndarray]:
    return {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in PYANNET_EXPECTED_WEIGHT_SHAPES.items()
    }


def test_from_reference_weights_rejects_missing_weight() -> None:
    reference_weights = _zero_reference_weights()
    reference_weights.pop("classifier.bias")

    with pytest.raises(ValueError, match="missingReference"):
        MlxPyanNetSegmentation.from_reference_weights(reference_weights)


def test_complete_zero_weight_set_builds_from_segmentation_config() -> None:
    config = MlxSegmentationConfig(
        sample_rate=16000,
        chunk_duration_seconds=10.0,
        output_classes=7,
        architecture_name=PYANNET_ARCHITECTURE_NAME,
    )

    model = build_mlx_segmentation(config, _zero_reference_weights())

    assert isinstance(model, MlxPyanNetSegmentation)
    assert model.output_classes == 7
    assert model.sample_rate == 16000
    assert model.chunk_duration_seconds == 10.0


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("sample_rate", 8000),
        ("chunk_duration_seconds", 5.0),
        ("output_classes", 3),
    ],
)
def test_build_mlx_segmentation_rejects_invalid_pyannet_config(
    field_name: str,
    field_value: int | float,
) -> None:
    config_kwargs = {
        "sample_rate": 16000,
        "chunk_duration_seconds": 10.0,
        "output_classes": 7,
        "architecture_name": PYANNET_ARCHITECTURE_NAME,
    }
    config_kwargs[field_name] = field_value
    config = MlxSegmentationConfig(**config_kwargs)

    with pytest.raises(ValueError, match=field_name):
        build_mlx_segmentation(config, _zero_reference_weights())


def test_forward_returns_expected_shape_for_mlx_waveform() -> None:
    model = MlxPyanNetSegmentation.from_reference_weights(_zero_reference_weights())
    waveform = mlx.zeros((1, 1, 160000), dtype=mlx.float32)

    output = model(waveform)

    assert tuple(output.shape) == PYANNET_EXPECTED_OUTPUT_SHAPE


def test_write_candidate_npz_writes_float32_output(tmp_path) -> None:
    model = MlxPyanNetSegmentation.from_reference_weights(_zero_reference_weights())
    waveform = np.zeros((1, 1, 160000), dtype=np.float32)
    output_path = tmp_path / "candidate.npz"

    model.write_candidate_npz(waveform, output_path)

    with np.load(output_path) as payload:
        output = payload["output"]

    assert output.shape == PYANNET_EXPECTED_OUTPUT_SHAPE
    assert output.dtype == np.float32
