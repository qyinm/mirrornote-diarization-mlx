import pytest

from mirrornote_diarization.mlx_segmentation import (
    MlxSegmentationConfig,
    UnsupportedArchitectureError,
)


def test_mlx_segmentation_config_to_dict_uses_public_camel_case_contract() -> None:
    config = MlxSegmentationConfig(
        sample_rate=16000,
        chunk_duration_seconds=10.0,
        output_classes=3,
        architecture_name="pyannote.audio.models.segmentation.PyanNet",
    )

    assert config.to_dict() == {
        "sampleRate": 16000,
        "chunkDurationSeconds": 10.0,
        "outputClasses": 3,
        "architectureName": "pyannote.audio.models.segmentation.PyanNet",
    }


def test_unsupported_architecture_error_includes_architecture_name() -> None:
    architecture_name = "pyannote.audio.models.segmentation.PyanNet"

    with pytest.raises(UnsupportedArchitectureError, match=architecture_name):
        raise UnsupportedArchitectureError(architecture_name)
