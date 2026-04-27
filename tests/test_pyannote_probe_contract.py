import json

import numpy as np
import pytest

from mirrornote_diarization.pyannote_probe import (
    PyannoteProbeMetadata,
    require_pyannote_enabled,
    write_probe_artifacts,
)


def test_probe_metadata_to_dict_uses_public_camel_case_contract() -> None:
    metadata = PyannoteProbeMetadata(
        model_class="pyannote.audio.models.segmentation.PyanNet",
        sample_rate=16000,
        chunk_duration_seconds=10.0,
        frame_resolution_seconds=0.016875,
        module_tree=["model", "model.sincnet", "model.lstm"],
        weight_shapes={"model.lstm.weight_ih_l0": [512, 60]},
        weight_dtypes={"model.lstm.weight_ih_l0": "float32"},
        parameter_count=30720,
        output_shape=[1, 589, 3],
    )

    assert metadata.to_dict() == {
        "modelClass": "pyannote.audio.models.segmentation.PyanNet",
        "sampleRate": 16000,
        "chunkDurationSeconds": 10.0,
        "frameResolutionSeconds": 0.016875,
        "moduleTree": ["model", "model.sincnet", "model.lstm"],
        "weightShapes": {"model.lstm.weight_ih_l0": [512, 60]},
        "weightDtypes": {"model.lstm.weight_ih_l0": "float32"},
        "parameterCount": 30720,
        "outputShape": [1, 589, 3],
    }


def test_probe_metadata_includes_weight_dtype_and_parameter_count() -> None:
    metadata = PyannoteProbeMetadata(
        model_class="FakeSegmentationModel",
        sample_rate=16000,
        chunk_duration_seconds=10.0,
        frame_resolution_seconds=0.016875,
        module_tree=["model", "model.encoder"],
        weight_shapes={"encoder.weight": [2, 3]},
        weight_dtypes={"encoder.weight": "float32"},
        parameter_count=6,
        output_shape=[1, 589, 7],
    )

    payload = metadata.to_dict()

    assert payload["weightDtypes"] == {"encoder.weight": "float32"}
    assert payload["parameterCount"] == 6


def test_require_pyannote_enabled_requires_runtime_flag() -> None:
    with pytest.raises(RuntimeError, match="MIRRORNOTE_RUN_PYANNOTE_PROBE=1"):
        require_pyannote_enabled({})


def test_require_pyannote_enabled_requires_huggingface_token_when_flag_set() -> None:
    with pytest.raises(RuntimeError, match="HUGGINGFACE_ACCESS_TOKEN"):
        require_pyannote_enabled({"MIRRORNOTE_RUN_PYANNOTE_PROBE": "1"})


@pytest.mark.parametrize("token", ["hf_token", "  hf_token  "])
def test_require_pyannote_enabled_accepts_non_empty_token(token: str) -> None:
    require_pyannote_enabled(
        {
            "MIRRORNOTE_RUN_PYANNOTE_PROBE": "1",
            "HUGGINGFACE_ACCESS_TOKEN": token,
        }
    )


def test_write_probe_artifacts_writes_metadata_and_float32_reference_output(tmp_path) -> None:
    metadata = PyannoteProbeMetadata(
        model_class="PyanNet",
        sample_rate=16000,
        chunk_duration_seconds=10.0,
        frame_resolution_seconds=0.016875,
        module_tree=["model"],
        weight_shapes={"model.weight": [2, 2]},
        weight_dtypes={"model.weight": "float32"},
        parameter_count=4,
        output_shape=[1, 2, 2],
    )
    reference_output = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64)

    write_probe_artifacts(metadata, reference_output, tmp_path / "nested" / "probe")

    metadata_path = tmp_path / "nested" / "probe" / "metadata.json"
    output_path = tmp_path / "nested" / "probe" / "reference-output.npz"
    assert json.loads(metadata_path.read_text()) == metadata.to_dict()

    with np.load(output_path) as payload:
        assert set(payload.files) == {"output"}
        assert payload["output"].dtype == np.float32
        np.testing.assert_array_equal(payload["output"], reference_output.astype(np.float32))
