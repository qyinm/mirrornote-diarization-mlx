import json

import numpy as np
import pytest

from mirrornote_diarization.probe_artifacts import load_probe_artifacts


def test_load_probe_artifacts_reads_metadata_and_reference_output(tmp_path):
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    metadata = {
        "modelClass": "FakeSegmentationModel",
        "sampleRate": 16000,
        "chunkDurationSeconds": 10.0,
        "frameResolutionSeconds": 0.016875,
        "moduleTree": ["<root>", "encoder", "classifier"],
        "weightShapes": {"encoder.weight": [2, 3], "classifier.bias": [7]},
        "outputShape": [1, 589, 7],
    }
    (probe_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.savez(probe_dir / "reference-output.npz", output=np.zeros((1, 589, 7), dtype=np.float64))

    artifacts = load_probe_artifacts(probe_dir)

    assert artifacts.metadata["modelClass"] == "FakeSegmentationModel"
    assert artifacts.reference_output.shape == (1, 589, 7)
    assert artifacts.reference_output.dtype == np.float32
    assert artifacts.parameter_count == 13
    assert artifacts.module_count == 3


def test_load_probe_artifacts_prefers_metadata_parameter_count(tmp_path):
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    metadata = {
        "modelClass": "FakeSegmentationModel",
        "sampleRate": 16000,
        "chunkDurationSeconds": 10.0,
        "frameResolutionSeconds": 0.016875,
        "moduleTree": ["<root>", "encoder"],
        "weightShapes": {"encoder.weight": [2, 3]},
        "weightDtypes": {"encoder.weight": "float32"},
        "parameterCount": 123,
        "outputShape": [1, 589, 7],
    }
    (probe_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.savez(probe_dir / "reference-output.npz", output=np.zeros((1, 589, 7), dtype=np.float32))

    artifacts = load_probe_artifacts(probe_dir)

    assert artifacts.parameter_count == 123


def test_load_probe_artifacts_rejects_missing_metadata(tmp_path):
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    np.savez(probe_dir / "reference-output.npz", output=np.zeros((1, 1, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="missing metadata.json"):
        load_probe_artifacts(probe_dir)


def test_load_probe_artifacts_rejects_missing_output_key(tmp_path):
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "metadata.json").write_text("{}", encoding="utf-8")
    np.savez(probe_dir / "reference-output.npz", logits=np.zeros((1, 1, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="missing output array"):
        load_probe_artifacts(probe_dir)


def test_load_probe_artifacts_rejects_non_object_metadata(tmp_path):
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "metadata.json").write_text("null", encoding="utf-8")
    np.savez(probe_dir / "reference-output.npz", output=np.zeros((1, 1, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="metadata.json must contain an object"):
        load_probe_artifacts(probe_dir)


def test_load_probe_artifacts_rejects_non_object_weight_shapes(tmp_path):
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "metadata.json").write_text(
        json.dumps({"weightShapes": ["bad"]}),
        encoding="utf-8",
    )
    np.savez(probe_dir / "reference-output.npz", output=np.zeros((1, 1, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="weightShapes must be an object"):
        load_probe_artifacts(probe_dir)


def test_load_probe_artifacts_rejects_null_weight_shapes(tmp_path):
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "metadata.json").write_text(
        json.dumps({"weightShapes": None}),
        encoding="utf-8",
    )
    np.savez(probe_dir / "reference-output.npz", output=np.zeros((1, 1, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="weightShapes must be an object"):
        load_probe_artifacts(probe_dir)


def test_load_probe_artifacts_rejects_invalid_weight_shape_dimensions(tmp_path):
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "metadata.json").write_text(
        json.dumps({"weightShapes": {"w": ["bad"]}}),
        encoding="utf-8",
    )
    np.savez(probe_dir / "reference-output.npz", output=np.zeros((1, 1, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="weightShapes dimensions must be positive integers"):
        load_probe_artifacts(probe_dir)
