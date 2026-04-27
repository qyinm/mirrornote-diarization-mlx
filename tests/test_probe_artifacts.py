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
