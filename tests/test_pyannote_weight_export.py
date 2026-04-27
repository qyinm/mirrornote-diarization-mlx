import numpy as np

from mirrornote_diarization.pyannote_probe import (
    PyannoteProbeMetadata,
    write_probe_artifacts,
)


class FakeTorchTensor:
    def __init__(self, array):
        self._array = np.asarray(array)
        self.detached = False
        self.moved_to_cpu = False

    def detach(self):
        self.detached = True
        return self

    def cpu(self):
        self.moved_to_cpu = True
        return self

    def numpy(self):
        return self._array


def _metadata() -> PyannoteProbeMetadata:
    return PyannoteProbeMetadata(
        model_class="PyanNet",
        sample_rate=16000,
        chunk_duration_seconds=10.0,
        frame_resolution_seconds=0.016875,
        module_tree=["model"],
        weight_shapes={"model.sincnet.conv1.weight": [2, 2]},
        weight_dtypes={"model.sincnet.conv1.weight": "float32"},
        parameter_count=4,
        output_shape=[1, 2, 2],
    )


def test_write_probe_artifacts_writes_optional_reference_weights(tmp_path) -> None:
    numpy_weight = np.array([[1.25, 2.5]], dtype=np.float64)
    tensor_weight = FakeTorchTensor([[3, 4], [5, 6]])

    write_probe_artifacts(
        _metadata(),
        np.zeros((1, 2, 2), dtype=np.float32),
        tmp_path,
        reference_weights={
            "model.sincnet.conv1.weight": numpy_weight,
            "model.lstm.weight_ih_l0": tensor_weight,
        },
    )

    weights_path = tmp_path / "reference-weights.npz"
    assert weights_path.exists()
    assert tensor_weight.detached is True
    assert tensor_weight.moved_to_cpu is True

    with np.load(weights_path) as payload:
        assert set(payload.files) == {
            "model.sincnet.conv1.weight",
            "model.lstm.weight_ih_l0",
        }
        assert payload["model.sincnet.conv1.weight"].dtype == np.float32
        assert payload["model.lstm.weight_ih_l0"].dtype == np.float32
        np.testing.assert_array_equal(
            payload["model.sincnet.conv1.weight"], numpy_weight.astype(np.float32)
        )
        np.testing.assert_array_equal(
            payload["model.lstm.weight_ih_l0"],
            np.array([[3, 4], [5, 6]], dtype=np.float32),
        )


def test_write_probe_artifacts_without_weights_preserves_existing_contract(tmp_path) -> None:
    write_probe_artifacts(
        _metadata(),
        np.zeros((1, 2, 2), dtype=np.float64),
        tmp_path,
    )

    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "reference-output.npz").exists()
    assert not (tmp_path / "reference-weights.npz").exists()
