import json

import numpy as np

from mirrornote_diarization.pyannote_probe import PyannoteProbeMetadata, write_probe_artifacts
from mirrornote_diarization.segmentation_parity import main


def _write_fake_probe(probe_dir):
    metadata = PyannoteProbeMetadata(
        model_class="FakeSegmentationModel",
        sample_rate=16000,
        chunk_duration_seconds=10.0,
        frame_resolution_seconds=0.016875,
        module_tree=["<root>", "encoder", "classifier"],
        weight_shapes={"encoder.weight": [2, 3], "classifier.bias": [7]},
        output_shape=[1, 999, 99],
    )
    reference_output = np.zeros((1, 589, 7), dtype=np.float64)

    write_probe_artifacts(metadata, reference_output, probe_dir)

    assert (probe_dir / "metadata.json").exists()
    assert (probe_dir / "reference-output.npz").exists()


def test_inspect_probe_prints_human_summary(tmp_path, capsys) -> None:
    probe_dir = tmp_path / "probe"
    _write_fake_probe(probe_dir)

    exit_code = main(["segmentation", "inspect-probe", str(probe_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "modelClass: FakeSegmentationModel" in captured.out
    assert "outputShape: [1, 589, 7]" in captured.out
    assert "moduleCount: 3" in captured.out
    assert "parameterCount: 13" in captured.out


def test_inspect_probe_writes_json_summary(tmp_path) -> None:
    probe_dir = tmp_path / "probe"
    summary_path = tmp_path / "summary" / "probe-summary.json"
    _write_fake_probe(probe_dir)

    exit_code = main(
        [
            "segmentation",
            "inspect-probe",
            str(probe_dir),
            "--json-out",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["modelClass"] == "FakeSegmentationModel"
    assert summary["moduleCount"] == 3
    assert summary["parameterCount"] == 13


def test_inspect_probe_reports_missing_probe_dir(tmp_path, capsys) -> None:
    exit_code = main(["segmentation", "inspect-probe", str(tmp_path / "missing")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "probe inspection failed" in captured.err
