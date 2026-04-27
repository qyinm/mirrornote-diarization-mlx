import json

import numpy as np

from mirrornote_diarization.segmentation_parity import main


def test_compare_npz_writes_passing_report_for_identical_outputs(tmp_path) -> None:
    reference_path = tmp_path / "reference.npz"
    candidate_path = tmp_path / "candidate.npz"
    report_path = tmp_path / "reports" / "segmentation-parity.json"
    output = np.array([[[0.1, 0.9], [0.4, 0.6]]], dtype=np.float32)
    np.savez(reference_path, output=output)
    np.savez(candidate_path, output=output)

    exit_code = main(
        [
            "segmentation",
            "compare-npz",
            "--reference",
            str(reference_path),
            "--candidate",
            str(candidate_path),
            "--source",
            "fixtures/single-speaker/system-track.wav",
            "--out",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["shape"]["matches"] is True
