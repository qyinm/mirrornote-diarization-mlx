import math

import numpy as np
import pytest

from mirrornote_diarization.parity_report import (
    DEFAULT_THRESHOLDS,
    ParityReport,
    compute_metrics,
    validate_report_dict,
)


def test_compute_metrics_for_identical_arrays_passes() -> None:
    reference = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    candidate = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    metrics = compute_metrics(reference, candidate, DEFAULT_THRESHOLDS)

    assert metrics.mean_abs_error == 0.0
    assert metrics.max_abs_error == 0.0
    assert math.isclose(metrics.cosine_similarity, 1.0, rel_tol=0.0, abs_tol=1e-7)
    assert metrics.passed is True


def test_compute_metrics_rejects_shape_mismatch() -> None:
    reference = np.zeros((1, 2, 3), dtype=np.float32)
    candidate = np.zeros((1, 2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="shape mismatch"):
        compute_metrics(reference, candidate, DEFAULT_THRESHOLDS)


def test_report_dict_contains_required_fields() -> None:
    report = ParityReport(
        reference_provider="pyannote-3.1-segmentation-pytorch",
        candidate_provider="pyannote-3.1-segmentation-mlx",
        audio_chunk={
            "source": "fixtures/single-speaker/system-track.wav",
            "startTimeSeconds": 0.0,
            "durationSeconds": 10.0,
            "sampleRate": 16000,
        },
        shape={"reference": [1, 3], "candidate": [1, 3], "matches": True},
        dtype={"reference": "float32", "candidate": "float32"},
        mean_abs_error=0.0,
        max_abs_error=0.0,
        cosine_similarity=1.0,
        thresholds=DEFAULT_THRESHOLDS,
        passed=True,
    )

    payload = report.to_dict()
    validate_report_dict(payload)

    assert payload["referenceProvider"] == "pyannote-3.1-segmentation-pytorch"
    assert payload["candidateProvider"] == "pyannote-3.1-segmentation-mlx"
    assert payload["passed"] is True


def test_validate_report_dict_requires_thresholds() -> None:
    payload = {
        "referenceProvider": "pyannote-3.1-segmentation-pytorch",
        "candidateProvider": "pyannote-3.1-segmentation-mlx",
        "audioChunk": {
            "source": "fixtures/single-speaker/system-track.wav",
            "startTimeSeconds": 0.0,
            "durationSeconds": 10.0,
            "sampleRate": 16000,
        },
        "shape": {"reference": [1, 3], "candidate": [1, 3], "matches": True},
        "dtype": {"reference": "float32", "candidate": "float32"},
        "meanAbsError": 0.0,
        "maxAbsError": 0.0,
        "cosineSimilarity": 1.0,
        "passed": True,
    }

    with pytest.raises(ValueError, match="missing required field: thresholds"):
        validate_report_dict(payload)
