"""Parity metric computation and JSON report contract validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_THRESHOLDS = {
    "meanAbsError": 1e-4,
    "maxAbsError": 1e-3,
    "cosineSimilarity": 0.999,
}

REQUIRED_REPORT_FIELDS = (
    "referenceProvider",
    "candidateProvider",
    "audioChunk",
    "shape",
    "dtype",
    "meanAbsError",
    "maxAbsError",
    "cosineSimilarity",
    "thresholds",
    "passed",
)

_REQUIRED_SHAPE_FIELDS = ("reference", "candidate", "matches")


@dataclass(frozen=True)
class ParityMetrics:
    """Numeric parity metrics for one reference/candidate comparison."""

    mean_abs_error: float
    max_abs_error: float
    cosine_similarity: float
    passed: bool


@dataclass(frozen=True)
class ParityReport:
    """JSON-serializable parity report with Python-friendly attribute names."""

    reference_provider: str
    candidate_provider: str
    audio_chunk: Mapping[str, Any]
    shape: Mapping[str, Any]
    dtype: Mapping[str, Any]
    mean_abs_error: float
    max_abs_error: float
    cosine_similarity: float
    thresholds: Mapping[str, float]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the report using the public camelCase JSON contract."""
        return {
            "referenceProvider": self.reference_provider,
            "candidateProvider": self.candidate_provider,
            "audioChunk": dict(self.audio_chunk),
            "shape": dict(self.shape),
            "dtype": dict(self.dtype),
            "meanAbsError": self.mean_abs_error,
            "maxAbsError": self.max_abs_error,
            "cosineSimilarity": self.cosine_similarity,
            "thresholds": dict(self.thresholds),
            "passed": self.passed,
        }


def compute_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> ParityMetrics:
    """Compute parity metrics for two arrays with identical shapes."""
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: reference {reference.shape} != candidate {candidate.shape}"
        )

    active_thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    reference_float = reference.astype(np.float32, copy=False)
    candidate_float = candidate.astype(np.float32, copy=False)

    delta = np.abs(reference_float - candidate_float)
    mean_abs_error = float(delta.mean()) if delta.size else 0.0
    max_abs_error = float(delta.max()) if delta.size else 0.0

    reference_flat = reference_float.ravel()
    candidate_flat = candidate_float.ravel()
    denominator = float(np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat))
    cosine_similarity = 1.0 if denominator == 0.0 else float(np.dot(reference_flat, candidate_flat) / denominator)

    passed = (
        mean_abs_error <= active_thresholds["meanAbsError"]
        and max_abs_error <= active_thresholds["maxAbsError"]
        and cosine_similarity >= active_thresholds["cosineSimilarity"]
    )

    return ParityMetrics(
        mean_abs_error=mean_abs_error,
        max_abs_error=max_abs_error,
        cosine_similarity=cosine_similarity,
        passed=passed,
    )


def validate_report_dict(payload: Mapping[str, Any]) -> None:
    """Validate the required public parity report fields and nested contracts."""
    for field in REQUIRED_REPORT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")

    shape = payload["shape"]
    if not isinstance(shape, Mapping):
        raise ValueError("field shape must be an object")
    for field in _REQUIRED_SHAPE_FIELDS:
        if field not in shape:
            raise ValueError(f"missing required shape field: {field}")

    thresholds = payload["thresholds"]
    if not isinstance(thresholds, Mapping):
        raise ValueError("field thresholds must be an object")
    for field in DEFAULT_THRESHOLDS:
        if field not in thresholds:
            raise ValueError(f"missing required thresholds field: {field}")
