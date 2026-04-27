"""Parity metric computation and JSON report contract validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
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
_REQUIRED_AUDIO_CHUNK_FIELDS = (
    "source",
    "startTimeSeconds",
    "durationSeconds",
    "sampleRate",
)
_REQUIRED_DTYPE_FIELDS = ("reference", "candidate")


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
    reference_norm = float(np.linalg.norm(reference_flat))
    candidate_norm = float(np.linalg.norm(candidate_flat))
    denominator = reference_norm * candidate_norm
    if denominator == 0.0:
        cosine_similarity = 1.0 if reference_norm == candidate_norm else 0.0
    else:
        cosine_similarity = float(np.dot(reference_flat, candidate_flat) / denominator)

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

    _validate_non_empty_string(payload["referenceProvider"], "referenceProvider")
    _validate_non_empty_string(payload["candidateProvider"], "candidateProvider")
    _validate_audio_chunk(payload["audioChunk"])
    _validate_shape(payload["shape"])
    _validate_dtype(payload["dtype"])
    _validate_non_negative_number(payload["meanAbsError"], "meanAbsError")
    _validate_non_negative_number(payload["maxAbsError"], "maxAbsError")
    _validate_cosine_similarity(payload["cosineSimilarity"])
    _validate_thresholds(payload["thresholds"])
    if not isinstance(payload["passed"], bool):
        raise ValueError("field passed must be a bool")


def _validate_audio_chunk(audio_chunk: Any) -> None:
    if not isinstance(audio_chunk, Mapping):
        raise ValueError("field audioChunk must be an object")
    for field in _REQUIRED_AUDIO_CHUNK_FIELDS:
        if field not in audio_chunk:
            raise ValueError(f"missing required audioChunk field: {field}")

    _validate_non_empty_string(audio_chunk["source"], "audioChunk.source")
    _validate_finite_number(
        audio_chunk["startTimeSeconds"], "audioChunk.startTimeSeconds"
    )
    _validate_positive_number(
        audio_chunk["durationSeconds"], "audioChunk.durationSeconds"
    )
    if not isinstance(audio_chunk["sampleRate"], int) or isinstance(
        audio_chunk["sampleRate"], bool
    ):
        raise ValueError("field audioChunk.sampleRate must be a positive int")
    if audio_chunk["sampleRate"] <= 0:
        raise ValueError("field audioChunk.sampleRate must be a positive int")


def _validate_shape(shape: Any) -> None:
    if not isinstance(shape, Mapping):
        raise ValueError("field shape must be an object")
    for field in _REQUIRED_SHAPE_FIELDS:
        if field not in shape:
            raise ValueError(f"missing required shape field: {field}")

    reference_shape = shape["reference"]
    candidate_shape = shape["candidate"]
    if not _is_int_list(reference_shape):
        raise ValueError("field shape.reference must be a list of ints")
    if not _is_int_list(candidate_shape):
        raise ValueError("field shape.candidate must be a list of ints")
    if not isinstance(shape["matches"], bool):
        raise ValueError("field shape.matches must be a bool")
    if shape["matches"] != (reference_shape == candidate_shape):
        raise ValueError("shape matches must equal reference == candidate")


def _validate_dtype(dtype: Any) -> None:
    if not isinstance(dtype, Mapping):
        raise ValueError("field dtype must be an object")
    for field in _REQUIRED_DTYPE_FIELDS:
        if field not in dtype:
            raise ValueError(f"missing required dtype field: {field}")
        _validate_non_empty_string(dtype[field], f"dtype.{field}")


def _validate_thresholds(thresholds: Any) -> None:
    if not isinstance(thresholds, Mapping):
        raise ValueError("field thresholds must be an object")
    for field in DEFAULT_THRESHOLDS:
        if field not in thresholds:
            raise ValueError(f"missing required thresholds field: {field}")
        _validate_finite_number(thresholds[field], f"thresholds.{field}")


def _validate_non_empty_string(value: Any, field: str) -> None:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"field {field} must be a non-empty string")


def _validate_finite_number(value: Any, field: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
    ):
        raise ValueError(f"field {field} must be a finite number")


def _validate_positive_number(value: Any, field: str) -> None:
    _validate_finite_number(value, field)
    if value <= 0:
        raise ValueError(f"field {field} must be positive")


def _validate_non_negative_number(value: Any, field: str) -> None:
    _validate_finite_number(value, field)
    if value < 0:
        raise ValueError(f"field {field} must be non-negative")


def _validate_cosine_similarity(value: Any) -> None:
    _validate_finite_number(value, "cosineSimilarity")
    if value < -1.0 or value > 1.0:
        raise ValueError("field cosineSimilarity must be between -1.0 and 1.0")


def _is_int_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    )
