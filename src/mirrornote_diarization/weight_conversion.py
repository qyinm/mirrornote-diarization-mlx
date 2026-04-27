"""Strict weight-name mapping contract for segmentation parity checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MappingRule:
    """Required mapping from a reference checkpoint key to a candidate key."""

    reference_key: str
    candidate_key: str
    expected_shape: tuple[int, ...]


@dataclass(frozen=True)
class MappingResult:
    """Result of validating required reference weights against mapping rules."""

    mapped: dict[str, str]
    missing_reference: list[str]
    shape_mismatches: list[dict[str, Any]]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the public JSON-compatible camelCase contract."""
        return {
            "mapped": dict(self.mapped),
            "missingReference": list(self.missing_reference),
            "shapeMismatches": [dict(mismatch) for mismatch in self.shape_mismatches],
            "passed": self.passed,
        }


def validate_weight_mapping(
    reference_weights: Mapping[str, np.ndarray],
    rules: Sequence[MappingRule],
) -> MappingResult:
    """Validate that all required reference weights can map to candidate keys."""
    _reject_duplicate_candidate_keys(rules)

    mapped: dict[str, str] = {}
    missing_reference: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []

    for rule in rules:
        weight = reference_weights.get(rule.reference_key)
        if weight is None:
            missing_reference.append(rule.reference_key)
            continue

        actual_shape = tuple(weight.shape)
        if actual_shape != rule.expected_shape:
            shape_mismatches.append(
                {
                    "referenceKey": rule.reference_key,
                    "candidateKey": rule.candidate_key,
                    "expectedShape": list(rule.expected_shape),
                    "actualShape": list(actual_shape),
                }
            )
            continue

        mapped[rule.reference_key] = rule.candidate_key

    return MappingResult(
        mapped=mapped,
        missing_reference=missing_reference,
        shape_mismatches=shape_mismatches,
        passed=not missing_reference and not shape_mismatches,
    )


def _reject_duplicate_candidate_keys(rules: Sequence[MappingRule]) -> None:
    seen: set[str] = set()
    for rule in rules:
        if rule.candidate_key in seen:
            raise ValueError(f"duplicate candidate key: {rule.candidate_key}")
        seen.add(rule.candidate_key)
