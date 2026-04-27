"""Strict weight-name mapping contract for segmentation parity checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mirrornote_diarization.pyannet_contract import PYANNET_EXPECTED_WEIGHT_SHAPES


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
    _reject_duplicate_keys(rules)

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

        mapped[rule.candidate_key] = rule.reference_key

    return MappingResult(
        mapped=mapped,
        missing_reference=missing_reference,
        shape_mismatches=shape_mismatches,
        passed=not missing_reference and not shape_mismatches,
    )


def load_npz_weights(path: str | Path) -> dict[str, np.ndarray]:
    """Load named weight arrays from a `.npz` file as float32 arrays."""
    with np.load(path) as payload:
        return {
            name: np.asarray(payload[name], dtype=np.float32)
            for name in payload.files
        }


def build_pyannet_mapping_rules() -> list[MappingRule]:
    """Build strict reference-to-MLX mapping rules for pyannote 3.1 PyanNet."""
    return [
        MappingRule(
            reference_key=name,
            candidate_key=_pyannet_candidate_key(name),
            expected_shape=shape,
        )
        for name, shape in PYANNET_EXPECTED_WEIGHT_SHAPES.items()
    ]


def _pyannet_candidate_key(reference_key: str) -> str:
    direct = {
        "classifier.bias": "classifier.bias",
        "classifier.weight": "classifier.weight",
        "linear.0.bias": "linear.layers.0.bias",
        "linear.0.weight": "linear.layers.0.weight",
        "linear.1.bias": "linear.layers.1.bias",
        "linear.1.weight": "linear.layers.1.weight",
        "sincnet.conv1d.0.filterbank.band_hz_": "sincnet.sinc_filterbank.band_hz",
        "sincnet.conv1d.0.filterbank.low_hz_": "sincnet.sinc_filterbank.low_hz",
        "sincnet.conv1d.0.filterbank.n_": "sincnet.sinc_filterbank.n",
        "sincnet.conv1d.0.filterbank.window_": "sincnet.sinc_filterbank.window",
        "sincnet.conv1d.1.bias": "sincnet.conv.layers.1.bias",
        "sincnet.conv1d.1.weight": "sincnet.conv.layers.1.weight",
        "sincnet.conv1d.2.bias": "sincnet.conv.layers.2.bias",
        "sincnet.conv1d.2.weight": "sincnet.conv.layers.2.weight",
        "sincnet.norm1d.0.bias": "sincnet.norm.layers.0.bias",
        "sincnet.norm1d.0.weight": "sincnet.norm.layers.0.weight",
        "sincnet.norm1d.1.bias": "sincnet.norm.layers.1.bias",
        "sincnet.norm1d.1.weight": "sincnet.norm.layers.1.weight",
        "sincnet.norm1d.2.bias": "sincnet.norm.layers.2.bias",
        "sincnet.norm1d.2.weight": "sincnet.norm.layers.2.weight",
        "sincnet.wav_norm1d.bias": "sincnet.wav_norm.bias",
        "sincnet.wav_norm1d.weight": "sincnet.wav_norm.weight",
    }
    if reference_key in direct:
        return direct[reference_key]

    if reference_key.startswith("lstm."):
        return _pyannet_lstm_candidate_key(reference_key)

    raise ValueError(f"unsupported PyanNet reference key: {reference_key}")


def _pyannet_lstm_candidate_key(reference_key: str) -> str:
    if not reference_key.startswith("lstm."):
        raise ValueError(f"unsupported PyanNet LSTM key: {reference_key}")

    body = reference_key.removeprefix("lstm.")
    parts = body.split("_")
    if len(parts) not in (3, 4):
        raise ValueError(f"unsupported PyanNet LSTM key: {reference_key}")

    parameter_kind, matrix_kind, layer_token = parts[:3]
    if parameter_kind not in {"weight", "bias"} or matrix_kind not in {"ih", "hh"}:
        raise ValueError(f"unsupported PyanNet LSTM key: {reference_key}")

    direction = "forward"
    if len(parts) == 4:
        if parts[3] != "reverse":
            raise ValueError(f"unsupported PyanNet LSTM key: {reference_key}")
        direction = "reverse"

    if not layer_token.startswith("l") or not layer_token[1:].isdigit():
        raise ValueError(f"unsupported PyanNet LSTM key: {reference_key}")

    layer_index = int(layer_token[1:])
    leaf = f"{parameter_kind}_{matrix_kind}"

    return f"lstm.layers.{layer_index}.{direction}.{leaf}"


def _reject_duplicate_keys(rules: Sequence[MappingRule]) -> None:
    seen_reference_keys: set[str] = set()
    seen_candidate_keys: set[str] = set()

    for rule in rules:
        if rule.reference_key in seen_reference_keys:
            raise ValueError(f"duplicate reference key: {rule.reference_key}")
        seen_reference_keys.add(rule.reference_key)

        if rule.candidate_key in seen_candidate_keys:
            raise ValueError(f"duplicate candidate key: {rule.candidate_key}")
        seen_candidate_keys.add(rule.candidate_key)
