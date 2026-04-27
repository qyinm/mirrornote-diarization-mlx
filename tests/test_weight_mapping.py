import numpy as np
import pytest

from mirrornote_diarization.weight_conversion import (
    MappingRule,
    validate_weight_mapping,
)


def test_validate_weight_mapping_maps_all_required_weights() -> None:
    reference_weights = {
        "encoder.layer1.weight": np.zeros((2, 3), dtype=np.float32),
        "encoder.layer1.bias": np.zeros((2,), dtype=np.float32),
    }
    rules = [
        MappingRule(
            reference_key="encoder.layer1.weight",
            candidate_key="segmentation.encoder.layer1.weight",
            expected_shape=(2, 3),
        ),
        MappingRule(
            reference_key="encoder.layer1.bias",
            candidate_key="segmentation.encoder.layer1.bias",
            expected_shape=(2,),
        ),
    ]

    result = validate_weight_mapping(reference_weights, rules)

    assert result.mapped == {
        "encoder.layer1.weight": "segmentation.encoder.layer1.weight",
        "encoder.layer1.bias": "segmentation.encoder.layer1.bias",
    }
    assert result.missing_reference == []
    assert result.shape_mismatches == []
    assert result.passed is True
    assert result.to_dict() == {
        "mapped": {
            "encoder.layer1.weight": "segmentation.encoder.layer1.weight",
            "encoder.layer1.bias": "segmentation.encoder.layer1.bias",
        },
        "missingReference": [],
        "shapeMismatches": [],
        "passed": True,
    }


def test_validate_weight_mapping_missing_reference_weight_fails() -> None:
    reference_weights = {
        "encoder.layer1.weight": np.zeros((2, 3), dtype=np.float32),
    }
    rules = [
        MappingRule(
            reference_key="encoder.layer1.weight",
            candidate_key="segmentation.encoder.layer1.weight",
            expected_shape=(2, 3),
        ),
        MappingRule(
            reference_key="encoder.layer1.bias",
            candidate_key="segmentation.encoder.layer1.bias",
            expected_shape=(2,),
        ),
    ]

    result = validate_weight_mapping(reference_weights, rules)

    assert result.mapped == {
        "encoder.layer1.weight": "segmentation.encoder.layer1.weight",
    }
    assert result.missing_reference == ["encoder.layer1.bias"]
    assert result.shape_mismatches == []
    assert result.passed is False
    assert result.to_dict()["missingReference"] == ["encoder.layer1.bias"]


def test_validate_weight_mapping_shape_mismatch_fails_with_exact_contract() -> None:
    reference_weights = {
        "encoder.layer1.weight": np.zeros((2, 4), dtype=np.float32),
    }
    rules = [
        MappingRule(
            reference_key="encoder.layer1.weight",
            candidate_key="segmentation.encoder.layer1.weight",
            expected_shape=(2, 3),
        ),
    ]

    result = validate_weight_mapping(reference_weights, rules)

    assert result.mapped == {}
    assert result.missing_reference == []
    assert result.shape_mismatches == [
        {
            "referenceKey": "encoder.layer1.weight",
            "candidateKey": "segmentation.encoder.layer1.weight",
            "expectedShape": [2, 3],
            "actualShape": [2, 4],
        }
    ]
    assert result.passed is False
    assert result.to_dict()["shapeMismatches"] == [
        {
            "referenceKey": "encoder.layer1.weight",
            "candidateKey": "segmentation.encoder.layer1.weight",
            "expectedShape": [2, 3],
            "actualShape": [2, 4],
        }
    ]


def test_validate_weight_mapping_duplicate_candidate_keys_raise() -> None:
    reference_weights = {
        "encoder.layer1.weight": np.zeros((2, 3), dtype=np.float32),
        "encoder.layer2.weight": np.zeros((2, 3), dtype=np.float32),
    }
    rules = [
        MappingRule(
            reference_key="encoder.layer1.weight",
            candidate_key="dup",
            expected_shape=(2, 3),
        ),
        MappingRule(
            reference_key="encoder.layer2.weight",
            candidate_key="dup",
            expected_shape=(2, 3),
        ),
    ]

    with pytest.raises(ValueError, match="duplicate candidate key: dup"):
        validate_weight_mapping(reference_weights, rules)
