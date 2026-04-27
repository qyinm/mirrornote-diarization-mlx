import numpy as np

from mirrornote_diarization.pyannet_contract import PYANNET_EXPECTED_WEIGHT_SHAPES
from mirrornote_diarization.weight_conversion import (
    build_pyannet_mapping_rules,
    load_npz_weights,
    validate_weight_mapping,
)


def test_pyannet_mapping_rules_cover_every_expected_weight() -> None:
    rules = build_pyannet_mapping_rules()

    assert len(rules) == len(PYANNET_EXPECTED_WEIGHT_SHAPES)
    assert {rule.reference_key for rule in rules} == set(PYANNET_EXPECTED_WEIGHT_SHAPES)


def test_pyannet_mapping_rules_use_unique_candidate_keys() -> None:
    rules = build_pyannet_mapping_rules()

    assert len({rule.candidate_key for rule in rules}) == len(rules)


def test_pyannet_mapping_rules_use_stable_candidate_namespace() -> None:
    rules = build_pyannet_mapping_rules()
    mapped = {rule.reference_key: rule.candidate_key for rule in rules}

    assert mapped["sincnet.wav_norm1d.weight"] == "sincnet.wav_norm.weight"
    assert (
        mapped["sincnet.conv1d.0.filterbank.low_hz_"]
        == "sincnet.sinc_filterbank.low_hz"
    )
    assert mapped["lstm.weight_ih_l0"] == "lstm.layers.0.forward.weight_ih"
    assert mapped["lstm.weight_ih_l0_reverse"] == "lstm.layers.0.reverse.weight_ih"
    assert mapped["linear.0.weight"] == "linear.layers.0.weight"
    assert mapped["classifier.bias"] == "classifier.bias"


def test_validate_realistic_pyannet_weight_shapes_passes() -> None:
    reference_weights = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in PYANNET_EXPECTED_WEIGHT_SHAPES.items()
    }

    result = validate_weight_mapping(reference_weights, build_pyannet_mapping_rules())

    assert result.passed is True
    assert result.missing_reference == []
    assert result.shape_mismatches == []


def test_load_npz_weights_returns_float32_arrays(tmp_path) -> None:
    path = tmp_path / "weights.npz"
    np.savez(path, **{"classifier.weight": np.ones((7, 128), dtype=np.float64)})

    weights = load_npz_weights(path)

    assert weights["classifier.weight"].shape == (7, 128)
    assert weights["classifier.weight"].dtype == np.float32
