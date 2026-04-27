import numpy as np
import pytest

from mirrornote_diarization.pyannet_contract import PYANNET_EXPECTED_WEIGHT_SHAPES
from mirrornote_diarization.weight_conversion import (
    _pyannet_lstm_candidate_key,
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

    assert mapped == {
        "classifier.bias": "classifier.bias",
        "classifier.weight": "classifier.weight",
        "linear.0.bias": "linear.layers.0.bias",
        "linear.0.weight": "linear.layers.0.weight",
        "linear.1.bias": "linear.layers.1.bias",
        "linear.1.weight": "linear.layers.1.weight",
        "lstm.bias_hh_l0": "lstm.layers.0.forward.bias_hh",
        "lstm.bias_hh_l0_reverse": "lstm.layers.0.reverse.bias_hh",
        "lstm.bias_hh_l1": "lstm.layers.1.forward.bias_hh",
        "lstm.bias_hh_l1_reverse": "lstm.layers.1.reverse.bias_hh",
        "lstm.bias_hh_l2": "lstm.layers.2.forward.bias_hh",
        "lstm.bias_hh_l2_reverse": "lstm.layers.2.reverse.bias_hh",
        "lstm.bias_hh_l3": "lstm.layers.3.forward.bias_hh",
        "lstm.bias_hh_l3_reverse": "lstm.layers.3.reverse.bias_hh",
        "lstm.bias_ih_l0": "lstm.layers.0.forward.bias_ih",
        "lstm.bias_ih_l0_reverse": "lstm.layers.0.reverse.bias_ih",
        "lstm.bias_ih_l1": "lstm.layers.1.forward.bias_ih",
        "lstm.bias_ih_l1_reverse": "lstm.layers.1.reverse.bias_ih",
        "lstm.bias_ih_l2": "lstm.layers.2.forward.bias_ih",
        "lstm.bias_ih_l2_reverse": "lstm.layers.2.reverse.bias_ih",
        "lstm.bias_ih_l3": "lstm.layers.3.forward.bias_ih",
        "lstm.bias_ih_l3_reverse": "lstm.layers.3.reverse.bias_ih",
        "lstm.weight_hh_l0": "lstm.layers.0.forward.weight_hh",
        "lstm.weight_hh_l0_reverse": "lstm.layers.0.reverse.weight_hh",
        "lstm.weight_hh_l1": "lstm.layers.1.forward.weight_hh",
        "lstm.weight_hh_l1_reverse": "lstm.layers.1.reverse.weight_hh",
        "lstm.weight_hh_l2": "lstm.layers.2.forward.weight_hh",
        "lstm.weight_hh_l2_reverse": "lstm.layers.2.reverse.weight_hh",
        "lstm.weight_hh_l3": "lstm.layers.3.forward.weight_hh",
        "lstm.weight_hh_l3_reverse": "lstm.layers.3.reverse.weight_hh",
        "lstm.weight_ih_l0": "lstm.layers.0.forward.weight_ih",
        "lstm.weight_ih_l0_reverse": "lstm.layers.0.reverse.weight_ih",
        "lstm.weight_ih_l1": "lstm.layers.1.forward.weight_ih",
        "lstm.weight_ih_l1_reverse": "lstm.layers.1.reverse.weight_ih",
        "lstm.weight_ih_l2": "lstm.layers.2.forward.weight_ih",
        "lstm.weight_ih_l2_reverse": "lstm.layers.2.reverse.weight_ih",
        "lstm.weight_ih_l3": "lstm.layers.3.forward.weight_ih",
        "lstm.weight_ih_l3_reverse": "lstm.layers.3.reverse.weight_ih",
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


@pytest.mark.parametrize(
    "reference_key",
    [
        "weight_ih_l0",
        "lstm.weight_ihh_l0",
        "lstm.weight_ih_projection_l0",
        "lstm.bias_ih_extra_l2_reverse",
        "lstm.weight_ih_lx",
        "lstm.weight_ih_l0_bidirectional",
    ],
)
def test_pyannet_lstm_candidate_key_rejects_malformed_keys(
    reference_key: str,
) -> None:
    with pytest.raises(ValueError, match="unsupported PyanNet LSTM key"):
        _pyannet_lstm_candidate_key(reference_key)


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
