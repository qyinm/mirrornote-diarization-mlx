from mirrornote_diarization.pyannet_contract import (
    PYANNET_ARCHITECTURE_NAME,
    PYANNET_EXPECTED_MODULE_TREE,
    PYANNET_EXPECTED_OUTPUT_SHAPE,
    PYANNET_EXPECTED_PARAMETER_COUNT,
    PYANNET_EXPECTED_WEIGHT_SHAPES,
    PYANNET_SAMPLE_RATE,
    PYANNET_CHUNK_DURATION_SECONDS,
)

EXPECTED_WEIGHT_SHAPES = {
    "classifier.bias": (7,),
    "classifier.weight": (7, 128),
    "linear.0.bias": (128,),
    "linear.0.weight": (128, 256),
    "linear.1.bias": (128,),
    "linear.1.weight": (128, 128),
    "lstm.bias_hh_l0": (512,),
    "lstm.bias_hh_l0_reverse": (512,),
    "lstm.bias_hh_l1": (512,),
    "lstm.bias_hh_l1_reverse": (512,),
    "lstm.bias_hh_l2": (512,),
    "lstm.bias_hh_l2_reverse": (512,),
    "lstm.bias_hh_l3": (512,),
    "lstm.bias_hh_l3_reverse": (512,),
    "lstm.bias_ih_l0": (512,),
    "lstm.bias_ih_l0_reverse": (512,),
    "lstm.bias_ih_l1": (512,),
    "lstm.bias_ih_l1_reverse": (512,),
    "lstm.bias_ih_l2": (512,),
    "lstm.bias_ih_l2_reverse": (512,),
    "lstm.bias_ih_l3": (512,),
    "lstm.bias_ih_l3_reverse": (512,),
    "lstm.weight_hh_l0": (512, 128),
    "lstm.weight_hh_l0_reverse": (512, 128),
    "lstm.weight_hh_l1": (512, 128),
    "lstm.weight_hh_l1_reverse": (512, 128),
    "lstm.weight_hh_l2": (512, 128),
    "lstm.weight_hh_l2_reverse": (512, 128),
    "lstm.weight_hh_l3": (512, 128),
    "lstm.weight_hh_l3_reverse": (512, 128),
    "lstm.weight_ih_l0": (512, 60),
    "lstm.weight_ih_l0_reverse": (512, 60),
    "lstm.weight_ih_l1": (512, 256),
    "lstm.weight_ih_l1_reverse": (512, 256),
    "lstm.weight_ih_l2": (512, 256),
    "lstm.weight_ih_l2_reverse": (512, 256),
    "lstm.weight_ih_l3": (512, 256),
    "lstm.weight_ih_l3_reverse": (512, 256),
    "sincnet.conv1d.0.filterbank.band_hz_": (40, 1),
    "sincnet.conv1d.0.filterbank.low_hz_": (40, 1),
    "sincnet.conv1d.0.filterbank.n_": (1, 125),
    "sincnet.conv1d.0.filterbank.window_": (125,),
    "sincnet.conv1d.1.bias": (60,),
    "sincnet.conv1d.1.weight": (60, 80, 5),
    "sincnet.conv1d.2.bias": (60,),
    "sincnet.conv1d.2.weight": (60, 60, 5),
    "sincnet.norm1d.0.bias": (80,),
    "sincnet.norm1d.0.weight": (80,),
    "sincnet.norm1d.1.bias": (60,),
    "sincnet.norm1d.1.weight": (60,),
    "sincnet.norm1d.2.bias": (60,),
    "sincnet.norm1d.2.weight": (60,),
    "sincnet.wav_norm1d.bias": (1,),
    "sincnet.wav_norm1d.weight": (1,),
}


def test_pyannet_reference_contract_matches_real_probe() -> None:
    assert PYANNET_ARCHITECTURE_NAME == (
        "pyannote.audio.models.segmentation.PyanNet.PyanNet"
    )
    assert PYANNET_SAMPLE_RATE == 16000
    assert PYANNET_CHUNK_DURATION_SECONDS == 10.0
    assert PYANNET_EXPECTED_OUTPUT_SHAPE == (1, 589, 7)
    assert PYANNET_EXPECTED_PARAMETER_COUNT == 1_473_515


def test_pyannet_module_tree_is_complete_and_ordered() -> None:
    assert PYANNET_EXPECTED_MODULE_TREE == (
        "model",
        "model.sincnet",
        "model.sincnet.wav_norm1d",
        "model.sincnet.conv1d",
        "model.sincnet.conv1d.0",
        "model.sincnet.conv1d.0.filterbank",
        "model.sincnet.conv1d.1",
        "model.sincnet.conv1d.2",
        "model.sincnet.pool1d",
        "model.sincnet.pool1d.0",
        "model.sincnet.pool1d.1",
        "model.sincnet.pool1d.2",
        "model.sincnet.norm1d",
        "model.sincnet.norm1d.0",
        "model.sincnet.norm1d.1",
        "model.sincnet.norm1d.2",
        "model.lstm",
        "model.linear",
        "model.linear.0",
        "model.linear.1",
        "model.classifier",
        "model.activation",
    )


def test_pyannet_expected_weight_shapes_match_reference_state_dict() -> None:
    assert PYANNET_EXPECTED_WEIGHT_SHAPES == EXPECTED_WEIGHT_SHAPES
