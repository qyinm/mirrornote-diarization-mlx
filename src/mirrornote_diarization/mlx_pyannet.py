"""Shape-correct MLX PyanNet segmentation candidate runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mirrornote_diarization.pyannet_contract import (
    PYANNET_CHUNK_DURATION_SECONDS,
    PYANNET_EXPECTED_OUTPUT_SHAPE,
    PYANNET_SAMPLE_RATE,
)
from mirrornote_diarization.weight_conversion import (
    build_pyannet_mapping_rules,
    validate_weight_mapping,
)

PYANNET_EXPECTED_WAVEFORM_SHAPE = (1, 1, 160000)
_INSTANCE_NORM_EPSILON = 1e-5
SINCNET_SAMPLE_RATE = 16000
_SINCNET_STRIDE = 10
_SINCNET_KERNEL_SIZE = 251
_SINCNET_HOP_WINDOW = 125
_SINCNET_HOP_SIZE = 2.0 * np.pi * np.arange(-_SINCNET_HOP_WINDOW, 0.0).reshape(1, -1) / float(
    SINCNET_SAMPLE_RATE
)
_SINCNET_MIN_LOW_HZ = 50.0
_SINCNET_MIN_BAND_HZ = 50.0


def _to_mx_array(values: Any) -> Any:
    import mlx.core as mx

    if isinstance(values, mx.array):
        return values
    return mx.array(values, dtype=mx.float32)


def _leaky_relu(x: Any) -> Any:
    import mlx.core as mx

    return mx.maximum(x, 0) + 0.01 * mx.minimum(x, 0)


def _sinc_filters(low_hz: np.ndarray, band_hz: np.ndarray) -> np.ndarray:
    low = _SINCNET_MIN_LOW_HZ + np.abs(low_hz)
    high = np.minimum(
        np.maximum(low + _SINCNET_MIN_BAND_HZ + np.abs(band_hz), _SINCNET_MIN_LOW_HZ),
        SINCNET_SAMPLE_RATE / 2,
    )

    band = (high - low)[:, 0]
    ft_low = low @ _SINCNET_HOP_SIZE
    ft_high = high @ _SINCNET_HOP_SIZE

    window = np.hamming(_SINCNET_KERNEL_SIZE)[:_SINCNET_HOP_WINDOW]

    left = ((np.sin(ft_high) - np.sin(ft_low)) / (_SINCNET_HOP_SIZE / 2)) * window
    center = 2.0 * band.reshape(-1, 1)
    right = np.flip(left, axis=1)
    cos_filters = np.concatenate([left, center, right], axis=1)

    left = ((np.cos(ft_low) - np.cos(ft_high)) / (_SINCNET_HOP_SIZE / 2)) * window
    center = np.zeros_like(band).reshape(-1, 1)
    right = -np.flip(left, axis=1)
    sin_filters = np.concatenate([left, center, right], axis=1)

    cos_filters = cos_filters / (2.0 * band.reshape(-1, 1))
    sin_filters = sin_filters / (2.0 * band.reshape(-1, 1))
    filters = np.concatenate([cos_filters, sin_filters], axis=0)
    return filters.reshape(-1, _SINCNET_KERNEL_SIZE, 1)


def _conv1d_nlc(
    waveform: Any,
    weight: np.ndarray,
    *,
    stride: int = 1,
    bias: np.ndarray | None = None,
) -> Any:
    import mlx.core as mx

    x = _to_mx_array(waveform)
    w = _to_mx_array(weight)
    if w.ndim != 3:
        raise ValueError(f"expected 3D conv weight, got shape {tuple(w.shape)}")

    input_channels = x.shape[2]
    if w.shape[2] != input_channels and w.shape[1] == input_channels:
        w = mx.transpose(w, (0, 2, 1))
    bias_array = None if bias is None else _to_mx_array(bias)

    # MLX conv1d uses N, L_in, C_in with C_out, K, C_in weights.
    output = mx.conv1d(x, w, stride=stride)
    if bias_array is None:
        return output

    if bias_array.shape != (w.shape[0],):
        raise ValueError(
            f"Unexpected conv1d bias shape {tuple(bias_array.shape)} for conv weight {tuple(w.shape)}"
        )
    return output + bias_array.reshape(1, 1, -1)


def _max_pool1d(x: Any, kernel_size: int, stride: int) -> Any:
    import mlx.core as mx

    x = _to_mx_array(x)
    if x.ndim != 3:
        raise ValueError(f"expected (batch, time, channels) for max pool, got {tuple(x.shape)}")

    x_len = int(x.shape[1])
    output_len = (x_len - kernel_size) // stride + 1

    pooled: list[Any] = []
    for offset in range(0, output_len * stride, stride):
        pooled.append(mx.max(x[:, offset : offset + kernel_size, :], axis=1))

    return mx.stack(pooled, axis=1)


def _instance_norm1d(x: Any, weight: np.ndarray, bias: np.ndarray) -> Any:
    import mlx.core as mx

    channels = int(weight.shape[0])
    if weight.shape != (channels,) or bias.shape != (channels,):
        raise ValueError("expected channel-wise InstanceNorm1d weight and bias")

    mean = mx.mean(x, axis=1, keepdims=True)
    var = mx.mean((x - mean) ** 2, axis=1, keepdims=True)
    denominator = mx.sqrt(var + _INSTANCE_NORM_EPSILON)
    x = (x - mean) / denominator

    w = _to_mx_array(weight).reshape(1, 1, channels)
    b = _to_mx_array(bias).reshape(1, 1, channels)
    return x * w + b


def _lstm_one_direction(
    inputs: Any,
    weight_ih: np.ndarray,
    weight_hh: np.ndarray,
    bias_ih: np.ndarray,
    bias_hh: np.ndarray,
    *,
    reverse: bool,
) -> Any:
    import mlx.core as mx

    x = _to_mx_array(inputs)
    if reverse:
        x = x[:, ::-1, :]

    batch_size, num_frames, in_dim = x.shape
    hidden_size = weight_hh.shape[1]
    if weight_ih.shape != (4 * hidden_size, in_dim):
        raise ValueError(
            "invalid LSTM weight_ih shape "
            f"{tuple(weight_ih.shape)} for in_dim={in_dim}, hidden_size={hidden_size}"
        )
    if weight_hh.shape != (4 * hidden_size, hidden_size):
        raise ValueError(
            "invalid LSTM weight_hh shape "
            f"{tuple(weight_hh.shape)} for hidden_size={hidden_size}"
        )

    w_ih = _to_mx_array(weight_ih)
    w_hh = _to_mx_array(weight_hh)
    b_ih = _to_mx_array(bias_ih).reshape(1, 4 * hidden_size)
    b_hh = _to_mx_array(bias_hh).reshape(1, 4 * hidden_size)

    h = mx.zeros((batch_size, hidden_size), dtype=mx.float32)
    c = mx.zeros((batch_size, hidden_size), dtype=mx.float32)
    outputs: list[Any] = []

    for t in range(int(num_frames)):
        xt = x[:, t, :]
        gates = mx.matmul(xt, mx.transpose(w_ih)) + mx.matmul(h, mx.transpose(w_hh))
        gates = gates + b_ih + b_hh

        i = gates[:, :hidden_size]
        f = gates[:, hidden_size : 2 * hidden_size]
        g = gates[:, 2 * hidden_size : 3 * hidden_size]
        o = gates[:, 3 * hidden_size :]

        i = mx.sigmoid(i)
        f = mx.sigmoid(f)
        g = mx.tanh(g)
        o = mx.sigmoid(o)

        c = f * c + i * g
        h = o * mx.tanh(c)
        outputs.append(h)

    output = mx.stack(outputs, axis=1)
    if reverse:
        output = output[:, ::-1, :]
    return output


def _lstm_bidirectional_layer(
    inputs: Any,
    layer: int,
    weights: Mapping[str, np.ndarray],
) -> Any:
    import mlx.core as mx

    forward = _lstm_one_direction(
        inputs,
        weight_ih=weights[f"lstm.layers.{layer}.forward.weight_ih"],
        weight_hh=weights[f"lstm.layers.{layer}.forward.weight_hh"],
        bias_ih=weights[f"lstm.layers.{layer}.forward.bias_ih"],
        bias_hh=weights[f"lstm.layers.{layer}.forward.bias_hh"],
        reverse=False,
    )
    reverse = _lstm_one_direction(
        inputs,
        weight_ih=weights[f"lstm.layers.{layer}.reverse.weight_ih"],
        weight_hh=weights[f"lstm.layers.{layer}.reverse.weight_hh"],
        bias_ih=weights[f"lstm.layers.{layer}.reverse.bias_ih"],
        bias_hh=weights[f"lstm.layers.{layer}.reverse.bias_hh"],
        reverse=True,
    )
    return mx.concatenate([forward, reverse], axis=2)


@dataclass(frozen=True)
class MlxPyanNetSegmentation:
    """MLX PyanNet candidate scaffold for segmentation parity plumbing."""

    reference_weights: dict[str, np.ndarray]
    output_classes: int = PYANNET_EXPECTED_OUTPUT_SHAPE[2]
    sample_rate: int = PYANNET_SAMPLE_RATE
    chunk_duration_seconds: float = PYANNET_CHUNK_DURATION_SECONDS

    @classmethod
    def from_reference_weights(
        cls,
        reference_weights: Mapping[str, np.ndarray],
    ) -> MlxPyanNetSegmentation:
        result = validate_weight_mapping(
            reference_weights,
            build_pyannet_mapping_rules(),
        )
        if not result.passed:
            raise ValueError(str(result.to_dict()))

        mapped = {
            candidate_key: np.asarray(reference_weights[reference_key], dtype=np.float32)
            for candidate_key, reference_key in result.mapped.items()
        }

        return cls(
            reference_weights=mapped,
        )

    @property
    def _reference(self) -> dict[str, np.ndarray]:
        return self.reference_weights

    def _sincnet(self, waveform: Any) -> Any:
        import mlx.core as mx

        outputs = mx.transpose(waveform, (0, 2, 1))
        outputs = _instance_norm1d(
            outputs,
            weight=self.reference_weights["sincnet.wav_norm.weight"],
            bias=self.reference_weights["sincnet.wav_norm.bias"],
        )

        # First layer is SincNet filterbank (no bias).
        sinc_filters = _sinc_filters(
            self.reference_weights["sincnet.sinc_filterbank.low_hz"],
            self.reference_weights["sincnet.sinc_filterbank.band_hz"],
        )
        outputs = _conv1d_nlc(
            outputs,
            sinc_filters,
            stride=_SINCNET_STRIDE,
        )
        outputs = mx.abs(outputs)
        outputs = _max_pool1d(outputs, kernel_size=3, stride=3)
        outputs = _instance_norm1d(
            outputs,
            weight=self.reference_weights["sincnet.norm.layers.0.weight"],
            bias=self.reference_weights["sincnet.norm.layers.0.bias"],
        )
        outputs = _leaky_relu(outputs)

        for layer in range(2):
            conv_weight = self.reference_weights[f"sincnet.conv.layers.{layer + 1}.weight"]
            conv_bias = self.reference_weights[f"sincnet.conv.layers.{layer + 1}.bias"]
            outputs = _conv1d_nlc(outputs, conv_weight, bias=conv_bias)
            outputs = _max_pool1d(outputs, kernel_size=3, stride=3)
            outputs = _instance_norm1d(
                outputs,
                weight=self.reference_weights[f"sincnet.norm.layers.{layer + 1}.weight"],
                bias=self.reference_weights[f"sincnet.norm.layers.{layer + 1}.bias"],
            )
            outputs = _leaky_relu(outputs)

        return outputs

    def _lstm(self, features: Any) -> Any:
        outputs = features
        for layer in range(4):
            outputs = _lstm_bidirectional_layer(outputs, layer, self.reference_weights)

        return outputs

    def __call__(self, waveform: Any) -> Any:
        import mlx.core as mx

        if tuple(waveform.shape) != PYANNET_EXPECTED_WAVEFORM_SHAPE:
            raise ValueError(
                "PyanNet MLX waveform must have shape "
                f"{PYANNET_EXPECTED_WAVEFORM_SHAPE}; got {tuple(waveform.shape)}"
            )

        x = _to_mx_array(waveform)
        x = self._sincnet(x)

        x = self._lstm(x)

        x = self.linear_head(x)
        max_per_frame = mx.max(x, axis=2, keepdims=True)
        x = x - max_per_frame
        return x - mx.log(mx.sum(mx.exp(x), axis=2, keepdims=True))

    def linear_head(self, features: Any) -> Any:
        import mlx.core as mx

        x = _dense(
            features,
            self.reference_weights["linear.layers.0.weight"],
            self.reference_weights["linear.layers.0.bias"],
        )
        x = _leaky_relu(x)
        x = _dense(
            x,
            self.reference_weights["linear.layers.1.weight"],
            self.reference_weights["linear.layers.1.bias"],
        )
        x = _leaky_relu(x)
        return _dense(
            x,
            self.reference_weights["classifier.weight"],
            self.reference_weights["classifier.bias"],
        )

    def write_candidate_npz(self, waveform: np.ndarray, path: str | Path) -> None:
        import mlx.core as mx

        output = self(_to_mx_array(waveform))
        np.savez(path, output=np.asarray(output, dtype=np.float32))


def _dense(x: Any, weight: np.ndarray, bias: np.ndarray) -> Any:
    import mlx.core as mx

    return mx.matmul(x, mx.array(weight.T, dtype=mx.float32)) + mx.array(
        bias,
        dtype=mx.float32,
    )
