"""Shape-correct MLX PyanNet segmentation candidate runtime."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass, field
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

try:
    from mlx.core.fast import fast as _mlx_fast_context
except Exception:  # pragma: no cover - fallback for older MLX module layouts
    _mlx_fast_context = None


def _with_mlx_fast_context(enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    if _mlx_fast_context is not None:
        if callable(_mlx_fast_context):
            return _mlx_fast_context()
        nested_fast = getattr(_mlx_fast_context, "fast", None)
        if callable(nested_fast):
            return nested_fast()
    import mlx.core as mx

    mx_fast = getattr(mx, "fast", None)
    if callable(mx_fast):
        return mx_fast()
    return contextlib.nullcontext()


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
    weight: Any,
    *,
    stride: int = 1,
    bias: Any | None = None,
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
    if kernel_size == stride and x_len % kernel_size == 0:
        trimmed = x[:, : output_len * kernel_size, :]
        reshaped = trimmed.reshape(int(x.shape[0]), output_len, kernel_size, int(x.shape[2]))
        return mx.max(reshaped, axis=2)

    pooled: list[Any] = []
    for offset in range(0, output_len * stride, stride):
        pooled.append(mx.max(x[:, offset : offset + kernel_size, :], axis=1))

    return mx.stack(pooled, axis=1)


def _instance_norm1d(x: Any, weight: Any, bias: Any) -> Any:
    import mlx.core as mx

    channels = int(weight.shape[0])
    if weight.shape != (channels,) or bias.shape != (channels,):
        raise ValueError("expected channel-wise InstanceNorm1d weight and bias")

    mean = mx.mean(x, axis=1, keepdims=True)
    mean_sq = mx.mean(x * x, axis=1, keepdims=True)
    variance = mean_sq - mean * mean
    variance = mx.maximum(variance, 0.0)
    denominator = mx.sqrt(variance + _INSTANCE_NORM_EPSILON)
    x = (x - mean) / denominator

    w = weight.reshape(1, 1, channels)
    b = bias.reshape(1, 1, channels)
    return x * w + b


def _lstm_one_direction(
    inputs: Any,
    weight_ih_t: Any,
    weight_hh_t: Any,
    bias: Any,
    *,
    reverse: bool,
) -> Any:
    import mlx.core as mx

    x = inputs
    if reverse:
        x = x[:, ::-1, :]

    batch_size, num_frames, in_dim = x.shape
    hidden_size = int(weight_ih_t.shape[1]) // 4

    projected = mx.matmul(x, weight_ih_t) + bias

    dtype = x.dtype
    h = mx.zeros((batch_size, hidden_size), dtype=dtype)
    c = mx.zeros((batch_size, hidden_size), dtype=dtype)
    outputs: list[Any] = []

    for t in range(int(num_frames)):
        gates = projected[:, t, :] + mx.matmul(h, weight_hh_t)

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


def _lstm_one_direction_nn(
    inputs: Any,
    lstm_module: Any,
    *,
    reverse: bool,
) -> Any:
    x = inputs
    if reverse:
        x = x[:, ::-1, :]

    output, _ = lstm_module(x)
    if reverse:
        return output[:, ::-1, :]
    return output


def _lstm_bidirectional_layer(
    inputs: Any,
    lstm_specs: tuple[tuple[Any, Any, Any], tuple[Any, Any, Any]],
) -> Any:
    import mlx.core as mx

    forward_weights = lstm_specs[0]
    reverse_weights = lstm_specs[1]

    forward = _lstm_one_direction(
        inputs,
        weight_ih_t=forward_weights[0],
        weight_hh_t=forward_weights[1],
        bias=forward_weights[2],
        reverse=False,
    )
    reverse = _lstm_one_direction(
        inputs,
        weight_ih_t=reverse_weights[0],
        weight_hh_t=reverse_weights[1],
        bias=reverse_weights[2],
        reverse=True,
    )
    return mx.concatenate([forward, reverse], axis=2)


def _lstm_bidirectional_layer_nn(
    inputs: Any,
    lstm_modules: tuple[Any, Any],
) -> Any:
    import mlx.core as mx

    forward = _lstm_one_direction_nn(
        inputs,
        lstm_module=lstm_modules[0],
        reverse=False,
    )
    reverse = _lstm_one_direction_nn(
        inputs,
        lstm_module=lstm_modules[1],
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
    _mx_weights: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _compiled_call: Any | None = field(default=None, init=False, repr=False)
    _compile_enabled: bool = field(default=True, init=False, repr=False)
    _use_mlx_lstm: bool = field(default=False, init=False, repr=False)
    _lstm_specs: tuple[
        tuple[tuple[Any, Any, Any, Any], tuple[Any, Any, Any, Any]],
        ...,
    ] = field(default_factory=tuple, init=False, repr=False)
    _lstm_specs_fast: tuple[
        tuple[tuple[Any, Any, Any], tuple[Any, Any, Any]],
        ...,
    ] = field(default_factory=tuple, init=False, repr=False)
    _lstm_modules: tuple[tuple[Any, Any], ...] = field(
        default_factory=tuple,
        init=False,
        repr=False,
    )
    _dense_weight_transposes: dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _conv_weight_transposes: dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _cached_sinc_filters: Any = field(default=None, init=False, repr=False)
    _fast_math: bool = field(default=False, init=False, repr=False)
    _use_fp16: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_reference_weights(
        cls,
        reference_weights: Mapping[str, np.ndarray],
    ) -> MlxPyanNetSegmentation:
        import os
        import mlx.core as mx

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

        model = cls(reference_weights=mapped)
        use_fp16 = os.getenv("PYANNOTE_MLX_FP16", "0") in {"1", "true", "True"}
        reference_weight_dtype = mx.float16 if use_fp16 else mx.float32
        mx_reference_weights = {
            candidate_key: mx.array(weights, dtype=reference_weight_dtype)
            for candidate_key, weights in mapped.items()
        }
        object.__setattr__(
            model,
            "_mx_weights",
            mx_reference_weights,
        )
        object.__setattr__(model, "_use_fp16", use_fp16)
        object.__setattr__(
            model,
            "_dense_weight_transposes",
            {
                "linear.layers.0.weight_t": mx_reference_weights["linear.layers.0.weight"].T,
                "linear.layers.1.weight_t": mx_reference_weights["linear.layers.1.weight"].T,
                "classifier.weight_t": mx_reference_weights["classifier.weight"].T,
            },
        )
        object.__setattr__(
            model,
            "_conv_weight_transposes",
            {
                f"sincnet.conv.layers.{layer}.weight": mx_reference_weights[
                    f"sincnet.conv.layers.{layer}.weight"
                ].transpose(0, 2, 1)
                for layer in (1, 2)
            },
        )
        object.__setattr__(
            model,
            "_lstm_specs",
            tuple(
                (
                    (
                        mx_reference_weights[f"lstm.layers.{layer}.forward.weight_ih"],
                        mx_reference_weights[f"lstm.layers.{layer}.forward.weight_hh"],
                        mx_reference_weights[f"lstm.layers.{layer}.forward.bias_ih"],
                        mx_reference_weights[f"lstm.layers.{layer}.forward.bias_hh"],
                    ),
                    (
                        mx_reference_weights[f"lstm.layers.{layer}.reverse.weight_ih"],
                        mx_reference_weights[f"lstm.layers.{layer}.reverse.weight_hh"],
                        mx_reference_weights[f"lstm.layers.{layer}.reverse.bias_ih"],
                        mx_reference_weights[f"lstm.layers.{layer}.reverse.bias_hh"],
                    ),
                )
                for layer in range(4)
            ),
        )
        object.__setattr__(
            model,
            "_lstm_specs_fast",
            tuple(
                (
                    (
                        mx_reference_weights[f"lstm.layers.{layer}.forward.weight_ih"].T,
                        mx_reference_weights[f"lstm.layers.{layer}.forward.weight_hh"].T,
                        (mx_reference_weights[f"lstm.layers.{layer}.forward.bias_ih"]
                         + mx_reference_weights[f"lstm.layers.{layer}.forward.bias_hh"]).reshape(
                            1, -1
                        ),
                    ),
                    (
                        mx_reference_weights[f"lstm.layers.{layer}.reverse.weight_ih"].T,
                        mx_reference_weights[f"lstm.layers.{layer}.reverse.weight_hh"].T,
                        (mx_reference_weights[f"lstm.layers.{layer}.reverse.bias_ih"]
                         + mx_reference_weights[f"lstm.layers.{layer}.reverse.bias_hh"]).reshape(
                            1, -1
                        ),
                    ),
                )
                for layer in range(4)
            ),
        )
        use_mlx_lstm = os.getenv("PYANNOTE_MLX_LSTM_BACKEND", "nn").strip().lower() != "legacy"
        object.__setattr__(model, "_use_mlx_lstm", use_mlx_lstm)
        if use_mlx_lstm:
            try:
                import mlx.nn as nn

                lstm_modules = []
                for forward_weights, reverse_weights in model._lstm_specs:
                    in_dim = int(forward_weights[0].shape[1])
                    hidden_size = int(forward_weights[0].shape[0] // 4)
                    forward_module = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, bias=True)
                    reverse_module = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, bias=True)
                    forward_module.Wx = forward_weights[0]
                    forward_module.Wh = forward_weights[1]
                    forward_module.bias = forward_weights[2] + forward_weights[3]
                    reverse_module.Wx = reverse_weights[0]
                    reverse_module.Wh = reverse_weights[1]
                    reverse_module.bias = reverse_weights[2] + reverse_weights[3]
                    lstm_modules.append((forward_module, reverse_module))

                object.__setattr__(model, "_lstm_modules", tuple(lstm_modules))
            except Exception:  # pragma: no cover - fallback when nn path unavailable in current runtime
                object.__setattr__(model, "_use_mlx_lstm", False)
        object.__setattr__(
            model,
            "_cached_sinc_filters",
            mx.array(
                _sinc_filters(
                    mapped["sincnet.sinc_filterbank.low_hz"],
                    mapped["sincnet.sinc_filterbank.band_hz"],
                ),
                dtype=reference_weight_dtype,
            ),
        )
        object.__setattr__(
            model,
            "_compile_enabled",
            os.getenv("PYANNOTE_MLX_COMPILE", "1") != "0",
        )
        object.__setattr__(
            model,
            "_fast_math",
            os.getenv("PYANNOTE_MLX_FAST_MATH", "0") in {"1", "true", "True"},
        )

        return model

    def _forward_impl(self, waveform: Any) -> Any:
        import mlx.core as mx
        if tuple(waveform.shape) != PYANNET_EXPECTED_WAVEFORM_SHAPE:
            raise ValueError(
                "PyanNet MLX waveform must have shape "
                f"{PYANNET_EXPECTED_WAVEFORM_SHAPE}; got {tuple(waveform.shape)}"
            )

        if self._use_fp16:
            waveform = waveform.astype(mx.float16)
        elif waveform.dtype not in (mx.float32, mx.float16):
            waveform = waveform.astype(mx.float32)

        with _with_mlx_fast_context(self._fast_math):
            x = self._sincnet(waveform)
            x = self._lstm(x)
            x = self.linear_head(x)
            return x - mx.logsumexp(x, axis=2, keepdims=True)

    @property
    def _reference(self) -> dict[str, np.ndarray]:
        return self.reference_weights

    @property
    def _lstm_backend_name(self) -> str:
        return "nn" if self._use_mlx_lstm else "manual"

    def _sincnet(self, waveform: Any) -> Any:
        import mlx.core as mx

        outputs = mx.transpose(waveform, (0, 2, 1))
        outputs = _instance_norm1d(
            outputs,
            weight=self._mx_weights["sincnet.wav_norm.weight"],
            bias=self._mx_weights["sincnet.wav_norm.bias"],
        )

        # First layer is SincNet filterbank (no bias).
        outputs = _conv1d_nlc(
            outputs,
            self._cached_sinc_filters,
            stride=_SINCNET_STRIDE,
        )
        outputs = mx.abs(outputs)
        outputs = _max_pool1d(outputs, kernel_size=3, stride=3)
        outputs = _instance_norm1d(
            outputs,
            weight=self._mx_weights["sincnet.norm.layers.0.weight"],
            bias=self._mx_weights["sincnet.norm.layers.0.bias"],
        )
        outputs = _leaky_relu(outputs)

        for layer in range(2):
            conv_weight = self._conv_weight_transposes[
                f"sincnet.conv.layers.{layer + 1}.weight"
            ]
            conv_bias = self._mx_weights[f"sincnet.conv.layers.{layer + 1}.bias"]
            outputs = _conv1d_nlc(outputs, conv_weight, bias=conv_bias)
            outputs = _max_pool1d(outputs, kernel_size=3, stride=3)
            outputs = _instance_norm1d(
                outputs,
                weight=self._mx_weights[f"sincnet.norm.layers.{layer + 1}.weight"],
                bias=self._mx_weights[f"sincnet.norm.layers.{layer + 1}.bias"],
            )
            outputs = _leaky_relu(outputs)

        return outputs

    def _lstm(self, features: Any) -> Any:
        outputs = features
        for layer in range(4):
            if self._use_mlx_lstm:
                outputs = _lstm_bidirectional_layer_nn(
                    outputs,
                    lstm_modules=self._lstm_modules[layer],
                )
            else:
                outputs = _lstm_bidirectional_layer(
                    outputs,
                    lstm_specs=self._lstm_specs_fast[layer],
                )

        return outputs

    def __call__(self, waveform: Any) -> Any:
        import mlx.core as mx

        input_waveform = _to_mx_array(waveform)
        if self._compile_enabled:
            if self._compiled_call is None:
                object.__setattr__(
                    self,
                    "_compiled_call",
                    mx.compile(self._forward_impl),
                )
            return self._compiled_call(input_waveform)

        return self._forward_impl(input_waveform)

    def linear_head(self, features: Any) -> Any:
        import mlx.core as mx

        x = _dense(
            features,
            self._dense_weight_transposes["linear.layers.0.weight_t"],
            self._mx_weights["linear.layers.0.bias"],
        )
        x = _leaky_relu(x)
        x = _dense(
            x,
            self._dense_weight_transposes["linear.layers.1.weight_t"],
            self._mx_weights["linear.layers.1.bias"],
        )
        x = _leaky_relu(x)
        return _dense(
            x,
            self._dense_weight_transposes["classifier.weight_t"],
            self._mx_weights["classifier.bias"],
        )

    def write_candidate_npz(self, waveform: np.ndarray, path: str | Path) -> None:
        import mlx.core as mx

        output = self(_to_mx_array(waveform))
        np.savez(path, output=np.asarray(output, dtype=np.float32))


def _dense(x: Any, weight: Any, bias: Any) -> Any:
    import mlx.core as mx

    return mx.matmul(x, weight) + bias
