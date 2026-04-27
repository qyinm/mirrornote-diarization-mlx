import numpy as np
import pytest

from mirrornote_diarization.chunking import FixedChunk, extract_fixed_chunk


def test_extract_fixed_chunk_exact_window_from_mono_float32_waveform() -> None:
    waveform = np.arange(10, dtype=np.float32)

    chunk = extract_fixed_chunk(
        waveform,
        sample_rate=2,
        start_seconds=1.0,
        duration_seconds=2.0,
    )

    np.testing.assert_array_equal(chunk.samples, np.array([2, 3, 4, 5], dtype=np.float32))
    assert chunk.sample_rate == 2
    assert chunk.start_seconds == 1.0
    assert chunk.duration_seconds == 2.0


def test_extract_fixed_chunk_pads_short_tail_with_zeros() -> None:
    waveform = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    chunk = extract_fixed_chunk(
        waveform,
        sample_rate=2,
        start_seconds=1.0,
        duration_seconds=2.0,
    )

    np.testing.assert_array_equal(
        chunk.samples,
        np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def test_extract_fixed_chunk_rejects_multichannel_input() -> None:
    waveform = np.zeros((2, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="mono waveform"):
        extract_fixed_chunk(
            waveform,
            sample_rate=16_000,
            start_seconds=0.0,
            duration_seconds=1.0,
        )


def test_fixed_chunk_as_model_input_returns_batched_float32_tensor() -> None:
    chunk = FixedChunk(
        samples=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        sample_rate=16_000,
        start_seconds=0.0,
        duration_seconds=3 / 16_000,
    )

    model_input = chunk.as_model_input()

    assert model_input.shape == (1, 1, 3)
    assert model_input.dtype == np.float32
    np.testing.assert_array_equal(
        model_input,
        np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32),
    )
