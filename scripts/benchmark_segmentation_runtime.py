"""Benchmark MLX vs pyannote 3.1 segmentation runtime on the same waveform."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import platform
import statistics
import time
from typing import Any

import numpy as np

from mirrornote_diarization.mlx_pyannet import MlxPyanNetSegmentation
from mirrornote_diarization.weight_conversion import load_npz_weights
from mirrornote_diarization.chunking import extract_fixed_chunk


DEFAULT_WEIGHTS_PATH = Path("artifacts/probe/librispeech-dummy-probe/reference-weights.npz")
DEFAULT_WAVEFORM_PATH = Path("artifacts/probe/librispeech-dummy-probe/waveform-input.npz")
DEFAULT_OUT_DIR = Path("reports/segmentation-benchmark")
DEFAULT_REPORT_PATH = DEFAULT_OUT_DIR / "runtime-benchmark.json"
DEFAULT_PLOT_PATH = DEFAULT_OUT_DIR / "runtime-benchmark.png"


def _parse_env_file(path: Path = Path(".env")) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"\'')
    return values


def _load_token() -> str | None:
    token = os.getenv("HUGGINGFACE_ACCESS_TOKEN", "").strip()
    if token:
        return token
    return _parse_env_file().get("HUGGINGFACE_ACCESS_TOKEN", "").strip() or None


def _to_finite_ms(value: float) -> float:
    return float(value) if float(value) > 0 else 0.0


def _to_json(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _synchronize_torch() -> None:
    import torch

    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _bench_mlx(
    *,
    weights: Path,
    waveform: np.ndarray,
    runs: int,
    warmup: int,
) -> dict[str, Any]:
    import mlx.core as mx

    reference_weights = load_npz_weights(weights)
    model = MlxPyanNetSegmentation.from_reference_weights(reference_weights)
    input_waveform = mx.array(waveform, dtype=mx.float32)

    for _ in range(max(1, warmup)):
        output = model(input_waveform)
        mx.eval(output)

    latencies_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        output = model(input_waveform)
        mx.eval(output)
        latencies_ms.append(_to_finite_ms((time.perf_counter() - start) * 1000.0))

    return _summary_stats("pyannote-3.1-segmentation-mlx", latencies_ms)


def _bench_pyannote(
    *,
    waveform: np.ndarray,
    runs: int,
    warmup: int,
    token: str,
) -> dict[str, Any]:
    import torch
    from pyannote.audio import Pipeline
    from mirrornote_diarization.pyannote_probe import _patch_huggingface_hub_token_compat

    _patch_huggingface_hub_token_compat()
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
    segmentation = getattr(pipeline, "_segmentation", None) or getattr(
        pipeline,
        "segmentation",
        None,
    )
    if segmentation is None:
        raise RuntimeError("pyannote pipeline does not expose a segmentation model")

    model = getattr(segmentation, "model", segmentation)
    model.eval()
    tensor = torch.from_numpy(waveform.astype(np.float32)).to(torch.float32)

    with torch.no_grad():
        for _ in range(max(1, warmup)):
            out = model(tensor)
            if torch.is_tensor(out):
                _ = out.cpu()
            _synchronize_torch()

    latencies_ms: list[float] = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            out = model(tensor)
            if torch.is_tensor(out):
                _ = out.cpu()
            _synchronize_torch()
            latencies_ms.append(_to_finite_ms((time.perf_counter() - start) * 1000.0))

    return _summary_stats("pyannote-3.1-segmentation-pytorch", latencies_ms)


def _summary_stats(provider: str, latencies_ms: list[float]) -> dict[str, Any]:
    sorted_latencies = sorted(latencies_ms)
    return {
        "provider": provider,
        "runs": len(sorted_latencies),
        "meanMs": statistics.fmean(sorted_latencies),
        "medianMs": statistics.median(sorted_latencies),
        "minMs": sorted_latencies[0],
        "maxMs": sorted_latencies[-1],
        "p95Ms": sorted_latencies[max(0, int(len(sorted_latencies) * 0.95) - 1)],
        "p99Ms": sorted_latencies[max(0, int(len(sorted_latencies) * 0.99) - 1)],
        "stdMs": statistics.pstdev(sorted_latencies),
        "samplesMs": sorted_latencies,
    }


def _build_chart(summary: dict[str, Any], plot_path: Path) -> None:
    from matplotlib import pyplot as plt

    providers = [
        entry["provider"]
        for entry in summary["providers"]
        if entry["status"] == "ok"
    ]
    means = [entry["meanMs"] for entry in summary["providers"] if entry["status"] == "ok"]

    if not providers:
        return

    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    bars = ax.bar(providers, means)
    ax.set_title("Segmentation Inference Latency (10s Chunk)")
    ax.set_ylabel("Mean Inference Time (ms)")
    ax.set_xlabel("Provider")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0f}ms",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)


def _load_waveform(path: Path) -> tuple[np.ndarray, int]:
    if path.suffix == ".npz":
        payload = np.load(path)
        waveform = np.asarray(payload["waveform"], dtype=np.float32)
        if waveform.ndim == 3 and waveform.shape[0] == 1 and waveform.shape[1] == 1:
            waveform = waveform[0, 0]
        elif waveform.ndim == 2:
            if waveform.shape[0] == 1:
                waveform = waveform[0]
            elif waveform.shape[1] == 1:
                waveform = waveform[:, 0]
            elif waveform.shape[0] > 1 and waveform.shape[0] == waveform.shape[1]:
                waveform = waveform[:, 0]
            else:
                waveform = waveform.mean(axis=0)
        elif waveform.ndim != 1:
            waveform = np.asarray(waveform).ravel()
        return np.asarray(waveform, dtype=np.float32), 16000

    import soundfile as sf

    data, sample_rate = sf.read(str(path), dtype="float32")
    return np.asarray(data, dtype=np.float32), int(sample_rate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--waveform", type=Path, default=DEFAULT_WAVEFORM_PATH)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--no-pyannote", action="store_true")
    args = parser.parse_args()

    waveform, sample_rate = _load_waveform(args.waveform)
    waveform_chunk = extract_fixed_chunk(
        waveform,
        sample_rate=sample_rate,
        start_seconds=0.0,
        duration_seconds=10.0,
    ).as_model_input()

    summary: dict[str, Any] = {
        "audio": {
            "source": str(args.waveform),
            "durationSeconds": float(waveform_chunk.shape[-1]) / 16000,
            "sampleRate": 16000,
        },
        "runs": args.runs,
        "warmup": args.warmup,
        "providers": [],
    }

    summary["providers"].append(
        {
            "provider": "pyannote-3.1-segmentation-mlx",
            **_bench_mlx(
                weights=args.weights,
                waveform=waveform_chunk,
                runs=args.runs,
                warmup=args.warmup,
            ),
            "status": "ok",
            "statusMessage": None,
        }
    )

    if not args.no_pyannote:
        token = _load_token()
        if token:
            try:
                summary["providers"].append(
                    {
                        "provider": "pyannote-3.1-segmentation-pytorch",
                        **_bench_pyannote(
                            waveform=waveform_chunk,
                            runs=args.runs,
                            warmup=args.warmup,
                            token=token,
                        ),
                        "status": "ok",
                        "statusMessage": None,
                    }
                )
            except Exception as exc:  # pragma: no cover
                summary["providers"].append(
                    {
                        "provider": "pyannote-3.1-segmentation-pytorch",
                        "status": "failed",
                        "statusMessage": str(exc),
                    }
                )
        else:
            summary["providers"].append(
                {
                    "provider": "pyannote-3.1-segmentation-pytorch",
                    "status": "skipped",
                    "statusMessage": "Missing HUGGINGFACE_ACCESS_TOKEN",
                }
            )

    ok_entries = [entry for entry in summary["providers"] if entry["status"] == "ok"]
    if len(ok_entries) >= 2:
        pytorch = next(
            entry
            for entry in ok_entries
            if entry["provider"] == "pyannote-3.1-segmentation-pytorch"
        )
        mlx = next(
            entry
            for entry in ok_entries
            if entry["provider"] == "pyannote-3.1-segmentation-mlx"
        )
        audio_seconds = summary["audio"]["durationSeconds"]
        summary["comparison"] = {
            "pyannoteMeanMs": _to_finite_ms(pytorch["meanMs"]),
            "mlxMeanMs": _to_finite_ms(mlx["meanMs"]),
            "pytorchFasterThanMlxX": _to_finite_ms(pytorch["meanMs"] / mlx["meanMs"])
            if mlx["meanMs"]
            else None,
            "mlxSpeedupVsPyannoteX": _to_finite_ms(mlx["meanMs"] / pytorch["meanMs"])
            if pytorch["meanMs"]
            else None,
            "pytorchRealTimeFactor": _to_finite_ms(audio_seconds / (pytorch["meanMs"] / 1000.0))
            if pytorch["meanMs"]
            else None,
            "mlxRealTimeFactor": _to_finite_ms(audio_seconds / (mlx["meanMs"] / 1000.0))
            if mlx["meanMs"]
            else None,
        }

    summary["environment"] = {
        "platform": platform.platform(),
    }

    summary["providers"].sort(key=lambda item: item.get("provider", ""))

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _build_chart(summary, args.plot)

    args.report.write_text(json.dumps(summary, indent=2, default=_to_json) + "\n", encoding="utf-8")
    print(f"wrote runtime benchmark: {args.report}")
    print(f"wrote runtime benchmark plot: {args.plot}")
    if "comparison" in summary:
        print("metric_name: mlx_speedup_vs_pyannote_mean_time_x")
        print(f"metric_value: {summary['comparison']['pytorchFasterThanMlxX']}")
        print("metric_unit: x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
