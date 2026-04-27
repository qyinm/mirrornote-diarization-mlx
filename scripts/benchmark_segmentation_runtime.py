"""Benchmark MLX vs pyannote 3.1 segmentation runtime on the same waveform."""

from __future__ import annotations

import contextlib
from pathlib import Path
import argparse
import json
import os
import platform
import statistics
import time
from typing import Any

import numpy as np
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


def _safe_division(numerator: float, denominator: float) -> float:
    if float(denominator) <= 0:
        return 0.0
    return _to_finite_ms(numerator / denominator)


def _bench_device() -> str:
    import torch

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


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
    profile_stages: bool,
    profile_with_compile: bool,
) -> dict[str, Any]:
    from mirrornote_diarization.mlx_pyannet import MlxPyanNetSegmentation
    from mirrornote_diarization.weight_conversion import load_npz_weights
    import mlx.core as mx

    reference_weights = load_npz_weights(weights)
    model = MlxPyanNetSegmentation.from_reference_weights(reference_weights)
    input_dtype = mx.float16 if model._use_fp16 else mx.float32
    input_waveform = mx.array(waveform, dtype=input_dtype)

    if profile_stages and not profile_with_compile:
        object.__setattr__(model, "_compile_enabled", False)

    for _ in range(max(1, warmup)):
        if profile_stages:
            _ = _bench_mlx_stages(
                model,
                input_waveform,
                compile_stages=(profile_stages and profile_with_compile),
            )
        else:
            output = model(input_waveform)
            mx.eval(output)

    latencies_ms: list[float] = []
    stage_stats: dict[str, list[float]] = {"sincnet": [], "lstm": [], "linear": [], "total": []}
    for _ in range(runs):
        start = time.perf_counter()
        if profile_stages:
            stage_latency = _bench_mlx_stages(
                model,
                input_waveform,
                compile_stages=(profile_stages and profile_with_compile),
            )
            latencies_ms.append(stage_latency["total"])
            for name in stage_stats:
                stage_stats[name].append(stage_latency[name])
        else:
            output = model(input_waveform)
            mx.eval(output)
            latencies_ms.append(_to_finite_ms((time.perf_counter() - start) * 1000.0))

    summary = _summary_stats("pyannote-3.1-segmentation-mlx", latencies_ms)
    summary["lstmBackend"] = model._lstm_backend_name
    summary["compileEnabled"] = model._compile_enabled
    summary["fastMathEnabled"] = model._fast_math
    summary["fp16Enabled"] = model._use_fp16
    if profile_stages:
        summary["stageProfilesMs"] = {
            name: _summary_stats(name, values)
            for name, values in stage_stats.items()
            if values
        }

        profile_total = summary["stageProfilesMs"]["total"]["meanMs"]
        summary["stagePercentMs"] = {
            stage: _to_finite_ms(profile["meanMs"] / profile_total)
            for stage, profile in summary["stageProfilesMs"].items()
        }
    return summary


def _bench_mlx_stages(
    model: Any,
    input_waveform: Any,
    *,
    compile_stages: bool,
) -> dict[str, float]:
    import mlx.core as mx

    sinc_stage = model._sincnet
    lstm_stage = model._lstm
    linear_stage = model.linear_head

    if compile_stages and getattr(model, "_compile_enabled", False):
        sinc_stage = mx.compile(sinc_stage)
        lstm_stage = mx.compile(lstm_stage)
        linear_stage = mx.compile(linear_stage)

    start = time.perf_counter()
    with _with_fast_context(model._fast_math):
        sinc_output = sinc_stage(input_waveform)
        mx.eval(sinc_output)
        sinc_ms = _to_finite_ms((time.perf_counter() - start) * 1000.0)

    start = time.perf_counter()
    with _with_fast_context(model._fast_math):
        lstm_output = lstm_stage(sinc_output)
        mx.eval(lstm_output)
        lstm_ms = _to_finite_ms((time.perf_counter() - start) * 1000.0)

    start = time.perf_counter()
    with _with_fast_context(model._fast_math):
        linear_output = linear_stage(lstm_output)
        mx.eval(linear_output)
        linear_ms = _to_finite_ms((time.perf_counter() - start) * 1000.0)

    total_ms = sinc_ms + lstm_ms + linear_ms
    return {
        "sincnet": sinc_ms,
        "lstm": lstm_ms,
        "linear": linear_ms,
        "total": total_ms,
    }


def _with_fast_context(enabled: bool):
    import mlx.core as mx

    try:
        from mlx.core.fast import fast

        if callable(fast):
            return fast() if enabled else contextlib.nullcontext()
    except Exception:
        pass

    if not enabled:
        return contextlib.nullcontext()

    mx_fast = getattr(mx, "fast", None)
    if callable(mx_fast):
        return mx_fast()
    return contextlib.nullcontext()


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
    device = _bench_device()
    tensor = torch.from_numpy(waveform.astype(np.float32)).to(device=device, dtype=torch.float32)
    model = model.to(device)

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

    summary = _summary_stats("pyannote-3.1-segmentation-pytorch", latencies_ms)
    summary["device"] = device
    return summary


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
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="Measure MLX stages (sincnet/lstm/linear) separately",
    )
    parser.add_argument(
        "--profile-with-compile",
        action="store_true",
        help="When profiling stages, keep model compilation enabled and profile compiled kernels",
    )
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
                profile_stages=args.profile_stages,
                profile_with_compile=args.profile_with_compile,
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
            "pytorchFasterThanMlxX": _safe_division(mlx["meanMs"], pytorch["meanMs"]),
            "mlxFasterThanPyannoteX": _safe_division(pytorch["meanMs"], mlx["meanMs"]),
            "pytorchRealTimeFactor": _safe_division(audio_seconds, pytorch["meanMs"] / 1000.0),
            "mlxRealTimeFactor": _safe_division(audio_seconds, mlx["meanMs"] / 1000.0),
        }
        summary["comparison"]["mlxSpeedupVsPyannoteX"] = summary["comparison"]["mlxFasterThanPyannoteX"]
        summary["comparison"]["speedupTargetMet"] = (
            summary["comparison"]["mlxFasterThanPyannoteX"] >= 3.0
        )

    summary["environment"] = {
        "platform": platform.platform(),
        "benchmarkTorchDevice": _bench_device() if not args.no_pyannote else "cpu",
    }

    summary["providers"].sort(key=lambda item: item.get("provider", ""))

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _build_chart(summary, args.plot)

    args.report.write_text(json.dumps(summary, indent=2, default=_to_json) + "\n", encoding="utf-8")
    print(f"wrote runtime benchmark: {args.report}")
    print(f"wrote runtime benchmark plot: {args.plot}")
    if "comparison" in summary:
        print("metric_name: mlx_faster_than_pyannote_mean_time_x")
        print(f"metric_value: {summary['comparison']['mlxFasterThanPyannoteX']}")
        print("metric_unit: x")
        print(f"metric_ok_target_3x: {summary['comparison']['speedupTargetMet']}")

    if args.profile_stages:
        stage_entry = next(
            (
                entry
                for entry in summary["providers"]
                if entry["provider"] == "pyannote-3.1-segmentation-mlx"
                and "stageProfilesMs" in entry
            ),
            None,
        )
        if stage_entry is not None:
            print("stage_profile_ms:")
            for stage in ("sincnet", "lstm", "linear", "total"):
                value = stage_entry["stageProfilesMs"].get(stage, {})
                if value:
                    print(f"  {stage}: mean={value['meanMs']:.2f}, p95={value['p95Ms']:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
