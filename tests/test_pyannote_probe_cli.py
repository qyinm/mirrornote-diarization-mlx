from __future__ import annotations

from mirrornote_diarization.segmentation_parity import main


def test_probe_cli_requires_explicit_runtime_gate_before_loading_audio(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.delenv("MIRRORNOTE_RUN_PYANNOTE_PROBE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_ACCESS_TOKEN", raising=False)
    audio_path = tmp_path / "not-a-real.wav"
    audio_path.write_text("not a wav", encoding="utf-8")

    exit_code = main(
        [
            "segmentation",
            "probe",
            "--audio",
            str(audio_path),
            "--out",
            str(tmp_path / "probe"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "MIRRORNOTE_RUN_PYANNOTE_PROBE=1" in captured.err

