import pytest


def test_pick_device_explicit_wins():
    from audio_denoise import _pick_device
    assert _pick_device("cuda", False) == "cuda"
    assert _pick_device("cpu", True) == "cpu"


def test_pick_device_auto():
    from audio_denoise import _pick_device
    assert _pick_device(None, True) == "cuda"
    assert _pick_device(None, False) == "cpu"


def test_load_demucs_builds_separator(monkeypatch):
    # fake Separator qua seam _ensure_demucs -> khong cham demucs/pip that
    import audio_denoise
    created = {}

    class FakeSeparator:
        def __init__(self, model=None, device=None):
            created["model"] = model
            created["device"] = device

    monkeypatch.setattr(audio_denoise, "_ensure_demucs", lambda: FakeSeparator)
    monkeypatch.setattr(audio_denoise, "_cuda_available", lambda: False)
    sep = audio_denoise.load_demucs()
    assert isinstance(sep, FakeSeparator)
    assert created["model"] == "htdemucs"
    assert created["device"] == "cpu"


def test_load_demucs_raises_when_unavailable(monkeypatch):
    # _ensure_demucs raise -> load_demucs propagate, KHONG chay pip that
    import audio_denoise

    def boom():
        raise ImportError("no demucs")

    monkeypatch.setattr(audio_denoise, "_ensure_demucs", boom)
    with pytest.raises(ImportError):
        audio_denoise.load_demucs()
