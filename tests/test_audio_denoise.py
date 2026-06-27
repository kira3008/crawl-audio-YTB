import sys
import types
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


def test_post_filter_from_constants():
    from audio_denoise import _post_filter
    f = _post_filter()
    assert "highpass=f=80:p=2" in f
    assert "loudnorm=I=-23:TP=-1.5:LRA=11" in f


def test_post_process_builds_cmd(monkeypatch):
    import audio_denoise
    captured = {}

    class R:
        returncode = 0

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return R()

    monkeypatch.setattr(audio_denoise.subprocess, "run", fake_run)
    ok = audio_denoise._post_process("in.wav", "out.wav", "ffmpeg")
    assert ok is True
    cmd = captured["cmd"]
    assert "ffmpeg" == cmd[0]
    assert "-af" in cmd and audio_denoise._post_filter() in cmd
    assert "22050" in cmd and "pcm_s16le" in cmd
    assert cmd[cmd.index("-ac") + 1] == "1"


# ── Task 3 tests ──────────────────────────────────────────────────────────────

from pathlib import Path


class _FakeTensor:
    def __init__(self, tag):
        self.tag = tag
    def cpu(self):
        return self


class _FakeSep:
    samplerate = 44100
    def __init__(self, raise_it=False):
        self.raise_it = raise_it
    def separate_audio_file(self, path):
        if self.raise_it:
            raise RuntimeError("boom")
        # tra nhieu stem; phai chon "vocals" theo TEN
        return None, {"drums": _FakeTensor("drums"),
                      "other": _FakeTensor("other"),
                      "vocals": _FakeTensor("vocals")}


def test_separate_vocals_picks_named_stem(monkeypatch, tmp_path):
    import audio_denoise
    saved = {}
    fake_ta = types.ModuleType("torchaudio")
    fake_ta.save = lambda p, t, sr: saved.update(path=p, tensor=t, sr=sr)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_ta)

    out = audio_denoise.separate_vocals("in.mp3", _FakeSep(), tmp_path / "v.wav")
    assert saved["tensor"].tag == "vocals"     # chon dung stem vocals theo ten
    assert saved["sr"] == 44100
    assert out == tmp_path / "v.wav"


def test_denoise_file_ok(monkeypatch, tmp_path):
    import audio_denoise
    monkeypatch.setattr(audio_denoise, "separate_vocals",
                        lambda i, s, t: Path(t))
    monkeypatch.setattr(audio_denoise, "_post_process",
                        lambda i, o, f: True)
    out = tmp_path / "out.wav"
    assert audio_denoise.denoise_file("in.mp3", out, _FakeSep(), "ffmpeg") is True


def test_denoise_file_returns_false_on_error(monkeypatch, tmp_path):
    import audio_denoise

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(audio_denoise, "separate_vocals", boom)
    out = tmp_path / "out.wav"
    assert audio_denoise.denoise_file("in.mp3", out, _FakeSep(), "ffmpeg") is False
