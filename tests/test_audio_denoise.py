import pytest


def test_pick_device_explicit_wins():
    from audio_denoise import _pick_device
    assert _pick_device("cuda", False) == "cuda"
    assert _pick_device("cpu", True) == "cpu"


def test_pick_device_auto():
    from audio_denoise import _pick_device
    assert _pick_device(None, True) == "cuda"
    assert _pick_device(None, False) == "cpu"


def test_load_demucs_loads_model(monkeypatch):
    # fake model qua seam _ensure_demucs_model -> khong cham demucs/pip that
    import audio_denoise
    rec = {}

    class FakeModel:
        def to(self, dev):
            rec["dev"] = dev
            return self

        def eval(self):
            rec["eval"] = True
            return self

    fake = FakeModel()
    monkeypatch.setattr(audio_denoise, "_ensure_demucs_model", lambda: fake)
    monkeypatch.setattr(audio_denoise, "_cuda_available", lambda: False)
    m = audio_denoise.load_demucs()
    assert m is fake
    assert rec["dev"] == "cpu"          # device chon dung
    assert rec["eval"] is True          # da chuyen sang eval mode


def test_load_demucs_raises_when_unavailable(monkeypatch):
    # _ensure_demucs_model raise -> load_demucs propagate, KHONG chay pip that
    import audio_denoise

    def boom():
        raise ImportError("no demucs")

    monkeypatch.setattr(audio_denoise, "_ensure_demucs_model", boom)
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


class _FakeModel:
    sources = ["drums", "bass", "other", "vocals"]   # thu tu stem cua htdemucs
    samplerate = 44100
    audio_channels = 2


def test_separate_vocals_picks_named_stem(monkeypatch, tmp_path):
    import audio_denoise
    saved = {}
    # 4 stem dung thu tu demucs; phai chon "vocals" theo TEN (= index 3)
    sources = [_FakeTensor("drums"), _FakeTensor("bass"),
               _FakeTensor("other"), _FakeTensor("vocals")]
    monkeypatch.setattr(audio_denoise, "_read_audio", lambda i, m: "wav")
    monkeypatch.setattr(audio_denoise, "_apply_demucs", lambda m, w: sources)
    monkeypatch.setattr(audio_denoise, "_save_wav",
                        lambda wav, p, sr: saved.update(tensor=wav, path=p, sr=sr))

    out = audio_denoise.separate_vocals("in.mp3", _FakeModel(), tmp_path / "v.wav")
    assert saved["tensor"].tag == "vocals"     # chon dung stem vocals theo ten
    assert saved["sr"] == 44100
    assert out == tmp_path / "v.wav"


def test_denoise_file_ok(monkeypatch, tmp_path):
    import audio_denoise
    monkeypatch.setattr(audio_denoise, "separate_vocals",
                        lambda i, m, t: Path(t))
    monkeypatch.setattr(audio_denoise, "_post_process",
                        lambda i, o, f: True)
    out = tmp_path / "out.wav"
    assert audio_denoise.denoise_file("in.mp3", out, _FakeModel(), "ffmpeg") is True


def test_denoise_file_returns_false_on_error(monkeypatch, tmp_path):
    import audio_denoise

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(audio_denoise, "separate_vocals", boom)
    out = tmp_path / "out.wav"
    assert audio_denoise.denoise_file("in.mp3", out, _FakeModel(), "ffmpeg") is False


# ── Task 4 tests ──────────────────────────────────────────────────────────────

def test_denoise_batch_loads_once_and_maps(monkeypatch, tmp_path):
    import audio_denoise
    calls = {"load": 0}
    monkeypatch.setattr(audio_denoise, "load_demucs",
                        lambda: calls.__setitem__("load", calls["load"] + 1) or object())
    monkeypatch.setattr(audio_denoise, "denoise_file",
                        lambda i, o, s, f: True)
    ins = [tmp_path / "a.mp3", tmp_path / "b.mp3"]
    out = audio_denoise.denoise_batch(ins, "ffmpeg")
    assert calls["load"] == 1
    assert set(out.keys()) == set(ins)
    assert out[ins[0]] == tmp_path / "audio_denoised" / "a.wav"


def test_denoise_batch_abort_on_load_fail(monkeypatch, tmp_path):
    import audio_denoise

    def boom():
        raise RuntimeError("no gpu")

    monkeypatch.setattr(audio_denoise, "load_demucs", boom)
    called = {"file": 0}
    monkeypatch.setattr(audio_denoise, "denoise_file",
                        lambda *a, **k: called.__setitem__("file", 1) or True)
    out = audio_denoise.denoise_batch([tmp_path / "a.mp3"], "ffmpeg")
    assert out == {}
    assert called["file"] == 0      # khong cat tho file nao


def test_denoise_batch_skips_failed_file(monkeypatch, tmp_path):
    import audio_denoise
    monkeypatch.setattr(audio_denoise, "load_demucs", lambda: object())

    def half(i, o, s, f):
        return i.name == "a.mp3"

    monkeypatch.setattr(audio_denoise, "denoise_file", half)
    ins = [tmp_path / "a.mp3", tmp_path / "b.mp3"]
    out = audio_denoise.denoise_batch(ins, "ffmpeg")
    assert (tmp_path / "a.mp3") in out
    assert (tmp_path / "b.mp3") not in out
