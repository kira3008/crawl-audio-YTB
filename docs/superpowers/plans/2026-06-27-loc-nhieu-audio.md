# Lọc Nhiễu Audio (Demucs-only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lọc nhiễu tín hiệu audio bằng Demucs (tách vocal) cho mọi file trước khi cắt segment, cho ra dataset WAV 22050Hz mono sạch và đồng đều phục vụ train TTS/ASR.

**Architecture:** Module mới `audio_denoise.py` lọc CẢ FILE (Demucs `htdemucs` tách stem `vocals` → ffmpeg highpass+loudnorm+resample → WAV 22050 mono). `split_audio.py` chạy `denoise_batch` (load model 1 lần) trước vòng cắt, rồi `split_one` cắt segment từ file WAV đã sạch. Mọi phép biến đổi bảo toàn độ-dài-theo-giây nên timestamps transcript vẫn khớp, không re-transcribe.

**Tech Stack:** Python 3.10+, demucs>=4.0 (`demucs.api.Separator`), torch/torchaudio (đã có qua whisperx), ffmpeg (qua imageio-ffmpeg), pytest.

## Global Constraints

- Python 3.10+ (`str | None`, `list[dict]`).
- **Một đường DUY NHẤT**: Demucs cho mọi file; KHÔNG tier nhẹ/router/`--mode`/`--no-denoise`/fallback denoiser khác.
- **Một dạng output DUY NHẤT**: WAV `pcm_s16le`, `22050` Hz, mono. Không sinh mp3 thô song song.
- **Fail loud, không fallback**: Demucs không khả dụng → báo lỗi rõ + abort denoise, KHÔNG cắt thô.
- **Bảo toàn timing**: chỉ dùng phép biến đổi giữ nguyên độ-dài-theo-giây (Demucs, highpass, loudnorm, resample). KHÔNG `silenceremove`/trim.
- **Lọc cả file rồi cắt** (không cắt-rồi-lọc-từng-mảnh). Không ghi đè mp3 gốc.
- Mọi tham số là HẰNG SỐ module ở đầu `audio_denoise.py`; chuỗi ffmpeg nối từ hằng số (không hardcode).
- Lazy-install demucs theo pattern `load_vad_model` (`split_audio.py:57-64`).
- Stem vocals lấy theo TÊN (`"vocals"`), KHÔNG hardcode index.
- Comment/code/commit tiếng Việt-không-dấu hoặc tiếng Anh; KHÔNG đưa xưng hô anh/em vào code.
- Mọi hàm pure trừ hàm chạy Demucs/ffmpeg (I/O) — các hàm I/O phải fail-safe theo đặc tả.

Hằng số verbatim (đầu `audio_denoise.py`):
```
TARGET_SR        = 22050
TARGET_CHANNELS  = 1
DEMUCS_MODEL     = "htdemucs"
DEMUCS_OVERLAP   = 0.25
LOUDNORM_I       = -23
LOUDNORM_TP      = -1.5
LOUDNORM_LRA     = 11
HIGHPASS_HZ      = 80
```

---

### Task 1: Hằng số + chọn device + `load_demucs`

**Files:**
- Create: `audio_denoise.py`
- Test: `tests/test_audio_denoise.py`

**Interfaces:**
- Produces:
  - Các hằng số module (verbatim ở Global Constraints).
  - `_pick_device(device: str | None, cuda_available: bool) -> str` — device tường minh thắng; `None` → `"cuda"` nếu có cuda, ngược lại `"cpu"`.
  - `load_demucs(device: str | None = None) -> object` — lazy-install demucs, tạo `demucs.api.Separator(model=DEMUCS_MODEL, device=...)`. Import/load lỗi → **raise** (không nuốt lỗi).

- [ ] **Step 1: Viết test thất bại**

```python
# tests/test_audio_denoise.py
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
    # fake demucs.api.Separator để không cần cài demucs that
    created = {}

    class FakeSeparator:
        def __init__(self, model=None, device=None):
            created["model"] = model
            created["device"] = device

    fake_api = types.ModuleType("demucs.api")
    fake_api.Separator = FakeSeparator
    fake_pkg = types.ModuleType("demucs")
    monkeypatch.setitem(sys.modules, "demucs", fake_pkg)
    monkeypatch.setitem(sys.modules, "demucs.api", fake_api)

    import audio_denoise
    monkeypatch.setattr(audio_denoise, "_cuda_available", lambda: False)
    sep = audio_denoise.load_demucs()
    assert isinstance(sep, FakeSeparator)
    assert created["model"] == "htdemucs"
    assert created["device"] == "cpu"


def test_load_demucs_raises_when_unavailable(monkeypatch):
    import audio_denoise
    # gia lap import demucs.api that bai va pip install cung that bai
    monkeypatch.setattr(audio_denoise, "_import_separator",
                        lambda: (_ for _ in ()).throw(ImportError("no demucs")))
    with pytest.raises(ImportError):
        audio_denoise.load_demucs()
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_audio_denoise.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'audio_denoise'`

- [ ] **Step 3: Viết `audio_denoise.py` (hằng số + device + load_demucs)**

```python
#!/usr/bin/env python3
"""audio_denoise.py — loc nhieu tin hieu audio bang Demucs (tach vocal)."""

import subprocess
import sys

TARGET_SR        = 22050
TARGET_CHANNELS  = 1
DEMUCS_MODEL     = "htdemucs"
DEMUCS_OVERLAP   = 0.25
LOUDNORM_I       = -23
LOUDNORM_TP      = -1.5
LOUDNORM_LRA     = 11
HIGHPASS_HZ      = 80


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _pick_device(device, cuda_available: bool) -> str:
    if device:
        return device
    return "cuda" if cuda_available else "cpu"


def _import_separator():
    from demucs.api import Separator
    return Separator


def load_demucs(device=None):
    try:
        Separator = _import_separator()
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "demucs"])
        Separator = _import_separator()
    dev = _pick_device(device, _cuda_available())
    return Separator(model=DEMUCS_MODEL, device=dev)
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_audio_denoise.py -v`
Expected: PASS (4 test)

- [ ] **Step 5: Ghi chú demucs vào `requirements.txt`**

Thêm vào cuối `requirements.txt` (Demucs lazy-install nên để dạng comment + optional):
```
# demucs>=4.0.0  # loc nhieu audio (audio_denoise.py) — lazy-install khi can, can GPU
```

- [ ] **Step 6: Commit**

```bash
git add audio_denoise.py tests/test_audio_denoise.py requirements.txt
git commit -m "feat(denoise): constants, device picker, load_demucs (lazy-install)"
```

---

### Task 2: Chuỗi ffmpeg post-process (`_post_filter` + `_post_process`)

**Files:**
- Modify: `audio_denoise.py`
- Test: `tests/test_audio_denoise.py`

**Interfaces:**
- Consumes: hằng số `HIGHPASS_HZ`, `LOUDNORM_I/TP/LRA`, `TARGET_SR`, `TARGET_CHANNELS`.
- Produces:
  - `_post_filter() -> str` — nối chuỗi `-af` từ hằng số: `"highpass=f=80:p=2,loudnorm=I=-23:TP=-1.5:LRA=11"`.
  - `_post_process(in_wav, out_path, ffmpeg_exe) -> bool` — chạy ffmpeg: `-af _post_filter()`, `-ar TARGET_SR`, `-ac TARGET_CHANNELS`, `-c:a pcm_s16le`; trả `returncode == 0`.

- [ ] **Step 1: Viết test thất bại**

```python
# them vao tests/test_audio_denoise.py
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
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_audio_denoise.py -k "post_filter or post_process" -v`
Expected: FAIL — `ImportError: cannot import name '_post_filter'`

- [ ] **Step 3: Thêm `_post_filter` + `_post_process`**

```python
def _post_filter() -> str:
    return (f"highpass=f={HIGHPASS_HZ}:p=2,"
            f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}")


def _post_process(in_wav, out_path, ffmpeg_exe) -> bool:
    cmd = [
        ffmpeg_exe, "-y", "-loglevel", "error",
        "-i", str(in_wav),
        "-af", _post_filter(),
        "-ar", str(TARGET_SR),
        "-ac", str(TARGET_CHANNELS),
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_audio_denoise.py -k "post_filter or post_process" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add audio_denoise.py tests/test_audio_denoise.py
git commit -m "feat(denoise): ffmpeg post-process chain (highpass+loudnorm+resample to wav)"
```

---

### Task 3: Tách vocal + `denoise_file`

**Files:**
- Modify: `audio_denoise.py`
- Test: `tests/test_audio_denoise.py`

**Interfaces:**
- Consumes: `_post_process`.
- Produces:
  - `separate_vocals(in_path, separator, tmp_wav) -> Path` — gọi `separator.separate_audio_file(str(in_path))` → `(_, stems)`; lấy `stems["vocals"]` (theo TÊN); `torchaudio.save(str(tmp_wav), vocals.cpu(), separator.samplerate)`; trả `tmp_wav`.
  - `denoise_file(in_path, out_path, separator, ffmpeg_exe) -> bool` — orchestrate `separate_vocals` → `_post_process`; dọn file tạm; mọi Exception → log + trả `False`.

- [ ] **Step 1: Viết test thất bại**

```python
# them vao tests/test_audio_denoise.py
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
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_audio_denoise.py -k "separate_vocals or denoise_file" -v`
Expected: FAIL — `ImportError: cannot import name 'separate_vocals'`

- [ ] **Step 3: Thêm `separate_vocals` + `denoise_file`**

```python
def separate_vocals(in_path, separator, tmp_wav):
    import torchaudio
    _, stems = separator.separate_audio_file(str(in_path))
    vocals = stems["vocals"]          # chon theo TEN, khong hardcode index
    torchaudio.save(str(tmp_wav), vocals.cpu(), separator.samplerate)
    return tmp_wav


def denoise_file(in_path, out_path, separator, ffmpeg_exe) -> bool:
    from pathlib import Path
    tmp = Path(out_path).with_suffix(".vocals.wav")
    try:
        separate_vocals(in_path, separator, tmp)
        ok = _post_process(tmp, out_path, ffmpeg_exe)
        return ok
    except Exception as e:
        print(f"[denoise] loi {in_path}: {e}")
        return False
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_audio_denoise.py -k "separate_vocals or denoise_file" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add audio_denoise.py tests/test_audio_denoise.py
git commit -m "feat(denoise): separate_vocals (by name) + denoise_file orchestration"
```

---

### Task 4: `denoise_batch` (load 1 lần, abort/skip)

**Files:**
- Modify: `audio_denoise.py`
- Test: `tests/test_audio_denoise.py`

**Interfaces:**
- Consumes: `load_demucs`, `denoise_file`.
- Produces:
  - `denoise_batch(in_paths, out_dir, ffmpeg_exe, console=None) -> dict` — `load_demucs()` MỘT lần; lỗi load → in cảnh báo + trả `{}` (abort). Thành công → `out_dir.mkdir(parents=True, exist_ok=True)`; mỗi file gọi `denoise_file` ghi `out_dir/<stem>.wav`; chỉ map `{in_path: out_path}` cho file OK.

- [ ] **Step 1: Viết test thất bại**

```python
# them vao tests/test_audio_denoise.py
def test_denoise_batch_loads_once_and_maps(monkeypatch, tmp_path):
    import audio_denoise
    calls = {"load": 0}
    monkeypatch.setattr(audio_denoise, "load_demucs",
                        lambda: calls.__setitem__("load", calls["load"] + 1) or object())
    monkeypatch.setattr(audio_denoise, "denoise_file",
                        lambda i, o, s, f: True)
    ins = [tmp_path / "a.mp3", tmp_path / "b.mp3"]
    out = audio_denoise.denoise_batch(ins, tmp_path / "den", "ffmpeg")
    assert calls["load"] == 1
    assert set(out.keys()) == set(ins)
    assert out[ins[0]] == tmp_path / "den" / "a.wav"


def test_denoise_batch_abort_on_load_fail(monkeypatch, tmp_path):
    import audio_denoise

    def boom():
        raise RuntimeError("no gpu")

    monkeypatch.setattr(audio_denoise, "load_demucs", boom)
    called = {"file": 0}
    monkeypatch.setattr(audio_denoise, "denoise_file",
                        lambda *a, **k: called.__setitem__("file", 1) or True)
    out = audio_denoise.denoise_batch([tmp_path / "a.mp3"], tmp_path / "den", "ffmpeg")
    assert out == {}
    assert called["file"] == 0      # khong cat tho file nao


def test_denoise_batch_skips_failed_file(monkeypatch, tmp_path):
    import audio_denoise
    monkeypatch.setattr(audio_denoise, "load_demucs", lambda: object())

    def half(i, o, s, f):
        return i.name == "a.mp3"

    monkeypatch.setattr(audio_denoise, "denoise_file", half)
    ins = [tmp_path / "a.mp3", tmp_path / "b.mp3"]
    out = audio_denoise.denoise_batch(ins, tmp_path / "den", "ffmpeg")
    assert (tmp_path / "a.mp3") in out
    assert (tmp_path / "b.mp3") not in out
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_audio_denoise.py -k denoise_batch -v`
Expected: FAIL — `ImportError: cannot import name 'denoise_batch'`

- [ ] **Step 3: Thêm `denoise_batch`**

```python
def denoise_batch(in_paths, out_dir, ffmpeg_exe, console=None):
    from pathlib import Path
    out_dir = Path(out_dir)

    def log(msg):
        if console:
            console.print(msg)
        else:
            print(msg)

    try:
        separator = load_demucs()
    except Exception as e:
        log(f"[red]✗ Demucs khong kha dung ({e}) — bo qua denoise, KHONG cat tho.[/red]")
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    for p in in_paths:
        p = Path(p)
        out_path = out_dir / f"{p.stem}.wav"
        if denoise_file(p, out_path, separator, ffmpeg_exe):
            result[p] = out_path
        else:
            log(f"[yellow]⚠ Skip (denoise loi): {p.name}[/yellow]")
    return result
```

- [ ] **Step 4: Chạy test — phải PASS (toàn bộ module)**

Run: `pytest tests/test_audio_denoise.py -v`
Expected: PASS toàn bộ

- [ ] **Step 5: Commit**

```bash
git add audio_denoise.py tests/test_audio_denoise.py
git commit -m "feat(denoise): denoise_batch — load model once, abort on fail, skip per-file"
```

---

### Task 5: Tích hợp `split_one` — nhận `source`, output `.wav`, field `denoise`

**Files:**
- Modify: `split_audio.py` (`split_one` chữ ký + `mp3_path` + tên file segment + manifest)
- Test: `tests/test_split_audio_clean.py`

**Interfaces:**
- Consumes: (không) — chỉ đổi `split_one`.
- Produces: `split_one(json_path, output_root, ffmpeg_exe, vad_model=None, console=None, inspect=False, breath_gap=0.2, source=None)` — nếu `source` có: dùng làm nguồn audio, segment kế thừa đuôi của `source` (`.wav`), manifest entry thêm `"denoise": True`. Nếu `source=None`: hành vi cũ (mp3, không field denoise).

- [ ] **Step 1: Viết test thất bại**

```python
# them vao tests/test_split_audio_clean.py
import json
from pathlib import Path
import subprocess
import split_audio


def _make_wav(path: Path, ffmpeg_exe: str, seconds: float = 3.0):
    # tao file wav test bang ffmpeg (sine 22050 mono)
    subprocess.run([ffmpeg_exe, "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"sine=frequency=220:duration={seconds}:sample_rate=22050",
                    "-ac", "1", "-c:a", "pcm_s16le", str(path)], check=True)


def test_split_one_uses_source_and_wav_output(tmp_path):
    ffmpeg_exe = split_audio.get_ffmpeg_exe()
    src = tmp_path / "clean.wav"
    _make_wav(src, ffmpeg_exe, seconds=4.0)
    entries = [
        {"start": "00:00:00.000", "end": "00:00:01.500", "text": "Xin chao cac ban."},
        {"start": "00:00:02.000", "end": "00:00:03.500", "text": "Hen gap lai nhe."},
    ]
    jpath = tmp_path / "clip.json"
    jpath.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")

    ok, err = split_audio.split_one(
        jpath, Path("segs"), ffmpeg_exe, vad_model=None,
        source=src,
    )
    assert err == 0 and ok >= 1
    seg_dir = Path(__file__).resolve().parent.parent / "segs" / "clip"
    wavs = list(seg_dir.glob("*.wav"))
    assert wavs, "segment phai la .wav"
    manifest = json.loads((seg_dir / "manifest.json").read_text(encoding="utf-8"))
    assert all(e.get("denoise") is True for e in manifest)
    # cleanup
    import shutil
    shutil.rmtree(Path(__file__).resolve().parent.parent / "segs", ignore_errors=True)
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_split_audio_clean.py -k source_and_wav -v`
Expected: FAIL — `TypeError: split_one() got an unexpected keyword argument 'source'`

- [ ] **Step 3: Sửa `split_one` trong `split_audio.py`**

Đổi chữ ký (dòng 183-191) thêm `source=None`:
```python
def split_one(
    json_path: Path,
    output_root: Path,
    ffmpeg_exe: str,
    vad_model=None,
    console=None,
    inspect: bool = False,
    breath_gap: float = 0.2,
    source: Path | None = None,
) -> tuple[int, int]:
```

Thay (dòng 199):
```python
    mp3_path = json_path.with_suffix(".mp3")
```
bằng:
```python
    src_path = source if source is not None else json_path.with_suffix(".mp3")
    seg_ext  = src_path.suffix          # ".wav" khi da denoise, ".mp3" khi khong
```
Và đổi mọi tham chiếu `mp3_path` còn lại trong hàm thành `src_path`
(dòng 200 kiểm tồn tại, dòng 217 `run_vad(str(mp3_path.resolve())...)`, dòng 287 `-i str(mp3_path)`).

Đổi tên file segment (dòng 267):
```python
        filename    = f"{i:04d}_{label}{seg_ext}"
```

Thêm field `denoise` vào manifest.append (dòng 297-304):
```python
        entry = {
            "file":        filename,
            "start":       start,
            "end":         end,
            "duration_ms": duration_ms,
            "entries":     len(ents),
            "text":        text,
        }
        if source is not None:
            entry["denoise"] = True
        manifest.append(entry)
```

- [ ] **Step 4: Chạy test — phải PASS (gồm test cũ)**

Run: `pytest tests/test_split_audio_clean.py -v`
Expected: PASS (test mới + test cũ vẫn xanh)

- [ ] **Step 5: Commit**

```bash
git add split_audio.py tests/test_split_audio_clean.py
git commit -m "feat(split): split_one accepts denoised source, outputs wav + denoise flag"
```

---

### Task 6: Nối `denoise_batch` vào `split_audio.main()` + menu

**Files:**
- Modify: `split_audio.py` (`main()` — vòng lặp dòng 421-447, thông báo)
- Test: `tests/test_split_audio_inspect.py`

**Interfaces:**
- Consumes: `audio_denoise.denoise_batch`, `split_one(source=...)`.
- Produces: `main()` chạy `denoise_batch` cho tất cả mp3 nguồn (theo `json_files`) TRƯỚC vòng split; chỉ split json có cleaned source; in thông báo "Đang lọc nhiễu (Demucs)…". Trích logic chọn-file-để-split thành hàm thuần `plan_split_sources(json_files, denoise_map) -> list[tuple[Path, Path]]` để test không cần Demucs.

- [ ] **Step 1: Viết test thất bại**

```python
# them vao tests/test_split_audio_inspect.py
from pathlib import Path
import split_audio


def test_plan_split_sources_filters_to_denoised():
    j1, j2 = Path("a.json"), Path("b.json")
    den = {Path("a.mp3"): Path("den/a.wav")}   # chi a.mp3 denoise OK
    # map theo mp3 sibling cua json
    pairs = split_audio.plan_split_sources([j1, j2], den)
    assert pairs == [(j1, Path("den/a.wav"))]   # b bi bo vi denoise loi
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `pytest tests/test_split_audio_inspect.py -k plan_split_sources -v`
Expected: FAIL — `AttributeError: module 'split_audio' has no attribute 'plan_split_sources'`

- [ ] **Step 3: Thêm `plan_split_sources` + nối vào `main()`**

Thêm hàm thuần (cạnh `collect_json_files`):
```python
def plan_split_sources(json_files, denoise_map):
    """Map json -> cleaned wav; bo json nao khong co source da denoise."""
    pairs = []
    for jf in json_files:
        src = denoise_map.get(jf.with_suffix(".mp3"))
        if src is not None:
            pairs.append((jf, src))
    return pairs
```

Trong `main()`, SAU khi có `json_files` và `ffmpeg_exe`, TRƯỚC vòng `with Progress(...)`:
```python
    from audio_denoise import denoise_batch
    mp3_inputs = [jf.with_suffix(".mp3") for jf in json_files]
    denoise_dir = json_files[0].parent / "audio_denoised"
    console.print("[bold]Đang lọc nhiễu (Demucs)…[/bold]")
    denoise_map = denoise_batch(mp3_inputs, denoise_dir, ffmpeg_exe, console=console)
    split_pairs = plan_split_sources(json_files, denoise_map)
    if not split_pairs:
        console.print("[red]Không có file nào denoise thành công — dừng.[/red]")
        return
```

Đổi vòng split (dòng 427 `for jf in json_files:`) sang dùng `split_pairs`:
```python
        for jf, src in split_pairs:
            out_root = Path(args.output) if args.output else jf.parent / "segments"
            task = progress.add_task(jf.stem[:50], total=None)

            ok, err = split_one(
                json_path=jf,
                output_root=out_root,
                ffmpeg_exe=ffmpeg_exe,
                vad_model=vad_model,
                console=console if args.inspect else None,
                inspect=args.inspect,
                breath_gap=args.breath_gap,
                source=src,
            )
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `pytest tests/test_split_audio_inspect.py -v`
Expected: PASS

- [ ] **Step 5: Chạy toàn bộ test — không vỡ gì**

Run: `pytest -q`
Expected: PASS toàn bộ (các test cũ vẫn xanh)

- [ ] **Step 6: Commit**

```bash
git add split_audio.py tests/test_split_audio_inspect.py
git commit -m "feat(split): run denoise_batch before split, cut only denoised sources"
```

---

## Self-Review

**Spec coverage:**
- Module `audio_denoise.py` (load/separate/post/batch) → Task 1-4. ✓
- Một đường Demucs, không tier/fallback → Task 1-4 (không có nhánh light); Global Constraints. ✓
- Output WAV 22050 mono pcm_s16le → Task 2 (`_post_process`) + Task 5 (đuôi `.wav`). ✓
- loudnorm -23, highpass 80, từ hằng số → Task 1 (hằng số) + Task 2 (`_post_filter`, test chống regression). ✓
- Lấy stem theo TÊN, không hardcode index → Task 3 (`separate_vocals`) + test `test_separate_vocals_picks_named_stem`. ✓
- Fail loud không fallback → Task 4 (`denoise_batch` abort) + test `abort_on_load_fail`. ✓
- Skip file lỗi → Task 4 test `skips_failed_file` + Task 6 `plan_split_sources`. ✓
- Lọc cả file rồi cắt; load model 1 lần → Task 4 + Task 6 (denoise_batch trước vòng split). ✓
- Bảo toàn timing (không re-transcribe) → không đổi transcript; cắt theo `-ss/-to` giây như cũ. ✓
- split_one nhận source, manifest field denoise → Task 5 + test. ✓
- Backward-compat (`source=None` → hành vi cũ; manifest cũ đọc được) → Task 5 (nhánh `source is None`). ✓
- Lazy-install demucs → Task 1 (`load_demucs`). ✓
- Không thêm `--mode`/`--no-denoise`/menu bật-tắt → Task 6 (luôn denoise, chỉ thông báo). ✓

**Placeholder scan:** không có TBD/TODO; mọi step có code/command cụ thể.

**Type consistency:** `load_demucs() -> Separator`, `_pick_device(device, cuda_available)`, `_post_filter() -> str`, `_post_process(in_wav, out_path, ffmpeg_exe) -> bool`, `separate_vocals(in_path, separator, tmp_wav) -> Path`, `denoise_file(in_path, out_path, separator, ffmpeg_exe) -> bool`, `denoise_batch(in_paths, out_dir, ffmpeg_exe, console=None) -> dict`, `split_one(..., source=None)`, `plan_split_sources(json_files, denoise_map) -> list[tuple]` — nhất quán giữa các task. Hằng số dùng cùng tên giữa Task 1/2.
