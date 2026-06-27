# split_audio Tự-detect Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cho `split_audio.py` tự nhận biết layout `downloads/{audio,transcript,transcript_clean}` của `main.py`, tự ghép cặp (transcript, mp3) và cắt segment — không cần gom file thủ công.

**Architecture:** Thêm vào `split_audio.py` các hàm thuần `_base_name`, `_segments_root`, `discover_split_jobs`; sửa `load_entries_for_split` xử lý `*.clean.json` trực tiếp; `split_one` đặt tên thư mục segment theo base; `main()` thay menu+`collect_json_files`+`plan_split_sources` bằng `discover_split_jobs`. Denoise và cách cắt giữ nguyên.

**Tech Stack:** Python 3.10+, stdlib (`pathlib`, `json`), pytest. Không thêm dependency.

## Global Constraints

- Python 3.10+ (`str | None`, `list[dict]`).
- CHỈ sửa `split_audio.py` (không thêm module mới).
- Ưu tiên `transcript_clean/<base>.clean.json` (giữ `type==dialogue`); fallback `transcript/<base>.json`.
- `discover_split_jobs` trả `list[tuple[Path, Path]]` = (transcript, mp3); **bỏ** job thiếu mp3 hoặc thiếu transcript.
- **KHÔNG** đụng `audio_denoise.denoise_batch` (giữ cơ chế per-parent → denoise output `downloads/audio/audio_denoised/<base>.wav`).
- segments → `downloads/segments/<base>/` (base đã bỏ đuôi `.clean`).
- `--inspect` truyền **mp3 gốc** làm `source` (dry-run, không denoise).
- Tương thích layout phẳng cũ (json + mp3 cùng thư mục).
- Bỏ menu questionary; bỏ `collect_json_files`, `plan_split_sources`.
- Comment/code/commit tiếng Việt-không-dấu hoặc tiếng Anh; KHÔNG đưa xưng hô anh/em vào code.
- Mọi hàm mới là pure (không I/O mạng); `discover_split_jobs` chỉ đọc filesystem (stat/glob).

---

### Task 1: Path helpers `_base_name` + `_segments_root`; dùng `_base_name` trong `split_one`

**Files:**
- Modify: `split_audio.py` (thêm 2 hàm gần `collect_json_files`; sửa `seg_dir` trong `split_one`, dòng ~256)
- Test: `tests/test_split_audio_discover.py`

**Interfaces:**
- Produces:
  - `_base_name(json_path: Path) -> str` — `"<base>.clean.json"` → `"<base>"`; ngược lại `json_path.stem`.
  - `_segments_root(mp3: Path) -> Path` — `mp3.parent.parent` nếu `mp3.parent.name == "audio"`, else `mp3.parent`.
  - `split_one` đặt tên thư mục segment bằng `_base_name(json_path)` thay `json_path.stem`.

- [ ] **Step 1: Viết test thất bại**

```python
# tests/test_split_audio_discover.py
from pathlib import Path
import split_audio


def test_base_name():
    assert split_audio._base_name(Path("x/a.clean.json")) == "a"
    assert split_audio._base_name(Path("x/a.json")) == "a"
    assert split_audio._base_name(Path("x/My Video.clean.json")) == "My Video"


def test_segments_root_organized_vs_flat():
    assert split_audio._segments_root(Path("downloads/audio/a.mp3")) == Path("downloads")
    assert split_audio._segments_root(Path("flat/a.mp3")) == Path("flat")
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `python -m pytest tests/test_split_audio_discover.py -k "base_name or segments_root" -v`
Expected: FAIL — `AttributeError: module 'split_audio' has no attribute '_base_name'`

- [ ] **Step 3: Thêm 2 hàm + sửa `split_one`**

Thêm gần `collect_json_files` (sau dòng ~331):
```python
def _base_name(json_path: Path) -> str:
    name = json_path.name
    if name.endswith(".clean.json"):
        return name[: -len(".clean.json")]
    return json_path.stem


def _segments_root(mp3: Path) -> Path:
    # organized: downloads/audio/x.mp3 -> downloads ; flat: dir/x.mp3 -> dir
    if mp3.parent.name == "audio":
        return mp3.parent.parent
    return mp3.parent
```

Trong `split_one`, sửa dòng tạo `seg_dir` (hiện `seg_dir = Path(__file__).parent / output_root / json_path.stem`):
```python
    seg_dir = Path(__file__).parent / output_root / _base_name(json_path)
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `python -m pytest tests/test_split_audio_discover.py -k "base_name or segments_root" -v`
Expected: PASS

- [ ] **Step 5: Chạy full suite — không vỡ**

Run: `python -m pytest -q`
Expected: PASS toàn bộ (test cũ vẫn xanh — với json thô `_base_name == stem` nên không đổi hành vi)

- [ ] **Step 6: Commit**

```bash
git add split_audio.py tests/test_split_audio_discover.py
git commit -m "feat(split): path helpers _base_name/_segments_root; segment folder by base"
```

---

### Task 2: Sửa `load_entries_for_split` lọc dialogue khi đưa `*.clean.json`

**Files:**
- Modify: `split_audio.py` (`load_entries_for_split`, dòng 173-178)
- Test: `tests/test_split_audio_discover.py`

**Interfaces:**
- Produces: `load_entries_for_split(json_path)` — nếu `json_path.name` kết thúc `.clean.json` → đọc & lọc `type=="dialogue"`; ngược lại giữ logic cũ (tìm sibling `.clean.json` cho layout phẳng, else đọc raw).

- [ ] **Step 1: Viết test thất bại**

```python
# them vao tests/test_split_audio_discover.py
import json


def test_load_entries_clean_json_filters_dialogue(tmp_path):
    cj = tmp_path / "a.clean.json"
    cj.write_text(json.dumps([
        {"start": "00:00:00.000", "end": "00:00:01.000", "text": "xin chao", "type": "dialogue"},
        {"start": "00:00:01.000", "end": "00:00:02.000", "text": "la la la", "type": "music"},
    ]), encoding="utf-8")
    out = split_audio.load_entries_for_split(cj)
    assert len(out) == 1 and out[0]["text"] == "xin chao"


def test_load_entries_raw_json_no_filter(tmp_path):
    rj = tmp_path / "b.json"
    rj.write_text(json.dumps([
        {"start": "00:00:00.000", "end": "00:00:01.000", "text": "cau mot"},
        {"start": "00:00:01.000", "end": "00:00:02.000", "text": "cau hai"},
    ]), encoding="utf-8")
    out = split_audio.load_entries_for_split(rj)
    assert len(out) == 2
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `python -m pytest tests/test_split_audio_discover.py -k load_entries -v`
Expected: FAIL — `test_load_entries_clean_json_filters_dialogue` trả 2 entries (chưa lọc) thay vì 1

- [ ] **Step 3: Sửa `load_entries_for_split`**

Thay toàn bộ hàm (dòng 173-178) bằng:
```python
def load_entries_for_split(json_path: Path) -> list[dict]:
    if json_path.name.endswith(".clean.json"):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return [e for e in data if e.get("type", "dialogue") == "dialogue"]
    clean = json_path.with_suffix(".clean.json")
    if clean.exists():
        data = json.loads(clean.read_text(encoding="utf-8"))
        return [e for e in data if e.get("type", "dialogue") == "dialogue"]
    return json.loads(json_path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `python -m pytest tests/test_split_audio_discover.py -k load_entries -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add split_audio.py tests/test_split_audio_discover.py
git commit -m "fix(split): load_entries_for_split filters dialogue for direct .clean.json"
```

---

### Task 3: `discover_split_jobs`

**Files:**
- Modify: `split_audio.py` (thêm `discover_split_jobs` gần `collect_json_files`)
- Test: `tests/test_split_audio_discover.py`

**Interfaces:**
- Consumes: `_base_name`.
- Produces: `discover_split_jobs(inputs: list[Path]) -> list[tuple[Path, Path]]` — danh sách (transcript, mp3). Rỗng → mặc định `[Path("downloads")]`. Organized (có `audio/` + `transcript_clean/` hoặc `transcript/`): ghép theo base, clean ưu tiên, bỏ base thiếu mp3. Flat: `*.json` (bỏ `manifest.json` và `*.clean.json` làm driver) → `<base>.mp3`. File `.json` đơn: resolve mp3 theo ngữ cảnh.

- [ ] **Step 1: Viết test thất bại**

```python
# them vao tests/test_split_audio_discover.py
def _touch(p: Path, text="[]"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_discover_organized_prefers_clean(tmp_path):
    _touch(tmp_path / "transcript" / "a.json")
    _touch(tmp_path / "transcript_clean" / "a.clean.json")
    (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
    (tmp_path / "audio" / "a.mp3").write_bytes(b"x")
    jobs = split_audio.discover_split_jobs([tmp_path])
    assert jobs == [(tmp_path / "transcript_clean" / "a.clean.json", tmp_path / "audio" / "a.mp3")]


def test_discover_organized_fallback_raw_and_skip_missing_mp3(tmp_path):
    _touch(tmp_path / "transcript" / "b.json")       # co transcript, KHONG co audio/b.mp3 -> bo
    _touch(tmp_path / "transcript" / "c.json")       # co transcript + audio/c.mp3 -> nhan (raw)
    (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
    (tmp_path / "audio" / "c.mp3").write_bytes(b"x")
    jobs = split_audio.discover_split_jobs([tmp_path])
    assert jobs == [(tmp_path / "transcript" / "c.json", tmp_path / "audio" / "c.mp3")]


def test_discover_flat_layout(tmp_path):
    (tmp_path / "d.json").write_text("[]", encoding="utf-8")
    (tmp_path / "d.mp3").write_bytes(b"x")
    (tmp_path / "d.clean.json").write_text("[]", encoding="utf-8")   # KHONG lam driver rieng
    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")  # bo qua
    jobs = split_audio.discover_split_jobs([tmp_path])
    assert jobs == [(tmp_path / "d.json", tmp_path / "d.mp3")]


def test_discover_single_clean_json_resolves_audio(tmp_path):
    _touch(tmp_path / "transcript_clean" / "e.clean.json")
    (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
    (tmp_path / "audio" / "e.mp3").write_bytes(b"x")
    cj = tmp_path / "transcript_clean" / "e.clean.json"
    jobs = split_audio.discover_split_jobs([cj])
    assert jobs == [(cj, tmp_path / "audio" / "e.mp3")]


def test_discover_empty_defaults_to_downloads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _touch(tmp_path / "downloads" / "transcript" / "f.json")
    (tmp_path / "downloads" / "audio").mkdir(parents=True, exist_ok=True)
    (tmp_path / "downloads" / "audio" / "f.mp3").write_bytes(b"x")
    jobs = split_audio.discover_split_jobs([])
    assert len(jobs) == 1
    tr, mp3 = jobs[0]
    assert tr == Path("downloads") / "transcript" / "f.json"
    assert mp3 == Path("downloads") / "audio" / "f.mp3"
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `python -m pytest tests/test_split_audio_discover.py -k discover -v`
Expected: FAIL — `AttributeError: module 'split_audio' has no attribute 'discover_split_jobs'`

- [ ] **Step 3: Thêm `discover_split_jobs`**

```python
def discover_split_jobs(inputs: list[Path]) -> list[tuple[Path, Path]]:
    if not inputs:
        inputs = [Path("downloads")]

    jobs: list[tuple[Path, Path]] = []
    seen: set[tuple[str, str]] = set()

    def add(tr: Path, mp3: Path):
        key = (str(tr), str(mp3))
        if tr.exists() and mp3.exists() and key not in seen:
            seen.add(key)
            jobs.append((tr, mp3))

    def resolve_mp3_for_json(jf: Path) -> Path:
        base = _base_name(jf)
        if jf.parent.name in ("transcript", "transcript_clean") \
                and (jf.parent.parent / "audio").is_dir():
            return jf.parent.parent / "audio" / f"{base}.mp3"
        return jf.with_suffix(".mp3")

    for p in inputs:
        p = Path(p)
        if p.is_file() and p.suffix.lower() == ".json":
            add(p, resolve_mp3_for_json(p))
            continue
        if not p.is_dir():
            continue

        audio = p / "audio"
        traw = p / "transcript"
        tclean = p / "transcript_clean"
        organized = audio.is_dir() and (tclean.is_dir() or traw.is_dir())

        if organized:
            by_base: dict[str, Path] = {}
            if traw.is_dir():
                for jf in sorted(traw.glob("*.json")):
                    if jf.name == "manifest.json" or jf.name.endswith(".clean.json"):
                        continue
                    by_base[_base_name(jf)] = jf
            if tclean.is_dir():
                for jf in sorted(tclean.glob("*.clean.json")):
                    by_base[_base_name(jf)] = jf          # clean ghi de raw
            for base, tr in sorted(by_base.items()):
                add(tr, audio / f"{base}.mp3")
        else:
            for jf in sorted(p.glob("*.json")):
                if jf.name == "manifest.json" or jf.name.endswith(".clean.json"):
                    continue
                add(jf, jf.with_suffix(".mp3"))

    return jobs
```

- [ ] **Step 4: Chạy test — phải PASS**

Run: `python -m pytest tests/test_split_audio_discover.py -k discover -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add split_audio.py tests/test_split_audio_discover.py
git commit -m "feat(split): discover_split_jobs auto-detects organized/flat layout"
```

---

### Task 4: Tích hợp `main()`; bỏ `collect_json_files`/`plan_split_sources`/menu; xóa test cũ

**Files:**
- Modify: `split_audio.py` (`main()` dòng ~398-446; xóa `collect_json_files` dòng 323-331 và `plan_split_sources` dòng 334-341)
- Modify: `tests/test_split_audio_inspect.py` (xóa `test_plan_split_sources_filters_to_denoised`)
- Test: `tests/test_split_audio_discover.py`

**Interfaces:**
- Consumes: `discover_split_jobs`, `_segments_root`, `audio_denoise.denoise_batch`, `split_one(source=...)`.
- Produces: `main()` dùng `discover_split_jobs`; không còn `collect_json_files`/`plan_split_sources`/menu.

- [ ] **Step 1: Viết test thất bại (khẳng định 2 hàm cũ bị bỏ)**

```python
# them vao tests/test_split_audio_discover.py
def test_old_helpers_removed():
    assert not hasattr(split_audio, "plan_split_sources")
    assert not hasattr(split_audio, "collect_json_files")
```

- [ ] **Step 2: Chạy test — phải FAIL**

Run: `python -m pytest tests/test_split_audio_discover.py -k old_helpers_removed -v`
Expected: FAIL (2 hàm vẫn còn)

- [ ] **Step 3: Xóa 2 hàm cũ + viết lại khối chọn file trong `main()`**

Xóa hẳn `collect_json_files` (dòng 323-331) và `plan_split_sources` (dòng 334-341).

Thay khối từ `if args.inputs:` … đến hết phần build `split_pairs` (dòng 398-446) bằng:
```python
    jobs = discover_split_jobs([Path(p) for p in args.inputs])
    if not jobs:
        console.print("[red]Không tìm thấy cặp transcript + audio nào trong downloads/.[/red]")
        return

    denoise_map: dict = {}
    if not args.inspect:
        from audio_denoise import denoise_batch
        console.print("[bold]Đang lọc nhiễu (Demucs)…[/bold]")
        denoise_map = denoise_batch([mp3 for _, mp3 in jobs], ffmpeg_exe, console=console)
        if not denoise_map:
            console.print("[red]Không có file nào denoise thành công — dừng.[/red]")
            return

    total_ok = total_err = 0

    with Progress(SpinnerColumn(), TextColumn("[bold white]{task.description}"),
                  BarColumn(), MofNCompleteColumn(),
                  console=console, transient=False) as progress:

        for tr, mp3 in jobs:
            src = mp3 if args.inspect else denoise_map.get(mp3)
            if src is None:
                continue
            out_root = Path(args.output) if args.output else _segments_root(mp3) / "segments"
            task = progress.add_task(_base_name(tr)[:50], total=None)

            ok, err = split_one(
                json_path=tr,
                output_root=out_root,
                ffmpeg_exe=ffmpeg_exe,
                vad_model=vad_model,
                console=console if args.inspect else None,
                inspect=args.inspect,
                breath_gap=args.breath_gap,
                source=src,
            )
            total_ok  += ok
            total_err += err

            status = f"[green]✓ {ok} đoạn[/green]"
            if err:
                status += f" [red]✗ {err} lỗi[/red]"
            progress.update(task, completed=1, total=1,
                            description=f"{_base_name(tr)[:40]} — {status}")
```

Đồng thời xóa import `questionary` không còn dùng trong `main()` (khối `try: import questionary …` đã nằm trong đoạn bị thay) và phần `if not json_files:` cũ (đã thay bằng `if not jobs:`).

Trong `tests/test_split_audio_inspect.py`, xóa hàm `test_plan_split_sources_filters_to_denoised` (và import không còn dùng nếu có).

- [ ] **Step 4: Chạy test — phải PASS**

Run: `python -m pytest tests/test_split_audio_discover.py -k old_helpers_removed -v`
Expected: PASS

- [ ] **Step 5: Chạy full suite — không vỡ**

Run: `python -m pytest -q`
Expected: PASS toàn bộ (đã xóa test `plan_split_sources`; các test khác xanh)

- [ ] **Step 6: Smoke import (main parse không lỗi)**

Run: `python -c "import split_audio; import argparse; print('import OK', callable(split_audio.main))"`
Expected: in `import OK True`

- [ ] **Step 7: Commit**

```bash
git add split_audio.py tests/test_split_audio_discover.py tests/test_split_audio_inspect.py
git commit -m "feat(split): wire discover_split_jobs into main, drop menu/collect/plan_split_sources"
```

---

## Self-Review

**Spec coverage:**
- Component 1 `discover_split_jobs` → Task 3. ✓
- Component 2 `load_entries_for_split` lọc dialogue → Task 2. ✓
- Component 3 tên thư mục segment theo base → Task 1 (`_base_name` + sửa `split_one`). ✓
- `_segments_root` → Task 1. ✓
- main() tích hợp (inspect dùng mp3 gốc; denoise; output `_segments_root`) → Task 4. ✓
- Bỏ menu/`collect_json_files`/`plan_split_sources` + xóa test cũ → Task 4. ✓
- Ưu tiên transcript_clean, fallback raw, bỏ thiếu mp3 → Task 3 logic + test. ✓
- Tương thích flat layout → Task 3 (nhánh flat) + test `test_discover_flat_layout`. ✓
- Không đụng `denoise_batch` (per-parent output) → không task nào sửa `audio_denoise.py`. ✓

**Placeholder scan:** không có TBD/TODO; mọi step có code/command cụ thể.

**Type consistency:** `_base_name(json_path)->str`, `_segments_root(mp3)->Path`, `discover_split_jobs(inputs)->list[tuple[Path,Path]]`, `load_entries_for_split(json_path)->list[dict]`, `split_one(..., source=)` — nhất quán giữa Task 1/2/3/4. `main()` lặp theo `jobs` giữ `mp3` gốc cho `_segments_root` và `denoise_map.get(mp3)`.
