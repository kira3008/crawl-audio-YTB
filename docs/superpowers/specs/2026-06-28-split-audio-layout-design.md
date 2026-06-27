# Thiết kế: split_audio tự-detect layout thư mục (v2)

**Ngày:** 2026-06-28
**Trạng thái:** Chờ duyệt thiết kế → lập kế hoạch
**Liên quan:** `2026-06-27-loc-nhieu-audio-design.md` (denoise) — nối tiếp, sửa chỗ split không khớp layout.

## Bối cảnh

`main.py` lưu output tách theo subdir:
- mp3 → `downloads/audio/<base>.mp3` (`AUDIO_SUBDIR`)
- transcript thô → `downloads/transcript/<base>.json` (`TRANSCRIPT_SUBDIR`)
- transcript đã lọc → `downloads/transcript_clean/<base>.clean.json` (`CLEAN_SUBDIR`)

Nhưng `split_audio.py` giả định **mp3 nằm kế bên json** (`jf.with_suffix(".mp3")`,
`split_audio.py:437`) và menu chỉ quét `downloads/*.json` (phẳng). Với layout của `main.py`,
json và mp3 ở 3 thư mục khác nhau → split **không khớp được cặp file** → không cắt được.

Mục tiêu: `split_audio.py` **tự nhận biết** layout `downloads/{audio,transcript,transcript_clean}`,
tự ghép đúng cặp (transcript, mp3) mà không cần gom file thủ công.

## Quyết định đã chốt (brainstorming)

| Vấn đề | Lựa chọn |
|---|---|
| Phạm vi | Chỉ sửa `split_audio.py` standalone tự-detect layout. **Không** tích hợp vào `main.py` (chạy split vẫn là bước riêng, có kiểm soát). |
| Nguồn transcript | **Ưu tiên `transcript_clean/<base>.clean.json`** (chỉ giữ `type==dialogue`); fallback `transcript/<base>.json` nếu thiếu clean. |
| Menu chọn file | **Bỏ** menu questionary (chỉ hợp layout phẳng). Không tham số = xử lý **tất cả** job trong `downloads/`. Vẫn cho truyền path cụ thể để chạy subset. |
| Vị trí denoise output | `downloads/audio/audio_denoised/<base>.wav` — giữ cơ chế **per-parent** của `denoise_batch`, **không sửa lại module denoise**. |
| Vị trí segments | `downloads/segments/<base>/*.wav` (thư mục theo `<base>`, đã bỏ đuôi `.clean`). |

## Component 1 — `discover_split_jobs(inputs) -> list[tuple[Path, Path]]`

Hàm khám phá, trả danh sách cặp `(transcript_path, mp3_path)`.

- `inputs` rỗng → mặc định `[Path("downloads")]`.
- Với mỗi `p` trong inputs:
  - **`p` là file `.json`**: resolve mp3 theo ngữ cảnh — nếu `p.parent.name` ∈ {`transcript`,`transcript_clean`}
    và `p.parent.parent/"audio"` tồn tại → mp3 = `p.parent.parent/"audio"/<base>.mp3`;
    ngược lại mp3 = `p.with_suffix(".mp3")` (sibling, layout phẳng). Nhận nếu mp3 tồn tại.
  - **`p` là thư mục "organized"** (`p/"audio"` là dir VÀ (`p/"transcript_clean"` HOẶC `p/"transcript"`) là dir):
    - Tập base = stem-base của `p/"transcript_clean"/*.clean.json` ∪ `p/"transcript"/*.json`.
    - Mỗi base: transcript = `transcript_clean/<base>.clean.json` nếu tồn tại, else `transcript/<base>.json`;
      mp3 = `p/"audio"/<base>.mp3`. Nhận job khi **đủ cả transcript + mp3**; thiếu → bỏ qua + log.
    - Dedup theo base (mỗi base 1 job, clean thắng raw).
  - **`p` là thư mục phẳng** (không organized): mỗi `<base>.json` (bỏ `manifest.json` và bỏ `*.clean.json`
    khỏi danh sách *driver* để tránh trùng) → mp3 = `<base>.mp3` sibling. Nhận nếu mp3 tồn tại.

Trích base: `"<base>.clean.json"` → bỏ hậu tố `.clean.json`; `"<base>.json"` → `stem`.

## Component 2 — Sửa `load_entries_for_split` (`split_audio.py:173-178`)

Hiện nếu đưa thẳng `<base>.clean.json` thì hàm tìm sibling `<base>.clean.clean.json` (không có)
rồi đọc raw **không lọc** dialogue. Sửa:

```
def load_entries_for_split(json_path):
    if json_path.name.endswith(".clean.json"):
        data = json.loads(json_path.read_text(...))
        return [e for e in data if e.get("type", "dialogue") == "dialogue"]
    clean = json_path.with_suffix(".clean.json")          # layout phang cu
    if clean.exists():
        data = json.loads(clean.read_text(...))
        return [e for e in data if e.get("type", "dialogue") == "dialogue"]
    return json.loads(json_path.read_text(...))
```

## Component 3 — Tên thư mục segment theo base

`split_one` đặt `seg_dir = Path(__file__).parent / output_root / json_path.stem`. Với transcript
`<base>.clean.json`, `stem == "<base>.clean"` → thư mục xấu. Thêm helper `_base_name(json_path)`
(bỏ `.clean`) và dùng nó thay `json_path.stem` cho **tên thư mục segment**. (Không đổi logic khác.)

## Luồng tích hợp `main()`

Thay khối `collect_json_files` + `mp3_inputs` + `plan_split_sources` (`split_audio.py:398-446`):

1. `jobs = discover_split_jobs([Path(p) for p in args.inputs])` (rỗng → `downloads/`).
2. Nếu không có job → in lỗi rõ + return.
3. Nếu **không** `--inspect`: `denoise_map = denoise_batch([mp3 for _, mp3 in jobs], ffmpeg_exe, console=console)`.
   Nếu denoise_map rỗng → in lỗi + return.
4. Vòng lặp **theo `jobs`** (giữ được mp3 gốc): với mỗi `(tr, mp3)`:
   - `src` = `mp3` nếu `--inspect`, ngược lại `denoise_map.get(mp3)`; nếu `src is None` (denoise lỗi) → bỏ qua.
   - `output_root` = `args.output` nếu có, ngược lại `_segments_root(mp3) / "segments"`, trong đó
     `_segments_root(mp3)` = `mp3.parent.parent` nếu `mp3.parent.name == "audio"`
     (organized → `downloads/segments`), else `mp3.parent` (flat → `<dir>/segments`).
   - `split_one(json_path=tr, output_root=output_root, ..., source=src)`.

Bỏ `collect_json_files` và `plan_split_sources` (thay bằng `discover_split_jobs`); bỏ menu questionary.
Giữ `--inspect`/`--no-vad`/`--breath-gap`/`--output`.

## Error handling

- Job thiếu mp3 hoặc thiếu transcript → bỏ qua + log (không raise).
- Không tìm thấy job nào → in thông báo rõ + return (fail loud, không cắt bừa).
- denoise lỗi 1 file → file đó vắng `denoise_map` → split bỏ qua (như cơ chế denoise hiện có).

## Testing (TDD)

- `discover_split_jobs`:
  - organized: ghép đúng (transcript_clean ưu tiên hơn transcript cùng base).
  - organized: base thiếu mp3 trong `audio/` → bỏ.
  - flat: json + mp3 sibling → cặp đúng; `*.clean.json` không bị nhận làm driver thừa.
  - file `.json` trong `transcript_clean/` → resolve mp3 ở `../audio/`.
  - input rỗng → mặc định `downloads/`.
- `load_entries_for_split`: `<base>.clean.json` đưa trực tiếp → chỉ giữ `dialogue`.
- `_base_name`: `<base>.clean.json` → `<base>`; `<base>.json` → `<base>`.

## Tương thích ngược

- Layout phẳng cũ (json + mp3 cùng thư mục) vẫn chạy qua nhánh "flat".
- `split_one(source=...)`, denoise, manifest `denoise:true` không đổi.
- CLI cũ (truyền file/thư mục) vẫn hoạt động; chỉ bỏ menu tương tác.

## Ngoài phạm vi (YAGNI)

- Tích hợp tự động vào `main.py` pipeline sau crawl.
- Menu chọn-từng-file tương tác.
- Re-transcribe / đổi cơ chế denoise.
- Hỗ trợ layout tùy biến ngoài 3 subdir chuẩn của `main.py`.
