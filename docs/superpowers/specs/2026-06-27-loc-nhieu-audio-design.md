# Thiết kế: Lọc nhiễu tín hiệu audio cho dataset (audio-denoise v1)

**Ngày:** 2026-06-27
**Trạng thái:** Chờ duyệt thiết kế → lập kế hoạch
**Liên quan:** `2026-06-24-loc-nhieu-v2-design.md` (lọc nhiễu *transcript* — bổ trợ, không trùng)

## Bối cảnh

Pipeline hiện tại: crawl YouTube → tải mp3 (`main.py`) → transcribe (`transcribe_backends.py`)
→ lọc nhiễu *transcript* gán `type` dialogue/music/noise/sound (`clean_transcript.py`,
ghi `.clean.json`) → cắt segment theo câu bằng WhisperX + Silero VAD (`split_audio.py`).

`split_audio.py` cắt bằng `ffmpeg -c copy` (stream copy, **không** lọc tín hiệu) và
`load_entries_for_split` chỉ giữ entry `type == "dialogue"` (`split_audio.py:177`).
Tức là audio segment hiện **chưa hề được lọc nhiễu tín hiệu** — chỉ copy nguyên từ mp3 gốc.

Mục tiêu: làm **dataset TTS/ASR** từ giọng nói tiếng Việt → cần giọng sạch, đồng đều,
ít nhạc nền, chuẩn hóa âm lượng. Môi trường chạy: **server headless có GPU 24GB VRAM**.

## Phân biệt "lọc nhiễu" (quan trọng)

| Loại | Ở đâu | Việc gì |
|---|---|---|
| Lọc nhiễu **transcript** (text) | `clean_transcript.py` (đã có spec v2) | bỏ hallucination/lời nhạc/câu lặp ở mức *nội dung* |
| Lọc nhiễu **audio** (tín hiệu) — **spec này** | `audio_denoise.py` (mới) | làm sạch *tín hiệu*: tách giọng khỏi nhạc/ồn nền |

Hai cơ chế **bổ trợ**, không trùng: transcript-clean loại đoạn nhạc/hát khỏi *danh sách cắt*;
audio-denoise làm sạch *tín hiệu* phần thoại được giữ lại.

## Quyết định đã chốt (brainstorming)

| Vấn đề | Lựa chọn |
|---|---|
| Loại lọc | Lọc nhiễu **tín hiệu audio** (không phải transcript). |
| Mục đích | Dataset training TTS/ASR. |
| Cách xử lý | **Một đường DUY NHẤT: Demucs tách vocal cho MỌI file.** Không tier nhẹ, không router, không cờ tắt. Có GPU 24GB nên ưu tiên chất lượng & đồng đều tối đa. |
| Định dạng output | WAV PCM 16-bit (`pcm_s16le`), **22050 Hz**, **mono** — **một dạng output duy nhất**. |
| Chuẩn hóa âm lượng | `loudnorm` target **-23 LUFS** (EBU R128). |
| Nhạc nền **có lời** | Không xử lý ở tầng audio (Demucs không tách được thoại khỏi giọng hát) → cậy transcript-clean đánh `music` để loại. |

> **Một đường, một dạng output.** Bỏ hẳn `--no-denoise` / cắt-thô: mọi segment đều cắt từ
> audio đã Demucs, đều là WAV 22050 mono. Không tồn tại đường nào sinh ra mp3 thô song song
> (tránh trộn hai loại artifact vào cùng dataset).

### Vì sao Demucs cho mọi file (kể cả file không có nhạc nền)
- File **không** nhạc nền (podcast/phỏng vấn): Demucs vẫn tách "vocals" = giọng người,
  đẩy ồn nền sang stem khác → giọng sạch hơn, **không hại**.
- Mọi file qua **đúng một** pipeline biến đổi → dataset đồng đều tuyệt đối.
- denoise+split là bước chạy **sau**, riêng với transcribe → Demucs độc chiếm GPU,
  **không tranh VRAM** với WhisperX local. 24GB thừa sức cho `htdemucs` (chỉ cần ~vài GB).

## Nguyên tắc xuyên suốt

- **Bảo toàn trục thời gian**: chỉ dùng phép biến đổi giữ nguyên độ dài-theo-giây
  (Demucs, highpass, loudnorm, resample) → timestamps từ transcript gốc vẫn khớp file đã lọc
  → **không re-transcribe**. (Loại `silenceremove`/trim-edge vì nó đổi timing.)
- **Lọc CẢ FILE trước, rồi mới cắt** — không cắt-rồi-lọc-từng-mảnh.
- **Không ghi đè** mp3 gốc (giữ để so sánh/rollback).
- Mọi tham số để thành **hằng số module** ở đầu `audio_denoise.py`; mọi chuỗi lệnh ffmpeg
  được **nối từ các hằng số đó** (không hardcode trong code — các lệnh mẫu dưới đây chỉ minh họa).
- Lazy-install Demucs theo pattern `load_vad_model` (`split_audio.py:57-64`) — không bắt buộc
  bỏ vào `requirements.txt` cứng.
- **Fail loud, không fallback**: Demucs là đường duy nhất; khi không khả dụng thì báo lỗi rõ
  chứ không lén cắt thô bằng đường khác (xem Error handling).

### Vì sao lọc-cả-file-rồi-cắt (đã verify primary-source)
Demucs `apply_model(model, mix, split=True, overlap=0.25)` tự chia file dài thành cửa sổ
`model.segment` (~7.8s với weights pretrained của `htdemucs`) overlap 25% **bên trong**.
Nếu cắt trước thành segment ngắn (1-15s) rồi lọc từng mảnh: (a) phải inference N lần thay vì 1
(chậm, mất batching); (b) segment < 7.8s thiếu context → tách vocal kém; (c) mỗi segment qua
overlap/biên khác nhau → **dataset không đồng đều**. ⇒ denoise(full) → file sạch → split cắt.

## Component 1 — Module mới `audio_denoise.py`

Hàm public:

```
load_demucs(device: str | None = None) -> object
    # Separator/get_model("htdemucs") đặt .eval().to(device); device = "cuda" nếu có,
    # ngược lại "cpu" (+ cảnh báo chậm). Lazy-install demucs nếu thiếu.
    # KHÔNG nuốt lỗi: import/load thất bại -> raise (caller xử lý fail loud).

denoise_file(in_path: Path, out_path: Path, model, ffmpeg_exe: str) -> bool
    # Demucs tách vocal + post-process (Component 2). Ghi WAV 22050 Hz mono pcm_s16le ra out_path.
    # Lỗi xử lý trên file này -> log + trả False (caller skip file, KHÔNG cắt thô).

denoise_batch(in_paths: list[Path], out_dir: Path, ffmpeg_exe: str,
              console=None) -> dict[Path, Path]
    # load_demucs MỘT lần. Nếu load thất bại -> báo lỗi rõ + abort (không denoise file nào,
    # trả map rỗng). Thành công -> loop denoise_file; chỉ map {in_path: out_path}
    # cho file denoise OK (file lỗi vắng khỏi map).
```

## Component 2 — Đường Demucs (đường duy nhất)

**Đường chính (đã chốt khi triển khai):** dùng API gốc cấp thấp
`demucs.pretrained.get_model("htdemucs")` + `demucs.apply.apply_model(..., overlap=0.25)` +
`demucs.audio.AudioFile(...).read(...)` / `demucs.audio.save_audio(...)`. Lý do: module
`demucs.api.Separator` **không có trong bản PyPI demucs 4.0.1** (kiểm chứng trực tiếp trên
máy chạy: thư mục package thiếu `api.py`), nên không dùng được. Đường low-level này có trong
**mọi** bản demucs 4.x.
- Đọc audio bằng `AudioFile(...).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)`
  (tự decode mp3 qua ffmpeg nội bộ, resample về 44100/stereo).
- Chuẩn hóa theo `demucs/separate.py` (`(wav-ref.mean())/ref.std()` rồi nhân/cộng lại) trước/sau `apply_model`.
- Lấy stem bằng `vocals_idx = model.sources.index("vocals")` — **không hardcode index**
  (thứ tự là `['drums','bass','other','vocals']` nhưng vẫn lấy theo tên cho an toàn).
- Ghi WAV tạm bằng `save_audio(vocals, tmp, model.samplerate)`.

Sau khi có stem vocals (stereo, sr gốc ~44100) → ghi WAV tạm → post-process bằng ffmpeg
(tối giản — Demucs đã khử nền nên **không** thêm afftdn). Lệnh **minh họa** (giá trị thật nối
từ hằng số module):
   ```
   ffmpeg -y -i vocals_tmp.wav -af "highpass=f=80:p=2,loudnorm=I=-23:TP=-1.5:LRA=11" \
          -ar 22050 -ac 1 -c:a pcm_s16le out.wav
   ```

Demucs: `demucs>=4.0.0` (MIT license), model `htdemucs` ~80MB tải lần đầu; torch đã có sẵn
qua whisperx. GPU 24GB: dư VRAM, chạy nhanh. (CPU vẫn chạy được nhưng ~5-10× realtime — chỉ
là phương án tình thế, không phải đường thiết kế.)

## Luồng tích hợp (`split_audio.py`)

Thứ tự: `transcribe(gốc) → clean → denoise CẢ FILE → split cắt từ file sạch`.

1. Trong `main()` của `split_audio.py`, **trước** vòng split: gọi `denoise_batch` cho toàn bộ
   file đầu vào → map `json_path → cleaned_wav` (đặt ở subdir mới `audio_denoised/<stem>.wav`,
   sibling của thư mục chứa json). Demucs load 1 lần.
2. **Chỉ split các json có trong map** (file denoise OK). File denoise lỗi → vắng khỏi map →
   **skip** (fail loud, không cắt thô). Không bao giờ cắt từ mp3 chưa lọc.
3. `split_one` nhận thêm tham số `source: Path` = WAV đã denoise (thay cho
   `json_path.with_suffix(".mp3")`). VAD (`_audio_to_tensor` đọc WAV bình thường) và phần cắt
   chạy trên file đã sạch. Output segment luôn `.wav` `pcm_s16le` 22050 mono;
   cắt `-ss/-to` trên PCM **chính xác đến mẫu**, `-c copy` (giữ nguyên codec nguồn WAV).
4. Menu questionary: **không** thêm câu hỏi bật/tắt (luôn denoise) — chỉ in thông báo
   "Đang lọc nhiễu (Demucs)…" và tiến độ.

`main.py` pipeline tự động: **ngoài phạm vi v1** (denoise+split chạy ở `split_audio.py`).

## Phối hợp với transcript-clean

`load_entries_for_split` (`split_audio.py:173-178`) và logic `type == "dialogue"`
(`split_audio.py:177`) **không đổi**. denoise tác động lên *audio*, transcript-clean lên *text*.
Đoạn nhạc-có-lời bị transcript-clean đánh `music` → không nằm trong danh sách cắt → không bị
denoise oan. Không cần re-transcribe trên audio đã sạch (v1).

## Error handling (fail loud, không fallback)

- **Demucs không khả dụng** (không import/lazy-install lỗi/không load được model):
  `denoise_batch` in lỗi rõ ràng và **abort** bước denoise (trả map rỗng) — KHÔNG tự cắt thô
  bằng đường khác. Người dùng sửa môi trường rồi chạy lại. (Server đảm bảo GPU nên hiếm.)
- **Lỗi trên một file cụ thể** (Demucs/ffmpeg fail riêng file đó): log + skip file đó,
  **tiếp tục** các file còn lại; file bị skip không có trong map → split bỏ qua.
- Timestamps lệch: không xảy ra vì mọi phép biến đổi bảo toàn độ dài-theo-giây.
- **Lưu ý thuật ngữ:** cơ chế "fallback VAD → no-VAD" có sẵn trong `split_audio.py:380` là
  của **VAD**, hoàn toàn **không liên quan** denoise; denoise tuyệt đối không fallback.

## Testing (TDD)

- `load_demucs`: mock import lỗi → **raise** (không nuốt lỗi); mock có/không cuda → chọn device đúng.
- Demucs path low-level (nếu dùng): mock `model.sources.index("vocals")` được gọi
  (khẳng định **không** hardcode index).
- Chuỗi lọc ffmpeg được **sinh từ hằng số module** (assert chứa `-23`, `highpass=f=80`,
  `22050`, `pcm_s16le`) — chống regression khi đổi hằng số.
- `denoise_file`: chạy ra WAV; kiểm header = 22050 Hz, mono, pcm_s16le.
- Bảo toàn timing: độ dài-theo-giây file sau denoise == file gốc (sai số < 1 frame).
- `denoise_batch`: load model đúng **1 lần** cho N file (đếm số lần gọi `load_demucs`);
  load thất bại → abort, trả map rỗng (không file nào bị cắt thô); lỗi 1 file → file đó vắng
  trong map, các file khác vẫn có.
- Tích hợp: `split_one(source=...)` cắt từ WAV đã denoise; segment ra `.wav`.

## Hằng số (tunable, đầu `audio_denoise.py`)

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

## Tương thích ngược

- `manifest.json` thêm field **optional** `denoise: true` mỗi entry (đánh dấu segment đã lọc
  bằng Demucs — nay luôn đặt vì mọi segment đều qua denoise); file manifest cũ thiếu field
  vẫn đọc bình thường (chỉ thêm khóa, không đổi khóa cũ).
- `.clean.json` cũ vẫn đọc; logic `type == "dialogue"` không đổi.
- **Thay đổi hành vi có chủ đích:** segment output đổi từ `.mp3` (cắt thô cũ) sang `.wav`
  22050 mono. Segment mp3 đã cắt từ trước vẫn nằm trên đĩa, không bị đụng.
- Dependency: `demucs>=4.0.0` (torch đã có), lazy-install. Cần GPU để chạy đúng thiết kế.

## Ngoài phạm vi (YAGNI)

- **Tier nhẹ / router / `--mode` / `--no-denoise` / fallback ffmpeg light** — đã bỏ:
  chỉ một đường Demucs, một dạng output.
- Re-transcribe trên audio đã denoise.
- `silenceremove` / trim silence đầu-cuối (đổi timing).
- DeepFilterNet, afftdn, arnndn (RNNoise) và mọi denoiser thay thế.
- Tích hợp tự động vào `main.py` pipeline sau crawl.
- Auto-tune tham số.
