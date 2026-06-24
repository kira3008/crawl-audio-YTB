# Thiết kế: Nâng độ chính xác bước lọc nhiễu (v2)

**Ngày:** 2026-06-24
**Trạng thái:** Đã duyệt thiết kế, chờ lập kế hoạch
**Nối tiếp:** `2026-06-24-crawl-video-cai-tien-design.md` (cải tiến #3 — lọc nhiễu)

## Bối cảnh

Bước lọc nhiễu hiện tại (`clean_transcript.py`) gồm:
- Tầng A heuristic: `is_sound_tag` (tag `[...]`/♪), `is_hallucination` (9 câu cứng), lặp giống-hệt-liên-tiếp ≥3 → music.
- Tầng B LLM Groq: `llama-3.1-8b-instant`, prompt đơn giản, phân loại + chuẩn hóa.

Điểm yếu về độ chính xác:
1. Không dùng tín hiệu confidence khách quan từ Whisper (`no_speech_prob`, `avg_logprob`, `compression_ratio`) — schema đang bỏ đi.
2. Bắt lặp chỉ khi giống hệt & liên tiếp → sót lời nhạc lặp gần giống, lặp trong 1 câu, và cụm lặp rải rác toàn bài.
3. LLM model 8b yếu, prompt thiếu ví dụ mẫu.

## Quyết định đã chốt (brainstorming)

| Vấn đề | Lựa chọn |
|---|---|
| Phạm vi | Làm #1 (confidence) + #2 (lặp/nhạc) + #4 (LLM). Bỏ #3 (mở rộng hallucination list). |
| #1 phạm vi backend | Groq luôn lấy metric; local **best-effort** (có thì lấy, không thì bỏ qua, không lỗi). |
| Thiên hướng lọc | **Giữ dialogue (thận trọng)** — khi không chắc thì để nguyên dialogue. |

## Nguyên tắc xuyên suốt: thận trọng

- Mọi heuristic chỉ **hạ cấp** (`dialogue → music/noise`) khi vượt **ngưỡng chặt**; không bao giờ tự ý nâng cấp hay đụng vào entry đã là `sound`.
- Tầng confidence chỉ chạy khi entry **có** metric; thiếu metric → bỏ qua hoàn toàn (không đổi gì).
- LLM được chỉ thị rõ: "không chắc → giữ dialogue".
- Mọi ngưỡng để thành **hằng số module** ở đầu `clean_transcript.py` cho dễ chỉnh.

## Schema (mở rộng tương thích ngược)

Entry được thêm các field **tùy chọn** (chỉ có khi transcription cung cấp):

```json
{ "start","end","text","words":[...],
  "no_speech_prob": float?, "avg_logprob": float?, "compression_ratio": float? }
```

- Groq `verbose_json` luôn có 3 field này → `groq_response_to_entries` copy vào.
- Local WhisperX thường KHÔNG lộ ra (bị cắt khi batch) → `transcribe_local` best-effort: có field thì copy, không có thì bỏ.
- `split_audio.py` không đọc các field này → không ảnh hưởng. Entry cũ không có field → mọi thứ chạy như cũ.

## Component 1 — Confidence filter (#1)

Trong `clean_transcript.py`, thêm hằng số + hàm:

```
NO_SPEECH_MAX   = 0.6
COMPRESSION_MAX = 2.4
LOGPROB_MIN     = -1.0
LOGPROB_NOSPEECH_COMBO = 0.4
```

`confidence_is_noise(entry) -> bool`: trả `True` (đánh `noise`) khi entry **có** metric VÀ thỏa BẤT KỲ:
- `no_speech_prob >= NO_SPEECH_MAX`, hoặc
- `compression_ratio >= COMPRESSION_MAX`, hoặc
- `avg_logprob <= LOGPROB_MIN` VÀ `no_speech_prob >= LOGPROB_NOSPEECH_COMBO`.

Nếu entry thiếu bất kỳ metric cần thiết cho một điều kiện → điều kiện đó bị bỏ (coi như không thỏa). Thiếu cả 3 → hàm trả `False` (bỏ qua, giữ nguyên).

## Component 2 — Bắt lặp/lời nhạc tốt hơn (#2)

3 bộ phát hiện, pure Python (`re`, `difflib`, `collections`), chỉ hạ `dialogue → music`:

1. **`is_repetitive_text(text) -> bool`** — lặp trong 1 câu: tách token; nếu số token ≥ `REPEAT_MIN_TOKENS` (vd 4) và tỉ lệ token unique ≤ `REPEAT_UNIQUE_RATIO` (vd 0.5) → `True` (vd "la la la la", "yeah yeah yeah").
2. **Near-duplicate liên tiếp** — thay so-sánh-giống-hệt bằng `difflib.SequenceMatcher(None, a, b).ratio() >= SIM_THRESHOLD` (vd 0.85) trên text đã `_norm_key`; chuỗi ≥ `repeat_threshold` câu gần-giống liên tiếp → music.
3. **Tần suất toàn bài** — đếm `_norm_key(text)` trên cả transcript; cụm có số từ trong `[1, GLOBAL_PHRASE_MAX_WORDS]` (vd ≤ 12) và xuất hiện ≥ `GLOBAL_FREQ_MIN` (vd 4) lần → đánh tất cả entry đó là music (điệp khúc/jingle/outro).

Các bộ này tích hợp vào `tag_entries_heuristic` (mở rộng hàm hiện có), giữ thứ tự ưu tiên: `sound` → `confidence noise` → `hallucination noise` → `dialogue`; rồi hậu xử lý lặp (intra + near-dup + global) chỉ nâng `dialogue → music`.

## Component 3 — Nâng LLM (#4)

- `llm_clean_batch(..., model="llama-3.3-70b-versatile")` — đổi mặc định (giữ tham số để override về 8b khi cần nhanh). `clean_entries`/`clean_file` truyền model xuống nếu cần.
- `build_llm_prompt(batch)` viết lại: thêm **2–3 ví dụ mẫu** (dialogue/music/noise), chỉ thị **"không chắc → giữ dialogue"**, quy tắc chuẩn hóa rõ hơn, và đính kèm **nhãn heuristic hiện tại** mỗi dòng (`"{i} [{type}]: {text}"`) làm gợi ý để LLM xác nhận/sửa. Contract JSON `{"items":[{index,type,text}]}` **giữ nguyên**.
- `apply_llm_result` giữ nguyên (đã validate type + chỉ cập nhật text non-empty + giữ `text_raw`).

## Luồng tích hợp

`clean_entries`:
1. `tag_entries_heuristic(entries)` — nay gồm confidence (#1) + lặp/nhạc (#2).
2. Nếu có client → `llm_clean_batch` theo lô với model mới + prompt mới (#4).

Không đổi `clean_file`, CLI, hay cách `split_audio.py` đọc `.clean.json`.

## Error handling

- Metric thiếu/sai kiểu → `confidence_is_noise` trả `False` (không raise).
- `difflib`/đếm tần suất là pure Python, không I/O.
- LLM lỗi → `llm_clean_batch` trả batch nguyên (đã có sẵn).

## Testing (TDD)

- `confidence_is_noise`: từng ngưỡng đơn (no_speech, compression, combo logprob), entry thiếu metric → False, entry tốt → False.
- `is_repetitive_text`: "la la la la" → True; câu thoại bình thường → False.
- Near-duplicate: 3 câu gần-giống liên tiếp → music; 3 câu khác nhau → giữ dialogue.
- Global frequency: cụm xuất hiện ≥4 lần → music; <4 → giữ.
- `tag_entries_heuristic` tổng hợp: thứ tự ưu tiên đúng; thận trọng (thiếu metric không demote).
- `build_llm_prompt`: chứa ví dụ mẫu + chỉ thị "giữ dialogue" + nhãn heuristic theo dòng.
- `groq_response_to_entries`: copy 3 metric khi segment có; bỏ khi không có.

## Hằng số (tunable, đầu file)

`NO_SPEECH_MAX=0.6`, `COMPRESSION_MAX=2.4`, `LOGPROB_MIN=-1.0`, `LOGPROB_NOSPEECH_COMBO=0.4`, `REPEAT_MIN_TOKENS=4`, `REPEAT_UNIQUE_RATIO=0.5`, `SIM_THRESHOLD=0.85`, `GLOBAL_FREQ_MIN=4`, `GLOBAL_PHRASE_MAX_WORDS=12`.

## Tương thích ngược

- Entry không có metric → confidence bỏ qua; data cũ chạy như trước.
- `.clean.json` cũ vẫn đọc được; `split_audio.py` không đổi.
- LLM model là tham số có default mới; gọi cũ vẫn hoạt động.

## Ngoài phạm vi (YAGNI)

- #3 (mở rộng + file hallucination ngoài) — không làm lần này.
- Phát hiện nhạc bằng mô hình audio.
- Lấy metric local bằng cách đào sâu faster-whisper (chỉ best-effort).
