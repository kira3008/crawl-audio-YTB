# Thiết kế: 3 cải tiến cho Crawl Video

**Ngày:** 2026-06-24
**Trạng thái:** Đã duyệt thiết kế, chờ lập kế hoạch triển khai

## Bối cảnh

Pipeline hiện tại (`main.py` + `split_audio.py`):

```
search (yt-dlp) → lọc tiếng Việt → download MP3 (≤4 luồng, chống 429 bằng sleep)
  → transcribe (WhisperX local + alignment) → JSON [{start,end,text,words}]
  → split_audio.py: gom câu + Silero VAD → cắt segment + manifest.json
```

Ba điểm cần cải tiến:

1. **Giới hạn 1 IP** — một IP chỉ kéo được ~100 video/lần thì dính rate-limit (429). Cần xoay IP để crawl khối lượng lớn làm data.
2. **WhisperX local chậm** — cần thêm lựa chọn chạy Whisper qua **Groq API** cho nhanh.
3. **Transcript lẫn nhiễu** — Whisper lấy cả lời nhạc / phần không phải đối thoại. Cần bước lọc & chuẩn hóa.

## Quyết định đã chốt (qua brainstorming)

| Vấn đề | Lựa chọn |
|---|---|
| #3 cách lọc | Hybrid: heuristic + LLM Groq |
| #3 xử lý đoạn nhiễu | Tag lại, **giữ bản gốc** (tạo `.clean.json`, không đụng `.json`) |
| #3 vị trí chạy | **Cả hai**: auto inline trong `main.py` + standalone `clean_transcript.py` |
| #1 nguồn proxy | Auto free + cho phép custom list |
| #1 cơ chế refresh | Background thread trong app |
| #2 Groq | Tùy chọn backend, **local vẫn mặc định**; chỉ bật Groq khi có `GROQ_API_KEY` |

## Kiến trúc tổng thể

Pipeline mới (phần viết hoa là thêm/sửa):

```
search → download (PROXY POOL) → transcribe (LOCAL | GROQ) → CLEAN (.clean.json) → split (đọc clean.json)
```

Tách tính năng mới thành module riêng để `main.py` không phình thêm và mỗi đơn vị có một nhiệm vụ rõ ràng, test/chạy độc lập được:

| Module | Nhiệm vụ | Phụ thuộc |
|---|---|---|
| `proxy_pool.py` (mới) | Fetch + validate + xoay proxy free/custom, background refresh | `requests` |
| `transcribe_backends.py` (mới) | Hai backend transcribe (WhisperX local + Groq), cùng schema output | `whisperx`, `groq`, ffmpeg |
| `clean_transcript.py` (mới) | Lọc nhiễu heuristic + LLM Groq; standalone + import được | `groq` (chỉ tầng B) |
| `main.py` (sửa) | Điều phối: chọn backend, bật proxy, gọi clean inline | các module trên |
| `split_audio.py` (sửa nhẹ) | Ưu tiên `.clean.json`, chỉ cắt segment `type=="dialogue"` | — |

**Schema transcript dùng chung** (không đổi so với hiện tại — đây là hợp đồng giữa các module):

```json
[{ "start": "hh:mm:ss.mmm", "end": "hh:mm:ss.mmm", "text": "...", "words": [{"word","start","end"}] }]
```

---

## Cải tiến #1 — Proxy pool xoay IP

### `proxy_pool.py`

Class `ProxyPool` (thread-safe):

- **Nguồn proxy:**
  - Tự fetch từ các API/list proxy free công khai (proxyscrape, free-proxy-list, …) — danh sách URL nguồn cấu hình được.
  - Merge thêm danh sách custom từ file `proxies.txt` (mỗi dòng `host:port` hoặc `scheme://host:port`). **Custom ưu tiên** xếp trước.
- **Validate:** test song song từng proxy (request HTTPS với timeout ngắn ~5s), chỉ giữ proxy sống. Lưu cache ra `proxy_cache.json` để tránh cold-start lần sau.
- **Xoay:** `get_proxy()` trả proxy theo round-robin. Khi yt-dlp báo 429 / lỗi mạng → `mark_bad(proxy)` rồi lấy proxy khác retry (giới hạn số lần retry/video, vd 3).
- **Background refresh:** một thread nền tự fetch + validate lại mỗi N phút (mặc định 10), loại proxy chết, cập nhật cache. Thread là daemon, dừng theo app.
- **Fallback an toàn:** khi pool rỗng → `get_proxy()` trả `None` ⇒ kết nối trực tiếp (không proxy). Không bao giờ làm crash pipeline.

### Tích hợp `main.py`

- Truyền `ydl_opts["proxy"]` vào **cả** `search_videos()` và `download_one()`.
- Trong `download_one()`: bọc vòng retry — fail/429 thì đổi proxy (qua `pool.mark_bad` + `pool.get_proxy`) và thử lại tới giới hạn; hết proxy thì thử trực tiếp.
- Menu thêm:
  - Bật/tắt proxy mode (mặc định tắt để giữ hành vi cũ; bật khi cần crawl số lượng lớn).
  - Khi bật proxy: cho tăng số luồng song song (mở rộng lựa chọn hiện tại 1–4 lên cao hơn, kèm cảnh báo).
  - Đường dẫn file proxy custom (mặc định `proxies.txt`).

### Rủi ro & giới hạn (ghi rõ)

Proxy free độ ổn định thấp (hay chết, chậm, bị YouTube chặn nhanh). Thiết kế validate gắt + fallback trực tiếp để giảm rủi ro. Code dạng "pool" nên muốn dùng proxy trả phí chỉ cần đổ vào `proxies.txt`, không sửa code.

---

## Cải tiến #2 — Backend Groq cho Whisper

### `transcribe_backends.py`

Xuất hai hàm cùng trả về **schema transcript dùng chung** ở trên:

- `transcribe_local(mp3_path, model, ...)` — di chuyển nguyên logic WhisperX hiện có từ `main.py` (`_load_whisper_model`, `transcribe_audio`) sang, **không đổi hành vi**.
- `transcribe_groq(mp3_path, model, ...)`:
  - **Xử lý giới hạn dung lượng Groq** (~25MB free / ~100MB dev tier):
    1. ffmpeg downsample audio → **16kHz mono FLAC** (giảm size mạnh).
    2. Nếu vẫn vượt ngưỡng → **chunk theo cửa sổ thời gian** (mặc định 10 phút, overlap ~5s), transcribe từng chunk.
    3. **Offset timestamp** theo vị trí chunk rồi merge; khử trùng lặp ở vùng overlap.
  - Gọi `client.audio.transcriptions.create(model=..., file=..., language="vi", response_format="verbose_json", timestamp_granularities=["segment","word"])`.
  - Map segment → schema chung. Có word-level thì điền `words`; không có thì để `[]` (split vẫn chạy vì core của `split_audio.py` dùng start/end segment + VAD; `words` chỉ dùng hiển thị `--inspect`).
  - Model Groq: `whisper-large-v3` (chuẩn) hoặc `whisper-large-v3-turbo` (nhanh hơn).

### Tích hợp `main.py`

- Menu thêm chọn backend: **WhisperX local (mặc định)** / **Groq API (nhanh)**.
- Option Groq chỉ hiện/chọn được khi có `GROQ_API_KEY` (đọc từ env hoặc `.env`). Không có key → ẩn + cảnh báo, dùng local.
- Menu model thích ứng theo backend (model local cũ giữ nguyên; Groq có 2 lựa chọn ở trên).
- Worker transcribe gọi hàm backend tương ứng — phần còn lại của pipeline không đổi.

### Cần verify lúc code

Groq có hỗ trợ `timestamp_granularities=["word"]` không. Nếu không → rơi về segment-level, **không ảnh hưởng việc cắt** (như giải thích trên).

---

## Cải tiến #3 — Lọc nhiễu & chuẩn hóa transcript

### `clean_transcript.py`

Chạy được standalone (giống `split_audio.py`) và import inline vào `main.py`. Hai tầng:

**Tầng A — Heuristic (luôn chạy, không cần API):** gắn cờ `type` mỗi segment:

- Tag nhạc/âm thanh: regex `[Âm nhạc]`, `[Music]`, `[Vỗ tay]`, `[Tiếng cười]`, `♪`, `🎵`, annotation trong ngoặc vuông → `music` / `sound`.
- Hallucination thường gặp của Whisper tiếng Việt: "Hãy subscribe…", "Cảm ơn các bạn đã theo dõi", "Hẹn gặp lại các bạn"… (đặc biệt lặp ở cuối) → `noise`.
- Câu lặp liên tiếp ≥ N lần (mặc định 3, đặc trưng lời nhạc/loop) → `music`.
- Mặc định còn lại → `dialogue`.

**Tầng B — LLM Groq (chạy khi có `GROQ_API_KEY`):**

- Gom segment theo lô (~30–50 câu/lần) gửi LLM (`llama-3.1-8b-instant`, JSON mode / `response_format`) để:
  1. Phân loại `dialogue` / `music` / `noise` chính xác theo ngữ nghĩa (sửa lại cờ của tầng A).
  2. **Chuẩn hóa** text: sửa dấu câu, viết hoa đầu câu, bỏ filler ("ờ", "à", lặp từ).
- Map kết quả về theo index. Lỗi API / lô lỗi → giữ kết quả tầng A cho lô đó (không chặn pipeline).

**Output (giữ bản gốc):**

- `<name>.clean.json` — mỗi segment thêm: `type` (`dialogue`/`music`/`noise`/`sound`), `text` (đã chuẩn hóa), `text_raw` (gốc); giữ nguyên `start`/`end`/`words`.
- `<name>.json` gốc **không đụng tới**.

### Tích hợp

- **Inline trong `main.py`:** sau khi transcribe ghi `.json`, gọi clean để sinh `.clean.json` (tầng A luôn chạy; tầng B nếu có key + bật).
- **Standalone:** `python clean_transcript.py downloads/` xử lý lại JSON cũ đã crawl.
- **`split_audio.py` sửa nhẹ:** tự ưu tiên `.clean.json` nếu tồn tại và chỉ cắt segment `type=="dialogue"`; không có thì dùng `.json` như cũ (tương thích ngược 100%).

---

## Pipeline đầy đủ sau cải tiến

```
search → download(proxy pool) → transcribe(local | groq) → clean(.clean.json) → split(đọc clean.json, chỉ dialogue)
```

## Cấu hình

- `GROQ_API_KEY` — env hoặc `.env` (đọc bằng `python-dotenv`). Gate cho Groq backend (#2) và tầng B lọc nhiễu (#3).
- `proxies.txt` — danh sách proxy custom (optional).
- `proxy_cache.json` — cache proxy đã validate (tự sinh).

## Dependencies thêm (`requirements.txt`)

- `groq` — SDK Groq (transcribe + LLM clean).
- `requests` — fetch/validate proxy.
- `python-dotenv` — đọc `.env` (optional).

## Tương thích ngược

- Proxy mode mặc định **tắt** → hành vi crawl cũ không đổi.
- Backend mặc định **local WhisperX** → không cần key vẫn chạy y như cũ.
- Không có `.clean.json` → `split_audio.py` dùng `.json` như trước.
- Schema `.json` gốc không đổi.

## Ngoài phạm vi (YAGNI)

- Proxy trả phí / residential (chỉ chừa đường nạp qua `proxies.txt`).
- Phân loại nhạc/thoại bằng mô hình audio (chỉ làm theo text).
- Song song hóa nhiều lời gọi Groq cùng lúc (giữ worker tuần tự cho an toàn; có thể thêm sau).
