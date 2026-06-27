# YouTube Vietnamese Audio Crawler

Công cụ terminal tự động **tìm kiếm video tiếng Việt trên YouTube → tải MP3 → nhận dạng giọng nói (transcript) → lọc nhiễu/chuẩn hóa**. Phù hợp để xây dựng dataset audio + transcript tiếng Việt phục vụ huấn luyện model.

## Tính năng

- Tìm kiếm video YouTube theo từ khóa, tự động lọc video tiếng Việt (đa tín hiệu: ký tự Unicode, từ khóa, tên kênh, metadata ngôn ngữ).
- Tải âm thanh chất lượng cao và convert sang MP3.
- **Pipeline song song**: vừa tải vừa transcribe, không phải chờ tải hết mới chạy.
- **3 backend nhận dạng giọng nói** lựa chọn linh hoạt:
  - **WhisperX local** — chạy trên máy (GPU/CPU), đa worker, chính xác cao nhất.
  - **Groq API** — nhanh, có rate-limit tự quản lý.
  - **OpenRouter** — xoay nhiều API key để scale, chỉ trả text.
- Lọc nhiễu / chuẩn hóa transcript (`clean_transcript.py`) → sinh file `.clean.json`.
- **Khử trùng lặp**: tự bỏ qua video đã tải (theo video ID và tên file).
- **Ưu tiên tải file ngắn trước** để giải phóng worker sớm.
- Proxy pool xoay IP free/custom (fetch + validate song song) để crawl số lượng lớn.
- Lưu lịch sử link đã crawl vào `crawled_links.md`.

## Yêu cầu

- Python 3.10 trở lên
- (Tùy chọn) GPU NVIDIA + CUDA nếu muốn chạy WhisperX local nhanh

## Cài đặt

```bash
git clone https://github.com/kira3008/crawl-audio-YTB.git
cd crawl-audio-YTB
pip install -r requirements.txt
```

> Lần đầu chạy, app tự cài các thư viện còn thiếu (gồm WhisperX + PyTorch, ~2GB) và tự tải `ffmpeg.exe` — không cần cài thủ công.

## Sử dụng

```bash
python main.py
```

App hỏi lần lượt qua menu tương tác:

1. **Từ khóa tìm kiếm** — hỗ trợ paste đầy đủ (Ctrl+V / chuột phải).
2. **Số lượng video** — 10 / 20 / 50 / tùy chỉnh.
3. **Số luồng tải song song** — 1–8 luồng.
4. **Bật proxy pool?** — xoay IP free khi crawl số lượng lớn.
5. **Backend nhận dạng giọng nói** — WhisperX local / Groq / OpenRouter (các tùy chọn API chỉ hiện khi đã cấu hình key trong `.env`).
6. **Model** — tùy backend.
7. **Số worker transcribe local** *(chỉ khi chọn local)* — mỗi worker là 1 model riêng trong VRAM/RAM (xem [Hiệu năng local](#hiệu-năng-local)).
8. **Thư mục lưu file**.

Sau đó app hiển thị bảng video tiếng Việt tìm được và cho **chọn từng video** (Space chọn/bỏ, Enter xác nhận) trước khi tải.

## Backend nhận dạng giọng nói

| Backend | Tốc độ | Độ chính xác | Timestamps | Ghi chú |
|---|---|---|---|---|
| **WhisperX local** | Tùy GPU | Cao nhất | Word-level | Cần GPU mạnh để nhanh; chính xác nhất cho train |
| **Groq API** | Nhanh | Cao | Word-level | Rate-limit: 20 req/phút · 7200s/giờ · 28800s/ngày |
| **OpenRouter** | Nhanh | Cao | ❌ Chỉ text | Xoay nhiều key để scale; không có timestamps |

> Nếu cần **word-level timestamps** cho việc train, dùng WhisperX local hoặc Groq. OpenRouter chỉ trả plain text theo từng đoạn ~10 phút.

### Hiệu năng local

WhisperX local cho phép chạy nhiều worker song song — **mỗi worker load 1 model riêng** vào VRAM (GPU) hoặc RAM (CPU).

- GPU dùng `float16` (chính xác hơn `int8`), `batch_size=24`.
- CPU dùng `int8`, `batch_size=4`.
- Có sẵn cơ chế **OOM auto-fallback**: tự giảm batch khi hết bộ nhớ thay vì crash.

Tham khảo chọn số worker theo VRAM (model `large-v3`, mỗi worker ~6–7GB ở batch 24):

| VRAM | Số worker khuyến nghị |
|---|---|
| < 8GB | 1 |
| 8–16GB | 2 |
| 16–24GB | 3 |
| ≥ 24GB | 3–4 |

## Cấu hình `.env`

Tạo file `.env` ở thư mục gốc (đã nằm trong `.gitignore`, không bị commit):

```bash
# Groq — dùng cho backend Groq và bước clean transcript
# Lấy key tại: https://console.groq.com/keys
GROQ_API_KEY=gsk_xxx

# OpenRouter — mỗi key là 1 account khác nhau để xoay rate limit
# (20 req/phút · 1000 req/ngày mỗi key). Phân cách bằng dấu phẩy, không khoảng trắng.
# Lấy key tại: https://openrouter.ai/keys
OPENROUTER_API_KEYS=sk-or-v1-aaa,sk-or-v1-bbb,sk-or-v1-ccc
```

> Backend Groq/OpenRouter chỉ xuất hiện trong menu khi key tương ứng đã được cấu hình.

### Proxy tùy chọn

Tạo file `proxies.txt` ở thư mục gốc (một proxy mỗi dòng) để nạp proxy riêng:

```
http://proxy1:port
http://proxy2:port
1.2.3.4:8080
```

Proxy pool tự fetch thêm proxy free, validate song song, và xoay IP khi crawl số lượng lớn hoặc gặp lỗi 429.

## Kết quả

```
downloads/
├── Nhạc Trẻ Hay Nhất 2024.mp3          # audio
├── Nhạc Trẻ Hay Nhất 2024.json         # transcript thô (có timestamps nếu backend hỗ trợ)
├── Nhạc Trẻ Hay Nhất 2024.clean.json   # transcript đã lọc nhiễu / chuẩn hóa
└── crawled_links.md                    # lịch sử link đã crawl
```

`crawled_links.md` lưu lại link theo từng lần chạy:

```markdown
# Danh sách video đã crawl

## 2026-06-27 09:30:00

- [Nhạc Trẻ Hay Nhất 2024](https://www.youtube.com/watch?v=...) — ACV Ballad `01:32:23`
```

## Lưu ý

- Lần đầu chạy, model Whisper (`large-v3` ~1.5GB) được tải về thư mục `models/` — các lần sau dùng lại nên nhanh.
- Bộ lọc tiếng Việt dựa trên ký tự Unicode đặc trưng (ă, ơ, ư, đ…), từ khóa tiếng Việt, tên kênh và metadata ngôn ngữ.
- Để tìm được nhiều video tiếng Việt hơn, nên dùng từ khóa tiếng Việt (vd: `nhạc trẻ`, `tin tức`, `podcast`).
- Bước clean transcript dùng `GROQ_API_KEY`; nếu không có key thì vẫn tạo `.json` thô, chỉ bỏ qua bước `.clean.json`.
