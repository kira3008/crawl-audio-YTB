# YouTube Vietnamese Audio Crawler

Công cụ chạy trên terminal, tự động tìm kiếm video tiếng Việt trên YouTube và tải về dưới dạng file MP3.

## Tính năng

- Tìm kiếm video theo từ khóa
- Tự động lọc video tiếng Việt
- Tải âm thanh và convert sang MP3
- Lưu danh sách link đã crawl vào file `crawled_links.md`

## Yêu cầu

- Python 3.10 trở lên

## Cài đặt

```bash
# 1. Clone repo
git clone https://github.com/kira3008/crawl-audio-YTB.git
cd crawl-audio-YTB

# 2. Cài dependencies
pip install -r requirements.txt
```

## Sử dụng

```bash
python main.py
```

Script sẽ hỏi lần lượt:

```
Từ khóa tìm kiếm: nhạc trẻ 2024
Số lượng video tìm kiếm [mặc định 20]: 10
Thư mục lưu file [mặc định 'downloads']:
```

Sau đó hiển thị danh sách video tiếng Việt tìm được và xác nhận trước khi tải:

```
  1. Nhạc Trẻ Hay Nhất 2024
     Kênh: ACV Ballad  |  Thời lượng: 01:32:23
  2. Top 30 Nhạc Remix TikTok 2024
     Kênh: H2O Remix   |  Thời lượng: 01:55:36

Tải 2 video dưới dạng MP3? (y/n):
```

## Kết quả

```
downloads/
├── Nhạc Trẻ Hay Nhất 2024.mp3
├── Top 30 Nhạc Remix TikTok 2024.mp3
└── crawled_links.md
```

File `crawled_links.md` lưu lại toàn bộ link đã crawl theo từng lần chạy:

```markdown
# Danh sách video đã crawl

## 2024-05-20 09:30:00

- [Nhạc Trẻ Hay Nhất 2024](https://youtube.com/watch?v=...) — ACV Ballad `01:32:23`
- [Top 30 Nhạc Remix TikTok 2024](https://youtube.com/watch?v=...) — H2O Remix `01:55:36`
```

## Lưu ý

- Lần đầu chạy, script tự động tải `ffmpeg.exe` (dùng để convert sang MP3), không cần cài thủ công.
- Bộ lọc tiếng Việt dựa trên ký tự Unicode đặc trưng (ă, ơ, ư, đ, ...) hoặc trường ngôn ngữ của video.
- Để tìm được nhiều video tiếng Việt hơn, nên dùng từ khóa tiếng Việt (vd: `nhạc trẻ`, `tin tức`, `hài hước`).
