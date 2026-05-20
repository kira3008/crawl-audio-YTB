# Audio Splitter - Module Documentation

## Cấu trúc

```
crawl-audio-YTB/
  ├── run_splitter.py      ← File chính để chạy
  ├── splitter_core.py     ← Logic tách audio (AudioSplitter class)
  ├── splitter_utils.py    ← Hàm utility (helper functions)
  ├── audio_splitter.py    ← File cũ (có thể xóa)
  └── ...
```

## Cách sử dụng

### 1. Tách audio 1 file JSON

```bash
python run_splitter.py "..\downloads\FAPtv Cơm Nguội_ Tập 52- Học Sinh Mới.json"
```

**Output:** `downloads/segments/FAPtv Cơm Nguội_ Tập 52- Học Sinh Mới/`
- `0001_ô_tô_theo_show_hẹn_tốt_Mày_ngu_như_bay.mp3`
- `0002_tối_mà_tao_theo_ta_không_biết_luôn_hả_có.mp3`
- ...

### 2. Tách TẤT CẢ JSON files trong thư mục

```bash
python run_splitter.py "..\downloads"
```

**Output:** `downloads/segments/{tên_mỗi_file}/`

### 3. Tách với output directory tùy chỉnh

```bash
python run_splitter.py "..\downloads" "my_segments"
```

### 4. Xem trợ giúp

```bash
python run_splitter.py --help
```

## Chi tiết các Module

### splitter_utils.py
Chứa các hàm utility:
- `sanitize_filename()` - Chuyển text thành tên file an toàn
- `find_media_file()` - Tìm file media tương ứng (mp3, mp4, v.v.)
- `get_ffmpeg_path()` - Lấy đường dẫn FFmpeg
- `format_time()` - Format thời gian sang HH:MM:SS.mmm
- `print_*()` - Các hàm in màu (success, error, warning, info)

### splitter_core.py
Chứa class `AudioSplitter`:
- `__init__()` - Khởi tạo
- `load_json()` - Load file JSON
- `validate_setup()` - Kiểm tra file media, FFmpeg
- `split_segment()` - Tách 1 đoạn audio
- `process()` - Xử lý tất cả entries
- `print_summary()` - In báo cáo kết quả

### run_splitter.py
File chính:
- `process_single_file()` - Xử lý 1 file
- `process_directory()` - Xử lý tất cả files trong thư mục
- `show_usage()` - Hiển thị trợ giúp

## Sửa code dễ dàng

**Muốn thay đổi logic tách audio?**
→ Sửa trong `splitter_core.py`, hàm `split_segment()`

**Muốn thay đổi tên file output?**
→ Sửa trong `splitter_core.py`, hàm `split_segment()` phần tạo filename

**Muốn thêm format output khác?**
→ Thêm format mới trong `splitter_utils.py` hoặc `splitter_core.py`

**Muốn thay đổi cách xử lý batch?**
→ Sửa trong `run_splitter.py`, hàm `process_directory()`

## Lưu ý

- Cần cài FFmpeg hoặc dùng `imageio_ffmpeg`
- JSON file phải cùng thư mục với media file (mp3, mp4, etc.)
- Tên file output được tạo từ `text` field trong JSON
