import subprocess
import sys
import json
import re
from pathlib import Path
from datetime import datetime

# Use python -m yt_dlp to avoid PATH issues on Windows
YTDLP_CMD = [sys.executable, "-m", "yt_dlp"]


def get_ffmpeg_dir() -> str | None:
    """Return a directory containing ffmpeg.exe, copying from imageio-ffmpeg if needed."""
    script_dir = Path(__file__).parent
    local_ffmpeg = script_dir / "ffmpeg.exe"

    if local_ffmpeg.exists():
        return str(script_dir)

    try:
        import imageio_ffmpeg
        import shutil
        src = imageio_ffmpeg.get_ffmpeg_exe()
        shutil.copy2(src, local_ffmpeg)
        return str(script_dir)
    except (ImportError, Exception):
        return None


VIETNAMESE_CHARS = set(
    "àáâãèéêìíòóôõùúýăđơư"
    "ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ"
    "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ"
    "ẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỶỸỴ"
)


def check_dependencies():
    try:
        subprocess.run(YTDLP_CMD + ["--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("yt-dlp chưa được cài. Đang cài đặt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp", "imageio-ffmpeg"])
        print("Cài đặt xong.\n")
        return

    if get_ffmpeg_dir() is None:
        print("imageio-ffmpeg chưa được cài. Đang cài đặt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"])
        print("Cài đặt xong.\n")


def is_vietnamese(text: str) -> bool:
    return any(c in VIETNAMESE_CHARS for c in (text or ""))


def sanitize_filename(title: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", title)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized[:180]


def search_videos(keyword: str, max_results: int) -> list[dict]:
    print(f"Đang tìm kiếm '{keyword}' ({max_results} kết quả)...")

    cmd = YTDLP_CMD + [
        "--dump-json",
        "--no-download",
        "--no-playlist",
        "--flat-playlist",
        f"ytsearch{max_results}:{keyword}",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )

    videos = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        try:
            info = json.loads(line)
            title = info.get("title") or ""
            lang = info.get("language") or ""
            vid_id = info.get("id") or info.get("url") or ""

            if not vid_id:
                continue

            url = (
                vid_id
                if vid_id.startswith("http")
                else f"https://www.youtube.com/watch?v={vid_id}"
            )

            videos.append(
                {
                    "id": vid_id,
                    "title": title,
                    "url": url,
                    "channel": info.get("uploader") or info.get("channel") or "N/A",
                    "duration": info.get("duration"),
                    "language": lang,
                    "is_vi": lang == "vi" or is_vietnamese(title),
                }
            )
        except json.JSONDecodeError:
            continue

    return videos


def filter_vietnamese(videos: list[dict]) -> list[dict]:
    vi_videos = [v for v in videos if v["is_vi"]]
    return vi_videos


def download_audio(video: dict, output_dir: str, ffmpeg_dir: str | None = None) -> bool:
    safe_title = sanitize_filename(video["title"])
    output_template = str(Path(output_dir) / f"{safe_title}.%(ext)s")

    cmd = YTDLP_CMD + [
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--output", output_template,
        "--no-playlist",
        "--retries", "3",
    ]

    if ffmpeg_dir:
        cmd += ["--ffmpeg-location", ffmpeg_dir]

    cmd.append(video["url"])

    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )

    if result.returncode != 0:
        # Print last line of stderr for quick diagnosis
        err = result.stderr.strip().splitlines()
        if err:
            print(f"     Lỗi: {err[-1]}")

    return result.returncode == 0


def update_links_md(videos: list[dict], output_dir: str):
    links_file = Path(output_dir) / "crawled_links.md"

    header = ""
    if not links_file.exists():
        header = "# Danh sách video đã crawl\n\n"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    section = f"## {timestamp}\n\n"
    for v in videos:
        duration_str = ""
        if v.get("duration"):
            mins, secs = divmod(int(v["duration"]), 60)
            duration_str = f" `{mins:02d}:{secs:02d}`"
        section += f"- [{v['title']}]({v['url']}) — {v['channel']}{duration_str}\n"
    section += "\n"

    with open(links_file, "a", encoding="utf-8") as f:
        f.write(header + section)


def format_duration(seconds) -> str:
    if not seconds:
        return "?:??"
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"


def main():
    print("=" * 50)
    print("  YouTube Vietnamese Audio Crawler")
    print("=" * 50)
    print()

    check_dependencies()

    keyword = input("Từ khóa tìm kiếm: ").strip()
    if not keyword:
        print("Từ khóa không được để trống.")
        sys.exit(1)

    max_input = input("Số lượng video tìm kiếm [mặc định 20]: ").strip()
    max_results = int(max_input) if max_input.isdigit() else 20

    output_dir = input("Thư mục lưu file [mặc định 'downloads']: ").strip() or "downloads"

    print()

    # Step 1: Search
    all_videos = search_videos(keyword, max_results)

    if not all_videos:
        print("Không tìm thấy kết quả nào.")
        sys.exit(0)

    print(f"Tổng tìm thấy: {len(all_videos)} video")

    # Step 2: Filter Vietnamese
    vi_videos = filter_vietnamese(all_videos)

    if not vi_videos:
        print("Không có video tiếng Việt nào trong kết quả tìm kiếm.")
        print("Tip: Thử thêm từ tiếng Việt vào từ khóa (vd: 'nhạc trẻ', 'tin tức', ...)")
        sys.exit(0)

    print(f"Video tiếng Việt: {len(vi_videos)}\n")

    for i, v in enumerate(vi_videos, 1):
        dur = format_duration(v.get("duration"))
        print(f"  {i:2}. {v['title']}")
        print(f"      Kênh: {v['channel']}  |  Thời lượng: {dur}")

    print()
    confirm = input(f"Tải {len(vi_videos)} video dưới dạng MP3? (y/n): ").strip().lower()
    if confirm != "y":
        print("Đã hủy.")
        sys.exit(0)

    # Step 3: Create output dir and download
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ffmpeg_path = get_ffmpeg_dir()

    success = []
    failed = []

    for i, video in enumerate(vi_videos, 1):
        print(f"\n[{i}/{len(vi_videos)}] {video['title']}")
        if download_audio(video, output_dir, ffmpeg_path):
            print("     ✓ Thành công")
            success.append(video)
        else:
            print("     ✗ Thất bại")
            failed.append(video)

    # Step 4: Update markdown log
    if success:
        update_links_md(success, output_dir)

    print()
    print("=" * 50)
    print(f"Hoàn tất: {len(success)} thành công, {len(failed)} thất bại")
    print(f"Thư mục: {Path(output_dir).resolve()}")
    if success:
        print(f"Log link: {Path(output_dir).resolve() / 'crawled_links.md'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
