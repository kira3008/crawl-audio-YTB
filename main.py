import subprocess
import sys
import json
import re
import time
import queue
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

YTDLP_CMD = [sys.executable, "-m", "yt_dlp"]
STALL_SECONDS = 60  # seconds without byte progress → show stall warning

_vi_base = (
    "àáâãèéêìíòóôõùúýăđơư"
    "ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ"
)
# Include uppercase variants automatically — avoids hand-coding Ạ, Ệ, etc.
VIETNAMESE_CHARS = set(_vi_base + _vi_base.upper())

# Minimum thresholds for the ratio-based filter
_MIN_VI_COUNT = 2    # số ký tự tiếng Việt tối thiểu trong văn bản
_MIN_VI_RATIO = 0.15 # tỉ lệ ký tự tiếng Việt / tổng chữ cái tối thiểu

# Từ vựng tiếng Việt đặc trưng — không tồn tại trong ngôn ngữ khác
# Dùng như tầng lọc thứ hai khi ratio thấp (title có nhiều từ tiếng Anh xen lẫn)
_VI_KEYWORDS = frozenset({
    "nhạc", "bài", "hát", "việt", "tiếng", "trẻ", "hay", "nhất",
    "tình", "yêu", "buồn", "vui", "đẹp", "mới", "lời", "triệu",
    "nghe", "nhiều", "không", "được", "người", "liên", "khúc",
    "remix", "cover", "phim", "kênh", "mùa", "thôi", "rồi",
    "chia", "sẻ", "quảng", "cáo", "gây", "nghiện", "tâm", "trạng",
})


# ── helpers ──────────────────────────────────────────────────────────────────

def get_ffmpeg_dir() -> str | None:
    script_dir = Path(__file__).parent
    local_ffmpeg = script_dir / "ffmpeg.exe"
    if local_ffmpeg.exists():
        return str(script_dir)
    try:
        import imageio_ffmpeg, shutil
        shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), local_ffmpeg)
        return str(script_dir)
    except Exception:
        return None


def check_dependencies():
    missing = []
    for pkg in ("yt_dlp", "rich", "questionary", "imageio_ffmpeg", "whisperx", "pydub"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg.replace("_", "-"))

    if not missing:
        return

    has_whisperx = "whisperx" in missing
    other = [p for p in missing if p != "whisperx"]

    if other:
        print(f"Đang cài dependencies: {', '.join(other)} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + other)

    if has_whisperx:
        print("Đang cài whisperx (bao gồm PyTorch ~2GB, có thể mất vài phút) ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "whisperx"])
        print("Cài whisperx xong.\n")


def _detect_gpu() -> bool:
    """Trả về True nếu có NVIDIA GPU và CUDA khả dụng."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def _load_whisper_model(model_name: str):
    """
    Load WhisperX (GPU nếu có, fallback CPU int8).
    Trả về (bundle, backend, device) với:
      bundle = {"asr": model, "align": align_model, "meta": metadata, "device": device}
      backend = "whisperx"
    """
    import warnings
    warnings.filterwarnings("ignore", message="torchcodec is not installed")
    import whisperx

    device = "cpu"
    compute_type = "int8"

    if _detect_gpu():
        try:
            asr_model = whisperx.load_model(model_name, device="cuda", compute_type="int8", language="vi")
            device = "cuda"
        except Exception:
            asr_model = whisperx.load_model(model_name, device="cpu", compute_type="int8", language="vi")
    else:
        asr_model = whisperx.load_model(model_name, device="cpu", compute_type="int8", language="vi")

    align_model, metadata = whisperx.load_align_model(language_code="vi", device=device)

    bundle = {"asr": asr_model, "align": align_model, "meta": metadata, "device": device}
    return bundle, "whisperx", device


def _vi_stats(text: str) -> tuple[int, float]:
    """Returns (vi_char_count, ratio_to_alpha) for the given text."""
    if not text:
        return 0, 0.0
    vi    = sum(1 for c in text if c in VIETNAMESE_CHARS)
    alpha = sum(1 for c in text if c.isalpha())
    return vi, (vi / alpha if alpha else 0.0)


def _is_vi_text(text: str) -> bool:
    """True when text itself is predominantly Vietnamese."""
    count, ratio = _vi_stats(text)
    return count >= _MIN_VI_COUNT and ratio >= _MIN_VI_RATIO


def _has_vi_keyword(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _VI_KEYWORDS)


def is_vietnamese_video(info: dict) -> bool:
    """
    Multi-signal filter — returns True only when confident the video is Vietnamese.

    Pass criteria (checked in order):
      1. YouTube metadata explicitly marks language = "vi"
      2. Title ratio: ≥2 vi-chars AND ≥15% of alpha chars are Vietnamese
      3. Title has ≥1 vi-char AND contains a Vietnamese keyword
         (catches mixed titles like "TOP 30 NHẠC REMIX TIKTOK TRIỆU VIEW 2024")
      4. Title has ≥1 vi-char with small ratio + channel name is predominantly Vietnamese
         (catches Vietnamese channels uploading with partly-English titles)
    """
    if (info.get("language") or "").lower() == "vi":
        return True

    title   = info.get("title")   or ""
    channel = info.get("uploader") or info.get("channel") or ""

    if _is_vi_text(title):
        return True

    t_count, t_ratio = _vi_stats(title)

    if t_count >= 1 and _has_vi_keyword(title):
        return True

    if t_count >= 1 and t_ratio >= 0.05 and _is_vi_text(channel):
        return True

    return False


def sanitize_filename(title: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", title)
    return re.sub(r"\s+", " ", sanitized).strip()[:180]


def format_duration(seconds) -> str:
    if not seconds:
        return "--:--"
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"


# ── search ────────────────────────────────────────────────────────────────────

def search_videos(keyword: str, fetch_count: int) -> list[dict]:
    """Fetch up to fetch_count results from YouTube search."""
    cmd = YTDLP_CMD + [
        "--dump-json", "--no-download", "--no-playlist", "--flat-playlist",
        "--extractor-args", "youtube:lang=vi",
        f"ytsearch{fetch_count}:{keyword}",
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
            vid_id = info.get("id") or info.get("url") or ""
            if not vid_id:
                continue
            url = vid_id if vid_id.startswith("http") else f"https://www.youtube.com/watch?v={vid_id}"
            videos.append({
                "id":       vid_id,
                "title":    info.get("title")   or "",
                "url":      url,
                "channel":  info.get("uploader") or info.get("channel") or "N/A",
                "duration": info.get("duration"),
                "is_vi":    is_vietnamese_video(info),
            })
        except json.JSONDecodeError:
            continue
    return videos


def search_until_enough(keyword: str, needed: int, status_fn=None) -> tuple[list[dict], int]:
    """
    Keep expanding the fetch window until we have `needed` Vietnamese videos
    or YouTube has no more results to give.

    Returns (vi_videos[:needed], total_fetched).
    """
    fetch = max(int(needed * 1.5), 10)  # start 1.5× to reduce loop rounds
    max_fetch = max(needed * 5, 100) # hard ceiling to avoid infinite loops
    prev_total = -1

    while True:
        if status_fn:
            status_fn(fetch)
        all_videos = search_videos(keyword, fetch)
        vi_videos  = [v for v in all_videos if v["is_vi"]]

        # enough results or YouTube is exhausted (no new results came in)
        if len(vi_videos) >= needed or len(all_videos) == prev_total or fetch >= max_fetch:
            return vi_videos[:needed], len(all_videos)

        prev_total = len(all_videos)
        fetch = min(fetch + needed, max_fetch)


# ── transcription ────────────────────────────────────────────────────────────

def _sec_to_hms(sec: float) -> str:
    """float seconds → 'hh:mm:ss.mmm'."""
    ms = round((sec % 1) * 1000)
    total = int(sec)
    if ms == 1000:
        ms = 0
        total += 1
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"



def transcribe_audio(safe_title: str, output_dir: str, model, backend: str = "whisperx") -> bool:
    """Dùng WhisperX + PhoWhisper nhận diện giọng nói từ MP3 → JSON [{start, end, text}].
    Alignment bước 2 căn chỉnh timestamp chính xác theo audio thực tế.
    """
    import logging
    mp3_path = Path(output_dir) / f"{safe_title}.mp3"
    if not mp3_path.exists():
        logging.warning(f"[transcribe] MP3 not found: {mp3_path}")
        return False
    try:
        import whisperx

        bundle = model  # {"asr", "align", "meta", "device"}
        device = bundle["device"]
        batch_size = 16 if device == "cuda" else 4

        audio = whisperx.load_audio(str(mp3_path))

        # Bước 1: nhận diện văn bản, tự giảm batch_size nếu OOM
        result = None
        while batch_size >= 1:
            try:
                import torch
                if device == "cuda":
                    torch.cuda.empty_cache()
                result = bundle["asr"].transcribe(audio, batch_size=batch_size, language="vi")
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    batch_size = batch_size // 2
                    logging.warning(f"[transcribe] OOM — giảm batch_size xuống {batch_size}")
                else:
                    raise
        if result is None:
            return False

        # Bước 2: căn chỉnh timestamp khớp chính xác với audio
        aligned = whisperx.align(
            result["segments"],
            bundle["align"],
            bundle["meta"],
            audio,
            device=device,
            return_char_alignments=False,
        )

        entries = []
        for seg in aligned["segments"]:
            text = seg.get("text", "").strip()
            if not text:
                continue
            # Dùng word-level boundaries để timestamp chính xác hơn segment-level ASR
            words = [w for w in seg.get("words", []) if "start" in w and "end" in w]
            if words:
                start = words[0]["start"]
                end   = words[-1]["end"]
            else:
                start = seg["start"]
                end   = seg["end"]
            entries.append({
                "start": _sec_to_hms(start),
                "end":   _sec_to_hms(end),
                "text":  text,
            })

        if not entries:
            logging.warning(f"[transcribe] No segments found in: {mp3_path}")
            return False

        json_path = Path(output_dir) / f"{safe_title}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logging.error(f"[transcribe] Error transcribing {mp3_path}: {e}", exc_info=True)
        return False


# ── download (single video) ───────────────────────────────────────────────────

def download_one(
    video: dict,
    output_dir: str,
    ffmpeg_dir: str | None,
    progress,        # rich.progress.Progress (shared, thread-safe)
    task_id: int,
) -> bool:
    import yt_dlp

    stall = {"last_bytes": 0, "last_time": time.time()}

    def hook(d: dict):
        if d["status"] == "downloading":
            downloaded = d.get("downloaded_bytes") or 0
            total      = d.get("total_bytes") or d.get("total_bytes_estimate") or 0

            # stall detection
            if downloaded > stall["last_bytes"]:
                stall["last_bytes"] = downloaded
                stall["last_time"]  = time.time()

            idle = time.time() - stall["last_time"]
            status = "[yellow]⚠ Stalled[/yellow]" if idle > STALL_SECONDS else "[cyan]⬇[/cyan]"

            progress.update(
                task_id,
                completed=downloaded,
                total=total if total > 0 else None,
                status=status,
            )

        elif d["status"] == "finished":
            final = d.get("total_bytes") or stall["last_bytes"] or 1
            progress.update(task_id, completed=final, total=final,
                            status="[magenta]⚙ Converting…[/magenta]")

    # >30 min → more fragment threads for faster downloads
    duration  = video.get("duration") or 0
    fragments = 8 if duration > 1800 else 4

    safe_title       = sanitize_filename(video["title"])
    output_template  = str(Path(output_dir) / f"{safe_title}.%(ext)s")

    ydl_opts: dict = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "0",
        }],
        "outtmpl":                        output_template,
        "progress_hooks":                 [hook],
        "concurrent_fragment_downloads":  fragments,
        "quiet":       True,
        "no_warnings": True,
        "noprogress":  True,
        "retries":          5,
        "fragment_retries": 5,
        # tránh 429: chờ 2-5s giữa các request
        "sleep_interval":         2,
        "max_sleep_interval":     5,
        "sleep_interval_requests": 1,
    }
    if ffmpeg_dir:
        ydl_opts["ffmpeg_location"] = ffmpeg_dir

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video["url"]])

        progress.update(task_id, status="[green]✓ Xong[/green]")
        return True
    except Exception:
        return False


# ── duplicate check ──────────────────────────────────────────────────────────

def get_downloaded_ids(output_dir: str) -> set[str]:
    """Return a set of video IDs already recorded in crawled_links.md."""
    links_file = Path(output_dir) / "crawled_links.md"
    if not links_file.exists():
        return set()
    content = links_file.read_text(encoding="utf-8", errors="replace")
    # Links are written as [title](https://www.youtube.com/watch?v=ID)
    return set(re.findall(r"watch\?v=([A-Za-z0-9_-]+)", content))


def get_downloaded_titles(output_dir: str) -> set[str]:
    """Return sanitized base-names of MP3 files already present in output_dir."""
    p = Path(output_dir)
    if not p.exists():
        return set()
    return {f.stem for f in p.glob("*.mp3")}


# ── markdown log ─────────────────────────────────────────────────────────────

def _init_links_section(output_dir: str):
    """Ghi header section một lần khi bắt đầu batch download."""
    links_file = Path(output_dir) / "crawled_links.md"
    doc_header = "" if links_file.exists() else "# Danh sách video đã crawl\n\n"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(links_file, "a", encoding="utf-8") as f:
        f.write(f"{doc_header}## {timestamp}\n\n")


def _append_link(video: dict, output_dir: str):
    """Append một dòng ngay khi video xử lý xong (thread-safe vì GIL bảo vệ file.write)."""
    links_file = Path(output_dir) / "crawled_links.md"
    line = f"- [{video['title']}]({video['url']}) — {video['channel']} `{format_duration(video.get('duration'))}`\n"
    with open(links_file, "a", encoding="utf-8") as f:
        f.write(line)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    check_dependencies()

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table, Column
    from rich.progress import (
        Progress, TextColumn, BarColumn,
        DownloadColumn, TransferSpeedColumn, TimeRemainingColumn,
    )
    from rich import box
    import questionary

    console = Console()

    console.print(Panel(
        "[bold cyan]YouTube Vietnamese Audio Crawler[/bold cyan]\n"
        "[dim]Tìm kiếm · Lọc tiếng Việt · Tải MP3[/dim]",
        border_style="cyan", padding=(1, 4),
    ))

    # ── inputs ────────────────────────────────────────────────────────────────
    keyword = questionary.text(
        "Từ khóa tìm kiếm:",
        validate=lambda v: True if v.strip() else "Không được để trống",
    ).ask()
    if keyword is None:
        return

    count_choice = questionary.select(
        "Số lượng video tìm kiếm:",
        choices=["10", "20", "50", "Tùy chỉnh"],
        default="20",
    ).ask()
    if count_choice is None:
        return

    if count_choice == "Tùy chỉnh":
        raw = questionary.text(
            "Nhập số lượng:",
            validate=lambda v: True if v.isdigit() and int(v) > 0 else "Phải là số nguyên dương",
        ).ask()
        if raw is None:
            return
        max_results = int(raw)
    else:
        max_results = int(count_choice)

    workers_choice = questionary.select(
        "Số luồng tải song song:",
        choices=["1 luồng", "2 luồng (mặc định)", "3 luồng", "4 luồng"],
        default="2 luồng (mặc định)",
    ).ask()
    if workers_choice is None:
        return
    max_workers = int(workers_choice[0])

    whisper_choice = questionary.select(
        "Whisper model — tiếng Việt (dùng khi không có caption VTT):",
        choices=[
            "tiny        — nhanh nhất, ít chính xác (~39MB)",
            "base        — cân bằng tốt (~74MB)",
            "small       — chính xác hơn (~244MB)",
            "medium      — rất tốt (~769MB) [mặc định]",
            "large-v2    — tốt, ổn định (~1.5GB)",
            "large-v3    — tốt nhất cho tiếng Việt (~1.5GB)",
        ],
        default="medium      — rất tốt (~769MB) [mặc định]",
    ).ask()
    if whisper_choice is None:
        return
    whisper_model = whisper_choice.split()[0]

    output_dir = questionary.text("Thư mục lưu file:", default="downloads").ask()
    if output_dir is None:
        return

    console.print()

    # ── search ────────────────────────────────────────────────────────────────
    with console.status("") as live_status:
        def status_fn(fetch_count: int):
            live_status.update(
                f"[bold green]Đang tìm kiếm '{keyword}'… "
                f"[dim](thử {fetch_count} kết quả)[/dim][/bold green]"
            )

        vi_videos, total_fetched = search_until_enough(keyword, max_results, status_fn)

    if not vi_videos:
        console.print(Panel(
            "[yellow]Không tìm thấy video tiếng Việt nào.[/yellow]\n"
            "[dim]Thử dùng từ khóa tiếng Việt: nhạc trẻ, tin tức, hài hước…[/dim]",
            border_style="yellow",
        ))
        return

    console.print(
        f"[green]Tìm thấy[/green] [bold]{len(vi_videos)}[/bold] video tiếng Việt "
        f"[dim](quét {total_fetched} kết quả từ YouTube)[/dim]\n"
    )

    # ── results table ─────────────────────────────────────────────────────────
    table = Table(box=box.ROUNDED, border_style="cyan", show_lines=False)
    table.add_column("#",          style="dim",        width=3,    justify="right")
    table.add_column("Tiêu đề",    style="bold white", max_width=52)
    table.add_column("Kênh",       style="cyan",       max_width=22)
    table.add_column("Thời lượng", style="green",      justify="right")

    for i, v in enumerate(vi_videos, 1):
        table.add_row(str(i), v["title"], v["channel"], format_duration(v.get("duration")))

    console.print(table)
    console.print()

    # ── checkbox ──────────────────────────────────────────────────────────────
    choices = [
        questionary.Choice(
            title=f"{v['title']} [{format_duration(v.get('duration'))}]",
            value=v, checked=True,
        )
        for v in vi_videos
    ]
    selected = questionary.checkbox(
        "Chọn video muốn tải (Space chọn/bỏ · Enter xác nhận):",
        choices=choices,
    ).ask()

    if not selected:
        console.print("[yellow]Không có video nào được chọn.[/yellow]")
        return

    # ── skip already-downloaded ───────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    existing_ids    = get_downloaded_ids(output_dir)
    existing_titles = get_downloaded_titles(output_dir)

    to_skip = [
        v for v in selected
        if v["id"] in existing_ids or sanitize_filename(v["title"]) in existing_titles
    ]
    to_download = [v for v in selected if v not in to_skip]

    if to_skip:
        skip_names = ", ".join(f"[dim]{v['title'][:40]}[/dim]" for v in to_skip)
        console.print(
            f"[yellow]⏭  Bỏ qua {len(to_skip)} video đã tải:[/yellow] {skip_names}\n"
        )

    if not to_download:
        console.print(Panel(
            "[green]Tất cả video đã được tải trước đó. Không có gì mới để tải.[/green]",
            border_style="green", padding=(1, 4),
        ))
        return

    console.print()
    console.print(
        f"[bold]Tải [cyan]{len(to_download)}[/cyan] video mới · "
        f"[cyan]{max_workers}[/cyan] luồng song song[/bold]\n"
    )

    # ── load model trước download để pipeline kịp thời ───────────────────────
    console.print(f"\n[bold]🎙 Load Whisper model [cyan]{whisper_model}[/cyan]…[/bold]")
    wmodel, backend, device = _load_whisper_model(whisper_model)
    console.print(f"[dim]Backend: WhisperX [{device.upper()}] + alignment vi[/dim]\n")

    # ── parallel download + pipeline transcription ────────────────────────────
    ffmpeg_dir = get_ffmpeg_dir()
    success, failed = [], []

    _init_links_section(output_dir)
    transcribe_q: queue.Queue = queue.Queue()

    with Progress(
        TextColumn("[bold white]{task.fields[title]}", table_column=Column(min_width=36, max_width=36, no_wrap=True)),
        BarColumn(bar_width=20),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}"),
        console=console,
        expand=False,
    ) as progress:

        task_map: dict[str, int] = {}
        for v in to_download:
            short = v["title"][:34] + ("…" if len(v["title"]) > 34 else "")
            tid = progress.add_task(
                "", total=None, title=short, status="[dim]Chờ…[/dim]",
            )
            task_map[v["id"]] = tid

        def _transcribe_worker():
            while True:
                item = transcribe_q.get()
                if item is None:
                    break
                vid, tid = item
                safe = sanitize_filename(vid["title"])
                progress.update(tid, status="[blue]🎙 Transcribing…[/blue]")
                has_script = transcribe_audio(safe, output_dir, wmodel, backend)
                _append_link(vid, output_dir)
                label = "[green]✓ +script[/green]" if has_script else "[green]✓ Xong[/green]"
                progress.update(tid, status=label)
                transcribe_q.task_done()

        t_worker = threading.Thread(target=_transcribe_worker, daemon=True)
        t_worker.start()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    download_one, v, output_dir, ffmpeg_dir,
                    progress, task_map[v["id"]],
                ): v
                for v in to_download
            }

            for future in as_completed(future_map):
                video = future_map[future]
                tid   = task_map[video["id"]]
                try:
                    ok = future.result()
                except Exception:
                    ok = False

                if ok:
                    success.append(video)
                    transcribe_q.put((video, tid))
                else:
                    progress.update(tid, status="[red]✗ Lỗi[/red]")
                    failed.append(video)

        # Chờ transcription worker xử lý hết queue trước khi đóng progress
        transcribe_q.put(None)
        t_worker.join()

    # count JSON script files actually saved
    out = Path(output_dir)
    script_count = sum(
        1 for v in success
        if (out / f"{sanitize_filename(v['title'])}.json").exists()
    )

    # ── summary panel ─────────────────────────────────────────────────────────
    lines = [f"[green]✓ {len(success)} file MP3 đã tải[/green]"]
    if script_count:
        lines.append(f"[green]📝 {script_count} script JSON (WhisperX)[/green]")
    no_script = len(success) - script_count
    if no_script:
        lines.append(f"[dim]   {no_script} video không có captions tiếng Việt[/dim]")
    if failed:
        lines.append(f"[red]✗ {len(failed)} thất bại[/red]")
    lines += [
        f"\n[dim]Thư mục : {Path(output_dir).resolve()}[/dim]",
        f"[dim]Log link: {Path(output_dir).resolve() / 'crawled_links.md'}[/dim]",
    ]

    console.print(Panel(
        "\n".join(lines),
        title="[bold]Hoàn tất[/bold]",
        border_style="green" if not failed else "yellow",
        padding=(1, 4),
    ))


if __name__ == "__main__":
    main()
