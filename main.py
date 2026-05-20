import subprocess
import sys
import json
import re
import time
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
    for pkg in ("yt_dlp", "rich", "questionary", "imageio_ffmpeg"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg.replace("_", "-"))
    if missing:
        print(f"Đang cài dependencies: {', '.join(missing)} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("Xong.\n")


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


# ── script extraction ────────────────────────────────────────────────────────

def _ts_to_sec(ts: str) -> float:
    """'HH:MM:SS.mmm' or 'HH:MM:SS,mmm' → float seconds."""
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _find_vtt(output_dir: str, safe_title: str) -> Path | None:
    """Locate the VTT subtitle file yt-dlp saved for this video."""
    for f in Path(output_dir).iterdir():
        if f.suffix == ".vtt":
            # yt-dlp names subs "<title>.<lang>.vtt" → stem = "<title>.<lang>"
            base = f.stem.rsplit(".", 1)[0] if "." in f.stem else f.stem
            if base == safe_title:
                return f
    return None


def _parse_vtt(vtt_path: Path) -> list[dict]:
    """Parse WebVTT → [{start, end, text}], deduplicating auto-caption repeats."""
    content = vtt_path.read_text(encoding="utf-8", errors="replace")

    entries: list[dict] = []
    # Match timestamp line + following text block
    block_re = re.compile(
        r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})[^\n]*\n"
        r"((?:(?!\n\n|\d{2}:\d{2}:\d{2}).)+)",
        re.DOTALL,
    )
    prev_text = ""
    for m in block_re.finditer(content):
        raw = m.group(3)
        # strip HTML tags (e.g. <c.color>, <b>, alignment markers)
        text = re.sub(r"<[^>]+>", "", raw)
        text = re.sub(r"\s+", " ", text).strip()
        if not text or text == prev_text:
            continue
        entries.append({
            "start": _ts_to_sec(m.group(1)),
            "end":   _ts_to_sec(m.group(2)),
            "text":  text,
        })
        prev_text = text

    return entries


def extract_script(safe_title: str, output_dir: str) -> bool:
    """Find the downloaded VTT, convert to JSON, delete VTT. Returns True if saved."""
    vtt = _find_vtt(output_dir, safe_title)
    if not vtt:
        return False
    try:
        entries = _parse_vtt(vtt)
        if not entries:
            vtt.unlink(missing_ok=True)
            return False
        json_path = Path(output_dir) / f"{safe_title}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        vtt.unlink(missing_ok=True)
        return True
    except Exception:
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
        # subtitle / caption options
        "writesubtitles":    True,   # manual subtitles
        "writeautomaticsub": True,   # auto-generated captions (fallback)
        "subtitleslangs":    ["vi", "vi-VN"],
        "subtitlesformat":   "vtt",
        "quiet":       True,
        "no_warnings": True,
        "noprogress":  True,
        "retries":     3,
    }
    if ffmpeg_dir:
        ydl_opts["ffmpeg_location"] = ffmpeg_dir

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video["url"]])

        # convert downloaded VTT → JSON script
        progress.update(task_id, status="[blue]📝 Script…[/blue]")
        has_script = extract_script(safe_title, output_dir)
        status_final = "[green]✓ +script[/green]" if has_script else "[green]✓[/green]"
        progress.update(task_id, status=status_final)
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

def update_links_md(videos: list[dict], output_dir: str):
    links_file = Path(output_dir) / "crawled_links.md"
    header = "" if links_file.exists() else "# Danh sách video đã crawl\n\n"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    section = f"## {timestamp}\n\n"
    for v in videos:
        section += f"- [{v['title']}]({v['url']}) — {v['channel']} `{format_duration(v.get('duration'))}`\n"
    section += "\n"
    with open(links_file, "a", encoding="utf-8") as f:
        f.write(header + section)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    check_dependencies()

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
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
        choices=["2 luồng", "3 luồng (mặc định)", "4 luồng", "5 luồng"],
        default="3 luồng (mặc định)",
    ).ask()
    if workers_choice is None:
        return
    max_workers = int(workers_choice[0])

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

    # ── parallel download ─────────────────────────────────────────────────────
    ffmpeg_dir = get_ffmpeg_dir()
    success, failed = [], []

    with Progress(
        TextColumn("[bold white]{task.fields[title]}", no_wrap=True, min_width=36, max_width=36),
        BarColumn(bar_width=20),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}"),
        console=console,
        expand=False,
    ) as progress:

        # register one task per video up-front so all rows appear immediately
        task_map: dict[str, int] = {}
        for v in to_download:
            short = v["title"][:34] + ("…" if len(v["title"]) > 34 else "")
            tid = progress.add_task(
                "", total=None, title=short, status="[dim]Chờ…[/dim]",
            )
            task_map[v["id"]] = tid

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
                else:
                    progress.update(tid, status="[red]✗ Lỗi[/red]")
                    failed.append(video)

    # ── markdown log ─────────────────────────────────────────────────────────
    if success:
        update_links_md(success, output_dir)

    # count JSON script files actually saved
    out = Path(output_dir)
    script_count = sum(
        1 for v in success
        if (out / f"{sanitize_filename(v['title'])}.json").exists()
    )

    # ── summary panel ─────────────────────────────────────────────────────────
    lines = [f"[green]✓ {len(success)} file MP3 đã tải[/green]"]
    if script_count:
        lines.append(f"[green]📝 {script_count} script JSON (có captions)[/green]")
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
