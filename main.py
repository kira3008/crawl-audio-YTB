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

VIETNAMESE_CHARS = set(
    "àáâãèéêìíòóôõùúýăđơư"
    "ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ"
    "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ"
    "ẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỶỸỴ"
)


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


def is_vietnamese(text: str) -> bool:
    return any(c in VIETNAMESE_CHARS for c in (text or ""))


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

def search_videos(keyword: str, max_results: int) -> list[dict]:
    cmd = YTDLP_CMD + [
        "--dump-json", "--no-download", "--no-playlist", "--flat-playlist",
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
            vid_id = info.get("id") or info.get("url") or ""
            if not vid_id:
                continue
            url = vid_id if vid_id.startswith("http") else f"https://www.youtube.com/watch?v={vid_id}"
            title = info.get("title") or ""
            lang  = info.get("language") or ""
            videos.append({
                "id":       vid_id,
                "title":    title,
                "url":      url,
                "channel":  info.get("uploader") or info.get("channel") or "N/A",
                "duration": info.get("duration"),
                "is_vi":    lang == "vi" or is_vietnamese(title),
            })
        except json.JSONDecodeError:
            continue
    return videos


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
        "retries":     3,
    }
    if ffmpeg_dir:
        ydl_opts["ffmpeg_location"] = ffmpeg_dir

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video["url"]])
        return True
    except Exception:
        return False


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
    with console.status(f"[bold green]Đang tìm kiếm '{keyword}'…[/bold green]"):
        all_videos = search_videos(keyword, max_results)

    vi_videos = [v for v in all_videos if v["is_vi"]]

    if not vi_videos:
        console.print(Panel(
            "[yellow]Không tìm thấy video tiếng Việt nào.[/yellow]\n"
            "[dim]Thử dùng từ khóa tiếng Việt: nhạc trẻ, tin tức, hài hước…[/dim]",
            border_style="yellow",
        ))
        return

    console.print(
        f"[green]Tìm thấy[/green] [bold]{len(vi_videos)}[/bold] video tiếng Việt "
        f"[dim](từ {len(all_videos)} kết quả)[/dim]\n"
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

    console.print()
    console.print(
        f"[bold]Tải [cyan]{len(selected)}[/cyan] video · "
        f"[cyan]{max_workers}[/cyan] luồng song song[/bold]\n"
    )

    # ── parallel download ─────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
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
        for v in selected:
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
                for v in selected
            }

            for future in as_completed(future_map):
                video = future_map[future]
                tid   = task_map[video["id"]]
                try:
                    ok = future.result()
                except Exception:
                    ok = False

                if ok:
                    t = progress.tasks[tid]
                    final = t.total or 1
                    progress.update(tid, completed=final, total=final,
                                    status="[green]✓ Xong[/green]")
                    success.append(video)
                else:
                    progress.update(tid, status="[red]✗ Lỗi[/red]")
                    failed.append(video)

    # ── markdown log ─────────────────────────────────────────────────────────
    if success:
        update_links_md(success, output_dir)

    # ── summary panel ─────────────────────────────────────────────────────────
    lines = [f"[green]✓ {len(success)} file MP3 đã tải[/green]"]
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
