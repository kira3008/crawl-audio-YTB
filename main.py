import subprocess
import sys
import json
import re
from pathlib import Path
from datetime import datetime

YTDLP_CMD = [sys.executable, "-m", "yt_dlp"]

VIETNAMESE_CHARS = set(
    "àáâãèéêìíòóôõùúýăđơư"
    "ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ"
    "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ"
    "ẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỶỸỴ"
)


def get_ffmpeg_dir() -> str | None:
    script_dir = Path(__file__).parent
    local_ffmpeg = script_dir / "ffmpeg.exe"
    if local_ffmpeg.exists():
        return str(script_dir)
    try:
        import imageio_ffmpeg
        import shutil
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
    if hours:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


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
            lang = info.get("language") or ""
            videos.append({
                "id": vid_id,
                "title": title,
                "url": url,
                "channel": info.get("uploader") or info.get("channel") or "N/A",
                "duration": info.get("duration"),
                "is_vi": lang == "vi" or is_vietnamese(title),
            })
        except json.JSONDecodeError:
            continue
    return videos


def download_audio(video: dict, output_dir: str, ffmpeg_dir: str | None = None) -> bool:
    safe_title = sanitize_filename(video["title"])
    output_template = str(Path(output_dir) / f"{safe_title}.%(ext)s")
    cmd = YTDLP_CMD + [
        "--extract-audio", "--audio-format", "mp3", "--audio-quality", "0",
        "--output", output_template, "--no-playlist", "--retries", "3",
    ]
    if ffmpeg_dir:
        cmd += ["--ffmpeg-location", ffmpeg_dir]
    cmd.append(video["url"])
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return result.returncode == 0


def update_links_md(videos: list[dict], output_dir: str):
    links_file = Path(output_dir) / "crawled_links.md"
    header = "" if links_file.exists() else "# Danh sách video đã crawl\n\n"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    section = f"## {timestamp}\n\n"
    for v in videos:
        dur = format_duration(v.get("duration"))
        section += f"- [{v['title']}]({v['url']}) — {v['channel']} `{dur}`\n"
    section += "\n"
    with open(links_file, "a", encoding="utf-8") as f:
        f.write(header + section)


def main():
    check_dependencies()

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich import box
    import questionary

    console = Console()

    console.print(
        Panel(
            "[bold cyan]YouTube Vietnamese Audio Crawler[/bold cyan]\n"
            "[dim]Tìm kiếm · Lọc tiếng Việt · Tải MP3[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )
    )

    # --- Inputs ---
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

    output_dir = questionary.text("Thư mục lưu file:", default="downloads").ask()
    if output_dir is None:
        return

    console.print()

    # --- Search ---
    with console.status(f"[bold green]Đang tìm kiếm '{keyword}'...[/bold green]"):
        all_videos = search_videos(keyword, max_results)

    vi_videos = [v for v in all_videos if v["is_vi"]]

    if not vi_videos:
        console.print(
            Panel(
                "[yellow]Không tìm thấy video tiếng Việt nào.[/yellow]\n"
                "[dim]Thử dùng từ khóa tiếng Việt như: nhạc trẻ, tin tức, hài hước[/dim]",
                border_style="yellow",
            )
        )
        return

    console.print(
        f"[green]Tìm thấy[/green] [bold]{len(vi_videos)}[/bold] video tiếng Việt "
        f"[dim](từ {len(all_videos)} kết quả)[/dim]\n"
    )

    # --- Results table ---
    table = Table(box=box.ROUNDED, border_style="cyan", show_lines=False)
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Tiêu đề", style="bold white", max_width=52)
    table.add_column("Kênh", style="cyan", max_width=22)
    table.add_column("Thời lượng", style="green", justify="right")

    for i, v in enumerate(vi_videos, 1):
        table.add_row(str(i), v["title"], v["channel"], format_duration(v.get("duration")))

    console.print(table)
    console.print()

    # --- Checkbox chọn video ---
    choices = [
        questionary.Choice(
            title=f"{v['title']} [{format_duration(v.get('duration'))}]",
            value=v,
            checked=True,
        )
        for v in vi_videos
    ]

    selected = questionary.checkbox(
        "Chọn video muốn tải (Space để chọn/bỏ, Enter để xác nhận):",
        choices=choices,
    ).ask()

    if not selected:
        console.print("[yellow]Không có video nào được chọn.[/yellow]")
        return

    console.print()

    # --- Download ---
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ffmpeg_dir = get_ffmpeg_dir()
    success, failed = [], []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[filename]}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("", total=len(selected), filename="Chuẩn bị...")

        for video in selected:
            short = video["title"][:45] + ("…" if len(video["title"]) > 45 else "")
            progress.update(task, filename=short)
            if download_audio(video, output_dir, ffmpeg_dir):
                success.append(video)
            else:
                failed.append(video)
            progress.advance(task)

    # --- Update markdown log ---
    if success:
        update_links_md(success, output_dir)

    # --- Summary ---
    lines = [f"[green]✓ {len(success)} file MP3 đã tải[/green]"]
    if failed:
        lines.append(f"[red]✗ {len(failed)} thất bại[/red]")
    lines.append(f"\n[dim]Thư mục : {Path(output_dir).resolve()}[/dim]")
    if success:
        lines.append(f"[dim]Log link: {Path(output_dir).resolve() / 'crawled_links.md'}[/dim]")

    console.print(
        Panel(
            "\n".join(lines),
            title="[bold]Hoàn tất[/bold]",
            border_style="green" if not failed else "yellow",
            padding=(1, 4),
        )
    )


if __name__ == "__main__":
    main()
