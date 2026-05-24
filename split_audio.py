#!/usr/bin/env python3
"""
split_audio.py — cắt audio theo câu (WhisperX + Silero VAD).

Usage:
    python split_audio.py                       # menu
    python split_audio.py downloads/file.json   # 1 file
    python split_audio.py downloads/            # cả thư mục
    python split_audio.py file.json --no-vad    # chỉ dùng Whisper timestamps
    python split_audio.py file.json --inspect   # xem cut points, không cắt
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def hms_to_sec(hms: str) -> float:
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def sec_to_hms(sec: float) -> str:
    ms = round((sec % 1) * 1000)
    total = int(sec)
    if ms == 1000:
        ms = 0
        total += 1
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def sanitize_filename(text: str, max_len: int = 80) -> str:
    s = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", text)
    return re.sub(r"\s+", " ", s).strip()[:max_len]


def get_ffmpeg_exe() -> str:
    local = Path(__file__).parent / "ffmpeg.exe"
    if local.exists():
        return str(local)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return "ffmpeg"


# ── Silero VAD ────────────────────────────────────────────────────────────────

def load_vad_model():
    try:
        import silero_vad  # noqa
    except ImportError:
        print("Installing silero-vad...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "silero-vad"])
    from silero_vad import load_silero_vad
    return load_silero_vad()


def _audio_to_tensor(mp3_path: str, ffmpeg_exe: str):
    import numpy as np
    import torch
    r = subprocess.run(
        [ffmpeg_exe, "-loglevel", "error",
         "-i", mp3_path, "-ar", "16000", "-ac", "1", "-f", "s16le", "-"],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace"))
    audio = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(audio)


def run_vad(mp3_path: str, model, ffmpeg_exe: str,
            threshold: float = 0.4,
            min_speech_ms: int = 200,
            min_silence_ms: int = 200) -> list[dict]:
    from silero_vad import get_speech_timestamps
    wav = _audio_to_tensor(mp3_path, ffmpeg_exe)
    return get_speech_timestamps(
        wav, model,
        return_seconds=True,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        sampling_rate=16000,
    )


# ── Layer 1: grouping theo câu ────────────────────────────────────────────────

_SENT_END = (".", "?", "!")


def build_semantic_groups(entries: list[dict],
                          music_gap: float = 2.0,
                          breath_gap: float = 0.2,
                          min_duration: float = 1.0) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []

    def seal():
        if current:
            groups.append(list(current))
            current.clear()

    for idx, entry in enumerate(entries):
        if current and idx > 0:
            gap = hms_to_sec(entry["start"]) - hms_to_sec(entries[idx - 1]["end"])
            if gap > music_gap:
                seal()

        current.append(entry)

        dur  = hms_to_sec(entry["end"]) - hms_to_sec(entry["start"])
        text = entry.get("text", "").strip()

        if dur >= min_duration and text.endswith(_SENT_END):
            next_gap = (
                hms_to_sec(entries[idx + 1]["start"]) - hms_to_sec(entry["end"])
                if idx + 1 < len(entries) else float("inf")
            )
            if next_gap > breath_gap:
                seal()

    seal()
    return groups


# ── Layer 2: VAD boundaries ───────────────────────────────────────────────────

def find_vad_boundaries(group_entries: list[dict],
                        all_vad: list[dict],
                        next_start_sec: float | None = None,
                        start_buf: float = 0.3) -> tuple[float, float]:
    t_start = hms_to_sec(group_entries[0]["start"])
    t_end   = hms_to_sec(group_entries[-1]["end"])

    start_candidates = [v for v in all_vad if abs(v["start"] - t_start) <= start_buf]
    if start_candidates:
        vad_start = min(start_candidates, key=lambda v: abs(v["start"] - t_start))["start"]
    else:
        vad_start = t_start

    # Dùng next_start_sec làm upper bound thay vì t_end + buffer nhỏ.
    # WhisperX hay bị nén timestamp nên t_end không đáng tin; gap tới segment
    # kế (music break) thì đáng tin hơn nhiều.
    if next_start_sec is not None:
        upper = next_start_sec - 0.3
        end_candidates = sorted(
            [v for v in all_vad if v["end"] >= vad_start and v["end"] <= upper],
            key=lambda v: v["end"], reverse=True,
        )
    else:
        end_candidates = sorted(
            [v for v in all_vad if v["end"] >= t_end - 0.1 and v["end"] <= t_end + 1.5],
            key=lambda v: v["end"],
        )

    vad_end = end_candidates[0]["end"] if end_candidates else t_end + 0.15
    return vad_start, vad_end


# ── core splitter ─────────────────────────────────────────────────────────────

def split_one(
    json_path: Path,
    output_root: Path,
    ffmpeg_exe: str,
    vad_model=None,
    console=None,
    inspect: bool = False,
    breath_gap: float = 0.2,
) -> tuple[int, int]:

    def log(msg):
        if console:
            console.print(msg)
        else:
            print(re.sub(r"\[/?[\w ]+\]", "", msg))

    mp3_path = json_path.with_suffix(".mp3")
    if not mp3_path.exists():
        log(f"[red]✗ Không tìm thấy MP3: {mp3_path.name}[/red]")
        return 0, 1

    entries: list[dict] = json.loads(json_path.read_text(encoding="utf-8"))
    if not entries:
        log(f"[yellow]⚠ Transcript rỗng: {json_path.name}[/yellow]")
        return 0, 0

    if inspect:
        log(f"\n[bold cyan]INSPECT: {json_path.name}[/bold cyan]  (breath_gap={breath_gap}s)")

    sem_groups = build_semantic_groups(entries, breath_gap=breath_gap)

    all_vad: list[dict] = []
    if vad_model is not None:
        try:
            all_vad = run_vad(str(mp3_path.resolve()), vad_model, ffmpeg_exe)
        except Exception as e:
            log(f"[yellow]⚠ VAD lỗi ({e}), dùng Whisper timestamps[/yellow]")

    all_groups: list[dict] = []
    for idx, sg in enumerate(sem_groups):
        next_start_sec = (
            hms_to_sec(sem_groups[idx + 1][0]["start"])
            if idx + 1 < len(sem_groups) else None
        )
        if all_vad:
            start_sec, end_sec = find_vad_boundaries(sg, all_vad,
                                                     next_start_sec=next_start_sec)
        else:
            start_sec = hms_to_sec(sg[0]["start"])
            end_sec   = hms_to_sec(sg[-1]["end"])

        all_groups.append({"start_sec": start_sec, "end_sec": end_sec, "entries": sg})

    # merge groups quá ngắn (<1.5s) với group liền kề
    MIN_MS    = 1500
    MERGE_GAP = 0.8
    merged: list[dict] = []
    for g in all_groups:
        dur = (g["end_sec"] - g["start_sec"]) * 1000
        if dur < MIN_MS and merged:
            prev = merged[-1]
            if g["start_sec"] - prev["end_sec"] <= MERGE_GAP:
                merged[-1] = {
                    "start_sec": prev["start_sec"],
                    "end_sec":   g["end_sec"],
                    "entries":   prev["entries"] + g["entries"],
                }
                continue
        merged.append(g)
    all_groups = merged

    seg_dir = Path(__file__).parent / output_root / json_path.stem
    if not inspect:
        seg_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    ok_count = err_count = 0

    for i, group in enumerate(all_groups, 1):
        start = sec_to_hms(group["start_sec"])
        end   = sec_to_hms(group["end_sec"])
        ents  = group["entries"]
        text  = " ".join(e.get("text", "").strip() for e in ents)
        label = sanitize_filename(ents[0].get("text", "") if ents else "", max_len=80)
        filename    = f"{i:04d}_{label}.mp3"
        out_path    = seg_dir / filename
        duration_ms = round((group["end_sec"] - group["start_sec"]) * 1000)

        if inspect:
            first_word = ents[0].get("words", [{}])[0].get("word", "?")
            last_words = ents[-1].get("words", [])
            last_word  = last_words[-1].get("word", "?") if last_words else "?"
            vad_note   = (f" | whisper=[{ents[0]['start']}→{ents[-1]['end']}]"
                          f" vad=[{start}→{end}]") if all_vad else ""
            log(f"  #{i:04d} {start}→{end} ({duration_ms}ms)"
                f" | {len(ents)} entries | '{first_word}…{last_word}'{vad_note}")
            log(f"         {text[:100]}")
            ok_count += 1
            continue

        cmd = [
            ffmpeg_exe, "-y", "-loglevel", "error",
            "-ss", start, "-to", end,
            "-i", str(mp3_path),
            "-c", "copy", str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace").strip()[-200:]
            log(f"[red]✗ {i:04d}: {err}[/red]")
            err_count += 1
            continue

        manifest.append({
            "file":        filename,
            "start":       start,
            "end":         end,
            "duration_ms": duration_ms,
            "entries":     len(ents),
            "text":        text,
        })
        ok_count += 1

    if manifest:
        (seg_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return ok_count, err_count


# ── batch ─────────────────────────────────────────────────────────────────────

def collect_json_files(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    for p in paths:
        if p.is_dir():
            result.extend(f for f in sorted(p.glob("*.json"))
                          if f.name != "manifest.json")
        elif p.suffix.lower() == ".json" and p.exists():
            result.append(p)
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cắt audio theo câu: WhisperX grouping + Silero VAD boundaries."
    )
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--no-vad", action="store_true",
                        help="Dùng WhisperX timestamps (không VAD)")
    parser.add_argument("--inspect", "-n", action="store_true",
                        help="Dry-run: in ra cut points, không cắt file")
    parser.add_argument("--breath-gap", type=float, default=0.2,
                        help="Ngưỡng khoảng ngắt câu (mặc định 0.2s)")
    args = parser.parse_args()

    import io, sys as _sys
    if hasattr(_sys.stdout, "buffer"):
        _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer,
                                       encoding="utf-8", errors="replace")

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import (Progress, SpinnerColumn,
                                   TextColumn, BarColumn, MofNCompleteColumn)
        console = Console(highlight=False)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import (Progress, SpinnerColumn,
                                   TextColumn, BarColumn, MofNCompleteColumn)
        console = Console(highlight=False)

    console.print(Panel(
        "[bold cyan]Audio Segment Splitter[/bold cyan]\n"
        "[dim]WhisperX grouping + Silero VAD boundaries[/dim]",
        border_style="cyan", padding=(0, 4),
    ))

    ffmpeg_exe = get_ffmpeg_exe()

    vad_model = None
    if not args.no_vad:
        with console.status("[bold green]Loading Silero VAD…[/bold green]"):
            try:
                vad_model = load_vad_model()
                console.print("[green]✓ Silero VAD ready[/green]\n")
            except Exception as e:
                console.print(
                    f"[yellow]⚠ VAD failed ({e}), using --no-vad fallback[/yellow]\n"
                )

    if args.inputs:
        json_files = collect_json_files([Path(p) for p in args.inputs])
    else:
        try:
            import questionary
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "questionary"])
            import questionary

        downloads = Path("downloads")
        candidates = sorted(f for f in downloads.glob("*.json")
                            if f.name != "manifest.json") if downloads.exists() else []
        if not candidates:
            candidates = sorted(f for f in Path(".").glob("*.json")
                                if f.name != "manifest.json")
        if not candidates:
            console.print("[red]Không tìm thấy file JSON nào.[/red]")
            return

        selected = questionary.checkbox(
            "Chọn file JSON:",
            choices=[questionary.Choice(p.name, value=p, checked=True)
                     for p in candidates],
        ).ask()
        if not selected:
            return
        json_files = selected

        default_out = str(candidates[0].parent / "segments")
        out_raw = questionary.text("Thư mục lưu:", default=default_out).ask()
        if out_raw is None:
            return
        args.output = out_raw

    if not json_files:
        console.print("[red]Không có file JSON nào.[/red]")
        return

    total_ok = total_err = 0

    with Progress(SpinnerColumn(), TextColumn("[bold white]{task.description}"),
                  BarColumn(), MofNCompleteColumn(),
                  console=console, transient=False) as progress:

        for jf in json_files:
            out_root = Path(args.output) if args.output else jf.parent / "segments"
            task = progress.add_task(jf.stem[:50], total=None)

            ok, err = split_one(
                json_path=jf,
                output_root=out_root,
                ffmpeg_exe=ffmpeg_exe,
                vad_model=vad_model,
                console=console if args.inspect else None,
                inspect=args.inspect,
                breath_gap=args.breath_gap,
            )
            total_ok  += ok
            total_err += err

            status = f"[green]✓ {ok} đoạn[/green]"
            if err:
                status += f" [red]✗ {err} lỗi[/red]"
            progress.update(task, completed=1, total=1,
                            description=f"{jf.stem[:40]} — {status}")

    console.print(Panel(
        "\n".join(filter(None, [
            f"[green]✓ {total_ok} đoạn đã cắt[/green]",
            f"[red]✗ {total_err} lỗi[/red]" if total_err else "",
            f"[dim]Thư mục: {Path(args.output).resolve()}[/dim]" if args.output else "",
        ])),
        title="[bold]Hoàn tất[/bold]",
        border_style="green" if not total_err else "yellow",
        padding=(0, 4),
    ))


if __name__ == "__main__":
    main()
