#!/usr/bin/env python3
"""
split_audio.py — 2-layer audio segmentation (clean architecture).

Hai layer KHÔNG phụ thuộc lẫn nhau:

  Layer 1 — WhisperX (text-only):
    • Dùng entry gaps > 2s  → detect music break (đáng tin cậy)
    • Dùng dấu .?!          → detect sentence boundary
    • Dùng entry gap ≤ 0.2s → biết hai entry cùng hơi thở
    → Output: "semantic groups" (danh sách entries theo câu)
    ❌ KHÔNG dùng start/end để quyết định cut point

  Layer 2 — Silero VAD (audio-only):
    • Tìm speech boundaries chính xác ±30ms
    → Output: danh sách (start, end) của từng speech burst
    ❌ KHÔNG dùng text để quyết định gì

  Kết hợp:
    • Semantic group → tìm VAD boundaries bằng rough time range
    • Cut tại VAD start/end (không phải WhisperX start/end)

Usage:
    python split_audio.py                       # menu
    python split_audio.py downloads/file.json   # 1 file
    python split_audio.py downloads/            # cả thư mục
    python split_audio.py file.json --no-vad    # fallback WhisperX
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

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
    """MP3 → 16kHz mono float32 tensor. Không dùng torchaudio."""
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
    """
    Layer 2: Silero VAD — trả về [{start, end}] tính bằng giây.
    min_silence_ms=200: mỗi khoảng ngừng ≥200ms tạo một ranh giới.
    """
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


# ── Layer 1: WhisperX text-only grouping ─────────────────────────────────────

_SENT_END = (".", "?", "!")


def build_semantic_groups(entries: list[dict],
                          music_gap: float = 2.0,
                          breath_gap: float = 0.2,
                          min_duration: float = 1.0) -> list[list[dict]]:
    """
    Layer 1 — Chỉ dùng TEXT + ENTRY GAPS. Không dùng timestamps để cut.

    Quy tắc:
      gap > music_gap (2s)  → music break, đóng group (gap này đáng tin)
      entry kết thúc .?!   → câu hoàn chỉnh
      gap đến entry tiếp ≤ breath_gap (0.2s) → cùng hơi, chưa đóng
      gap đến entry tiếp > breath_gap        → ngừng thực, đóng group
    """
    groups: list[list[dict]] = []
    current: list[dict] = []

    def seal():
        if current:
            groups.append(list(current))
            current.clear()

    for idx, entry in enumerate(entries):
        # Music break → seal trước khi thêm entry mới
        if current and idx > 0:
            gap = (hms_to_sec(entry["start"])
                   - hms_to_sec(entries[idx - 1]["end"]))
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


# ── Layer 2: tìm VAD boundaries cho mỗi semantic group ───────────────────────

def find_vad_boundaries(group_entries: list[dict],
                        all_vad: list[dict],
                        start_buf: float = 0.3,
                        end_buf: float = 0.8) -> tuple[float, float]:
    """
    Dùng rough time range của entries để lọc VAD segments liên quan.
    Trả về (vad_start, vad_end) chính xác — hoàn toàn từ VAD, không phải Whisper.

    Lý do dùng timestamps ở đây: chỉ để LỌC VAD (tìm đúng đoạn trong file),
    không phải làm điểm cắt. VAD start/end mới là điểm cắt thực.
    """
    t_start = hms_to_sec(group_entries[0]["start"])
    t_end   = hms_to_sec(group_entries[-1]["end"])

    # Lọc VAD có start trong [t_start - start_buf, t_end + end_buf]
    # Dùng v["start"] để tránh lấy VAD từ music section kế bên
    relevant = [
        v for v in all_vad
        if v["start"] >= t_start - start_buf and v["start"] < t_end + end_buf
    ]

    if not relevant:
        # Fallback: dùng WhisperX timestamps (ít chính xác nhưng có còn hơn không)
        return t_start, t_end

    relevant.sort(key=lambda x: x["start"])

    # Tìm chuỗi VAD liên tiếp lớn nhất (loại bỏ nhảy vọt do nhạc nền)
    chains: list[list[dict]] = []
    cur_chain = [relevant[0]]
    for v in relevant[1:]:
        if v["start"] - cur_chain[-1]["end"] > 2.0:   # music gap
            chains.append(cur_chain)
            cur_chain = [v]
        else:
            cur_chain.append(v)
    chains.append(cur_chain)

    # Chọn chain có tổng overlap với [t_start, t_end] lớn nhất
    def overlap(chain):
        return (min(t_end, chain[-1]["end"])
                - max(t_start, chain[0]["start"]))

    best = max(chains, key=overlap)
    return best[0]["start"], best[-1]["end"]


# ── core splitter ─────────────────────────────────────────────────────────────

def split_one(
    json_path: Path,
    output_root: Path,
    ffmpeg_exe: str,
    vad_model=None,
    console=None,
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

    # ── Layer 1: semantic groups (text only) ──────────────────────────────────
    sem_groups = build_semantic_groups(entries)

    # ── Layer 2: VAD boundaries ───────────────────────────────────────────────
    all_vad: list[dict] = []
    if vad_model is not None:
        try:
            all_vad = run_vad(str(mp3_path.resolve()), vad_model, ffmpeg_exe)
        except Exception as e:
            log(f"[yellow]⚠ VAD lỗi ({e}), dùng Whisper timestamps[/yellow]")

    # ── Kết hợp: semantic group → VAD boundaries ─────────────────────────────
    all_groups: list[dict] = []
    for sg in sem_groups:
        if all_vad:
            start_sec, end_sec = find_vad_boundaries(sg, all_vad)
        else:
            start_sec = hms_to_sec(sg[0]["start"])
            end_sec   = hms_to_sec(sg[-1]["end"])

        all_groups.append({
            "start_sec": start_sec,
            "end_sec":   end_sec,
            "entries":   sg,
        })

    # ── merge groups quá ngắn ─────────────────────────────────────────────────
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

    # ── output ────────────────────────────────────────────────────────────────
    seg_dir = Path(__file__).parent / output_root / json_path.stem
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

        cmd = [
            ffmpeg_exe, "-y", "-loglevel", "error",
            "-i", str(mp3_path),
            "-ss", start, "-to", end,
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
        description="2-layer split: WhisperX text grouping + Silero VAD boundaries."
    )
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--no-vad", action="store_true",
                        help="Dùng WhisperX timestamps (không VAD)")
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
        "[dim]Layer 1: WhisperX text → semantic groups[/dim]\n"
        "[dim]Layer 2: Silero VAD   → audio boundaries[/dim]",
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
                console=None,
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
