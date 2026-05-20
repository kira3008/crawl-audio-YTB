"""
Core logic for splitting audio files based on WhisperX JSON timestamps.

Cắt chính xác theo ms bằng cách decode MP3 → PCM vào RAM một lần,
slice theo sample thay vì dùng ffmpeg seeking (vốn chỉ chính xác tới
~26ms do MP3 frame boundary). Export song song để tăng tốc.
"""
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

from splitter_utils import (
    find_media_file,
    get_ffmpeg_path,
    sanitize_filename,
    print_error,
    print_warning,
    print_info,
)


def _hms_to_sec(ts) -> float:
    """'HH:MM:SS.mmm' string hoặc số float → float seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    h, m, s = str(ts).split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


_OK   = "ok"
_SKIP = "skip"
_FAIL = "fail"


class AudioSplitter:
    def __init__(
        self,
        json_file: str,
        output_dir: Optional[str] = None,
        min_dur: float = 0.3,
        max_dur: float = 30.0,
        padding_ms: int = 50,          # padding nhỏ — smart-capped bởi gap thực tế
        max_workers: Optional[int] = None,
    ):
        self.json_path   = Path(json_file).resolve()
        self.min_dur     = min_dur
        self.max_dur     = max_dur
        self.padding_ms  = padding_ms
        self.max_workers = max_workers or min(8, os.cpu_count() or 4)
        self.entries: list[dict] = []
        self.media_file: Optional[Path] = None
        self.ffmpeg_cmd  = get_ffmpeg_path()

        if output_dir is None:
            self.output_dir = self.json_path.parent / "segments" / self.json_path.stem
        else:
            self.output_dir = Path(output_dir)

        self.stats    = {"total": 0, "success": 0, "skipped": 0, "failed": 0}
        self.manifest: list[dict] = []
        self._lock    = Lock()

    # ── setup ─────────────────────────────────────────────────────────────────

    def load_json(self) -> bool:
        if not self.json_path.exists():
            print_error(f"File không tồn tại: {self.json_path}")
            return False
        if self.json_path.suffix.lower() != ".json":
            print_error(f"Cần file .json, nhận được: {self.json_path.suffix}")
            return False
        try:
            with open(self.json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print_error(f"Lỗi đọc JSON: {e}")
            return False
        if not isinstance(data, list):
            print_error("JSON phải là một mảng entries")
            return False
        self.entries = data
        self.stats["total"] = len(data)
        return True

    def validate_setup(self) -> bool:
        if not self.entries:
            print_error("JSON không có entries")
            return False
        self.media_file = find_media_file(self.json_path)
        if not self.media_file:
            print_error(f"Không tìm thấy file MP3: {self.json_path.stem}.mp3")
            return False
        try:
            subprocess.run([self.ffmpeg_cmd, "-version"], capture_output=True, timeout=5)
        except Exception as e:
            print_error(f"FFmpeg không khả dụng: {e}")
            return False
        return True

    def prepare_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── single export (chạy trong thread) ────────────────────────────────────

    @staticmethod
    def _export_chunk(
        audio,              # pydub AudioSegment (shared, read-only)
        audio_len_ms: int,
        idx: int,
        raw_start,
        raw_end,
        text: str,
        prev_end_ms: int,
        next_start_ms: int,
        padding_ms: int,
        min_dur: float,
        max_dur: float,
        output_dir: Path,
        ffmpeg_location: str,
    ) -> tuple[str, Optional[dict]]:
        """
        Slice AudioSegment đã load sẵn và export ra file MP3.
        staticmethod → thread-safe hoàn toàn, không truy cập self.
        """
        if raw_start is None or raw_end is None:
            return _SKIP, None

        start_sec = _hms_to_sec(raw_start)
        end_sec   = _hms_to_sec(raw_end)
        duration  = end_sec - start_sec

        if not (min_dur <= duration <= max_dur):
            return _SKIP, None

        start_ms = int(round(start_sec * 1000))
        end_ms   = int(round(end_sec   * 1000))

        # Smart padding: cap bởi ranh giới segment kề và độ dài file
        cut_start_ms = max(prev_end_ms,   start_ms - padding_ms)
        cut_end_ms   = min(next_start_ms, end_ms   + padding_ms)
        cut_end_ms   = min(cut_end_ms, audio_len_ms)

        safe_text = sanitize_filename(text)[:80] if text else "segment"
        filename  = f"{idx:04d}_{safe_text}.mp3"
        out_path  = output_dir / filename

        try:
            # Slice PCM — chính xác tới ms (pydub dùng index = millisecond)
            chunk = audio[cut_start_ms:cut_end_ms]
            chunk.export(
                str(out_path),
                format="mp3",
                parameters=["-q:a", "2"],
                # Truyền ffmpeg location nếu không phải system ffmpeg
                **({"codec": "libmp3lame"} if False else {}),
            )
        except Exception as e:
            return _FAIL, None

        meta = {
            "file":     filename,
            "start":    raw_start,
            "end":      raw_end,
            "duration": round(duration, 3),
            "text":     text,
        }
        return _OK, meta

    # ── parallel process ──────────────────────────────────────────────────────

    def process(self) -> bool:
        try:
            from pydub import AudioSegment
        except ImportError:
            print_error("Thiếu pydub. Chạy: pip install pydub")
            return False

        import io

        # Lấy path ffmpeg thật từ imageio_ffmpeg (không phụ thuộc PATH hệ thống)
        try:
            import imageio_ffmpeg as _iio
            _ffmpeg_exe = _iio.get_ffmpeg_exe()
        except Exception:
            _ffmpeg_exe = self.ffmpeg_cmd

        # Set converter cho pydub export (cần cho bước ghi MP3 sau)
        AudioSegment.converter = _ffmpeg_exe

        total = len(self.entries)
        print_info(f"Media   : {self.media_file.name}")
        print_info(f"Entries : {total}  →  filter {self.min_dur}s–{self.max_dur}s · pad {self.padding_ms}ms")
        print_info(f"Workers : {self.max_workers} luồng song song")
        print_info(f"Output  : {self.output_dir}\n")

        # Decode MP3 → WAV thô trong RAM bằng ffmpeg trực tiếp.
        # Dùng pipe thay vì pydub.from_file để tránh hoàn toàn ffprobe.
        # pydub.from_wav đọc WAV header bằng Python wave module — không cần ffmpeg.
        print("  📂 Đang decode audio vào bộ nhớ...", flush=True)
        try:
            proc = subprocess.run(
                [_ffmpeg_exe, "-y", "-i", str(self.media_file), "-f", "wav", "pipe:1"],
                capture_output=True, timeout=600,
            )
            if proc.returncode != 0:
                print_error(f"ffmpeg decode lỗi:\n{proc.stderr[-300:].decode(errors='replace')}")
                return False
            audio = AudioSegment.from_wav(io.BytesIO(proc.stdout))
        except Exception as e:
            print_error(f"Không load được audio: {e}")
            return False
        audio_len_ms = len(audio)
        print(f"  ✓ {audio_len_ms / 1000:.1f}s · {audio.channels}ch · {audio.frame_rate}Hz\n", flush=True)

        # Pre-compute ms boundaries để tính smart padding
        bounds_ms: list[tuple[int, int]] = []
        for e in self.entries:
            s  = int(round(_hms_to_sec(e["start"]) * 1000)) if e.get("start") is not None else 0
            en = int(round(_hms_to_sec(e["end"])   * 1000)) if e.get("end")   is not None else 0
            bounds_ms.append((s, en))

        done_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map: dict = {}
            for idx, entry in enumerate(self.entries, 1):
                i             = idx - 1
                prev_end_ms   = bounds_ms[i - 1][1] if i > 0               else 0
                next_start_ms = bounds_ms[i + 1][0] if i < len(bounds_ms) - 1 else audio_len_ms

                future = executor.submit(
                    self._export_chunk,
                    audio, audio_len_ms,
                    idx,
                    entry.get("start"), entry.get("end"), entry.get("text", "").strip(),
                    prev_end_ms, next_start_ms,
                    self.padding_ms, self.min_dur, self.max_dur,
                    self.output_dir, self.ffmpeg_cmd,
                )
                future_map[future] = idx

            for future in as_completed(future_map):
                status, meta = future.result()
                done_count += 1

                with self._lock:
                    if status == _OK:
                        self.stats["success"] += 1
                        self.manifest.append(meta)
                    elif status == _SKIP:
                        self.stats["skipped"] += 1
                    else:
                        self.stats["failed"] += 1

                pct = done_count * 100 // total
                print(
                    f"\r  ⚡ {done_count}/{total} ({pct}%) "
                    f"✅{self.stats['success']} ⏭{self.stats['skipped']} ❌{self.stats['failed']}",
                    end="", flush=True,
                )

        print()

        self.manifest.sort(key=lambda x: x["file"])
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)

        return self.stats["failed"] == 0

    def print_summary(self):
        print("\n" + "─" * 50)
        print(f"✅ Thành công : {self.stats['success']}")
        print(f"⏭  Bỏ qua    : {self.stats['skipped']}  (ngoài {self.min_dur}s–{self.max_dur}s)")
        if self.stats["failed"]:
            print(f"❌ Thất bại  : {self.stats['failed']}")
        print(f"📄 Manifest  : {self.output_dir / 'manifest.json'}")
        print("─" * 50 + "\n")
