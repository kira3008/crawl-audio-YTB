"""
Core logic for splitting audio files based on WhisperX JSON timestamps.
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
    format_time,
    print_success,
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


# Kết quả trả về từ mỗi worker
_OK   = "ok"
_SKIP = "skip"
_FAIL = "fail"


class AudioSplitter:
    def __init__(
        self,
        json_file: str,
        output_dir: Optional[str] = None,
        min_dur: float = 1.0,
        max_dur: float = 30.0,
        padding_ms: int = 100,
        max_workers: Optional[int] = None,   # None → tự động theo CPU
    ):
        self.json_path   = Path(json_file).resolve()
        self.min_dur     = min_dur
        self.max_dur     = max_dur
        self.padding_sec = padding_ms / 1000.0
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
            print_error("JSON phải là một mảng các entries")
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
            print_error(f"Không tìm thấy file MP3 tương ứng: {self.json_path.stem}.mp3")
            return False
        try:
            subprocess.run([self.ffmpeg_cmd, "-version"], capture_output=True, timeout=5)
        except Exception as e:
            print_error(f"FFmpeg không khả dụng: {e}")
            return False
        return True

    def prepare_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── single segment (chạy trong thread) ───────────────────────────────────

    def _cut_segment(
        self,
        idx: int,
        entry: dict,
        prev_end: float,       # end của segment trước (giây)
        next_start: float,     # start của segment sau (giây)
    ) -> tuple[str, Optional[dict]]:
        """
        Trả về (_OK/_SKIP/_FAIL, manifest_entry_or_None).
        Không print gì — chỉ trả kết quả để process() tổng hợp.
        """
        raw_start = entry.get("start")
        raw_end   = entry.get("end")
        text      = entry.get("text", "").strip()

        if raw_start is None or raw_end is None:
            return _SKIP, None

        start_sec = _hms_to_sec(raw_start)
        end_sec   = _hms_to_sec(raw_end)
        duration  = end_sec - start_sec

        if not (self.min_dur <= duration <= self.max_dur):
            return _SKIP, None

        # Smart padding: không tràn qua ranh giới segment kề bên.
        # Ví dụ: gap chỉ 20ms → padding 100ms bị cap về 20ms/2 = 10ms.
        cut_start = max(prev_end,     start_sec - self.padding_sec)
        cut_end   = min(next_start,   end_sec   + self.padding_sec)

        safe_text = sanitize_filename(text)[:80] if text else "segment"
        filename  = f"{idx:04d}_{safe_text}.mp3"
        out_path  = self.output_dir / filename

        # Input-side seeking: -ss trước -i để ffmpeg nhảy thẳng đến vị trí
        # thay vì decode từ đầu file (quan trọng với podcast dài hàng giờ).
        # Dùng -to thay -t để chỉ điểm kết thúc tuyệt đối trong file gốc,
        # tránh lỗi tích lũy khi tính relative duration.
        cmd = [
            self.ffmpeg_cmd, "-y",
            "-ss",     f"{cut_start:.3f}",   # input-side seek
            "-to",     f"{cut_end:.3f}",      # absolute end trong file gốc
            "-i",      str(self.media_file),
            "-acodec", "libmp3lame",
            "-q:a",    "2",
            str(out_path),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=120,
            )
        except subprocess.TimeoutExpired:
            return _FAIL, None
        except Exception:
            return _FAIL, None

        if result.returncode != 0:
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
        total = len(self.entries)
        print_info(f"Media   : {self.media_file.name}")
        print_info(f"Entries : {total}  →  filter {self.min_dur}s–{self.max_dur}s · pad {int(self.padding_sec*1000)}ms")
        print_info(f"Workers : {self.max_workers} luồng song song")
        print_info(f"Output  : {self.output_dir}\n")

        # Pre-compute boundaries để smart padding biết giới hạn của segment kề
        bounds: list[tuple[float, float]] = []
        for e in self.entries:
            s = _hms_to_sec(e["start"]) if e.get("start") is not None else 0.0
            en = _hms_to_sec(e["end"])  if e.get("end")   is not None else 0.0
            bounds.append((s, en))

        done_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {}
            for idx, entry in enumerate(self.entries, 1):
                i          = idx - 1
                prev_end   = bounds[i - 1][1] if i > 0                  else 0.0
                next_start = bounds[i + 1][0] if i < len(bounds) - 1    else float("inf")
                future = executor.submit(self._cut_segment, idx, entry, prev_end, next_start)
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

                # Cập nhật progress trên cùng một dòng
                pct = done_count * 100 // total
                print(
                    f"\r  ⚡ {done_count}/{total} ({pct}%) "
                    f"✅{self.stats['success']} ⏭{self.stats['skipped']} ❌{self.stats['failed']}",
                    end="", flush=True,
                )

        print()  # xuống dòng sau progress

        # Lưu manifest
        manifest_path = self.output_dir / "manifest.json"
        # Sắp xếp theo tên file (= thứ tự index)
        self.manifest.sort(key=lambda x: x["file"])
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
