"""
Core logic for splitting audio files based on WhisperX JSON timestamps.
"""
import json
import subprocess
from pathlib import Path
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


class AudioSplitter:
    def __init__(
        self,
        json_file: str,
        output_dir: Optional[str] = None,
        min_dur: float = 1.0,       # giây — bỏ qua segment ngắn hơn
        max_dur: float = 30.0,      # giây — bỏ qua segment dài hơn
        padding_ms: int = 100,      # ms thêm vào mỗi đầu/cuối khi cắt
    ):
        self.json_path   = Path(json_file).resolve()
        self.min_dur     = min_dur
        self.max_dur     = max_dur
        self.padding_sec = padding_ms / 1000.0
        self.entries: list[dict] = []
        self.media_file: Optional[Path] = None
        self.ffmpeg_cmd  = get_ffmpeg_path()

        if output_dir is None:
            self.output_dir = self.json_path.parent / "segments" / self.json_path.stem
        else:
            self.output_dir = Path(output_dir)

        self.stats = {"total": 0, "success": 0, "skipped": 0, "failed": 0}
        self.manifest: list[dict] = []

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
            subprocess.run(
                [self.ffmpeg_cmd, "-version"],
                capture_output=True, timeout=5,
            )
        except Exception as e:
            print_error(f"FFmpeg không khả dụng: {e}")
            return False
        return True

    def prepare_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── split ─────────────────────────────────────────────────────────────────

    def split_segment(self, idx: int, entry: dict) -> bool:
        raw_start = entry.get("start")
        raw_end   = entry.get("end")
        text      = entry.get("text", "").strip()

        if raw_start is None or raw_end is None:
            print_warning(f"[{idx}] Thiếu start/end — bỏ qua")
            self.stats["skipped"] += 1
            return False

        start_sec = _hms_to_sec(raw_start)
        end_sec   = _hms_to_sec(raw_end)
        duration  = end_sec - start_sec

        if duration < self.min_dur:
            self.stats["skipped"] += 1
            return False
        if duration > self.max_dur:
            self.stats["skipped"] += 1
            return False

        # Áp dụng padding — không vượt qua 0 ở đầu
        cut_start = max(0.0, start_sec - self.padding_sec)
        cut_dur   = (end_sec + self.padding_sec) - cut_start

        safe_text = sanitize_filename(text)[:80] if text else "segment"
        filename  = f"{idx:04d}_{safe_text}.mp3"
        out_path  = self.output_dir / filename

        # Re-encode (libmp3lame) để cắt chính xác theo ms.
        # -c copy với MP3 snap về frame boundary (~26ms error).
        cmd = [
            self.ffmpeg_cmd, "-y",
            "-i",      str(self.media_file),
            "-ss",     f"{cut_start:.3f}",
            "-t",      f"{cut_dur:.3f}",
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
            print_error(f"[{idx}] Timeout")
            self.stats["failed"] += 1
            return False
        except Exception as e:
            print_error(f"[{idx}] {e}")
            self.stats["failed"] += 1
            return False

        if result.returncode != 0:
            print_error(f"[{idx}] ffmpeg lỗi: {result.stderr[-200:]}")
            self.stats["failed"] += 1
            return False

        self.stats["success"] += 1
        self.manifest.append({
            "file":     filename,
            "start":    raw_start,
            "end":      raw_end,
            "duration": round(duration, 3),
            "text":     text,
        })
        return True

    # ── process ───────────────────────────────────────────────────────────────

    def process(self) -> bool:
        print_info(f"Media : {self.media_file.name}")
        print_info(f"Entries: {len(self.entries)}")
        print_info(f"Filter : {self.min_dur}s – {self.max_dur}s · padding {int(self.padding_sec*1000)}ms")
        print_info(f"Output : {self.output_dir}\n")

        for idx, entry in enumerate(self.entries, 1):
            start_sec = _hms_to_sec(entry.get("start", 0))
            end_sec   = _hms_to_sec(entry.get("end",   0))
            duration  = end_sec - start_sec
            text      = entry.get("text", "")[:60]

            ok = self.split_segment(idx, entry)

            if ok:
                print(f"✅ [{idx:04d}] {format_time(start_sec)} → {format_time(end_sec)} ({duration:.2f}s)")
                print(f"        {text}\n")

        # Lưu manifest chứa toàn bộ metadata của các segment đã tạo
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)

        return self.stats["failed"] == 0

    def print_summary(self):
        print("\n" + "=" * 60)
        print(f"✅ Thành công : {self.stats['success']}")
        print(f"⏭  Bỏ qua    : {self.stats['skipped']}  (ngoài {self.min_dur}s–{self.max_dur}s)")
        if self.stats["failed"]:
            print(f"❌ Thất bại  : {self.stats['failed']}")
        print(f"📄 Manifest  : {self.output_dir / 'manifest.json'}")
        print("=" * 60 + "\n")
