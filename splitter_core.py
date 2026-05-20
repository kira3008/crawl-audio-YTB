"""
Core logic for splitting audio files
"""
import json
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from splitter_utils import (
    sanitize_filename,
    find_media_file,
    get_ffmpeg_path,
    format_time,
    print_success,
    print_error,
    print_warning,
    print_info,
)


class AudioSplitter:
    """Main audio splitting class"""
    
    def __init__(self, json_file: str, output_dir: Optional[str] = None):
        """
        Initialize AudioSplitter
        
        Args:
            json_file: Path to JSON file with timestamps
            output_dir: Output directory (default: json_dir/segments/{json_stem})
        """
        self.json_path = Path(json_file).resolve()
        self.entries: List[Dict] = []
        self.media_file: Optional[Path] = None
        self.ffmpeg_cmd: str = get_ffmpeg_path()
        
        # Set output directory
        if output_dir is None:
            self.output_dir = self.json_path.parent / "segments" / self.json_path.stem
        else:
            self.output_dir = Path(output_dir)
        
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
        }
        self.failed_entries: List[Tuple[int, str, str]] = []
    
    def load_json(self) -> bool:
        """Load and validate JSON file"""
        if not self.json_path.exists():
            print_error(f"File không tồn tại: {self.json_path}")
            return False
        
        # Check file extension
        if self.json_path.suffix.lower() != '.json':
            print_error(f"File phải có extension .json, nhưng bạn cung cấp: {self.json_path.suffix}")
            print_info(f"Hãy chạy với: python run_splitter.py \"{self.json_path.with_suffix('.json')}\"")
            return False
        
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print_error(f"Lỗi đọc JSON: {e}")
            return False
        except UnicodeDecodeError as e:
            print_error(f"Lỗi encoding file JSON: {e}")
            print_info("File JSON có thể bị hỏng. Hãy kiểm tra lại.")
            return False
        
        if not isinstance(data, list):
            print_error("JSON phải là một mảng các entries")
            return False
        
        self.entries = data
        self.stats["total"] = len(data)
        return True
    
    def validate_setup(self) -> bool:
        """Validate all required files exist"""
        if not self.entries:
            print_error("Không có entries trong JSON")
            return False
        
        # Find media file
        self.media_file = find_media_file(self.json_path)
        if not self.media_file:
            print_error(f"Không tìm thấy file media: {self.json_path.name}")
            return False
        
        # Check FFmpeg
        try:
            subprocess.run(
                [self.ffmpeg_cmd, "-version"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=5
            )
        except Exception as e:
            print_error(f"FFmpeg không khả dụng: {e}")
            return False
        
        return True
    
    def prepare_output_dir(self):
        """Create output directory"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def split_segment(self, idx: int, entry: Dict) -> bool:
        """
        Extract a single audio segment
        
        Args:
            idx: Entry index (1-based)
            entry: Entry dict with start, end, text
        
        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = entry.get('start')
            end_time = entry.get('end')
            text = entry.get('text', '').strip()
            
            # Validate timestamps
            if start_time is None or end_time is None:
                print_warning(f"[{idx}] Thiếu start/end time")
                self.stats["skipped"] += 1
                return False
            
            duration = end_time - start_time
            
            # Skip segments that are too short (< 0.1 seconds)
            min_duration = 0.1
            if duration < min_duration:
                print_warning(f"[{idx}] Segment quá ngắn ({duration:.3f}s < {min_duration}s) → Bỏ qua")
                self.stats["skipped"] += 1
                return False
            
            # Create filename
            if text:
                safe_text = sanitize_filename(text)
                filename = f"{idx:04d}_{safe_text}.mp3"
            else:
                filename = f"{idx:04d}_segment.mp3"
            
            output_file = self.output_dir / filename
            
            # Run FFmpeg
            cmd = [
                self.ffmpeg_cmd,
                "-i", str(self.media_file),
                "-ss", str(start_time),
                "-t", str(duration),
                "-c", "copy",
                "-y",
                str(output_file)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300
            )
            
            if result.returncode == 0:
                self.stats["success"] += 1
                return True
            else:
                self.failed_entries.append((idx, filename, result.stderr[:100]))
                self.stats["failed"] += 1
                return False
        
        except subprocess.TimeoutExpired:
            self.failed_entries.append((idx, filename, "Timeout"))
            self.stats["failed"] += 1
            return False
        except Exception as e:
            self.failed_entries.append((idx, filename, str(e)[:100]))
            self.stats["failed"] += 1
            return False
    
    def process(self) -> bool:
        """Process all entries"""
        print_info(f"File media: {self.media_file.name}")
        print_info(f"Total entries: {len(self.entries)}")
        print_info(f"Min duration to extract: 0.1s (bỏ qua nếu < 0.1s)")
        print_info(f"Output directory: {self.output_dir}\n")
        
        for idx, entry in enumerate(self.entries, 1):
            text = entry.get('text', '')[:50]
            start = entry.get('start', 0)
            end = entry.get('end', 0)
            duration = end - start
            
            success = self.split_segment(idx, entry)
            
            # Only print successful and failed - skip short segments from output
            if duration >= 0.1:
                status = "✅" if success else "❌"
                print(f"{status} [{idx}/{len(self.entries)}] {format_time(start)} → {format_time(end)} ({duration:.3f}s)")
                print(f"   └─ {text}...\n")
        
        return self.stats["failed"] == 0
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*70)
        print(f"✅ Thành công: {self.stats['success']}/{self.stats['total']}")
        
        if self.stats["failed"] > 0:
            print(f"❌ Thất bại: {self.stats['failed']}")
        
        if self.stats["skipped"] > 0:
            print(f"⏭️  Bỏ qua (quá ngắn): {self.stats['skipped']}")
        
        if self.failed_entries:
            print("\n📋 Failed entries:")
            for idx, filename, error in self.failed_entries[:10]:  # Show first 10
                print(f"  [{idx}] {filename}: {error}")
            
            if len(self.failed_entries) > 10:
                print(f"  ... và {len(self.failed_entries) - 10} entries khác")
        
        print("="*70 + "\n")
