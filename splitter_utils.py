"""
Utilities for audio splitting
"""
import re
import sys
from pathlib import Path
from typing import Optional


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """Convert text to safe filename"""
    safe = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", text)
    safe = re.sub(r"\s+", " ", safe).strip()
    return safe[:max_length]


def find_media_file(json_path: Path) -> Optional[Path]:
    """Find corresponding media file (mp3, mp4, webm, etc.)"""
    base_name = json_path.stem
    parent_dir = json_path.parent
    
    # Try common audio/video formats
    for ext in ['.mp3', '.mp4', '.webm', '.m4a', '.wav', '.flac']:
        media_file = parent_dir / (base_name + ext)
        if media_file.exists():
            return media_file
    
    # Also try with .part extension
    for ext in ['.webm', '.mp4', '.mkv']:
        media_file = parent_dir / (base_name + ext + '.part')
        if media_file.exists():
            return media_file
    
    return None


def get_ffmpeg_path() -> str:
    """Get FFmpeg executable path"""
    ffmpeg_cmd = "ffmpeg"
    
    # Check local ffmpeg
    local_ffmpeg = Path(__file__).parent / "ffmpeg.exe"
    if local_ffmpeg.exists():
        return str(local_ffmpeg)
    
    # Try imageio_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except:
        pass
    
    return ffmpeg_cmd


def format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def print_section(title: str, width: int = 70):
    """Print formatted section header"""
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def print_success(msg: str):
    """Print success message"""
    print(f"✅ {msg}")


def print_error(msg: str):
    """Print error message"""
    print(f"❌ {msg}")


def print_warning(msg: str):
    """Print warning message"""
    print(f"⚠️  {msg}")


def print_info(msg: str):
    """Print info message"""
    print(f"ℹ️  {msg}")
