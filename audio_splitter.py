"""
Audio Splitter - Tách audio thành từng đoạn dựa trên timestamp trong JSON
"""
import json
import subprocess
import sys
import re
from pathlib import Path
from typing import Optional


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """Convert text to safe filename"""
    # Remove invalid characters
    safe = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", text)
    # Replace multiple spaces with single space
    safe = re.sub(r"\s+", " ", safe).strip()
    # Truncate to max length
    return safe[:max_length]


def find_media_file(json_path: Path) -> Optional[Path]:
    """Find corresponding media file (mp3, mp4, webm, etc.)"""
    base_name = json_path.stem  # Remove .json extension
    parent_dir = json_path.parent
    
    # Try common audio/video formats
    for ext in ['.mp3', '.mp4', '.webm', '.m4a', '.wav', '.flac']:
        media_file = parent_dir / (base_name + ext)
        if media_file.exists():
            return media_file
    
    # Also try without .part if it exists
    for ext in ['.webm', '.mp4', '.mkv']:
        media_file = parent_dir / (base_name + ext + '.part')
        if media_file.exists():
            return media_file
    
    return None


def split_audio(json_file: str, output_dir: Optional[str] = None):
    """
    Split audio file into segments based on JSON timestamps
    
    Args:
        json_file: Path to JSON file with timestamps
        output_dir: Output directory for segments (default: json_dir/segments)
    """
    json_path = Path(json_file).resolve()
    
    if not json_path.exists():
        print(f"❌ File không tồn tại: {json_path}")
        return
    
    # Read JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Lỗi đọc JSON: {e}")
        return
    
    if not isinstance(entries, list):
        print("❌ JSON phải là một mảng các entries")
        return
    
    # Find media file
    media_file = find_media_file(json_path)
    if not media_file:
        print(f"❌ Không tìm thấy file media tương ứng cho: {json_path.name}")
        return
    
    print(f"📁 File media: {media_file.name}")
    print(f"📊 Tổng entries: {len(entries)}\n")
    
    # Create output directory
    if output_dir is None:
        output_dir = json_path.parent / "segments" / json_path.stem
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Output directory: {output_path}\n")
    
    # Get FFmpeg path
    ffmpeg_cmd = "ffmpeg"
    
    # Check if ffmpeg is available locally
    local_ffmpeg = Path(__file__).parent / "ffmpeg.exe"
    if local_ffmpeg.exists():
        ffmpeg_cmd = str(local_ffmpeg)
    
    # Try imageio_ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
    except:
        pass
    
    # Extract segments
    failed = []
    success_count = 0
    
    for idx, entry in enumerate(entries, 1):
        try:
            start_time = entry.get('start')
            end_time = entry.get('end')
            text = entry.get('text', '').strip()
            
            if start_time is None or end_time is None:
                print(f"⚠️  Entry {idx}: Thiếu start/end time")
                continue
            
            # Create filename from text
            if text:
                filename = f"{idx:04d}_{sanitize_filename(text)}.mp3"
            else:
                filename = f"{idx:04d}_segment.mp3"
            
            output_file = output_path / filename
            
            # Calculate duration
            duration = end_time - start_time
            
            # FFmpeg command
            cmd = [
                ffmpeg_cmd,
                "-i", str(media_file),
                "-ss", str(start_time),
                "-t", str(duration),
                "-c", "copy",  # Copy codec without re-encoding (faster)
                "-y",  # Overwrite output file
                str(output_file)
            ]
            
            # Run FFmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                success_count += 1
                print(f"✅ [{idx}/{len(entries)}] {filename}")
                print(f"   ├─ Time: {start_time:.2f}s → {end_time:.2f}s ({duration:.2f}s)")
                print(f"   └─ Text: {text[:60]}{'...' if len(text) > 60 else ''}\n")
            else:
                print(f"❌ [{idx}/{len(entries)}] Lỗi: {filename}")
                print(f"   {result.stderr[:200]}\n")
                failed.append((idx, filename))
        
        except subprocess.TimeoutExpired:
            print(f"⏱️  [{idx}/{len(entries)}] Timeout: {filename}\n")
            failed.append((idx, filename))
        except Exception as e:
            print(f"❌ [{idx}/{len(entries)}] Exception: {str(e)[:100]}\n")
            failed.append((idx, filename))
    
    # Summary
    print("\n" + "="*60)
    print(f"✅ Thành công: {success_count}/{len(entries)}")
    if failed:
        print(f"❌ Thất bại: {len(failed)}")
        print("\nFailed entries:")
        for idx, filename in failed:
            print(f"  - [{idx}] {filename}")
    print("="*60)


def process_all_json_in_directory(directory: str, output_base_dir: Optional[str] = None):
    """Process all JSON files in a directory"""
    dir_path = Path(directory).resolve()
    
    if not dir_path.is_dir():
        print(f"❌ Thư mục không tồn tại: {dir_path}")
        return
    
    json_files = sorted(dir_path.glob("*.json"))
    
    if not json_files:
        print(f"❌ Không tìm thấy JSON file nào trong: {dir_path}")
        return
    
    print(f"📂 Found {len(json_files)} JSON files\n")
    print("="*70)
    
    for i, json_file in enumerate(json_files, 1):
        print(f"\n🔄 [{i}/{len(json_files)}] Processing: {json_file.name}")
        print("-"*70)
        
        output_dir = None
        if output_base_dir:
            output_dir = Path(output_base_dir) / json_file.stem
        
        split_audio(str(json_file), output_dir)
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_splitter.py <json_file_or_directory> [output_dir]")
        print("\nExamples:")
        print("  # Process single file:")
        print("  python audio_splitter.py downloads/video.json")
        print()
        print("  # Process all JSON files in directory:")
        print("  python audio_splitter.py downloads")
        print()
        print("  # Process all with custom output directory:")
        print("  python audio_splitter.py downloads output_segments")
        sys.exit(1)
    
    path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    path_obj = Path(path)
    
    if path_obj.is_dir():
        # Process all JSON files in directory
        process_all_json_in_directory(path, output_dir)
    else:
        # Process single file
        split_audio(path, output_dir)
