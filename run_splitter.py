"""
Main script to run audio splitting
Usage:
    python run_splitter.py <json_file_or_directory> [output_dir]
"""
import sys
from pathlib import Path

from splitter_core import AudioSplitter
from splitter_utils import print_section, print_error, print_info


def process_single_file(json_file: str, output_dir: str = None):
    """Process a single JSON file"""
    print_section(f"Processing: {Path(json_file).name}")
    
    splitter = AudioSplitter(json_file, output_dir)
    
    # Load and validate
    if not splitter.load_json():
        return False
    
    if not splitter.validate_setup():
        return False
    
    splitter.prepare_output_dir()
    
    # Process
    splitter.process()
    splitter.print_summary()
    
    return True


def process_directory(directory: str, output_base_dir: str = None):
    """Process all JSON files in a directory"""
    dir_path = Path(directory).resolve()
    
    if not dir_path.is_dir():
        print_error(f"Thư mục không tồn tại: {dir_path}")
        return False
    
    json_files = sorted(dir_path.glob("*.json"))
    
    if not json_files:
        print_error(f"Không tìm thấy JSON file nào trong: {dir_path}")
        return False
    
    print_section(f"Found {len(json_files)} JSON files")
    
    success_count = 0
    failed_count = 0
    
    for i, json_file in enumerate(json_files, 1):
        output_dir = None
        if output_base_dir:
            output_dir = str(Path(output_base_dir) / json_file.stem)
        
        success = process_single_file(str(json_file), output_dir)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
    
    # Final summary
    print_section("Batch Processing Summary")
    print(f"Total files: {len(json_files)}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print()
    
    return failed_count == 0


def show_usage():
    """Show usage information"""
    usage = """
╔════════════════════════════════════════════════════════════╗
║              Audio Splitter - Usage Guide                  ║
╚════════════════════════════════════════════════════════════╝

⚠️  LƯU Ý: Input phải là file .JSON (không phải .mp3)

USAGE:
  python run_splitter.py <json_file_or_directory> [output_dir]

EXAMPLES:

1. Process single JSON file (đúng ✅):
   python run_splitter.py downloads/video.json
   python run_splitter.py "downloads/FAPtv Cơm Nguội_ Tập 52- Học Sinh Mới.json"
   
2. Process all JSON files in directory (đúng ✅):
   python run_splitter.py downloads
   
3. ❌ SAI - Không dùng file .mp3:
   python run_splitter.py downloads/video.mp3  ← Lỗi!
   
4. Process with custom output directory (đúng ✅):
   python run_splitter.py downloads output_segments
   python run_splitter.py downloads/video.json my_segments

OUTPUT STRUCTURE:
   
   For single file:
     downloads/
       video.json
       video.mp3
       segments/
         video/
           0001_segment_text.mp3
           0002_next_segment.mp3
           ...

   For directory:
     downloads/
       video1.json
       video1.mp3
       video2.json
       video2.mp3
     segments/
       video1/
         0001_segment.mp3
         ...
       video2/
         0001_segment.mp3
         ...
"""
    print(usage)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help", "help"]:
        show_usage()
        sys.exit(0 if len(sys.argv) >= 2 else 1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    path_obj = Path(input_path)
    
    if path_obj.is_dir():
        success = process_directory(input_path, output_dir)
    else:
        success = process_single_file(input_path, output_dir)
    
    sys.exit(0 if success else 1)
