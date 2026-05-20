"""
Main script to run audio splitting
Usage:
    python run_splitter.py <json_file_or_directory> [output_dir] [--min-dur N] [--max-dur N] [--pad N]
"""
import sys
import argparse
from pathlib import Path

from splitter_core import AudioSplitter
from splitter_utils import print_section, print_error, print_info


def process_single_file(json_file: str, output_dir: str = None, **kwargs):
    print_section(f"Processing: {Path(json_file).name}")
    splitter = AudioSplitter(json_file, output_dir, **kwargs)
    if not splitter.load_json():
        return False
    if not splitter.validate_setup():
        return False
    splitter.prepare_output_dir()
    splitter.process()
    splitter.print_summary()
    return True


def process_directory(directory: str, output_base_dir: str = None, **kwargs):
    dir_path = Path(directory).resolve()
    if not dir_path.is_dir():
        print_error(f"Thư mục không tồn tại: {dir_path}")
        return False

    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        print_error(f"Không tìm thấy JSON file nào trong: {dir_path}")
        return False

    print_section(f"Batch: {len(json_files)} files")
    ok = failed = 0

    for i, f in enumerate(json_files, 1):
        out = str(Path(output_base_dir) / f.stem) if output_base_dir else None
        print_section(f"[{i}/{len(json_files)}] {f.name}")
        if process_single_file(str(f), out, **kwargs):
            ok += 1
        else:
            failed += 1

    print_section("Kết quả")
    print(f"✅ {ok}/{len(json_files)}  ❌ {failed}")
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cắt audio thành segments theo JSON timestamps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python run_splitter.py downloads/video.json
  python run_splitter.py downloads/
  python run_splitter.py downloads/video.json --min-dur 0.3 --max-dur 20
  python run_splitter.py downloads/ --min-dur 0.5 --pad 30
""",
    )
    parser.add_argument("input",      help="File .json hoặc thư mục chứa .json")
    parser.add_argument("output_dir", nargs="?", default=None, help="Thư mục output (tùy chọn)")
    parser.add_argument("--min-dur",  type=float, default=0.3,  metavar="S", help="Thời lượng tối thiểu (giây, mặc định 0.3)")
    parser.add_argument("--max-dur",  type=float, default=30.0, metavar="S", help="Thời lượng tối đa (giây, mặc định 30)")
    parser.add_argument("--pad",      type=int,   default=50,   metavar="MS", help="Padding mỗi đầu (ms, mặc định 50)")

    args = parser.parse_args()

    kwargs = {
        "min_dur":    args.min_dur,
        "max_dur":    args.max_dur,
        "padding_ms": args.pad,
    }

    path = Path(args.input)
    success = (
        process_directory(str(path), args.output_dir, **kwargs)
        if path.is_dir()
        else process_single_file(str(path), args.output_dir, **kwargs)
    )
    sys.exit(0 if success else 1)
