"""
Standalone script: tách audio thành segments theo JSON timestamps.
Usage: python audio_splitter.py <json_file_or_dir> [output_dir]
"""
import sys
from pathlib import Path

from splitter_core import AudioSplitter
from splitter_utils import print_error, print_section


def process_file(json_file: str, output_dir: str = None):
    splitter = AudioSplitter(json_file, output_dir)
    if not splitter.load_json():
        return False
    if not splitter.validate_setup():
        return False
    splitter.prepare_output_dir()
    splitter.process()
    splitter.print_summary()
    return True


def process_directory(directory: str, output_base: str = None):
    dir_path = Path(directory).resolve()
    json_files = sorted(dir_path.glob("*.json"))

    if not json_files:
        print_error(f"Không có JSON file trong: {dir_path}")
        return False

    print_section(f"Batch: {len(json_files)} files")
    ok = failed = 0

    for i, f in enumerate(json_files, 1):
        out = str(Path(output_base) / f.stem) if output_base else None
        print_section(f"[{i}/{len(json_files)}] {f.name}")
        if process_file(str(f), out):
            ok += 1
        else:
            failed += 1

    print_section("Kết quả")
    print(f"✅ {ok}/{len(json_files)}  ❌ {failed}")
    return failed == 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_splitter.py <json_file_or_dir> [output_dir]")
        sys.exit(1)

    path = Path(sys.argv[1])
    out  = sys.argv[2] if len(sys.argv) > 2 else None

    success = process_directory(str(path), out) if path.is_dir() else process_file(str(path), out)
    sys.exit(0 if success else 1)
