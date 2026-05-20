"""
Full Pipeline: YouTube Download → Audio Splitting → Final Report
Chạy toàn bộ quy trình từ tìm kiếm video → tách audio thành segments
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json

from splitter_utils import print_section, print_success, print_error, print_info, print_warning


class FullPipeline:
    """Orchestrate the entire workflow"""
    
    def __init__(self, downloads_dir: str = "downloads"):
        self.downloads_dir = Path(downloads_dir).resolve()
        self.segments_dir = self.downloads_dir / "segments"
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report = {
            "timestamp": self.timestamp,
            "download_stats": {},
            "split_stats": {},
            "total_segments": 0,
            "errors": []
        }
    
    def step_1_download_videos(self):
        """Step 1: Run main.py to download videos"""
        print_section("STEP 1: Download Videos from YouTube")
        
        main_script = Path(__file__).parent / "main.py"
        
        if not main_script.exists():
            print_error(f"Không tìm thấy: {main_script}")
            return False
        
        print_info(f"Chạy: {main_script.name}")
        print_info(f"Output sẽ lưu vào: {self.downloads_dir}\n")
        
        try:
            result = subprocess.run(
                [sys.executable, str(main_script)],
                cwd=str(Path(__file__).parent)
            )
            
            if result.returncode != 0:
                print_warning("main.py không hoàn tất hoặc có lỗi")
                self.report["errors"].append("Download step failed or incomplete")
            else:
                print_success("✅ Download hoàn tất!")
            
            return True
        
        except Exception as e:
            print_error(f"Lỗi chạy main.py: {e}")
            self.report["errors"].append(f"Download error: {str(e)}")
            return False
    
    def step_2_analyze_downloads(self):
        """Step 2: Analyze downloaded files"""
        print_section("STEP 2: Analyze Downloaded Files")
        
        if not self.downloads_dir.exists():
            print_error(f"Thư mục downloads không tồn tại: {self.downloads_dir}")
            return False
        
        json_files = list(self.downloads_dir.glob("*.json"))
        mp3_files = list(self.downloads_dir.glob("*.mp3"))
        
        print_info(f"📁 JSON files: {len(json_files)}")
        print_info(f"📁 MP3 files: {len(mp3_files)}\n")
        
        if not json_files:
            print_warning("Không có JSON file nào được tải")
            return False
        
        # Store stats
        self.report["download_stats"] = {
            "json_files": len(json_files),
            "mp3_files": len(mp3_files),
            "json_list": [f.stem for f in json_files]
        }
        
        for json_file in json_files[:5]:  # Show first 5
            size = json_file.stat().st_size
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                entries = len(data) if isinstance(data, list) else 0
            print_info(f"  • {json_file.name}: {entries} entries ({size:,} bytes)")
        
        if len(json_files) > 5:
            print_info(f"  ... và {len(json_files) - 5} files khác\n")
        else:
            print()
        
        return True
    
    def step_3_split_audio_segments(self):
        """Step 3: Split audio into segments"""
        print_section("STEP 3: Split Audio into Segments")
        
        json_files = sorted(self.downloads_dir.glob("*.json"))
        
        if not json_files:
            print_error("Không có JSON file để xử lý")
            return False
        
        print_info(f"Sẽ xử lý {len(json_files)} JSON files\n")
        
        splitter_script = Path(__file__).parent / "run_splitter.py"
        
        if not splitter_script.exists():
            print_error(f"Không tìm thấy: {splitter_script}")
            return False
        
        success_count = 0
        failed_count = 0
        total_segments = 0
        
        for i, json_file in enumerate(json_files, 1):
            print_info(f"[{i}/{len(json_files)}] Xử lý: {json_file.name}")
            
            try:
                video_seg_dir = self.segments_dir / json_file.stem
                result = subprocess.run(
                    [sys.executable, str(splitter_script), str(json_file), str(video_seg_dir)],
                    capture_output=True,
                    text=True,
                    timeout=3600,
                    encoding='utf-8',
                    errors='replace',
                    cwd=str(Path(__file__).parent),  # đảm bảo import splitter_core/utils đúng
                )

                if result.returncode == 0:
                    success_count += 1
                    segments = list(video_seg_dir.glob("*.mp3")) if video_seg_dir.exists() else []
                    total_segments += len(segments)
                    print_success(f"  ✅ {len(segments)} segments")
                else:
                    failed_count += 1
                    # In ra stderr/stdout để dễ debug
                    err = (result.stderr or result.stdout or "").strip()
                    print_error(f"  ❌ Lỗi (returncode={result.returncode})")
                    if err:
                        for line in err.splitlines()[-10:]:  # 10 dòng cuối
                            print(f"     {line}")
            
            except subprocess.TimeoutExpired:
                failed_count += 1
                print_error(f"  ❌ Timeout")
            except Exception as e:
                failed_count += 1
                print_error(f"  ❌ {str(e)[:100]}")
        
        print()
        print_info(f"✅ Thành công: {success_count}/{len(json_files)}")
        
        if failed_count > 0:
            print_warning(f"❌ Thất bại: {failed_count}/{len(json_files)}")
        
        self.report["split_stats"] = {
            "total_files": len(json_files),
            "success": success_count,
            "failed": failed_count
        }
        self.report["total_segments"] = total_segments
        
        return failed_count == 0
    
    def step_4_final_report(self):
        """Step 4: Generate final report"""
        print_section("STEP 4: Final Report")
        
        # Count final segments
        if self.segments_dir.exists():
            all_segments = list(self.segments_dir.rglob("*.mp3"))
            self.report["total_segments"] = len(all_segments)
        
        # Print summary
        print()
        print(f"📊 Timeline: {self.timestamp}")
        print()
        
        print("📥 Downloaded:")
        print(f"   • JSON files: {self.report['download_stats'].get('json_files', 0)}")
        print(f"   • MP3 files: {self.report['download_stats'].get('mp3_files', 0)}")
        print()
        
        print("✂️  Split Results:")
        split_stats = self.report['split_stats']
        print(f"   • Processed: {split_stats.get('success', 0)}/{split_stats.get('total_files', 0)}")
        print(f"   • Failed: {split_stats.get('failed', 0)}")
        print()
        
        print("🎵 Final Output:")
        print(f"   • Total segments: {self.report['total_segments']}")
        print(f"   • Location: {self.segments_dir}")
        print()
        
        if self.report["errors"]:
            print("⚠️  Warnings/Errors:")
            for error in self.report["errors"]:
                print(f"   • {error}")
            print()
        
        # Save report to JSON
        report_file = self.downloads_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.report, f, ensure_ascii=False, indent=2)
            print_info(f"📄 Report lưu tại: {report_file}")
        except Exception as e:
            print_warning(f"Không lưu được report: {e}")
        
        print()
    
    def run(self):
        """Run the entire pipeline"""
        print_section("🚀 FULL PIPELINE: DOWNLOAD → SPLIT → REPORT")
        print()
        print_info("Các bước:")
        print_info("  1️⃣  Tải video từ YouTube (chạy main.py)")
        print_info("  2️⃣  Phân tích file tải")
        print_info("  3️⃣  Tách audio thành segments")
        print_info("  4️⃣  Báo cáo cuối cùng")
        print()
        
        # Step 1
        if not self.step_1_download_videos():
            print_warning("⏭️  Bỏ qua download, tiếp tục với files hiện có...")
        
        print()
        
        # Step 2
        if not self.step_2_analyze_downloads():
            print_error("Không thể tiếp tục: không có file để xử lý")
            return False
        
        print()
        
        # Step 3
        if not self.step_3_split_audio_segments():
            print_warning("Có một số lỗi trong quá trình tách audio")
        
        print()
        
        # Step 4
        self.step_4_final_report()
        
        return True


def show_help():
    """Show usage information"""
    help_text = """
╔════════════════════════════════════════════════════════════╗
║     Full Pipeline: Download → Split Audio → Report         ║
╚════════════════════════════════════════════════════════════╝

Chạy toàn bộ quy trình tự động từ tìm kiếm video YouTube 
đến tách audio thành các segments nhỏ.

USAGE:
  python full_pipeline.py [downloads_dir]

EXAMPLES:

1. Chạy với thư mục mặc định (downloads):
   python full_pipeline.py
   
2. Chạy với thư mục tùy chỉnh:
   python full_pipeline.py my_downloads

QUY TRÌNH:
  
  Step 1: Download Videos
    ├─ Chạy main.py (tìm kiếm → tải MP3 → tải captions)
    └─ Output: downloads/*.mp3 + downloads/*.json
  
  Step 2: Analyze
    ├─ Kiểm tra số lượng JSON và MP3 files
    └─ Liệt kê danh sách files
  
  Step 3: Split Audio
    ├─ Chạy run_splitter.py trên tất cả JSON files
    └─ Output: downloads/segments/{file_name}/*.mp3
  
  Step 4: Final Report
    ├─ Thống kê kết quả
    ├─ Lưu report JSON
    └─ Tổng kết

OUTPUT:

  downloads/
    ├── crawled_links.md
    ├── video1.json
    ├── video1.mp3
    ├── video2.json
    ├── video2.mp3
    ├── pipeline_report_YYYYMMDD_HHMMSS.json
    └── segments/
        ├── video1/
        │   ├── 0001_segment.mp3
        │   ├── 0002_segment.mp3
        │   └── ...
        └── video2/
            ├── 0001_segment.mp3
            └── ...

LƯỚI ĐỒ:

  main.py
     │
     ├─→ Search YouTube
     ├─→ Download MP3
     ├─→ Extract captions (JSON)
     └─→ Save to downloads/
           │
           ↓
  full_pipeline.py (Step 2)
           │
           ├─→ Count JSON/MP3
           └─→ Store stats
                 │
                 ↓
  run_splitter.py
           │
           ├─→ Read JSON timestamps
           ├─→ Split audio
           └─→ Save to segments/
                 │
                 ↓
  full_pipeline.py (Step 4)
           │
           ├─→ Generate report
           ├─→ Save report.json
           └─→ Print summary
"""
    print(help_text)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_help()
        sys.exit(0)
    
    downloads_dir = sys.argv[1] if len(sys.argv) > 1 else "downloads"
    
    pipeline = FullPipeline(downloads_dir)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)
