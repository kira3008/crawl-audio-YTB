# tests/test_split_audio_clean.py
import json
from pathlib import Path
import subprocess
import split_audio


def _make_wav(path: Path, ffmpeg_exe: str, seconds: float = 3.0):
    # tao file wav test bang ffmpeg (sine 22050 mono)
    subprocess.run([ffmpeg_exe, "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"sine=frequency=220:duration={seconds}:sample_rate=22050",
                    "-ac", "1", "-c:a", "pcm_s16le", str(path)], check=True)


def test_split_one_uses_source_and_wav_output(tmp_path):
    ffmpeg_exe = split_audio.get_ffmpeg_exe()
    src = tmp_path / "clean.wav"
    _make_wav(src, ffmpeg_exe, seconds=4.0)
    entries = [
        {"start": "00:00:00.000", "end": "00:00:01.500", "text": "Xin chao cac ban."},
        {"start": "00:00:02.000", "end": "00:00:03.500", "text": "Hen gap lai nhe."},
    ]
    jpath = tmp_path / "clip.json"
    jpath.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")

    output_root = tmp_path / "segs"
    ok, err = split_audio.split_one(
        jpath, output_root, ffmpeg_exe, vad_model=None,
        source=src,
    )
    assert err == 0 and ok >= 1
    seg_dir = tmp_path / "segs" / "clip"
    wavs = list(seg_dir.glob("*.wav"))
    assert wavs, "segment phai la .wav"
    manifest = json.loads((seg_dir / "manifest.json").read_text(encoding="utf-8"))
    assert all(e.get("denoise") is True for e in manifest)


def test_prefers_clean_and_filters_dialogue(tmp_path):
    base = tmp_path / "a.json"
    base.write_text(json.dumps([
        {"start": "00:00:00.000", "end": "00:00:01.000", "text": "raw1", "words": []},
        {"start": "00:00:01.000", "end": "00:00:02.000", "text": "raw2", "words": []},
    ], ensure_ascii=False), encoding="utf-8")
    clean = tmp_path / "a.clean.json"
    clean.write_text(json.dumps([
        {"start": "00:00:00.000", "end": "00:00:01.000", "text": "Thoai", "type": "dialogue", "words": []},
        {"start": "00:00:01.000", "end": "00:00:02.000", "text": "[Âm nhạc]", "type": "sound", "words": []},
    ], ensure_ascii=False), encoding="utf-8")

    entries = split_audio.load_entries_for_split(base)
    assert len(entries) == 1
    assert entries[0]["text"] == "Thoai"


def test_falls_back_to_json(tmp_path):
    base = tmp_path / "b.json"
    base.write_text(json.dumps([
        {"start": "00:00:00.000", "end": "00:00:01.000", "text": "x", "words": []},
    ], ensure_ascii=False), encoding="utf-8")
    entries = split_audio.load_entries_for_split(base)
    assert len(entries) == 1
    assert entries[0]["text"] == "x"
