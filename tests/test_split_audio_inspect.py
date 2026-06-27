# tests/test_split_audio_inspect.py
"""Regression test for FIX A: inspect mode must not IndexError on empty words list."""
import json
from pathlib import Path
import split_audio


def test_inspect_empty_words_no_error(tmp_path):
    """split_one inspect=True must not raise when entries have words=[]."""
    # Build a minimal JSON with two entries that have empty words (Groq style)
    entries = [
        {"start": "00:00:00.000", "end": "00:00:02.000",
         "text": "Xin chao.", "words": []},
        {"start": "00:00:02.500", "end": "00:00:04.000",
         "text": "Tam biet.", "words": []},
    ]
    json_path = tmp_path / "sample.json"
    json_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")

    # Inspect mode short-circuits before any ffmpeg call, but split_one still
    # checks that the sibling .mp3 file exists before proceeding.
    mp3_path = tmp_path / "sample.mp3"
    mp3_path.write_bytes(b"\x00")   # 1-byte stub; ffmpeg is never called in inspect mode

    ok_count, err_count = split_audio.split_one(
        json_path=json_path,
        output_root=tmp_path / "segments",
        ffmpeg_exe="ffmpeg",      # never invoked in inspect mode
        vad_model=None,
        console=None,
        inspect=True,
    )

    assert err_count == 0
    assert ok_count > 0


def test_plan_split_sources_filters_to_denoised():
    j1, j2 = Path("a.json"), Path("b.json")
    den = {Path("a.mp3"): Path("den/a.wav")}   # chi a.mp3 denoise OK
    # map theo mp3 sibling cua json
    pairs = split_audio.plan_split_sources([j1, j2], den)
    assert pairs == [(j1, Path("den/a.wav"))]   # b bi bo vi denoise loi
