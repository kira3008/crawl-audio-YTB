# tests/test_split_audio_clean.py
import json
from pathlib import Path
import split_audio


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
