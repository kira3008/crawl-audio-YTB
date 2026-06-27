# tests/test_split_audio_discover.py
import json
from pathlib import Path
import split_audio


def test_base_name():
    assert split_audio._base_name(Path("x/a.clean.json")) == "a"
    assert split_audio._base_name(Path("x/a.json")) == "a"
    assert split_audio._base_name(Path("x/My Video.clean.json")) == "My Video"


def test_segments_root_organized_vs_flat():
    assert split_audio._segments_root(Path("downloads/audio/a.mp3")) == Path("downloads")
    assert split_audio._segments_root(Path("flat/a.mp3")) == Path("flat")


def test_load_entries_clean_json_filters_dialogue(tmp_path):
    cj = tmp_path / "a.clean.json"
    cj.write_text(json.dumps([
        {"start": "00:00:00.000", "end": "00:00:01.000", "text": "xin chao", "type": "dialogue"},
        {"start": "00:00:01.000", "end": "00:00:02.000", "text": "la la la", "type": "music"},
    ]), encoding="utf-8")
    out = split_audio.load_entries_for_split(cj)
    assert len(out) == 1 and out[0]["text"] == "xin chao"


def test_load_entries_raw_json_no_filter(tmp_path):
    rj = tmp_path / "b.json"
    rj.write_text(json.dumps([
        {"start": "00:00:00.000", "end": "00:00:01.000", "text": "cau mot"},
        {"start": "00:00:01.000", "end": "00:00:02.000", "text": "cau hai"},
    ]), encoding="utf-8")
    out = split_audio.load_entries_for_split(rj)
    assert len(out) == 2
