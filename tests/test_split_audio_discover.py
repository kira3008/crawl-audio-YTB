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


def _touch(p: Path, text="[]"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_discover_organized_prefers_clean(tmp_path):
    _touch(tmp_path / "transcript" / "a.json")
    _touch(tmp_path / "transcript_clean" / "a.clean.json")
    (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
    (tmp_path / "audio" / "a.mp3").write_bytes(b"x")
    jobs = split_audio.discover_split_jobs([tmp_path])
    assert jobs == [(tmp_path / "transcript_clean" / "a.clean.json", tmp_path / "audio" / "a.mp3")]


def test_discover_organized_fallback_raw_and_skip_missing_mp3(tmp_path):
    _touch(tmp_path / "transcript" / "b.json")       # co transcript, KHONG co audio/b.mp3 -> bo
    _touch(tmp_path / "transcript" / "c.json")       # co transcript + audio/c.mp3 -> nhan (raw)
    (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
    (tmp_path / "audio" / "c.mp3").write_bytes(b"x")
    jobs = split_audio.discover_split_jobs([tmp_path])
    assert jobs == [(tmp_path / "transcript" / "c.json", tmp_path / "audio" / "c.mp3")]


def test_discover_flat_layout(tmp_path):
    (tmp_path / "d.json").write_text("[]", encoding="utf-8")
    (tmp_path / "d.mp3").write_bytes(b"x")
    (tmp_path / "d.clean.json").write_text("[]", encoding="utf-8")   # KHONG lam driver rieng
    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")  # bo qua
    jobs = split_audio.discover_split_jobs([tmp_path])
    assert jobs == [(tmp_path / "d.json", tmp_path / "d.mp3")]


def test_discover_single_clean_json_resolves_audio(tmp_path):
    _touch(tmp_path / "transcript_clean" / "e.clean.json")
    (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
    (tmp_path / "audio" / "e.mp3").write_bytes(b"x")
    cj = tmp_path / "transcript_clean" / "e.clean.json"
    jobs = split_audio.discover_split_jobs([cj])
    assert jobs == [(cj, tmp_path / "audio" / "e.mp3")]


def test_discover_single_flat_clean_json(tmp_path):
    cj = tmp_path / "g.clean.json"
    cj.write_text("[]", encoding="utf-8")
    (tmp_path / "g.mp3").write_bytes(b"x")
    jobs = split_audio.discover_split_jobs([cj])
    assert jobs == [(cj, tmp_path / "g.mp3")]


def test_discover_empty_defaults_to_downloads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _touch(tmp_path / "downloads" / "transcript" / "f.json")
    (tmp_path / "downloads" / "audio").mkdir(parents=True, exist_ok=True)
    (tmp_path / "downloads" / "audio" / "f.mp3").write_bytes(b"x")
    jobs = split_audio.discover_split_jobs([])
    assert len(jobs) == 1
    tr, mp3 = jobs[0]
    assert tr == Path("downloads") / "transcript" / "f.json"
    assert mp3 == Path("downloads") / "audio" / "f.mp3"
