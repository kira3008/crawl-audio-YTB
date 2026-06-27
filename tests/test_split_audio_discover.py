# tests/test_split_audio_discover.py
from pathlib import Path
import split_audio


def test_base_name():
    assert split_audio._base_name(Path("x/a.clean.json")) == "a"
    assert split_audio._base_name(Path("x/a.json")) == "a"
    assert split_audio._base_name(Path("x/My Video.clean.json")) == "My Video"


def test_segments_root_organized_vs_flat():
    assert split_audio._segments_root(Path("downloads/audio/a.mp3")) == Path("downloads")
    assert split_audio._segments_root(Path("flat/a.mp3")) == Path("flat")
