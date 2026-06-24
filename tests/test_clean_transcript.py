from clean_transcript import (
    is_sound_tag, is_hallucination, tag_entries_heuristic,
    TYPE_DIALOGUE, TYPE_MUSIC, TYPE_SOUND, TYPE_NOISE,
)


def test_is_sound_tag():
    assert is_sound_tag("[Âm nhạc]")
    assert is_sound_tag("♪ la la la ♪")
    assert is_sound_tag("[Music]")
    assert not is_sound_tag("Xin chào các bạn")


def test_is_hallucination():
    assert is_hallucination("Hãy subscribe cho kênh để xem thêm nhiều video")
    assert is_hallucination("Cảm ơn các bạn đã theo dõi")
    assert not is_hallucination("Hôm nay chúng ta nói về chủ đề này")


def _e(text):
    return {"start": "00:00:00.000", "end": "00:00:01.000", "text": text, "words": []}


def test_tag_entries_types():
    entries = [_e("Xin chào"), _e("[Âm nhạc]"), _e("Hãy subscribe cho kênh nhé")]
    out = tag_entries_heuristic(entries)
    assert out[0]["type"] == TYPE_DIALOGUE
    assert out[1]["type"] == TYPE_SOUND
    assert out[2]["type"] == TYPE_NOISE
    assert out[0]["text_raw"] == "Xin chào"


def test_tag_repetition_as_music():
    entries = [_e("la la la"), _e("la la la"), _e("la la la")]
    out = tag_entries_heuristic(entries, repeat_threshold=3)
    assert all(o["type"] == TYPE_MUSIC for o in out)
