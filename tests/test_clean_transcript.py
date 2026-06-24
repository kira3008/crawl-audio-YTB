import json
from clean_transcript import (
    is_sound_tag, is_hallucination, tag_entries_heuristic,
    build_llm_prompt, apply_llm_result, llm_clean_batch,
    TYPE_DIALOGUE, TYPE_MUSIC, TYPE_SOUND, TYPE_NOISE,
    confidence_is_noise, is_repetitive_text,
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


def _te(text, typ="dialogue"):
    return {"start": "00:00:00.000", "end": "00:00:01.000",
            "text": text, "text_raw": text, "type": typ, "words": []}


def test_build_prompt_lists_indexes():
    p = build_llm_prompt([_te("Xin chao"), _te("la la la")])
    assert "0" in p and "Xin chao" in p and "la la la" in p


def test_apply_llm_result_updates():
    batch = [_te("xin chao cac ban"), _te("la la la")]
    result = [{"index": 0, "type": "dialogue", "text": "Xin chào các bạn."},
              {"index": 1, "type": "music", "text": "la la la"}]
    out = apply_llm_result(batch, result)
    assert out[0]["text"] == "Xin chào các bạn."
    assert out[0]["type"] == "dialogue"
    assert out[1]["type"] == "music"
    assert out[0]["text_raw"] == "xin chao cac ban"     # giu goc


def test_llm_clean_batch_error_returns_input(monkeypatch):
    class BadClient:
        class chat:
            class completions:
                @staticmethod
                def create(**k):
                    raise RuntimeError("boom")
    batch = [_te("a")]
    out = llm_clean_batch(BadClient(), batch)
    assert out == batch


def test_clean_entries_heuristic_only():
    from clean_transcript import clean_entries
    entries = [{"start": "00:00:00.000", "end": "00:00:01.000", "text": "[Âm nhạc]", "words": []},
               {"start": "00:00:01.000", "end": "00:00:02.000", "text": "Xin chào", "words": []}]
    out = clean_entries(entries, client=None)
    assert out[0]["type"] == "sound"
    assert out[1]["type"] == "dialogue"


def test_clean_file_writes_clean_json(tmp_path):
    from pathlib import Path
    from clean_transcript import clean_file
    src = tmp_path / "a.json"
    src.write_text(json.dumps(
        [{"start": "00:00:00.000", "end": "00:00:01.000", "text": "Xin chào", "words": []}],
        ensure_ascii=False), encoding="utf-8")
    out = clean_file(src, client=None)
    assert out == tmp_path / "a.clean.json"
    assert out.exists()
    assert src.exists()      # goc khong bi dung
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data[0]["type"] == "dialogue"


def test_confidence_high_no_speech():
    assert confidence_is_noise({"no_speech_prob": 0.7}) is True


def test_confidence_high_compression():
    assert confidence_is_noise({"compression_ratio": 2.5}) is True


def test_confidence_logprob_combo():
    assert confidence_is_noise({"avg_logprob": -1.2, "no_speech_prob": 0.5}) is True
    # logprob thap nhung no_speech thap -> khong demote (than trong)
    assert confidence_is_noise({"avg_logprob": -1.2, "no_speech_prob": 0.1}) is False


def test_confidence_no_metrics_false():
    assert confidence_is_noise({"text": "Xin chào"}) is False


def test_confidence_good_values_false():
    assert confidence_is_noise({"no_speech_prob": 0.1, "avg_logprob": -0.2,
                                "compression_ratio": 1.5}) is False


def test_repetitive_true():
    assert is_repetitive_text("la la la la") is True
    assert is_repetitive_text("yeah yeah yeah yeah yeah") is True


def test_repetitive_false_normal_sentence():
    assert is_repetitive_text("Hôm nay chúng ta bàn về hạnh phúc") is False


def test_repetitive_false_too_short():
    assert is_repetitive_text("la la la") is False   # 3 token < nguong


def test_near_dup_true():
    from clean_transcript import _is_near_dup
    assert _is_near_dup("la la la", "la la la la") is True


def test_near_dup_false():
    from clean_transcript import _is_near_dup
    assert _is_near_dup("xin chao cac ban", "tam biet hen gap lai") is False


def _e_full(text):
    return {"start": "00:00:00.000", "end": "00:00:01.000", "text": text,
            "text_raw": text, "words": []}


def test_global_music_keys_threshold():
    from clean_transcript import find_global_music_keys
    entries = [_e_full("nhớ đăng ký kênh")] * 4 + [_e_full("Nội dung hôm nay rất hay")]
    keys = find_global_music_keys(entries)
    assert "nhớ đăng ký kênh" in keys
    assert "nội dung hôm nay rất hay" not in keys


def test_global_music_keys_below_threshold():
    from clean_transcript import find_global_music_keys
    entries = [_e_full("điệp khúc lặp")] * 3      # 3 < GLOBAL_FREQ_MIN
    assert find_global_music_keys(entries) == set()
