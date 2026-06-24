from transcribe_backends import _sec_to_hms, groq_response_to_entries


def test_sec_to_hms_basic():
    assert _sec_to_hms(0) == "00:00:00.000"
    assert _sec_to_hms(61.5) == "00:01:01.500"
    assert _sec_to_hms(3661.250) == "01:01:01.250"


def test_sec_to_hms_ms_rounding_carry():
    assert _sec_to_hms(0.9996) == "00:00:01.000"


def test_groq_map_basic_with_words():
    resp = {"segments": [
        {"start": 1.0, "end": 2.5, "text": " Xin chao ",
         "words": [{"word": "Xin", "start": 1.0, "end": 1.4},
                   {"word": "chao", "start": 1.5, "end": 2.5}]},
        {"start": 3.0, "end": 3.0, "text": "  "},   # rong -> bo
    ]}
    out = groq_response_to_entries(resp)
    assert len(out) == 1
    assert out[0]["start"] == "00:00:01.000"
    assert out[0]["end"] == "00:00:02.500"
    assert out[0]["text"] == "Xin chao"
    assert out[0]["words"][1]["word"] == "chao"


def test_groq_map_offset_and_missing_words():
    resp = {"segments": [{"start": 0.0, "end": 1.0, "text": "A"}]}
    out = groq_response_to_entries(resp, offset_sec=600.0)
    assert out[0]["start"] == "00:10:00.000"
    assert out[0]["words"] == []
