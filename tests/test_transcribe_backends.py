import transcribe_backends as tb
from transcribe_backends import _sec_to_hms, groq_response_to_entries, plan_chunks, merge_chunk_entries, _attach_metrics


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


def test_plan_chunks_single_when_short():
    assert plan_chunks(300, max_chunk_sec=600) == [(0.0, 300.0)]


def test_plan_chunks_multiple_with_overlap():
    chunks = plan_chunks(1300, max_chunk_sec=600, overlap_sec=5)
    assert chunks[0] == (0.0, 600.0)
    assert chunks[1][0] == 595.0           # lui lai overlap
    assert chunks[1][1] == 1195.0
    assert chunks[-1][1] == 1300.0         # phu het toi cuoi


def test_merge_dedups_overlap():
    a = [{"start": "00:00:01.000", "end": "00:00:02.000", "text": "A", "words": []},
         {"start": "00:09:59.000", "end": "00:10:00.000", "text": "B", "words": []}]
    b = [{"start": "00:09:59.300", "end": "00:10:00.200", "text": "B", "words": []},  # trung B
         {"start": "00:10:05.000", "end": "00:10:06.000", "text": "C", "words": []}]
    out = merge_chunk_entries([a, b])
    assert [e["text"] for e in out] == ["A", "B", "C"]


class FakeTranscriptions:
    def __init__(self, payloads):
        self._payloads = payloads
        self.calls = 0

    def create(self, **kwargs):
        idx = self.calls
        self.calls += 1
        class R:
            def model_dump(self_inner):
                return self._payloads[idx]
        return R()


class FakeAudio:
    def __init__(self, payloads):
        self.transcriptions = FakeTranscriptions(payloads)


class FakeClient:
    def __init__(self, payloads):
        self.audio = FakeAudio(payloads)


def test_transcribe_groq_two_chunks(monkeypatch, tmp_path):
    mp3 = tmp_path / "a.mp3"
    mp3.write_bytes(b"x")

    # duration 1300s -> 3 chunk (600 / 595->1195 / 1190->1300)
    monkeypatch.setattr(tb, "_probe_duration", lambda *a, **k: 1300.0)
    # moi chunk tao file flac gia + tra 1 segment
    def _fake_extract(mp3p, start, end, ffmpeg, outp):
        open(outp, "wb").close()
        return outp
    monkeypatch.setattr(tb, "_extract_chunk_flac", _fake_extract)
    payloads = [
        {"segments": [{"start": 0.0, "end": 1.0, "text": "chunk0"}]},
        {"segments": [{"start": 0.0, "end": 1.0, "text": "chunk1"}]},
        {"segments": [{"start": 0.0, "end": 1.0, "text": "chunk2"}]},
    ]
    client = FakeClient(payloads)
    out = tb.transcribe_groq(str(mp3), client, "ffmpeg", model="whisper-large-v3")
    texts = [e["text"] for e in out]
    assert texts == ["chunk0", "chunk1", "chunk2"]
    # offset ap dung: chunk1 bat dau ~595s
    assert out[1]["start"].startswith("00:09:5")


def test_attach_metrics_copies_present():
    entry = {}
    _attach_metrics(entry, {"no_speech_prob": 0.7, "avg_logprob": -0.5,
                            "compression_ratio": 1.8, "other": 1})
    assert entry == {"no_speech_prob": 0.7, "avg_logprob": -0.5, "compression_ratio": 1.8}


def test_attach_metrics_skips_missing_and_none():
    entry = {}
    _attach_metrics(entry, {"no_speech_prob": None})
    assert entry == {}


def test_groq_map_carries_metrics():
    resp = {"segments": [
        {"start": 0.0, "end": 1.0, "text": "A",
         "no_speech_prob": 0.9, "avg_logprob": -1.2, "compression_ratio": 3.0},
        {"start": 1.0, "end": 2.0, "text": "B"},   # khong co metric
    ]}
    out = groq_response_to_entries(resp)
    assert out[0]["no_speech_prob"] == 0.9
    assert out[0]["compression_ratio"] == 3.0
    assert "no_speech_prob" not in out[1]
