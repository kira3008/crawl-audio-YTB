"""clean_transcript.py — loc nhieu & chuan hoa transcript (heuristic + LLM Groq)."""

import re

TYPE_DIALOGUE = "dialogue"
TYPE_MUSIC = "music"
TYPE_SOUND = "sound"
TYPE_NOISE = "noise"

_SOUND_TAG_RE = re.compile(r"[♪♫\U0001F3B5]|\[[^\]]*\]")
_BRACKET_ONLY_RE = re.compile(r"^\s*\[[^\]]*\]\s*$")

_HALLUCINATION_PATTERNS = [
    "hãy subscribe", "đăng ký kênh", "subscribe cho kênh",
    "cảm ơn các bạn đã theo dõi", "cảm ơn các bạn đã xem",
    "hẹn gặp lại các bạn", "hẹn gặp lại trong video",
    "like và đăng ký", "nhấn chuông thông báo",
]


def is_sound_tag(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if _BRACKET_ONLY_RE.match(t):
        return True
    return bool(re.search(r"[♪♫\U0001F3B5]", t))


def is_hallucination(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in _HALLUCINATION_PATTERNS)


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tag_entries_heuristic(entries: list[dict], repeat_threshold: int = 3) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        raw = e.get("text", "")
        new = dict(e)
        new["text_raw"] = raw
        new["text"] = raw
        if is_sound_tag(raw):
            new["type"] = TYPE_SOUND
        elif is_hallucination(raw):
            new["type"] = TYPE_NOISE
        else:
            new["type"] = TYPE_DIALOGUE
        out.append(new)

    # cau lap lien tiep >= threshold -> music
    i = 0
    n = len(out)
    while i < n:
        j = i
        key = _norm_key(out[i]["text_raw"])
        while j + 1 < n and _norm_key(out[j + 1]["text_raw"]) == key and key:
            j += 1
        if (j - i + 1) >= repeat_threshold:
            for k in range(i, j + 1):
                if out[k]["type"] == TYPE_DIALOGUE:
                    out[k]["type"] = TYPE_MUSIC
        i = j + 1
    return out
