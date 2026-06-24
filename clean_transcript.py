"""clean_transcript.py — loc nhieu & chuan hoa transcript (heuristic + LLM Groq)."""

import re
import json
from pathlib import Path
import difflib
from collections import Counter

TYPE_DIALOGUE = "dialogue"
TYPE_MUSIC = "music"
TYPE_SOUND = "sound"
TYPE_NOISE = "noise"

NO_SPEECH_MAX = 0.6
COMPRESSION_MAX = 2.4
LOGPROB_MIN = -1.0
LOGPROB_NOSPEECH_COMBO = 0.4

REPEAT_MIN_TOKENS = 4
REPEAT_UNIQUE_RATIO = 0.5

SIM_THRESHOLD = 0.85
GLOBAL_FREQ_MIN = 4
GLOBAL_PHRASE_MAX_WORDS = 12

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


def confidence_is_noise(entry: dict) -> bool:
    ns = entry.get("no_speech_prob")
    cr = entry.get("compression_ratio")
    lp = entry.get("avg_logprob")
    if ns is not None and ns >= NO_SPEECH_MAX:
        return True
    if cr is not None and cr >= COMPRESSION_MAX:
        return True
    if lp is not None and ns is not None and lp <= LOGPROB_MIN and ns >= LOGPROB_NOSPEECH_COMBO:
        return True
    return False


def is_repetitive_text(text: str) -> bool:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < REPEAT_MIN_TOKENS:
        return False
    return (len(set(tokens)) / len(tokens)) <= REPEAT_UNIQUE_RATIO


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _is_near_dup(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, a, b).ratio() >= SIM_THRESHOLD


def find_global_music_keys(entries: list[dict]) -> set[str]:
    counts: Counter = Counter()
    for e in entries:
        key = _norm_key(e.get("text_raw", e.get("text", "")))
        if key and 1 <= len(key.split()) <= GLOBAL_PHRASE_MAX_WORDS:
            counts[key] += 1
    return {k for k, c in counts.items() if c >= GLOBAL_FREQ_MIN}


def tag_entries_heuristic(entries: list[dict], repeat_threshold: int = 3) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        raw = e.get("text", "")
        new = dict(e)
        new["text_raw"] = raw
        new["text"] = raw
        if is_sound_tag(raw):
            new["type"] = TYPE_SOUND
        elif confidence_is_noise(new):
            new["type"] = TYPE_NOISE
        elif is_hallucination(raw):
            new["type"] = TYPE_NOISE
        elif is_repetitive_text(raw):
            new["type"] = TYPE_MUSIC
        else:
            new["type"] = TYPE_DIALOGUE
        out.append(new)

    # tan suat toan bai -> music (chi dialogue)
    music_keys = find_global_music_keys(out)
    if music_keys:
        for o in out:
            if o["type"] == TYPE_DIALOGUE and _norm_key(o["text_raw"]) in music_keys:
                o["type"] = TYPE_MUSIC

    # near-duplicate lien tiep >= threshold -> music (chi dialogue)
    i = 0
    n = len(out)
    while i < n:
        j = i
        ki = _norm_key(out[i]["text_raw"])
        while j + 1 < n and ki and _is_near_dup(ki, _norm_key(out[j + 1]["text_raw"])):
            j += 1
        if (j - i + 1) >= repeat_threshold:
            for k in range(i, j + 1):
                if out[k]["type"] == TYPE_DIALOGUE:
                    out[k]["type"] = TYPE_MUSIC
        i = j + 1
    return out


def build_llm_prompt(batch: list[dict]) -> str:
    lines = [
        f"{i} [{e.get('type', 'dialogue')}]: {e.get('text_raw', e.get('text', ''))}"
        for i, e in enumerate(batch)
    ]
    return (
        "Bạn là bộ lọc transcript tiếng Việt. Phân loại mỗi dòng thành một "
        "trong: dialogue (lời thoại/nội dung nói), music (lời bài hát/giai điệu), "
        "noise (câu thừa, hallucination, lời chào câu view không thuộc nội dung).\n"
        "QUAN TRỌNG: nếu KHÔNG chắc chắn, GIỮ là dialogue (thận trọng, tránh cắt nhầm lời thoại).\n"
        "Nhãn trong [ ] là phỏng đoán sơ bộ của hệ thống — hãy xác nhận hoặc sửa lại.\n"
        "Ví dụ:\n"
        "  'Hôm nay chúng ta bàn về hạnh phúc.' -> dialogue\n"
        "  'La la la la la' -> music\n"
        "  'Nhớ like và đăng ký kênh nhé các bạn.' -> noise\n"
        "Đồng thời chuẩn hóa 'text': sửa dấu câu, viết hoa đầu câu, bỏ từ đệm lặp.\n"
        "Trả về JSON: {\"items\": [{\"index\": int, \"type\": str, \"text\": str}, ...]}.\n\n"
        + "\n".join(lines)
    )


def apply_llm_result(batch: list[dict], result: list[dict]) -> list[dict]:
    out = [dict(e) for e in batch]
    for item in result:
        try:
            idx = int(item["index"])
        except (KeyError, ValueError, TypeError):
            continue
        if 0 <= idx < len(out):
            if item.get("type") in (TYPE_DIALOGUE, TYPE_MUSIC, TYPE_SOUND, TYPE_NOISE):
                out[idx]["type"] = item["type"]
            if isinstance(item.get("text"), str) and item["text"].strip():
                out[idx]["text"] = item["text"].strip()
    return out


def llm_clean_batch(client, batch: list[dict],
                    model: str = "llama-3.3-70b-versatile") -> list[dict]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": build_llm_prompt(batch)}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        items = data.get("items", data if isinstance(data, list) else [])
        return apply_llm_result(batch, items)
    except Exception:
        return batch


def clean_entries(entries: list[dict], client=None, batch_size: int = 40) -> list[dict]:
    tagged = tag_entries_heuristic(entries)
    if client is None:
        return tagged
    out: list[dict] = []
    for i in range(0, len(tagged), batch_size):
        out.extend(llm_clean_batch(client, tagged[i:i + batch_size]))
    return out


def clean_file(json_path, client=None) -> Path:
    json_path = Path(json_path)
    entries = json.loads(json_path.read_text(encoding="utf-8"))
    cleaned = clean_entries(entries, client=client)
    out_path = json_path.with_suffix(".clean.json")
    out_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _collect_json(paths):
    result = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            result.extend(f for f in sorted(p.glob("*.json"))
                          if not f.name.endswith(".clean.json") and f.name != "manifest.json")
        elif p.suffix.lower() == ".json" and p.exists() and not p.name.endswith(".clean.json"):
            result.append(p)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Loc nhieu & chuan hoa transcript JSON.")
    parser.add_argument("inputs", nargs="*", default=["downloads"])
    parser.add_argument("--no-llm", action="store_true", help="Chi heuristic, khong goi Groq")
    args = parser.parse_args()

    client = None
    if not args.no_llm:
        try:
            from transcribe_backends import load_groq_client
            client = load_groq_client()
            print("[clean] Dung LLM Groq de chuan hoa.")
        except Exception as e:
            print(f"[clean] Khong co Groq ({e}) -> chi heuristic.")

    files = _collect_json(args.inputs or ["downloads"])
    if not files:
        print("Khong tim thay file JSON.")
        return
    for jf in files:
        out = clean_file(jf, client=client)
        print(f"  ✓ {jf.name} -> {out.name}")


if __name__ == "__main__":
    main()
