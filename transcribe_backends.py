"""transcribe_backends.py — backend transcribe: WhisperX local + Groq API.

Tat ca tra ve schema chung:
  [{"start","end","text","words":[{"word","start","end"}]}]  (hms hh:mm:ss.mmm)
"""

import subprocess
import logging
from pathlib import Path


def _sec_to_hms(sec: float) -> str:
    ms = round((sec % 1) * 1000)
    total = int(sec)
    if ms == 1000:
        ms = 0
        total += 1
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _detect_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def load_local_model(model_name: str) -> dict:
    import warnings
    warnings.filterwarnings("ignore", message="torchcodec is not installed")
    import whisperx

    device = "cpu"
    download_root = str(Path(__file__).parent / "models")
    if _detect_gpu():
        try:
            asr_model = whisperx.load_model(model_name, device="cuda", compute_type="int8",
                                            language="vi", download_root=download_root)
            device = "cuda"
        except Exception:
            asr_model = whisperx.load_model(model_name, device="cpu", compute_type="int8",
                                            language="vi", download_root=download_root)
    else:
        asr_model = whisperx.load_model(model_name, device="cpu", compute_type="int8",
                                        language="vi", download_root=download_root)

    align_model, metadata = whisperx.load_align_model(language_code="vi", device=device)
    return {"asr": asr_model, "align": align_model, "meta": metadata, "device": device}


def transcribe_local(mp3_path: str, bundle: dict) -> list[dict]:
    import whisperx
    device = bundle["device"]
    batch_size = 16 if device == "cuda" else 4

    audio = whisperx.load_audio(str(mp3_path))

    result = None
    while batch_size >= 1:
        try:
            import torch
            if device == "cuda":
                torch.cuda.empty_cache()
            result = bundle["asr"].transcribe(audio, batch_size=batch_size, language="vi")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1:
                batch_size = batch_size // 2
                logging.warning(f"[transcribe] OOM — giam batch_size xuong {batch_size}")
            else:
                raise
    if result is None:
        return []

    aligned = whisperx.align(
        result["segments"], bundle["align"], bundle["meta"], audio,
        device=device, return_char_alignments=False,
    )

    entries = []
    for seg in aligned["segments"]:
        text = seg.get("text", "").strip()
        if not text:
            continue
        words = [w for w in seg.get("words", []) if "start" in w and "end" in w]
        if words:
            start, end = words[0]["start"], words[-1]["end"]
        else:
            start, end = seg["start"], seg["end"]
        entries.append({
            "start": _sec_to_hms(start),
            "end": _sec_to_hms(end),
            "text": text,
            "words": [
                {"word": w["word"], "start": _sec_to_hms(w["start"]), "end": _sec_to_hms(w["end"])}
                for w in words
            ],
        })
    return entries


def groq_response_to_entries(resp: dict, offset_sec: float = 0.0) -> list[dict]:
    """Map Groq verbose_json response to common schema.

    Reads resp["segments"], applies offset_sec, skips empty-text segments,
    returns list of dicts with hms timestamps via _sec_to_hms.
    """
    entries: list[dict] = []
    for seg in resp.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        words_out = []
        for w in seg.get("words") or []:
            token = w.get("word", w.get("text", ""))
            if "start" in w and "end" in w:
                words_out.append({
                    "word": token,
                    "start": _sec_to_hms(float(w["start"]) + offset_sec),
                    "end": _sec_to_hms(float(w["end"]) + offset_sec),
                })
        entries.append({
            "start": _sec_to_hms(float(seg["start"]) + offset_sec),
            "end": _sec_to_hms(float(seg["end"]) + offset_sec),
            "text": text,
            "words": words_out,
        })
    return entries
