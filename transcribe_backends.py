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


def _hms_to_sec(hms: str) -> float:
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


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


def plan_chunks(duration_sec: float, max_chunk_sec: float = 600.0,
                overlap_sec: float = 5.0) -> list[tuple[float, float]]:
    """Split [0, duration] into windows ≤ max_chunk_sec, with overlap.

    Each non-first window steps back overlap_sec to preserve context.
    If duration <= max_chunk_sec, returns single window [0, duration].

    Args:
        duration_sec: Total audio duration in seconds.
        max_chunk_sec: Maximum window size (default 600s).
        overlap_sec: Overlap to step back on each new window (default 5s).

    Returns:
        List of (start, end) tuples in seconds.
    """
    if duration_sec <= max_chunk_sec:
        return [(0.0, float(duration_sec))]
    chunks: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_sec:
        end = min(start + max_chunk_sec, duration_sec)
        chunks.append((start, end))
        if end >= duration_sec:
            break
        start = end - overlap_sec
    return chunks


def merge_chunk_entries(entry_lists: list[list[dict]]) -> list[dict]:
    """Merge entries from multiple chunks, deduplicate overlaps.

    Flattens all entries, sorts by start time, and drops entries that have
    the same text and start time within 1.0s of the previously kept entry.
    """
    flat = [e for lst in entry_lists for e in lst]
    flat.sort(key=lambda e: _hms_to_sec(e["start"]))
    merged: list[dict] = []
    for e in flat:
        if merged:
            prev = merged[-1]
            same_text = e["text"].strip() == prev["text"].strip()
            close = abs(_hms_to_sec(e["start"]) - _hms_to_sec(prev["start"])) < 1.0
            if same_text and close:
                continue
        merged.append(e)
    return merged


def load_groq_client():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    import os
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Thieu GROQ_API_KEY")
    from groq import Groq
    return Groq(api_key=key)


def _probe_duration(mp3_path: str, ffmpeg_exe: str) -> float:
    import re
    r = subprocess.run(
        [ffmpeg_exe, "-i", mp3_path, "-hide_banner"],
        capture_output=True, text=True, errors="replace",
    )
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", r.stderr)
    if not m:
        return 0.0
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


def _extract_chunk_flac(mp3_path: str, start: float, end: float,
                        ffmpeg_exe: str, out_path: str) -> str:
    subprocess.run(
        [ffmpeg_exe, "-y", "-loglevel", "error",
         "-ss", str(start), "-to", str(end), "-i", mp3_path,
         "-ar", "16000", "-ac", "1", "-c:a", "flac", out_path],
        check=True, capture_output=True,
    )
    return out_path


def transcribe_groq(mp3_path: str, client, ffmpeg_exe: str | None,
                    model: str = "whisper-large-v3-turbo") -> list[dict]:
    import tempfile
    import os
    ffmpeg = ffmpeg_exe or "ffmpeg"
    if ffmpeg and Path(ffmpeg).is_dir():
        ffmpeg = str(Path(ffmpeg) / "ffmpeg.exe")
    duration = _probe_duration(mp3_path, ffmpeg)
    windows = plan_chunks(duration) if duration > 0 else [(0.0, 0.0)]

    entry_lists: list[list[dict]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (start, end) in enumerate(windows):
            flac = os.path.join(tmp, f"chunk_{i}.flac")
            try:
                if end > start:
                    _extract_chunk_flac(mp3_path, start, end, ffmpeg, flac)
                    src = flac
                else:
                    # duration unknown: downsample whole file to stay under Groq size limit
                    subprocess.run(
                        [ffmpeg, "-y", "-loglevel", "error",
                         "-i", mp3_path,
                         "-ar", "16000", "-ac", "1", "-c:a", "flac", flac],
                        check=True, capture_output=True,
                    )
                    src = flac
                with open(src, "rb") as fh:
                    resp = client.audio.transcriptions.create(
                        model=model, file=(os.path.basename(src), fh.read()),
                        language="vi", response_format="verbose_json",
                        timestamp_granularities=["segment", "word"],
                    )
                data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
                entry_lists.append(groq_response_to_entries(data, offset_sec=start))
            except Exception as e:
                logging.error(f"[groq] chunk {i} loi: {e}")
                continue
    return merge_chunk_entries(entry_lists)


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
