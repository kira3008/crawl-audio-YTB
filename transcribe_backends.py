"""transcribe_backends.py — backend transcribe: WhisperX local + Groq API.

Tat ca tra ve schema chung:
  [{"start","end","text","words":[{"word","start","end"}]}]  (hms hh:mm:ss.mmm)
"""

import subprocess
import logging
import threading
import time
from pathlib import Path


class _GroqRateLimiter:
    """Thread-safe rate limiter theo Groq Whisper limits:
    20 req/min · 7200 audio-sec/hour · 28800 audio-sec/day.
    """
    RPM             = 20
    AUDIO_PER_HOUR  = 7_200
    AUDIO_PER_DAY   = 28_800

    def __init__(self):
        self._lock       = threading.Lock()
        self._req_times: list[float] = []   # timestamps trong 60s gần nhất
        self._hour_audio = 0.0
        self._day_audio  = 0.0
        self._hour_t     = time.monotonic()
        self._day_t      = time.monotonic()

    def acquire(self, duration_sec: float) -> bool:
        """Block cho đến khi rate limit cho phép. Trả False nếu hết quota ngày."""
        deadline = time.monotonic() + 4000   # chờ tối đa ~1h trước khi bỏ qua
        while True:
            with self._lock:
                now = time.monotonic()

                # reset counter giờ / ngày
                if now - self._hour_t >= 3600:
                    self._hour_audio = 0.0
                    self._hour_t = now
                if now - self._day_t >= 86400:
                    self._day_audio = 0.0
                    self._day_t = now

                # sliding window 60s cho RPM
                self._req_times = [t for t in self._req_times if now - t < 60]

                rpm_ok  = len(self._req_times) < self.RPM
                hour_ok = self._hour_audio + duration_sec <= self.AUDIO_PER_HOUR
                day_ok  = self._day_audio  + duration_sec <= self.AUDIO_PER_DAY

                if not day_ok:
                    logging.warning("[groq] Hết quota ngày (28800s audio). Bỏ qua chunk.")
                    return False

                if rpm_ok and hour_ok:
                    self._req_times.append(now)
                    self._hour_audio += duration_sec
                    self._day_audio  += duration_sec
                    return True

                # tính thời gian cần chờ và log để anh biết
                wait_reason = []
                if not rpm_ok:
                    oldest = self._req_times[0] if self._req_times else now
                    wait_reason.append(f"RPM ({len(self._req_times)}/20)")
                if not hour_ok:
                    wait_reason.append(
                        f"giờ ({self._hour_audio:.0f}/{self.AUDIO_PER_HOUR}s)"
                    )
                logging.info(f"[groq] Rate limit — chờ... ({', '.join(wait_reason)})")

            if time.monotonic() > deadline:
                logging.warning("[groq] Chờ rate limit quá lâu. Bỏ qua chunk.")
                return False

            time.sleep(2.0)


_groq_limiter = _GroqRateLimiter()


_METRIC_KEYS = ("no_speech_prob", "avg_logprob", "compression_ratio")


def _attach_metrics(entry: dict, seg: dict) -> None:
    for k in _METRIC_KEYS:
        v = seg.get(k)
        if v is not None:
            entry[k] = float(v)


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
    warnings.filterwarnings("ignore", message="torchcodec is not installed", category=UserWarning)
    warnings.filterwarnings("ignore", module="pyannote")
    import whisperx

    device = "cpu"
    download_root = str(Path(__file__).parent / "models")
    if _detect_gpu():
        try:
            # float16: chính xác hơn int8, phù hợp khi GPU có đủ VRAM (≥6GB cho large-v3)
            asr_model = whisperx.load_model(model_name, device="cuda", compute_type="float16",
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
    batch_size = 24 if device == "cuda" else 4

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
        entry = {
            "start": _sec_to_hms(start),
            "end": _sec_to_hms(end),
            "text": text,
            "words": [
                {"word": w["word"], "start": _sec_to_hms(w["start"]), "end": _sec_to_hms(w["end"])}
                for w in words
            ],
        }
        _attach_metrics(entry, seg)
        entries.append(entry)
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


class _OpenRouterKeyPool:
    """Xoay API key + per-key rate limiting cho OpenRouter Whisper.

    Limits áp dụng PER KEY (= per account):
      20 req/phút · 1000 req/ngày (account đã nạp ≥$10 credits).

    Lưu ý: nhiều key cùng 1 account KHÔNG bypass limit — chỉ có tác dụng
    khi key đến từ các account khác nhau.
    """
    RPM = 20
    RPD = 1000

    def __init__(self, keys: list[str]):
        self._keys = list(keys)
        self._lock = threading.Lock()
        self._req_times: dict[str, list[float]] = {k: [] for k in keys}
        self._day_reqs:  dict[str, int]         = {k: 0  for k in keys}
        self._day_t:     dict[str, float]        = {k: time.monotonic() for k in keys}
        # key bị block cứng đến thời điểm này (do 429 thực từ API)
        self._blocked_until: dict[str, float]   = {k: 0.0 for k in keys}

    def acquire(self) -> str | None:
        """Trả API key sẵn sàng. Block nếu cần, trả None nếu hết quota ngày."""
        deadline = time.monotonic() + 120
        while True:
            with self._lock:
                now = time.monotonic()
                for key in self._keys:
                    # bị block cứng (do 429 thực) → bỏ qua
                    if now < self._blocked_until[key]:
                        continue
                    # reset counter ngày
                    if now - self._day_t[key] >= 86400:
                        self._day_reqs[key] = 0
                        self._day_t[key] = now
                    # sliding window RPM
                    self._req_times[key] = [t for t in self._req_times[key] if now - t < 60]

                    if len(self._req_times[key]) < self.RPM and self._day_reqs[key] < self.RPD:
                        self._req_times[key].append(now)
                        self._day_reqs[key] += 1
                        return key

                exhausted_today = all(self._day_reqs[k] >= self.RPD for k in self._keys)
                if exhausted_today:
                    logging.warning("[openrouter] Tất cả key hết quota ngày (1000 req).")
                    return None

            if time.monotonic() > deadline:
                logging.warning("[openrouter] Chờ rate limit quá lâu. Bỏ qua.")
                return None
            time.sleep(2.0)

    def mark_rate_limited(self, key: str, retry_after: float = 60.0) -> None:
        """Gọi khi nhận 429 thực từ API — block key này trong retry_after giây."""
        with self._lock:
            self._blocked_until[key] = time.monotonic() + retry_after
            short = key[-8:]
            logging.warning(
                f"[openrouter] Key ...{short} bị 429 → block {retry_after:.0f}s, xoay sang key khác"
            )

    def status(self) -> str:
        """Tóm tắt trạng thái các key để hiển thị."""
        with self._lock:
            now = time.monotonic()
            parts = []
            for i, k in enumerate(self._keys, 1):
                rpm  = len([t for t in self._req_times[k] if now - t < 60])
                day  = self._day_reqs[k]
                blocked = now < self._blocked_until[k]
                state = "🔴blocked" if blocked else f"{rpm}rpm·{day}req"
                parts.append(f"key{i}:[{state}]")
            return "  ".join(parts)

    def count(self) -> int:
        return len(self._keys)


def load_openrouter_pool() -> "_OpenRouterKeyPool":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    import os
    raw = os.environ.get("OPENROUTER_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise RuntimeError("Thiếu OPENROUTER_API_KEYS trong .env (phân cách bằng dấu phẩy)")
    return _OpenRouterKeyPool(keys)


_OR_ENDPOINT = "https://openrouter.ai/api/v1/audio/transcriptions"


def _or_headers(key: str) -> dict:
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/crawl-video",
        "X-OpenRouter-Title": "Vietnamese Audio Crawler",
    }


def transcribe_openrouter(mp3_path: str, key_pool: "_OpenRouterKeyPool",
                          ffmpeg_exe: str | None,
                          model: str = "openai/whisper-large-v3") -> list[dict]:
    import requests as _req
    import base64
    import json as _json
    import re as _re
    import tempfile
    import os

    ffmpeg = ffmpeg_exe or "ffmpeg"
    if ffmpeg and Path(ffmpeg).is_dir():
        ffmpeg = str(Path(ffmpeg) / "ffmpeg.exe")

    duration = _probe_duration(mp3_path, ffmpeg)
    windows  = plan_chunks(duration) if duration > 0 else [(0.0, 0.0)]

    entry_lists: list[list[dict]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (start, end) in enumerate(windows):
            flac = os.path.join(tmp, f"chunk_{i}.flac")
            try:
                if end > start:
                    _extract_chunk_flac(mp3_path, start, end, ffmpeg, flac)
                else:
                    subprocess.run(
                        [ffmpeg, "-y", "-loglevel", "error",
                         "-i", mp3_path, "-ar", "16000", "-ac", "1", "-c:a", "flac", flac],
                        check=True, capture_output=True,
                    )
            except Exception as e:
                logging.error(f"[openrouter] chunk {i} ffmpeg loi: {e}")
                continue

            with open(flac, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("utf-8")

            payload = _json.dumps({
                "model": model,
                "input_audio": {"data": b64, "format": "flac"},
            })

            max_retries = key_pool.count() + 2
            conn_failures = 0
            for attempt in range(max_retries):
                key = key_pool.acquire()
                if key is None:
                    break
                try:
                    resp = _req.post(
                        _OR_ENDPOINT,
                        headers=_or_headers(key),
                        data=payload,
                        timeout=120,
                    )
                    if resp.status_code == 429:
                        m = _re.search(r"retry.after[=: ]+(\d+)", resp.text.lower())
                        retry_after = float(m.group(1)) if m else 60.0
                        key_pool.mark_rate_limited(key, retry_after)
                        if attempt < max_retries - 1:
                            continue
                        logging.error(f"[openrouter] chunk {i} 429 — hết key")
                        break
                    resp.raise_for_status()
                    text = (resp.json().get("text") or "").strip()
                    if text:
                        # OpenRouter chỉ trả text, không có timestamps
                        entry_lists.append([{
                            "start": _sec_to_hms(start),
                            "end":   _sec_to_hms(end if end > start else duration),
                            "text":  text,
                            "words": [],
                        }])
                    break
                except _req.exceptions.ConnectionError as e:
                    conn_failures += 1
                    wait = min(5 * conn_failures, 30)
                    logging.warning(
                        f"[openrouter] chunk {i} lỗi kết nối (lần {conn_failures})"
                        f" — thử lại sau {wait}s"
                    )
                    time.sleep(wait)
                    if attempt < max_retries - 1:
                        continue
                    break
                except Exception as e:
                    logging.error(f"[openrouter] chunk {i} attempt {attempt+1} loi: {e}")
                    break

    return merge_chunk_entries(entry_lists)


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
                    model: str = "whisper-large-v3") -> list[dict]:
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
            chunk_dur = (end - start) if end > start else duration
            if not _groq_limiter.acquire(chunk_dur):
                continue   # quota ngày hết, bỏ qua chunk này

            flac = os.path.join(tmp, f"chunk_{i}.flac")
            try:
                if end > start:
                    _extract_chunk_flac(mp3_path, start, end, ffmpeg, flac)
                else:
                    subprocess.run(
                        [ffmpeg, "-y", "-loglevel", "error",
                         "-i", mp3_path,
                         "-ar", "16000", "-ac", "1", "-c:a", "flac", flac],
                        check=True, capture_output=True,
                    )
            except Exception as e:
                logging.error(f"[groq] chunk {i} ffmpeg loi: {e}")
                continue

            for attempt in range(3):
                try:
                    with open(flac, "rb") as fh:
                        resp = client.audio.transcriptions.create(
                            model=model, file=(os.path.basename(flac), fh.read()),
                            language="vi", response_format="verbose_json",
                            timestamp_granularities=["segment", "word"],
                        )
                    data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
                    entry_lists.append(groq_response_to_entries(data, offset_sec=start))
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    is_429 = "429" in err_str or "rate limit" in err_str
                    if is_429:
                        logging.info(f"[groq] chunk {i} 429 — chờ rate limiter rồi thử lại")
                        # limiter sẽ tự block lần acquire tiếp theo
                    else:
                        logging.error(f"[groq] chunk {i} attempt {attempt+1} loi: {e}")
                    if attempt == 2:
                        break
                    time.sleep(5 * (attempt + 1))
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
        entry = {
            "start": _sec_to_hms(float(seg["start"]) + offset_sec),
            "end": _sec_to_hms(float(seg["end"]) + offset_sec),
            "text": text,
            "words": words_out,
        }
        _attach_metrics(entry, seg)
        entries.append(entry)
    return entries
