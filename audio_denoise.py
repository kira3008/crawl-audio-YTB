#!/usr/bin/env python3
"""audio_denoise.py — loc nhieu tin hieu audio bang Demucs (tach vocal)."""

import subprocess
import sys

TARGET_SR        = 22050
TARGET_CHANNELS  = 1
DEMUCS_MODEL     = "htdemucs"
DEMUCS_OVERLAP   = 0.25
LOUDNORM_I       = -23
LOUDNORM_TP      = -1.5
LOUDNORM_LRA     = 11
HIGHPASS_HZ      = 80


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _pick_device(device, cuda_available: bool) -> str:
    if device:
        return device
    return "cuda" if cuda_available else "cpu"


def _ensure_demucs():
    # seam testable: lazy-install demucs neu thieu, tra ve class Separator
    try:
        from demucs.api import Separator
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "demucs"])
        from demucs.api import Separator
    return Separator


def load_demucs(device=None):
    Separator = _ensure_demucs()
    dev = _pick_device(device, _cuda_available())
    return Separator(model=DEMUCS_MODEL, device=dev)


def _post_filter() -> str:
    return (f"highpass=f={HIGHPASS_HZ}:p=2,"
            f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}")


def _post_process(in_wav, out_path, ffmpeg_exe) -> bool:
    cmd = [
        ffmpeg_exe, "-y", "-loglevel", "error",
        "-i", str(in_wav),
        "-af", _post_filter(),
        "-ar", str(TARGET_SR),
        "-ac", str(TARGET_CHANNELS),
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def separate_vocals(in_path, separator, tmp_wav):
    import torchaudio
    _, stems = separator.separate_audio_file(str(in_path))
    vocals = stems["vocals"]          # chon theo TEN, khong hardcode index
    torchaudio.save(str(tmp_wav), vocals.cpu(), separator.samplerate)
    return tmp_wav


def denoise_file(in_path, out_path, separator, ffmpeg_exe) -> bool:
    from pathlib import Path
    tmp = Path(out_path).with_suffix(".vocals.wav")
    try:
        separate_vocals(in_path, separator, tmp)
        ok = _post_process(tmp, out_path, ffmpeg_exe)
        return ok
    except Exception as e:
        print(f"[denoise] loi {in_path}: {e}")
        return False
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
