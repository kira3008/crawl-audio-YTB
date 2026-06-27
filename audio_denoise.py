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
