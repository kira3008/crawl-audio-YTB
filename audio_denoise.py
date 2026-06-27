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
DENOISE_SUBDIR   = "audio_denoised"


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


def _ensure_demucs_model():
    # seam testable: lazy-install demucs neu thieu, tra ve model htdemucs.
    # Dung API goc cap thap (pretrained.get_model) — co trong moi ban demucs 4.x.
    # (demucs.api.Separator KHONG co trong ban PyPI 4.0.1.)
    try:
        from demucs.pretrained import get_model
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "demucs>=4.0.0"])
        from demucs.pretrained import get_model
    return get_model(DEMUCS_MODEL)


def load_demucs(device=None):
    model = _ensure_demucs_model()
    dev = _pick_device(device, _cuda_available())
    model.to(dev)
    model.eval()
    return model


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


def _read_audio(in_path, model):
    # doc audio ve dung samplerate/channels cua model (44100/stereo cho htdemucs)
    from demucs.audio import AudioFile
    return AudioFile(str(in_path)).read(
        streams=0, samplerate=model.samplerate, channels=model.audio_channels)


def _apply_demucs(model, wav):
    # chuan hoa theo demucs/separate.py roi tach nguon; tra tensor [sources, ch, len]
    import torch
    from demucs.apply import apply_model
    device = next(model.parameters()).device
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(model, wav[None].to(device),
                              overlap=DEMUCS_OVERLAP, device=device)[0]
    return sources * (ref.std() + 1e-8) + ref.mean()


def _save_wav(wav, path, samplerate):
    from demucs.audio import save_audio
    save_audio(wav.cpu(), str(path), samplerate)


def separate_vocals(in_path, model, tmp_wav):
    wav = _read_audio(in_path, model)
    sources = _apply_demucs(model, wav)
    vocals_idx = model.sources.index("vocals")   # chon theo TEN, khong hardcode index
    _save_wav(sources[vocals_idx], tmp_wav, model.samplerate)
    return tmp_wav


def denoise_file(in_path, out_path, model, ffmpeg_exe) -> bool:
    from pathlib import Path
    tmp = Path(out_path).with_suffix(".vocals.wav")
    try:
        separate_vocals(in_path, model, tmp)
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


def denoise_batch(in_paths, ffmpeg_exe, console=None):
    from pathlib import Path

    def log(msg):
        if console:
            console.print(msg)
        else:
            print(msg)

    try:
        model = load_demucs()
    except Exception as e:
        log(f"[red]✗ Demucs khong kha dung ({e}) — bo qua denoise, KHONG cat tho.[/red]")
        return {}

    result: dict = {}
    for p in in_paths:
        p = Path(p)
        out_dir = p.parent / DENOISE_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{p.stem}.wav"
        if denoise_file(p, out_path, model, ffmpeg_exe):
            result[p] = out_path
        else:
            log(f"[yellow]⚠ Skip (denoise loi): {p.name}[/yellow]")
    return result
