"""
Keyword-spotting dataset augmentation for tiny embedded targets.

Expands a small per-class corpus (e.g. 40 clips/class) to 2000+ using:
  - time shifting (+/- 100 ms)
  - Gaussian noise + optional "office" noise (WAV mix or synthetic hum + pink)
  - pitch scaling (librosa pitch_shift)

Requires: librosa, soundfile (and numpy).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import librosa
import numpy as np

try:
    import soundfile as sf
except ImportError as e:  # pragma: no cover
    raise ImportError("Install soundfile: pip install soundfile") from e


# Folder names under --data-root (must match exactly; used as class index order)
KEYWORDS_DEFAULT = ("up", "down", "left", "right", "yes", "no", "on", "off")


def _load_audio(path: Path, sr: int) -> Tuple[np.ndarray, int]:
    y, file_sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y, sr


def augment_time_shift(y: np.ndarray, sr: int, max_ms: float = 100.0, rng: random.Random | None = None) -> np.ndarray:
    rng = rng or random.Random()
    max_samples = int(abs(max_ms) * sr / 1000.0)
    if max_samples <= 0:
        return y
    shift = rng.randint(-max_samples, max_samples)
    return np.roll(y, shift)


def augment_gaussian_noise(
    y: np.ndarray,
    snr_db: float,
    rng: random.Random | None = None,
    np_rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or random.Random()
    if np_rng is None:
        np_rng = np.random.default_rng(rng.randrange(1 << 30))
    power = float(np.mean(y * y) + 1e-12)
    snr = 10.0 ** (snr_db / 10.0)
    noise_power = power / snr
    noise_sig = np_rng.standard_normal(len(y)).astype(np.float32)
    noise_sig *= np.sqrt(noise_power / (float(np.mean(noise_sig * noise_sig)) + 1e-12))
    return np.clip(y + noise_sig, -1.0, 1.0)


def _synthetic_office_noise(n: int, sr: int, rng: random.Random) -> np.ndarray:
    """Cheap stand-in when no office WAV is available: 50/60 Hz hum + band-limited noise."""
    t = np.arange(n, dtype=np.float32) / float(sr)
    f_hum = 60.0 if (sr % 60 == 0 or rng.random() > 0.5) else 50.0
    hum = 0.02 * np.sin(2.0 * np.pi * f_hum * t).astype(np.float32)
    hum += 0.01 * np.sin(2.0 * np.pi * 2 * f_hum * t).astype(np.float32)
    pink = librosa.util.normalize(np.random.randn(n).astype(np.float32), axis=0) * 0.03
    return hum + pink


def augment_office_noise(
    y: np.ndarray,
    sr: int,
    noise_wav: Optional[Path],
    mix_db: float,
    rng: random.Random | None = None,
) -> np.ndarray:
    rng = rng or random.Random()
    n = len(y)
    if noise_wav and noise_wav.is_file():
        n_src, _ = _load_audio(noise_wav, sr)
        if len(n_src) < n:
            reps = int(np.ceil(n / len(n_src)))
            n_src = np.tile(n_src, reps)[:n]
        else:
            start = rng.randint(0, max(0, len(n_src) - n))
            n_src = n_src[start : start + n]
        n_src = librosa.util.normalize(n_src.astype(np.float32), axis=0)
        y_n = librosa.util.normalize(y.astype(np.float32), axis=0)
        return librosa.util.normalize(y_n + (10.0 ** (mix_db / 20.0)) * n_src, axis=0)
    office = _synthetic_office_noise(n, sr, rng)
    y_n = librosa.util.normalize(y.astype(np.float32), axis=0)
    return librosa.util.normalize(y_n + (10.0 ** (mix_db / 20.0)) * office, axis=0)


def augment_pitch_scale(y: np.ndarray, sr: int, semitones: float, rng: random.Random | None = None) -> np.ndarray:
    rng = rng or random.Random()
    if abs(semitones) < 1e-6:
        return y
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=float(semitones)).astype(np.float32)


def one_randomized_variant(
    y: np.ndarray,
    sr: int,
    office_wav: Optional[Path],
    rng: random.Random,
) -> np.ndarray:
    x = y.copy()
    np_rng = np.random.default_rng(rng.randrange(1 << 30))
    # Order randomized lightly each call
    if rng.random() < 0.9:
        x = augment_time_shift(x, sr, max_ms=100.0, rng=rng)
    if rng.random() < 0.85:
        snr = rng.uniform(18.0, 35.0)
        x = augment_gaussian_noise(x, snr_db=snr, rng=rng, np_rng=np_rng)
    if rng.random() < 0.7:
        mix_db = rng.uniform(-22.0, -10.0)
        x = augment_office_noise(x, sr, noise_wav=office_wav, mix_db=mix_db, rng=rng)
    if rng.random() < 0.75:
        st = rng.uniform(-2.5, 2.5)
        x = augment_pitch_scale(x, sr, semitones=st, rng=rng)
    # final safety clip
    peak = float(np.max(np.abs(x)) + 1e-12)
    if peak > 0.99:
        x = (x / peak * 0.99).astype(np.float32)
    return x


def expand_class_directory(
    class_dir: Path,
    out_dir: Path,
    target_count: int,
    sr: int,
    office_wav: Optional[Path],
    seed: int,
) -> List[Path]:
    """
    Read all audio files in class_dir, repeat + augment until >= target_count files in out_dir.
    """
    rng = random.Random(seed ^ hash(class_dir.name) & 0xFFFF)
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = {".wav", ".flac", ".ogg", ".mp3"}
    sources = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in exts])
    if not sources:
        raise FileNotFoundError(f"No audio files in {class_dir}")

    written: List[Path] = []
    idx = 0
    # First pass: copy originals (resampled) for fidelity anchor
    for src in sources:
        y, _ = _load_audio(src, sr)
        dst = out_dir / f"{class_dir.name}_{idx:05d}_orig.wav"
        sf.write(str(dst), y, sr, subtype="PCM_16")
        written.append(dst)
        idx += 1

    # Augment until target
    while len(written) < target_count:
        src = rng.choice(sources)
        y, _ = _load_audio(src, sr)
        y2 = one_randomized_variant(y, sr, office_wav=office_wav, rng=rng)
        dst = out_dir / f"{class_dir.name}_{idx:05d}_aug.wav"
        sf.write(str(dst), y2, sr, subtype="PCM_16")
        written.append(dst)
        idx += 1
    return written


def expand_dataset_tree(
    data_root: Path,
    output_root: Path,
    per_class_target: int = 2000,
    sr: int = 16000,
    office_noise_wav: Optional[Path] = None,
    seed: int = 42,
    class_names: Iterable[str] = KEYWORDS_DEFAULT,
) -> None:
    """
    Expect layout: data_root/<ClassName>/*.wav
    Writes: output_root/<ClassName>/*.wav
    """
    output_root.mkdir(parents=True, exist_ok=True)
    for name in class_names:
        c_in = data_root / name
        if not c_in.is_dir():
            raise FileNotFoundError(f"Missing class folder: {c_in}")
        c_out = output_root / name
        expand_class_directory(c_in, c_out, per_class_target, sr, office_noise_wav, seed)


def main() -> None:
    p = argparse.ArgumentParser(description="Augment KWS dataset for ATmega32A TinyML pipeline.")
    p.add_argument("--data-root", type=Path, required=True, help="Input root with one subfolder per keyword.")
    p.add_argument("--out-root", type=Path, required=True, help="Output root for augmented audio.")
    p.add_argument("--per-class", type=int, default=2048, help="Minimum augmented files per class (default 2048).")
    p.add_argument("--sr", type=int, default=16000, help="Sample rate.")
    p.add_argument("--office-noise", type=Path, default=None, help="Optional office noise WAV to mix.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    expand_dataset_tree(
        args.data_root,
        args.out_root,
        per_class_target=args.per_class,
        sr=args.sr,
        office_noise_wav=args.office_noise,
        seed=args.seed,
    )
    print(f"Done. Augmented dataset written under {args.out_root.resolve()}")


if __name__ == "__main__":
    main()
