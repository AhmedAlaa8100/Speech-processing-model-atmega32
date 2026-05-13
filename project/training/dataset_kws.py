"""Shared KWS dataset helpers (MFCC, class order) for train.py and export.py."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

# Must match `data_engine/augment.py` KEYWORDS_DEFAULT and README class order (0..7).
CLASS_NAMES = ("up", "down", "left", "right", "yes", "no", "on", "off")


def wav_to_mfcc_13x16(y: np.ndarray, sr: int) -> np.ndarray:
    """Full-clip MFCC map (13, 16). Pads/truncates along time to exactly 16 frames."""
    m = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_mels=26,
        n_fft=256,
        hop_length=128,
    ).astype(np.float32)
    t = m.shape[1]
    if t >= 16:
        start = max(0, (t - 16) // 2)
        m = m[:, start : start + 16]
    else:
        pad = np.zeros((13, 16 - t), dtype=np.float32)
        m = np.concatenate([m, pad], axis=1)
    return m


def normalize_map_per_sample(m: np.ndarray) -> np.ndarray:
    """Per-map z-score (legacy path for DS-CNN experiments)."""
    mean = float(m.mean())
    std = float(m.std() + 1e-6)
    x = (m - mean) / std
    return np.clip(x, -4.0, 4.0).astype(np.float32)


def normalize_map_global(m: np.ndarray, mean13: np.ndarray, std13: np.ndarray) -> np.ndarray:
    """Per-coefficient norm — matches MCU mfcc.c (mean/std per MFCC index)."""
    x = (m - mean13[:, None]) / (std13[:, None] + 1e-6)
    return np.clip(x, -10.0, 10.0).astype(np.float32)


def compute_global_mfcc_mean_std(
    aug_root: Path,
    sr: int,
    class_names: Tuple[str, ...] = CLASS_NAMES,
    max_files: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and std per MFCC coeff (13,) over all frames and clips under aug_root."""
    sum_c = np.zeros(13, dtype=np.float64)
    sumsq_c = np.zeros(13, dtype=np.float64)
    n_frames = 0
    n_files = 0
    for name in class_names:
        d = aug_root / name
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.wav")):
            if max_files is not None and n_files >= max_files:
                break
            y, file_sr = sf.read(str(p), always_2d=False)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            y = y.astype(np.float32)
            if file_sr != sr:
                y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
            m = wav_to_mfcc_13x16(y, sr)
            sum_c += m.sum(axis=1)
            sumsq_c += (m * m).sum(axis=1)
            n_frames += m.shape[1]
            n_files += 1
        if max_files is not None and n_files >= max_files:
            break
    if n_frames == 0:
        raise FileNotFoundError(f"No WAVs under {aug_root} to compute MFCC stats.")
    mean = (sum_c / float(n_frames)).astype(np.float32)
    var = sumsq_c / float(n_frames) - (mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    return mean, std


class AugWavDataset(Dataset):
    def __init__(
        self,
        aug_root: Path,
        sr: int,
        class_names: Tuple[str, ...] = CLASS_NAMES,
        mfcc_mean: Optional[np.ndarray] = None,
        mfcc_std: Optional[np.ndarray] = None,
    ) -> None:
        self.sr = sr
        self.mfcc_mean = mfcc_mean
        self.mfcc_std = mfcc_std
        self.samples: List[Tuple[Path, int]] = []
        for label, name in enumerate(class_names):
            d = aug_root / name
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.wav")):
                self.samples.append((p, label))
        if not self.samples:
            raise FileNotFoundError(
                f"No WAVs under {aug_root}/<class>/. Run data_engine/augment.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        y, file_sr = sf.read(str(path), always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32)
        if file_sr != self.sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=self.sr)
        m = wav_to_mfcc_13x16(y, self.sr)
        if self.mfcc_mean is not None and self.mfcc_std is not None:
            m = normalize_map_global(m, self.mfcc_mean, self.mfcc_std)
        else:
            m = normalize_map_per_sample(m)
        x = torch.from_numpy(m).unsqueeze(0)
        return x, torch.tensor(label, dtype=torch.long)


def load_training_checkpoint(
    path: Path, device: torch.device
) -> Tuple[str, dict, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (arch, state_dict, mfcc_mean, mfcc_std).
    Supports legacy flat state_dict files (treated as dscnn).
    """
    try:
        blob = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(str(path), map_location=device)
    if isinstance(blob, dict) and "state_dict" in blob:
        arch = str(blob.get("arch", "mlp"))
        mean = blob.get("mfcc_mean")
        std = blob.get("mfcc_std")
        if mean is not None:
            mean = np.asarray(mean, dtype=np.float32)
        if std is not None:
            std = np.asarray(std, dtype=np.float32)
        return arch, blob["state_dict"], mean, std
    return "dscnn", blob, None, None
