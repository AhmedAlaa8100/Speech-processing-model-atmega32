"""
Bit-exact fixed-point MFCC simulator (Python reference for C firmware).

Implement mel filterbank, log-energy, and DCT-II in fixed-point to match
`firmware/src/mfcc.c`. This module is a scaffold: wire dtypes / Q formats to
your `config.h` once scales are frozen.
"""

from __future__ import annotations

import numpy as np


def mfcc_frame_float(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """Floating MFCC for training parity checks only (not bit-exact)."""
    import librosa

    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=256, hop_length=128).astype(np.float32)[:, 0]


def mfcc_frame_fixed_stub(frame_fft_mag_q15: np.ndarray) -> np.ndarray:
    """Placeholder for Q15 spectrum -> Q15 MFCC vector (13,)."""
    raise NotImplementedError("Match mfcc.c Q formats and LUT log2 here.")
