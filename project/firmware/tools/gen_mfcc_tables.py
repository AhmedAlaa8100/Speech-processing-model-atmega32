"""Generate mfcc_tables.c PROGMEM arrays for ATmega32 KWS firmware."""
import math
import os

import numpy as np

N_FFT = 256
WIN = 200
N_MELS = 26
N_MFCC = 13
LOG2_Q = 12
FFT_BINS = N_FFT // 2 + 1


def main() -> None:
    tw_cos = []
    tw_sin = []
    for k in range(N_FFT // 2):
        th = -2 * math.pi * k / N_FFT
        tw_cos.append(int(round(math.cos(th) * 32767)))
        tw_sin.append(int(round(math.sin(th) * 32767)))

    win = np.hanning(WIN)
    win_q15 = np.round(win * 32767).astype(int)

    lut = [int(round(math.log2(1 + i / 256.0) * (1 << LOG2_Q))) for i in range(256)]

    def hz_to_mel(f: float) -> float:
        return 2595 * math.log10(1 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700 * (10 ** (m / 2595) - 1)

    sr = 8000.0
    mel_min, mel_max = hz_to_mel(0.0), hz_to_mel(sr / 2)
    mels = np.linspace(mel_min, mel_max, N_MELS + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((N_FFT + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, N_FFT // 2)
    fb = np.zeros((N_MELS, FFT_BINS))
    for i in range(N_MELS):
        fL, fC, fH = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if fC == fL:
            fC += 1
        if fH == fC:
            fH += 1
        for k in range(fL, min(fC, FFT_BINS)):
            fb[i, k] = (k - fL) / max(fC - fL, 1)
        for k in range(fC, min(fH, FFT_BINS)):
            fb[i, k] = (fH - k) / max(fH - fC, 1)

    fb_q15 = np.zeros_like(fb, dtype=int)
    for i in range(N_MELS):
        row = fb[i]
        mx = float(row.max() or 1.0)
        fb_q15[i] = np.round(row / mx * 32767).astype(int)

    dct = np.zeros((N_MFCC, N_MELS))
    for c in range(N_MFCC):
        for m in range(N_MELS):
            dct[c, m] = math.sqrt(2.0 / N_MELS) * math.cos(math.pi * c * (m + 0.5) / N_MELS)
    dct_q15 = np.round(dct * 32767).astype(int)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(root, "src", "mfcc_tables.c")

    lines: list[str] = []
    lines += [
        '#include <avr/pgmspace.h>',
        '#include "mfcc.h"',
        '#include "mfcc_tables.h"',
        "",
        "const int16_t kws_twiddle_cos_q15[128] PROGMEM = {",
        "  " + ", ".join(str(int(x)) for x in tw_cos[:128]),
        "};",
        "",
        "const int16_t kws_twiddle_sin_q15[128] PROGMEM = {",
        "  " + ", ".join(str(int(x)) for x in tw_sin[:128]),
        "};",
        "",
        "const int16_t kws_window_q15[MFCC_WIN_LENGTH] PROGMEM = {",
    ]
    for i in range(0, WIN, 8):
        chunk = ", ".join(str(int(x)) for x in win_q15[i : i + 8])
        comma = "," if i + 8 < WIN else ""
        lines.append("  " + chunk + comma)
    lines.append("};")
    lines += ["", "const uint16_t kws_log2_lut_q12[256] PROGMEM = {", "  " + ", ".join(str(int(x)) for x in lut), "};", ""]

    lines.append("const uint16_t kws_mel_filters_q15[MFCC_N_MELS][FFT_BINS] PROGMEM = {")
    for i in range(N_MELS):
        row = ", ".join(str(int(x)) for x in fb_q15[i])
        lines.append("  { " + row + " }" + ("," if i < N_MELS - 1 else ""))
    lines.append("};")
    lines.append("")

    lines.append("const int16_t kws_dct_q15[KWS_N_MFCC][MFCC_N_MELS] PROGMEM = {")
    for c in range(N_MFCC):
        row = ", ".join(str(int(x)) for x in dct_q15[c])
        lines.append("  { " + row + " }" + ("," if c < N_MFCC - 1 else ""))
    lines.append("};")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Wrote", path)


if __name__ == "__main__":
    main()
