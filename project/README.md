# TinyML KWS (ATmega32A) — full cycle

Paths below are relative to the **`project`** folder (where `data/`, `training/`, `firmware/` live).

## 1. Python environment

```powershell
cd project
python -m venv .venv
.\.venv\Scripts\activate
pip install torch librosa soundfile numpy
```

## 2. Dataset layout (8 classes)

Put **raw** WAVs here:

```text
project/data/raw/up/*.wav
project/data/raw/down/*.wav
project/data/raw/left/*.wav
project/data/raw/right/*.wav
project/data/raw/yes/*.wav
project/data/raw/no/*.wav
project/data/raw/on/*.wav
project/data/raw/off/*.wav
```

Use **exactly** these folder names (lowercase). Class **index 0–7** follows this list everywhere (training, export, LCD).

## 3. Augment

```powershell
cd data_engine
python augment.py --data-root ..\data\raw --out-root ..\data\aug --per-class 2048 --sr 8000
```

Optional office-noise mix: `--office-noise path\to\noise.wav`

## 4. Train (float, MLP for MCU)

The AVR firmware uses a **208 → 96 → 8** int8 MLP (`kws_model.h`). Train that architecture with **global MFCC mean/std** per coefficient (same idea as `firmware/src/mfcc.c`).

```powershell
cd ..\training
python train.py --data-root ..\data\aug --save checkpoints\kws_mlp.pt --epochs 30 --batch-size 32
```

- Default `--arch` is **`mlp`** (omit the flag for the normal flow).
- **`--arch dscnn`** trains the depthwise CNN in `model.py` for experiments; **int8 export to the current firmware is only implemented for `mlp`.**
- Optional: `--metrics-json checkpoints\metrics.json` for train/val/all accuracy and confusion.
- Re-evaluate only: `python train.py --eval-only --checkpoint checkpoints\kws_mlp.pt --data-root ..\data\aug`

Smoke-test uninitialized weights:

```powershell
python model.py --arch mlp --save checkpoints\init_mlp.pt
```

## 5. Export (int8 PTQ → C)

`export.py` loads the **mlp** checkpoint (dict with `state_dict`, `mfcc_mean`, `mfcc_std`), runs **min–max calibration** on augmented WAVs, and overwrites:

- `firmware/src/kws_model.c` — int8 weights, int32 biases, float **mean/std** (13 coeffs), plus **fixed-point** tables `kws_mfcc_mean_q12[]` and `kws_mfcc_i8_mul[]` for the MCU MFCC path.
- `firmware/include/kws_model_scales.generated.h` — integer **FC1** rescale macros `KWS_FC1_RESCALE_MUL` / `KWS_FC1_RESCALE_SHR` (replaces float scale math in `main.c`).

`kws_model.h` includes `kws_model_scales.generated.h` (defaults ship in-repo until you export).

```powershell
python export.py --checkpoint checkpoints\kws_mlp.pt --data-root ..\data\aug
```

Flags: `--calib-max-samples 4096` (default), `--sr 8000`, paths `--out-c` / `--out-scales-h` if you want non-default outputs.

The script prints **calibration argmax match** (fraction of samples where quantized pipeline agrees with float argmax on the calibration batch).

## 6. MFCC tables (C)

After you lock FFT/mel settings to training:

```powershell
cd ..\firmware\tools
python gen_mfcc_tables.py
```

## 7. Build and flash firmware

- Toolchain: **ATmega32**, **F_CPU=11059200**, sources under `firmware/src`, includes under `firmware/include`.
- Typical flow: `avr-gcc` → `avr-objcopy` → `avrdude` (or Atmel Studio / MPLAB).

Adjust `firmware/include/sram.h` (SPI / CS pins) and LCD wiring in `LCD1602.h` to match your board.

### Flash / linker: `region text overflowed`

ATmega32 has **32 KiB** flash; soft-float and large mel tables add up quickly. This repo is tuned to fit by:

- **26 mel bins** (`MFCC_N_MELS` in `mfcc.h`, `n_mels=26` in `dataset_kws.py` and `firmware/tools/gen_mfcc_tables.py`) — run `gen_mfcc_tables.py` after changing mel count.
- **No float** on the inference/MFCC hot path (`mfcc.c`, `main.c` FC1 + confidence display) — `export.py` emits matching int tables and `KWS_FC1_*` macros.
- In MPLAB / Makefile use **`-Os`** (optimize for size), keep **`-ffunction-sections -fdata-sections`** and **`-Wl,--gc-sections`**.

Copy the updated `firmware/` sources into your MPLAB project tree, clean, and rebuild.

## 8. Sanity checks

- **Bit-exact MFCC** (optional): fixed-point Python vs MCU (`test_vector.h`).
- On device: check sample rate, hop/window, and that **exported mean/std** match what you trained with (re-run `export.py` after retraining).

## End-to-end command list

```powershell
cd project
.\.venv\Scripts\activate
cd data_engine
python augment.py --data-root ..\data\raw --out-root ..\data\aug --per-class 2048 --sr 8000
cd ..\training
python train.py --data-root ..\data\aug --save checkpoints\kws_mlp.pt --metrics-json checkpoints\metrics.json
python export.py --checkpoint checkpoints\kws_mlp.pt --data-root ..\data\aug
cd ..\firmware\tools
python gen_mfcc_tables.py
```

Then build/flash the `firmware` tree.
