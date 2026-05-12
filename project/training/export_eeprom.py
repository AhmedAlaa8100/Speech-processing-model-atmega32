"""
Export int8 MLP weights to a raw EEPROM image for external 24Cxx storage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_kws import (
    CLASS_NAMES,
    AugWavDataset,
    compute_global_mfcc_mean_std,
    load_training_checkpoint,
)
from export import manual_ptq_mlp
from model import MLP208_KWS


def build_eeprom_image(ptq: Dict[str, object]) -> bytes:
    w1 = np.asarray(ptq["w1_q"], dtype=np.int8)
    b1 = np.asarray(ptq["b1_i32"], dtype=np.int32)
    w2 = np.asarray(ptq["w2_q"], dtype=np.int8)
    b2 = np.asarray(ptq["b2_i32"], dtype=np.int32)

    blob = bytearray()
    blob += w1.tobytes(order="C")
    blob += b1.astype("<i4").tobytes(order="C")
    blob += w2.tobytes(order="C")
    blob += b2.astype("<i4").tobytes(order="C")
    return bytes(blob)


def main() -> None:
    p = argparse.ArgumentParser(description="Export EEPROM image for external weights.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "aug",
        help="Augmented WAVs for PTQ calibration.",
    )
    p.add_argument("--sr", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--calib-max-samples", type=int, default=4096)
    p.add_argument(
        "--out-bin",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "training" / "checkpoints" / "kws_eeprom.bin",
    )
    p.add_argument("--map-json", type=Path, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch, state, mean_ckpt, std_ckpt = load_training_checkpoint(args.checkpoint, device)
    if arch != "mlp":
        raise SystemExit("EEPROM export requires an MLP checkpoint.")

    if mean_ckpt is None or std_ckpt is None:
        mean, std = compute_global_mfcc_mean_std(args.data_root, args.sr, CLASS_NAMES)
    else:
        mean, std = mean_ckpt, std_ckpt

    ds = AugWavDataset(args.data_root, sr=args.sr, mfcc_mean=mean, mfcc_std=std)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    xs = []
    n = 0
    for xb, _ in loader:
        xs.append(xb)
        n += xb.size(0)
        if n >= args.calib_max_samples:
            break
    if not xs:
        raise SystemExit("No calibration batches.")
    calib_x = torch.cat(xs, dim=0)[: args.calib_max_samples].to(device)

    model = MLP208_KWS(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(state)

    ptq = manual_ptq_mlp(model, calib_x)
    blob = build_eeprom_image(ptq)

    args.out_bin.parent.mkdir(parents=True, exist_ok=True)
    args.out_bin.write_bytes(blob)
    print("Wrote", args.out_bin.resolve())

    if args.map_json:
        sizes = {
            "fc1_w_bytes": int(ptq["w1_q"].size),
            "fc1_b_bytes": int(np.asarray(ptq["b1_i32"]).size * 4),
            "fc2_w_bytes": int(ptq["w2_q"].size),
            "fc2_b_bytes": int(np.asarray(ptq["b2_i32"]).size * 4),
            "total_bytes": len(blob),
        }
        offsets = {
            "fc1_w_offset": 0,
            "fc1_b_offset": sizes["fc1_w_bytes"],
            "fc2_w_offset": sizes["fc1_w_bytes"] + sizes["fc1_b_bytes"],
            "fc2_b_offset": sizes["fc1_w_bytes"] + sizes["fc1_b_bytes"] + sizes["fc2_w_bytes"],
        }
        payload = {"sizes": sizes, "offsets": offsets}
        args.map_json.parent.mkdir(parents=True, exist_ok=True)
        args.map_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Wrote", args.map_json.resolve())


if __name__ == "__main__":
    main()
