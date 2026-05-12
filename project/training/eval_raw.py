"""
Evaluate a trained KWS checkpoint on the non-augmented dataset under data/raw.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_kws import (
    CLASS_NAMES,
    AugWavDataset,
    compute_global_mfcc_mean_std,
    load_training_checkpoint,
)
from model import DS_CNN_KWS, MLP208_KWS


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, np.ndarray]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss_sum += float(loss_fn(logits, y).item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)
        for t, p in zip(y.view(-1), pred.view(-1)):
            conf[int(t), int(p)] += 1
    acc = correct / max(1, total)
    loss_mean = loss_sum / max(1, total)
    return loss_mean, acc, conf


def metrics_from_confusion(conf: np.ndarray, class_names: Tuple[str, ...]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        row = conf[i].sum()
        out[name] = float(conf[i, i] / row) if row > 0 else float("nan")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate KWS checkpoint on raw (non-augmented) WAVs under data/raw."
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
        help="Folder containing class subdirs (up, down, ...).",
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--arch",
        type=str,
        choices=("mlp", "dscnn"),
        default=None,
        help="Override architecture stored in checkpoint.",
    )
    p.add_argument(
        "--stats-root",
        type=Path,
        default=None,
        help="If checkpoint lacks MFCC stats (MLP only), compute from this root (defaults to data/aug).",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--sr", type=int, default=8000)
    p.add_argument("--metrics-json", type=Path, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    arch, state, mean, std = load_training_checkpoint(args.checkpoint, device)
    if args.arch is not None:
        arch = args.arch

    if arch == "mlp":
        model = MLP208_KWS(num_classes=len(CLASS_NAMES)).to(device)
        if mean is None or std is None:
            stats_root = (
                args.stats_root
                if args.stats_root is not None
                else Path(__file__).resolve().parent.parent / "data" / "aug"
            )
            mean, std = compute_global_mfcc_mean_std(stats_root, args.sr)
            print("Computed MFCC stats from:", stats_root)
        ds = AugWavDataset(args.data_root, sr=args.sr, mfcc_mean=mean, mfcc_std=std)
    else:
        model = DS_CNN_KWS(num_classes=len(CLASS_NAMES)).to(device)
        ds = AugWavDataset(args.data_root, sr=args.sr)

    model.load_state_dict(state)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    loss_fn = nn.CrossEntropyLoss()
    loss_mean, acc, conf = evaluate_loader(model, loader, loss_fn, device, len(CLASS_NAMES))
    per_class = metrics_from_confusion(conf, CLASS_NAMES)

    print(
        f"Loaded {args.checkpoint}\n"
        f"  samples: {len(ds)}\n"
        f"  loss: {loss_mean:.4f}\n"
        f"  accuracy: {acc:.4f} ({100.0 * acc:.2f}%)"
    )
    print("  per-class recall:", ", ".join(f"{k}={v:.3f}" for k, v in per_class.items()))

    if args.metrics_json:
        metrics = {
            "arch": arch,
            "checkpoint": str(args.checkpoint.resolve()),
            "samples": len(ds),
            "loss": loss_mean,
            "acc": acc,
            "per_class_recall": per_class,
            "confusion": conf.tolist(),
        }
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("Wrote", args.metrics_json.resolve())


if __name__ == "__main__":
    main()
