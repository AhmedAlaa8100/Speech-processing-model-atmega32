"""
Train KWS on augmented WAVs under `data/aug/<class>/*.wav`.

- `--arch mlp`: 208→96→8 MLP matching `firmware/include/kws_model.h` (use for int8 PTQ export).
- `--arch dscnn`: depthwise separable CNN in `model.py` (MCU export not wired for this path).

MFCC: n_fft=256, hop_length=128, sr=8000. MLP training uses global per-coefficient mean/std
(aligned with `mfcc.c` + `kws_input_mean` / `kws_input_std`).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    """
    Returns (mean loss, accuracy in [0,1], confusion matrix [C, C] counts).
    """
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


def metrics_from_confusion(
    conf: np.ndarray, class_names: Tuple[str, ...]
) -> Dict[str, float]:
    """Per-class recall (accuracy of true class i among samples of class i)."""
    out: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        row = conf[i].sum()
        out[name] = float(conf[i, i] / row) if row > 0 else float("nan")
    return out


def train(
    aug_root: Path,
    out_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    sr: int,
    device: torch.device,
    seed: int,
    arch: str,
) -> Dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if arch == "mlp":
        mean, std = compute_global_mfcc_mean_std(aug_root, sr)
        ds = AugWavDataset(aug_root, sr=sr, mfcc_mean=mean, mfcc_std=std)
    else:
        mean, std = None, None
        ds = AugWavDataset(aug_root, sr=sr)

    print("samples:", len(ds), "arch:", arch)
    n_val = max(1, len(ds) // 10)
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    if arch == "mlp":
        model = MLP208_KWS(num_classes=len(CLASS_NAMES)).to(device)
    else:
        model = DS_CNN_KWS(num_classes=len(CLASS_NAMES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += x.size(0)
        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)

        model.eval()
        v_total, v_correct, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                v_loss += float(loss_fn(logits, y).item()) * x.size(0)
                v_correct += int((logits.argmax(dim=1) == y).sum().item())
                v_total += x.size(0)
        val_acc = v_correct / max(1, v_total)
        val_loss = v_loss / max(1, v_total)
        print(
            f"epoch {ep}/{epochs}  train loss {train_loss:.4f} acc {train_acc:.3f}  "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if arch == "mlp" and mean is not None and std is not None:
        torch.save(
            {
                "arch": "mlp",
                "state_dict": model.state_dict(),
                "mfcc_mean": mean.astype(np.float64).tolist(),
                "mfcc_std": std.astype(np.float64).tolist(),
            },
            str(out_path),
        )
    else:
        torch.save(model.state_dict(), str(out_path))
    print("Wrote", out_path.resolve())

    # Final accuracy (same weights as saved checkpoint).
    full_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    tr_loss, tr_acc, tr_conf = evaluate_loader(model, train_loader, loss_fn, device, len(CLASS_NAMES))
    va_loss, va_acc, va_conf = evaluate_loader(model, val_loader, loss_fn, device, len(CLASS_NAMES))
    al_loss, al_acc, al_conf = evaluate_loader(model, full_loader, loss_fn, device, len(CLASS_NAMES))
    print(
        "\nFinal evaluation (trained weights):\n"
        f"  train  loss {tr_loss:.4f}  acc {tr_acc:.4f}  ({100.0 * tr_acc:.2f}%)\n"
        f"  val    loss {va_loss:.4f}  acc {va_acc:.4f}  ({100.0 * va_acc:.2f}%)\n"
        f"  all    loss {al_loss:.4f}  acc {al_acc:.4f}  ({100.0 * al_acc:.2f}%)  (full aug set)"
    )
    per_all = metrics_from_confusion(al_conf, CLASS_NAMES)
    print("  per-class recall (all data):", ", ".join(f"{k}={v:.3f}" for k, v in per_all.items()))

    return {
        "arch": arch,
        "checkpoint": str(out_path.resolve()),
        "train_loss": tr_loss,
        "train_acc": tr_acc,
        "val_loss": va_loss,
        "val_acc": va_acc,
        "all_loss": al_loss,
        "all_acc": al_acc,
        "per_class_recall_all": per_all,
        "confusion_all": al_conf.tolist(),
    }


def eval_checkpoint(
    aug_root: Path,
    checkpoint: Path,
    batch_size: int,
    sr: int,
    device: torch.device,
    arch_override: Optional[str] = None,
) -> Dict[str, object]:
    arch, state, mean, std = load_training_checkpoint(checkpoint, device)
    if arch_override is not None:
        arch = arch_override
    if arch == "mlp":
        model = MLP208_KWS(num_classes=len(CLASS_NAMES)).to(device)
        if mean is None or std is None:
            mean, std = compute_global_mfcc_mean_std(aug_root, sr)
        ds = AugWavDataset(aug_root, sr=sr, mfcc_mean=mean, mfcc_std=std)
    else:
        model = DS_CNN_KWS(num_classes=len(CLASS_NAMES)).to(device)
        ds = AugWavDataset(aug_root, sr=sr)
    model.load_state_dict(state)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    loss_fn = nn.CrossEntropyLoss()
    loss_mean, acc, conf = evaluate_loader(model, loader, loss_fn, device, len(CLASS_NAMES))
    per_class = metrics_from_confusion(conf, CLASS_NAMES)
    print(
        f"Loaded {checkpoint}\n"
        f"  samples: {len(ds)}\n"
        f"  loss: {loss_mean:.4f}\n"
        f"  accuracy: {acc:.4f} ({100.0 * acc:.2f}%)"
    )
    print("  per-class recall:", ", ".join(f"{k}={v:.3f}" for k, v in per_class.items()))
    return {
        "arch": arch,
        "checkpoint": str(checkpoint.resolve()),
        "samples": len(ds),
        "loss": loss_mean,
        "acc": acc,
        "per_class_recall": per_class,
        "confusion": conf.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train KWS (MLP for AVR export, or DS-CNN) on data/aug WAVs.")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "aug",
        help="Folder containing class subdirs (up, down, ...).",
    )
    p.add_argument("--save", type=Path, default=Path("checkpoints/kws_mlp.pt"))
    p.add_argument(
        "--arch",
        type=str,
        choices=("mlp", "dscnn"),
        default=None,
        help="mlp = AVR kws_model topology (default). dscnn = conv model (no int8 export yet).",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sr", type=int, default=8000, help="Must match augment.py sample rate.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Write train/val/all accuracy and confusion (after training or --eval-only).",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Load --checkpoint and print accuracy on data under --data-root (no training).",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="state_dict path; required with --eval-only.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    arch = args.arch if args.arch is not None else "mlp"

    if args.eval_only:
        if args.checkpoint is None:
            raise SystemExit("--eval-only requires --checkpoint path/to/kws_mlp.pt")
        metrics = eval_checkpoint(
            aug_root=args.data_root,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            sr=args.sr,
            device=device,
            arch_override=args.arch,
        )
        if args.metrics_json:
            args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print("Wrote", args.metrics_json.resolve())
        return

    metrics = train(
        aug_root=args.data_root,
        out_path=args.save,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sr=args.sr,
        device=device,
        seed=args.seed,
        arch=arch,
    )
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("Wrote", args.metrics_json.resolve())


if __name__ == "__main__":
    main()
