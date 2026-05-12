"""
Depthwise Separable CNN (DS-CNN) for 13x16 MFCC maps, KWS (8 classes),
plus `MLP208_KWS` (208→96→8) for the AVR int8 export path in `export.py`.

Input:  (N, 1, 13, 16) float after feature normalization.
Blocks: Conv3x3 (8) -> DW3x3 -> PW1x1 (16) -> Global Avg Pool -> Linear (8).

PTQ: static int8 export for the **MLP** is implemented in `export.py` (min–max
calibration). The DS-CNN is float-only in this repo unless you extend export.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


class DS_CNN_KWS(nn.Module):
    """
    Compact DS-CNN matching the firmware-oriented topology:
      Conv2d(1->8, 3x3, padding=1) + BN + ReLU
      DepthwiseConv2d(8 groups, 3x3, padding=1) + BN + ReLU
      Pointwise Conv2d(8->16, 1x1) + BN + ReLU
      AdaptiveAvgPool2d(1)
      Linear(16 -> num_classes)
    """

    def __init__(self, num_classes: int = 8, in_ch: int = 1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(16, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, mode="fan_in", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class MLP208_KWS(nn.Module):
    """
    Two-layer MLP matching `firmware/include/kws_model.h`:
      flatten (1,13,16) -> 208,
      Linear(208 -> 96) + ReLU,
      Linear(96 -> num_classes).

    Use this architecture for int8 PTQ export to the current AVR inference path.
    """

    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(208, 96)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(96, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def model_size_macs(inp: Tuple[int, int, int, int] = (1, 1, 13, 16)) -> None:
    """Rough MAC estimate for logging (not cycle-accurate)."""
    n, c, h, w = inp
    # stem conv 3x3
    macs = n * 8 * c * 3 * 3 * h * w
    # dw 3x3
    macs += n * 8 * 3 * 3 * h * w
    # pw 1x1
    macs += n * 16 * 8 * 1 * 1 * h * w
    # fc
    macs += n * 16 * 8
    print(f"Approx MACs per frame (H=W from input): {macs}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--save", type=Path, default=None, help="Optional path to save TorchScript or state_dict.")
    p.add_argument("--script", action="store_true", help="Save TorchScript model.")
    p.add_argument("--arch", choices=("dscnn", "mlp"), default="dscnn", help="mlp matches AVR kws_model.h (208->96->8).")
    args = p.parse_args()

    if args.arch == "mlp":
        m = MLP208_KWS()
    else:
        m = DS_CNN_KWS()
    x = torch.randn(2, 1, 13, 16)
    y = m(x)
    print("out:", y.shape, "arch:", args.arch)
    if args.arch == "dscnn":
        model_size_macs()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        if args.script:
            traced = torch.jit.trace(m.eval(), x[:1])
            traced.save(str(args.save))
        else:
            torch.save(m.state_dict(), str(args.save))
        print("Wrote", args.save)


if __name__ == "__main__":
    main()
