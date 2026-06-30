"""
Building blocks for the deep UNet architecture.
"""
import torch
import torch.nn as nn
from typing import Optional


class DoubleConv(nn.Module):
    """Two consecutive (Conv2D → BN → LeakyReLU) blocks."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        pad = kernel_size // 2
        layers: list[nn.Module] = []

        for i in range(2):
            _in = in_ch if i == 0 else out_ch
            layers.append(nn.Conv2d(_in, out_ch, kernel_size,
                                     padding=pad, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Down-sampling: MaxPool2d → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """
    Up-sampling: TransposedConv2d (×2) → DoubleConv.
    Skip connection via concatenation.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # After concat with skip: out_ch + out_ch → out_ch
        self.conv = DoubleConv(out_ch * 2, out_ch, kernel_size,
                                use_bn=use_bn, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch (due to odd dimensions)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = torch.nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                                         diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class Bottleneck(nn.Module):
    """Bottleneck block at the deepest layer."""

    def __init__(self, in_ch: int, out_ch: int, **kwargs):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
