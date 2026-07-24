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


class AttentionGate(nn.Module):
    """Attention gate for filtering skip features with decoder context."""

    def __init__(self, skip_ch: int, gate_ch: int, inter_ch: int):
        super().__init__()
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.attn = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inter_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        alpha = self.attn(self.skip_proj(skip) + self.gate_proj(gate))
        return skip * alpha


class UpBlock(nn.Module):
    """
    Up-sampling: TransposedConv2d (×2) → DoubleConv.
    Skip connection via concatenation.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 use_bn: bool = True, dropout: float = 0.0,
                 upsample_mode: str = "transpose",
                 use_attention: bool = False):
        super().__init__()
        if upsample_mode == "transpose":
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        elif upsample_mode == "bilinear":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
            )
        else:
            raise ValueError(f"Unsupported upsample_mode: {upsample_mode}")
        self.attention = (
            AttentionGate(out_ch, out_ch, max(out_ch // 2, 1))
            if use_attention else None
        )
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
        if self.attention is not None:
            skip = self.attention(skip, x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class Bottleneck(nn.Module):
    """Bottleneck block at the deepest layer."""

    def __init__(self, in_ch: int, out_ch: int, **kwargs):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
