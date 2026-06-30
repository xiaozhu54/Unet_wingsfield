"""
Deep UNet for airfoil compressible flow field prediction.

Architecture (as described in the paper, ~75M params):
  - 6 down-sampling layers
  - 6 up-sampling layers
  - Skip connections between encoder and decoder
  - Input:  12 channels (3 IC + 9 grid metrics)
  - Output: 4 channels  (rho, p, u, v)
"""
import torch
import torch.nn as nn

from .blocks import DoubleConv, DownBlock, UpBlock, Bottleneck


class UNet(nn.Module):
    """
    Deep UNet with configurable depth.

    Default: 6 down/up levels, base_filters=32

    Down:  [in_ch, 32, 64, 128, 256, 512, 1024]
    Up:    [1024, 512, 256, 128, 64, 32]
    """

    def __init__(self, in_channels: int = 12, out_channels: int = 4,
                 base_filters: int = 32, n_down: int = 6,
                 kernel_size: int = 3, use_bn: bool = True,
                 dropout: float = 0.0):
        super().__init__()

        self.n_down = n_down

        # ── Initial convolution (no down-sampling yet) ──
        self.entry = DoubleConv(in_channels, base_filters,
                                kernel_size=kernel_size,
                                use_bn=use_bn, dropout=dropout)

        # ── Encoder (down-sampling path) ──
        self.encoders = nn.ModuleList()
        in_ch = base_filters
        for i in range(n_down):
            out_ch = base_filters * (2 ** (i + 1))
            self.encoders.append(
                DownBlock(in_ch, out_ch, kernel_size=kernel_size,
                          use_bn=use_bn, dropout=dropout)
            )
            in_ch = out_ch

        # ── Bottleneck (same channel count, no expansion) ──
        self.bottleneck = Bottleneck(in_ch, in_ch,
                                     kernel_size=kernel_size,
                                     use_bn=use_bn, dropout=dropout)

        # ── Decoder (up-sampling path) ──
        self.decoders = nn.ModuleList()
        for i in range(n_down):
            _in = base_filters * (2 ** (n_down - i))
            _out = base_filters * (2 ** (n_down - i - 1))
            self.decoders.append(
                UpBlock(_in, _out, kernel_size=kernel_size,
                        use_bn=use_bn, dropout=dropout)
            )

        # ── Output projection ──
        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entry
        x = self.entry(x)
        skips = []

        # Encoder
        for encoder in self.encoders:
            skips.append(x)
            x = encoder(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[-(i + 1)])

        return self.final(x)


def build_unet(config) -> UNet:
    """Factory function to build a UNet from config."""
    return UNet(
        in_channels=config.input_channels,
        out_channels=config.output_channels,
        base_filters=config.base_filters,
        n_down=config.n_down,
        kernel_size=config.kernel_size,
        use_bn=config.use_batch_norm,
        dropout=config.dropout,
    )
