"""
Filename parsing and normalization for airfoil flow field data.
"""
import re
import numpy as np
from typing import Tuple


# Pattern: {airfoil}(gen{iteration})?_{mach}_{aoa}_{reynolds}.npz
_FILENAME_PATTERN = re.compile(
    r"(.+?)(?:gen\d+)?_(\d+)_(-?\d+)_(\d+)\.npz"
)


def parse_filename(filename: str) -> Tuple[str, float, float, float]:
    """
    Parse a training data filename.

    Returns:
        (airfoil_name, Ma, AoA_degrees, Re)

    Example:
        "rae2822gen0_50_-100_3000.npz" -> ("rae2822", 0.50, -1.00, 3000000)
    """
    m = _FILENAME_PATTERN.match(filename)
    if m is None:
        raise ValueError(f"Could not parse filename: {filename}")

    name = m.group(1)
    ma_code = int(m.group(2))
    aoa_code = int(m.group(3))
    re_code = int(m.group(4))

    ma = ma_code / 100.0
    aoa = aoa_code / 100.0
    re_val = re_code * 1000

    return name, ma, aoa, re_val


class Normalizer:
    """
    Per-channel min-max normalizer for the 16-channel data.

    The 16 channels are:
      0-2:   IC (Ma, AoA, Re)         — constant per sample
      3-11:  grid metrics              — spatially varying
      12-15: flow field (rho, p, u, v) — spatially varying
    """

    def __init__(self, ic_range: Tuple[float, float] = (0.0, 1.0)):
        self.ic_range = ic_range
        self._fitted = False
        self.channel_mins: np.ndarray | None = None
        self.channel_maxs: np.ndarray | None = None

    def fit(self, data: np.ndarray):
        """
        Compute per-channel min/max over the batch dimension (first axis).

        Args:
            data: shape (N, 16, H, W)
        """
        N, C, H, W = data.shape
        # Reshape to (C, N*H*W) for efficient min/max
        flat = data.transpose(1, 0, 2, 3).reshape(C, -1)
        self.channel_mins = flat.min(axis=1)
        self.channel_maxs = flat.max(axis=1)
        self._fitted = True

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Min-max normalize the 16-channel data to [0, 1]."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted — call .fit() first.")
        out = data.copy().astype(np.float32)
        for c in range(data.shape[1]):
            lo = self.channel_mins[c]
            hi = self.channel_maxs[c]
            span = hi - lo
            if span > 1e-12:
                out[:, c] = (data[:, c] - lo) / span
            else:
                out[:, c] = 0.0
        return out

    def denormalize(self, data: np.ndarray, channels: list[int]) -> np.ndarray:
        """Reverse normalization for specific channels."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted.")
        out = data.copy()
        for c in channels:
            lo = self.channel_mins[c]
            hi = self.channel_maxs[c]
            span = hi - lo
            out[..., c, :, :] = data[..., c, :, :] * span + lo
        return out

    def save(self, path: str):
        np.savez_compressed(path,
                            mins=self.channel_mins,
                            maxs=self.channel_maxs)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        obj = cls()
        loaded = np.load(path)
        obj.channel_mins = loaded["mins"]
        obj.channel_maxs = loaded["maxs"]
        obj._fitted = True
        return obj


def split_input_target(raw_data: np.ndarray, config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the 16-channel data into input (12ch) and target (4ch).

    Args:
        raw_data: shape (N, 16, H, W)

    Returns:
        inputs:  shape (N, 12, H, W)
        targets: shape (N, 4, H, W)
    """
    ic = raw_data[:, config.ic_channels]
    grid = raw_data[:, config.grid_channels]
    flow = raw_data[:, config.flow_channels]
    inputs = np.concatenate([ic, grid], axis=1)
    return inputs, flow
