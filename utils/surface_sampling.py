"""
Surface Sampling Contract for near-wall Cp extraction.

The current dataset does not carry a true wall mask or surface index. This
module keeps the approximate transformed-grid sampling rule in one place, so
training losses and evaluation scripts use the same interface.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from utils.metrics import compute_pressure_coefficient


def _line_indices(length: int, index: Optional[int], offset: int) -> Tuple[int, int]:
    center = length // 2 if index is None else int(index)
    offset = max(1, int(offset))
    lower_idx = max(0, center - offset)
    upper_idx = min(length - 1, center + offset)
    return upper_idx, lower_idx


def sample_cp_lines(flow: np.ndarray, ma: float, axis: str = "row",
                    index: Optional[int] = None, offset: int = 2,
                    pressure_channel: int = 1, gamma: float = 1.4,
                    p_inf: Optional[float] = None
                    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, int]]:
    """
    Extract upper/lower near-wall Cp lines from transformed-grid rows/columns.

    This is an approximation until the preprocessing pipeline provides a true
    wall mask or surface index.
    """
    if axis not in ("row", "col"):
        raise ValueError(f"axis must be 'row' or 'col', got {axis!r}")

    pressure = flow[pressure_channel]
    cp = compute_pressure_coefficient(pressure, ma=ma, gamma=gamma, p_inf=p_inf)
    h, w = cp.shape

    if axis == "row":
        upper_idx, lower_idx = _line_indices(h, index, offset)
        x = np.linspace(0.0, 1.0, w)
        lines = {
            "upper": cp[upper_idx, :],
            "lower": cp[lower_idx, :],
        }
    else:
        upper_idx, lower_idx = _line_indices(w, index, offset)
        x = np.linspace(0.0, 1.0, h)
        lines = {
            "upper": cp[:, upper_idx],
            "lower": cp[:, lower_idx],
        }

    return x, lines, {"upper": upper_idx, "lower": lower_idx}


def summarize_cp_error(sample_name: str, meta: Dict[str, object],
                       cp_true: Dict[str, np.ndarray],
                       cp_pred: Dict[str, np.ndarray],
                       cp_indices: Dict[str, int]) -> list[Dict[str, object]]:
    rows = []
    for side in ("upper", "lower"):
        diff = cp_pred[side] - cp_true[side]
        rows.append({
            "sample": sample_name,
            **meta,
            "side": side,
            "line_index": cp_indices[side],
            "mae": float(np.mean(np.abs(diff))),
            "mse": float(np.mean(diff ** 2)),
            "max_abs": float(np.max(np.abs(diff))),
            "true_min": float(np.min(cp_true[side])),
            "true_max": float(np.max(cp_true[side])),
            "pred_min": float(np.min(cp_pred[side])),
            "pred_max": float(np.max(cp_pred[side])),
        })
    return rows


@dataclass(frozen=True)
class SurfaceSamplingContract:
    """Shared near-wall Cp sampling rule for numpy evaluation and torch loss."""

    axis: str = "row"
    index: Optional[int] = None
    offset: int = 2
    gamma: float = 1.4
    p_inf: Optional[float] = None
    focus_weight: float = 1.0
    focus_leading_fraction: float = 0.08
    focus_mid_center: float = 0.50
    focus_mid_width: float = 0.08
    focus_trailing_fraction: float = 0.08
    negative_aoa_lower_weight: float = 1.0

    def __post_init__(self):
        if self.axis not in ("row", "col"):
            raise ValueError(f"axis must be 'row' or 'col', got {self.axis!r}")

    @property
    def effective_p_inf(self) -> float:
        if self.p_inf is None:
            return 1.0 / self.gamma
        return float(self.p_inf)

    def sample_cp_lines(self, flow: np.ndarray, ma: float,
                        pressure_channel: int = 1
                        ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, int]]:
        return sample_cp_lines(
            flow, ma=ma, axis=self.axis, index=self.index, offset=self.offset,
            pressure_channel=pressure_channel, gamma=self.gamma, p_inf=self.p_inf,
        )

    def _torch_pressure_lines(self, pressure: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = pressure.shape
        if self.axis == "row":
            upper_idx, lower_idx = _line_indices(h, self.index, self.offset)
            return pressure[:, :, upper_idx, :], pressure[:, :, lower_idx, :]

        upper_idx, lower_idx = _line_indices(w, self.index, self.offset)
        return pressure[:, :, :, upper_idx], pressure[:, :, :, lower_idx]

    def _torch_position_weights(self, length: int, dtype, device) -> torch.Tensor:
        weights = torch.ones(length, dtype=dtype, device=device)
        if self.focus_weight <= 1.0:
            return weights.view(1, -1)

        x = torch.linspace(0.0, 1.0, length, dtype=dtype, device=device)
        focus = (
            (x <= self.focus_leading_fraction) |
            (torch.abs(x - self.focus_mid_center) <= self.focus_mid_width) |
            (x >= 1.0 - self.focus_trailing_fraction)
        )
        weights = torch.where(focus, weights * self.focus_weight, weights)
        return weights.view(1, -1)

    @staticmethod
    def _weighted_line_l1(pred: torch.Tensor, target: torch.Tensor,
                          weights: torch.Tensor) -> torch.Tensor:
        weights = weights.expand_as(pred)
        return (torch.abs(pred - target) * weights).sum() / weights.sum()

    def torch_cp_loss(self, pred: torch.Tensor, target: torch.Tensor,
                      inputs: Optional[torch.Tensor],
                      mach_range: torch.Tensor, aoa_range: torch.Tensor,
                      pressure_range: torch.Tensor) -> torch.Tensor:
        if inputs is None:
            raise ValueError("Cp loss is enabled but batch inputs were not provided.")

        mach_min, mach_max = mach_range.to(dtype=pred.dtype, device=pred.device)
        aoa_min, aoa_max = aoa_range.to(dtype=pred.dtype, device=pred.device)
        p_min, p_max = pressure_range.to(dtype=pred.dtype, device=pred.device)
        ma = inputs[:, 0, 0, 0] * (mach_max - mach_min) + mach_min
        aoa = inputs[:, 1, 0, 0] * (aoa_max - aoa_min) + aoa_min
        q_inf = 0.5 * self.gamma * ma.clamp_min(1e-6).pow(2)
        q_inf = q_inf.view(-1, 1)

        pred_p = pred[:, 1:2] * (p_max - p_min) + p_min
        target_p = target[:, 1:2] * (p_max - p_min) + p_min
        pred_upper, pred_lower = self._torch_pressure_lines(pred_p)
        target_upper, target_lower = self._torch_pressure_lines(target_p)

        p_inf = pred.new_tensor(self.effective_p_inf)
        pred_cp_upper = (pred_upper.squeeze(1) - p_inf) / q_inf
        pred_cp_lower = (pred_lower.squeeze(1) - p_inf) / q_inf
        target_cp_upper = (target_upper.squeeze(1) - p_inf) / q_inf
        target_cp_lower = (target_lower.squeeze(1) - p_inf) / q_inf

        position_weights = self._torch_position_weights(
            pred_cp_upper.shape[1], pred.dtype, pred.device
        )
        lower_weights = position_weights
        if self.negative_aoa_lower_weight > 1.0:
            side_multiplier = torch.where(
                aoa < 0,
                torch.full_like(aoa, self.negative_aoa_lower_weight),
                torch.ones_like(aoa),
            ).view(-1, 1)
            lower_weights = lower_weights * side_multiplier

        return 0.5 * (
            self._weighted_line_l1(pred_cp_upper, target_cp_upper, position_weights) +
            self._weighted_line_l1(pred_cp_lower, target_cp_lower, lower_weights)
        )
