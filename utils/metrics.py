"""
Evaluation metrics for flow field prediction.
"""
import numpy as np
import torch
from typing import Dict, List, Optional

CHANNEL_NAMES = ("rho", "p", "u", "v")


def _channel_axes(arr: np.ndarray) -> tuple:
    if arr.ndim == 3:
        return (1, 2)
    if arr.ndim == 4:
        return (0, 2, 3)
    raise ValueError(f"Expected (C,H,W) or (N,C,H,W), got shape {arr.shape}")


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MSE between true and predicted flow fields."""
    return float(np.mean((y_true - y_pred) ** 2))


def relative_l2_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Per-sample relative L2 error.

    Args:
        y_true: (N, C, H, W)
        y_pred: (N, C, H, W)
    Returns:
        (N,) array of relative L2 errors.
    """
    diff = y_true - y_pred
    norm_diff = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
    norm_true = np.linalg.norm(y_true.reshape(y_true.shape[0], -1), axis=1)
    return norm_diff / (norm_true + 1e-12)


def compute_error_map(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the normalized error map e_map as defined in the paper.

    e_map(n) = (|F(n) - F_hat(n)| - min(|F - F_hat|))
               / (max(|F - F_hat|) - min(|F - F_hat|))

    Args:
        y_true: (C, H, W)
        y_pred: (C, H, W)

    Returns:
        (C, H, W) error maps normalized to [0, 1]
    """
    abs_diff = np.abs(y_true - y_pred)                     # (C, H, W)
    flat = abs_diff.reshape(abs_diff.shape[0], -1)        # (C, H*W)
    min_vals = flat.min(axis=1, keepdims=True)             # (C, 1)
    max_vals = flat.max(axis=1, keepdims=True)             # (C, 1)
    span = max_vals - min_vals
    span = np.where(span < 1e-12, 1.0, span)
    e_map = (flat - min_vals) / span                       # (C, H*W)
    return e_map.reshape(abs_diff.shape)


def per_channel_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute paper-style MSE for each flow channel."""
    axes = _channel_axes(y_true)
    return np.mean((y_true - y_pred) ** 2, axis=axes)


def per_channel_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute mean absolute error for each flow channel."""
    axes = _channel_axes(y_true)
    return np.mean(np.abs(y_true - y_pred), axis=axes)


def per_channel_relative_l2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute relative L2 error for each channel of one sample."""
    if y_true.ndim == 3:
        diff = (y_true - y_pred).reshape(y_true.shape[0], -1)
        ref = y_true.reshape(y_true.shape[0], -1)
    elif y_true.ndim == 4:
        diff = np.moveaxis(y_true - y_pred, 1, 0).reshape(y_true.shape[1], -1)
        ref = np.moveaxis(y_true, 1, 0).reshape(y_true.shape[1], -1)
    else:
        raise ValueError(f"Expected (C,H,W) or (N,C,H,W), got shape {y_true.shape}")
    return np.linalg.norm(diff, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-12)


def summarize_flow_error(y_true: np.ndarray, y_pred: np.ndarray,
                         channel_names=CHANNEL_NAMES) -> List[Dict[str, object]]:
    """
    Summarize one predicted flow field with per-channel error metrics.

    Args:
        y_true: (4, H, W) reference CFD field
        y_pred: (4, H, W) predicted field

    Returns:
        List of dictionaries suitable for CSV/Markdown reporting.
    """
    mse = per_channel_mse(y_true, y_pred)
    mae = per_channel_mae(y_true, y_pred)
    rel_l2 = per_channel_relative_l2(y_true, y_pred)
    max_abs = np.max(np.abs(y_true - y_pred), axis=(1, 2))

    rows = []
    for i, name in enumerate(channel_names):
        rows.append({
            "channel": name,
            "mse": float(mse[i]),
            "mae": float(mae[i]),
            "relative_l2": float(rel_l2[i]),
            "max_abs": float(max_abs[i]),
            "true_min": float(np.min(y_true[i])),
            "true_max": float(np.max(y_true[i])),
            "pred_min": float(np.min(y_pred[i])),
            "pred_max": float(np.max(y_pred[i])),
        })
    return rows


def compute_pressure_coefficient(p: np.ndarray, ma: float,
                                  gamma: float = 1.4,
                                  p_inf: Optional[float] = None) -> np.ndarray:
    """
    Compute pressure coefficient Cp.

    Cp = (p - p∞) / (0.5 * ρ∞ * U∞²)
    For non-dimensionalized data where freestream values are 1:
    Cp = (p - 1) / (0.5 * gamma * Ma²)

    Args:
        p:    pressure field (H, W)
        ma:   freestream Mach number
        gamma: specific heat ratio (default 1.4 for air)
        p_inf: optional freestream pressure override
    """
    if p_inf is None:
        p_inf = 1.0 / gamma  # non-dimensional far-field pressure
    q_inf = 0.5 * gamma * ma ** 2
    cp = (p - p_inf) / q_inf
    return cp


@torch.no_grad()
def evaluate_model(model, loader, device="cpu") -> dict:
    """
    Full evaluation on a dataloader.

    Returns dict with average loss and per-channel MSE.
    """
    model.eval()
    criterion = torch.nn.L1Loss()
    total_loss = 0.0
    n_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return {"loss": total_loss / n_samples}
