"""
Visualization utilities for flow field prediction results.
"""
import os
import tempfile
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional

from utils.metrics import compute_error_map

# Global font settings for Chinese-friendly labels
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
})

_CHANNEL_LABELS = ["rho", "p", "u", "v"]
_CHANNEL_TITLES = ["Density rho", "Pressure p", "x-velocity u", "y-velocity v"]


def _ensure_parent_dir(save_path: Optional[str]):
    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def _imshow_field(ax, field: np.ndarray, title: str, cmap: str,
                  extent: Optional[tuple] = None, vmin=None, vmax=None):
    im = ax.imshow(field, cmap=cmap, origin="lower", extent=extent,
                   aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    return im


def plot_flow_field_comparison(true: np.ndarray, pred: np.ndarray,
                                 title: str = "",
                                 save_path: Optional[str] = None):
    """
    Plot true vs predicted flow fields side by side with error map.

    Args:
        true:  (4, H, W) ground-truth flow field
        pred:  (4, H, W) predicted flow field
        title: overall title
        save_path: if given, save figure to this path
    """
    n_vars = true.shape[0]
    fig, axes = plt.subplots(n_vars, 3, figsize=(12, 3 * n_vars))

    for i in range(n_vars):
        vmin = min(true[i].min(), pred[i].min())
        vmax = max(true[i].max(), pred[i].max())
        err = np.abs(true[i] - pred[i])

        axes[i, 0].imshow(true[i], cmap="viridis", vmin=vmin, vmax=vmax,
                           origin="lower")
        axes[i, 0].set_title(f"True {_CHANNEL_LABELS[i]}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred[i], cmap="viridis", vmin=vmin, vmax=vmax,
                           origin="lower")
        axes[i, 1].set_title(f"Pred {_CHANNEL_LABELS[i]}")
        axes[i, 1].axis("off")

        im = axes[i, 2].imshow(err, cmap="hot", origin="lower")
        axes[i, 2].set_title(f"Error {_CHANNEL_LABELS[i]}")
        axes[i, 2].axis("off")
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046)

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_pressure_coefficient(x_surf: np.ndarray, cp_true: np.ndarray,
                                cp_pred: np.ndarray, title: str = "",
                                save_path: Optional[str] = None):
    """
    Plot pressure coefficient comparison along the airfoil surface.

    Args:
        x_surf:  surface x-coordinates (normalized, 0-1)
        cp_true: ground-truth Cp values
        cp_pred: predicted Cp values
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_surf, cp_true, "b-", label="CFD (true)", linewidth=1.5)
    ax.plot(x_surf, cp_pred, "r--", label="UNet (pred)", linewidth=1.5)
    ax.set_xlabel("x/c")
    ax.set_ylabel("Cp")
    ax.set_title(title or "Pressure Coefficient")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_paper_flow_comparison(true: np.ndarray, pred: np.ndarray,
                               metrics: Optional[list] = None,
                               title: str = "",
                               save_path: Optional[str] = None,
                               extent: Optional[tuple] = None):
    """
    Plot CFD, UNet, and normalized e_map columns following the paper.

    Args:
        true: (4, H, W) denormalized CFD/reference field
        pred: (4, H, W) denormalized UNet prediction
        metrics: optional rows from summarize_flow_error()
        title: figure title
        save_path: output PNG path
        extent: optional imshow extent, e.g. (-0.25, 1.25, -0.4, 1.1)
    """
    n_vars = true.shape[0]
    error_maps = compute_error_map(true, pred)
    fig, axes = plt.subplots(n_vars, 3, figsize=(11.5, 2.7 * n_vars),
                             constrained_layout=True)

    for i in range(n_vars):
        vmin = min(float(np.min(true[i])), float(np.min(pred[i])))
        vmax = max(float(np.max(true[i])), float(np.max(pred[i])))
        metric_suffix = ""
        if metrics is not None:
            metric_suffix = f"\nMSE={metrics[i]['mse']:.4g}"

        im0 = _imshow_field(
            axes[i, 0], true[i], f"CFD {_CHANNEL_TITLES[i]}",
            "viridis", extent=extent, vmin=vmin, vmax=vmax,
        )
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.02)

        im1 = _imshow_field(
            axes[i, 1], pred[i], f"UNet {_CHANNEL_TITLES[i]}",
            "viridis", extent=extent, vmin=vmin, vmax=vmax,
        )
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.02)

        im2 = _imshow_field(
            axes[i, 2], error_maps[i],
            f"e_map {_CHANNEL_LABELS[i]}{metric_suffix}",
            "Greys", extent=extent, vmin=0.0, vmax=1.0,
        )
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.02)

    if title:
        fig.suptitle(title, fontsize=12)

    if save_path:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_pressure_error_map(true: np.ndarray, pred: np.ndarray,
                            mse: Optional[float] = None,
                            title: str = "",
                            save_path: Optional[str] = None,
                            extent: Optional[tuple] = None,
                            pressure_channel: int = 1):
    """Plot the paper-style normalized pressure error map."""
    error_map = compute_error_map(true, pred)[pressure_channel]

    fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)
    suffix = f"  MSE={mse:.4g}" if mse is not None else ""
    im = _imshow_field(
        ax, error_map, title or f"Pressure e_map{suffix}",
        "Greys", extent=extent, vmin=0.0, vmax=1.0,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    if save_path:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_surface_cp_comparison(x_surf: np.ndarray,
                               cp_true_lines: dict,
                               cp_pred_lines: dict,
                               title: str = "",
                               save_path: Optional[str] = None):
    """
    Plot upper/lower near-wall Cp curves extracted from transformed-grid rows.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    styles = {
        "upper": ("tab:blue", "-"),
        "lower": ("tab:orange", "-"),
    }
    for name, cp_true in cp_true_lines.items():
        color, line_style = styles.get(name, ("tab:gray", "-"))
        ax.plot(x_surf, cp_true, color=color, linestyle=line_style,
                linewidth=1.4, label=f"CFD {name}")
        ax.plot(x_surf, cp_pred_lines[name], color=color, linestyle="--",
                linewidth=1.4, label=f"UNet {name}")

    ax.set_xlabel("x/c")
    ax.set_ylabel("Cp")
    ax.set_title(title or "Surface Pressure Coefficient")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    if save_path:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_training_history(history: dict, save_path: Optional[str] = "results/training_history.png"):
    """Plot training and validation loss curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(epochs, history["train_loss"], label="Train Loss", alpha=0.8)
    ax.semilogy(epochs, history["val_loss"], label="Val Loss", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 Loss")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_prediction_sample(inputs: np.ndarray, true: np.ndarray,
                            pred: np.ndarray, idx: int = 0,
                            save_path: Optional[str] = None):
    """Convenience: plot flow field comparison for a single prediction."""
    plot_flow_field_comparison(
        true[idx], pred[idx],
        title=f"Sample {idx}",
        save_path=save_path,
    )
