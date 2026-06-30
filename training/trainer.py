"""
Training loop for the UNet flow field predictor, with MPS/AMP support.
"""
import os
import time
import json
import math
import subprocess
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

try:
    import psutil
except ImportError:  # pragma: no cover - optional local dependency
    psutil = None


class FlowGradientLoss(nn.Module):
    """L1(flow) + weight * L1(spatial gradients)."""

    def __init__(self, gradient_weight: float = 0.1):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.l1 = nn.L1Loss()

    @staticmethod
    def _gradients(field: torch.Tensor):
        dx = field[..., :, 1:] - field[..., :, :-1]
        dy = field[..., 1:, :] - field[..., :-1, :]
        return dx, dy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        flow_loss = self.l1(pred, target)
        pred_dx, pred_dy = self._gradients(pred)
        target_dx, target_dy = self._gradients(target)
        grad_loss = 0.5 * (
            self.l1(pred_dx, target_dx) + self.l1(pred_dy, target_dy)
        )
        return flow_loss + self.gradient_weight * grad_loss


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


class ProgressMonitor:
    """Lightweight terminal monitor for CPU/GPU occupancy and ETA."""

    def __init__(self, device: str):
        self.device = device
        self._cuda_query_failed = False
        if psutil is not None:
            psutil.cpu_percent(interval=None)

    def cpu_text(self) -> str:
        if psutil is not None:
            return f"{psutil.cpu_percent(interval=None):4.1f}%"
        try:
            load1 = os.getloadavg()[0]
            cores = os.cpu_count() or 1
            return f"{min(100.0, 100.0 * load1 / cores):4.1f}%*"
        except OSError:
            return "n/a"

    def gpu_text(self) -> str:
        if self.device == "cuda" and torch.cuda.is_available():
            util = self._cuda_utilization()
            alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            if util is not None:
                return f"{util:3d}% {alloc_gb:.1f}/{reserved_gb:.1f}GB"
            return f"{alloc_gb:.1f}/{reserved_gb:.1f}GB"
        if self.device == "mps" and hasattr(torch, "mps"):
            try:
                alloc_gb = torch.mps.current_allocated_memory() / (1024 ** 3)
                driver_gb = torch.mps.driver_allocated_memory() / (1024 ** 3)
                return f"MPS {alloc_gb:.1f}/{driver_gb:.1f}GB"
            except Exception:
                return "MPS n/a"
        return "n/a"

    def _cuda_utilization(self) -> Optional[int]:
        if self._cuda_query_failed:
            return None
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1,
            )
            return int(out.strip().splitlines()[0])
        except Exception:
            self._cuda_query_failed = True
            return None

    @staticmethod
    def trend(current: float, previous: Optional[float]) -> str:
        if previous is None or not math.isfinite(previous):
            return "start"
        delta = current - previous
        flat_tol = max(1e-6, abs(previous) * 1e-3)
        if delta < -flat_tol:
            return f"down {abs(delta):.2e}"
        if delta > flat_tol:
            return f"up {delta:.2e}"
        return "flat"

    @staticmethod
    def progress_bar(epoch: int, epochs: int, width: int = 18) -> str:
        filled = int(width * epoch / max(1, epochs))
        return "[" + "#" * filled + "." * (width - filled) + "]"


class Trainer:
    """Encapsulates the train / validate loop with mixed-precision support."""

    def __init__(self, model: nn.Module, optimizer: Optimizer,
                 scheduler: LRScheduler, config,
                 device: str = "cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.criterion = FlowGradientLoss(
            gradient_weight=getattr(config, "gradient_loss_weight", 0.1)
        )
        self.monitor = ProgressMonitor(device)

        # ── MPS thread tuning ──
        if device == "mps":
            torch.set_num_threads(2)  # keep CPU threads minimal for GPU

        # ── Mixed Precision (AMP) ──
        self.use_amp = config.use_amp and device in ("mps", "cuda")
        if self.use_amp:
            self.scaler = torch.amp.GradScaler(device=device)
            self.autocast = torch.amp.autocast(device_type=device)
        else:
            self.scaler = None
            self.autocast = nullcontext()

        self.best_val_loss = float("inf")
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            # AMP forward pass
            with self.autocast:
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)

            # AMP backward + step
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * x.size(0)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            with self.autocast:
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)

            total_loss += loss.item() * x.size(0)

        return total_loss / len(loader.dataset)

    def run(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = None) -> dict:
        epochs = epochs or self.config.epochs
        save_dir = self.config.save_dir
        os.makedirs(save_dir, exist_ok=True)

        device_tag = f"[{self.device.upper()}"
        if self.use_amp:
            device_tag += " AMP"
        device_tag += "]"

        print(f"Device: {device_tag}")
        print(f"Loss: L1(flow) + {self.criterion.gradient_weight:g} * L1(grad(flow))")
        print(f"Training config: {self.config}")
        print(f"Train samples: {len(train_loader.dataset)}, "
              f"Val samples: {len(val_loader.dataset)}")
        print(
            f"{'Prog':<20} {'Epoch':>9} {'Train':>11} {'Val':>11} "
            f"{'Trend':>13} {'LR':>9} {'EpochTime':>10} {'Total':>8} "
            f"{'ETA':>8} {'CPU':>7} {'GPU':>16}"
        )
        print("-" * 132)

        total_t0 = time.perf_counter()
        epoch_times = []
        prev_val_loss = None
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            elapsed = time.perf_counter() - t0
            epoch_times.append(elapsed)
            avg_epoch_time = sum(epoch_times[-10:]) / min(len(epoch_times), 10)
            eta = avg_epoch_time * (epochs - epoch)
            total_elapsed = time.perf_counter() - total_t0
            trend = self.monitor.trend(val_loss, prev_val_loss)
            prev_val_loss = val_loss

            print(
                f"{self.monitor.progress_bar(epoch, epochs):<20} "
                f"{epoch:4d}/{epochs:<4d} {train_loss:11.4e} {val_loss:11.4e} "
                f"{trend:>13} {current_lr:9.2e} {_format_duration(elapsed):>10} "
                f"{_format_duration(total_elapsed):>8} {_format_duration(eta):>8} "
                f"{self.monitor.cpu_text():>7} {self.monitor.gpu_text():>16}",
                flush=True,
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_path)

            # Periodic checkpoint
            if epoch % self.config.save_interval == 0:
                ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "history": self.history,
                }, ckpt_path)

        hist_path = os.path.join(save_dir, "training_history.json")
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.6e}")
        return self.history


def build_optimizer(model: nn.Module, config) -> Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
