"""
Training loop for the UNet flow field predictor, with CUDA/MPS AMP support.
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

from utils.surface_sampling import SurfaceSamplingContract

try:
    import psutil
except ImportError:  # pragma: no cover - optional local dependency
    psutil = None


class FlowGradientLoss(nn.Module):
    """Weighted flow/gradient loss with an optional near-wall Cp objective."""

    def __init__(self, gradient_weight: float = 0.1,
                 channel_weights: Optional[list[float]] = None,
                 cp_loss_weight: float = 0.0,
                 cp_surface_axis: str = "row",
                 cp_surface_index: Optional[int] = None,
                 cp_surface_offset: int = 2,
                 cp_gamma: float = 1.4,
                 cp_p_inf: Optional[float] = None,
                 cp_focus_weight: float = 1.0,
                 cp_focus_leading_fraction: float = 0.08,
                 cp_focus_mid_center: float = 0.50,
                 cp_focus_mid_width: float = 0.08,
                 cp_focus_trailing_fraction: float = 0.08,
                 cp_negative_aoa_lower_weight: float = 1.0,
                 mach_min: Optional[float] = None,
                 mach_max: Optional[float] = None,
                 aoa_min: Optional[float] = None,
                 aoa_max: Optional[float] = None,
                 pressure_min: Optional[float] = None,
                 pressure_max: Optional[float] = None):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.cp_loss_weight = float(cp_loss_weight)
        self.surface_contract = SurfaceSamplingContract(
            axis=cp_surface_axis,
            index=cp_surface_index,
            offset=cp_surface_offset,
            gamma=cp_gamma,
            p_inf=cp_p_inf,
            focus_weight=cp_focus_weight,
            focus_leading_fraction=cp_focus_leading_fraction,
            focus_mid_center=cp_focus_mid_center,
            focus_mid_width=cp_focus_mid_width,
            focus_trailing_fraction=cp_focus_trailing_fraction,
            negative_aoa_lower_weight=cp_negative_aoa_lower_weight,
        )
        weights = channel_weights if channel_weights is not None else [1.0, 1.0, 1.0, 1.0]
        self.register_buffer(
            "channel_weights",
            torch.tensor(weights, dtype=torch.float32).view(1, -1, 1, 1),
        )
        self.register_buffer(
            "mach_range",
            torch.tensor([
                float(0.0 if mach_min is None else mach_min),
                float(1.0 if mach_max is None else mach_max),
            ], dtype=torch.float32),
        )
        self.register_buffer(
            "pressure_range",
            torch.tensor([
                float(0.0 if pressure_min is None else pressure_min),
                float(1.0 if pressure_max is None else pressure_max),
            ], dtype=torch.float32),
        )
        self.register_buffer(
            "aoa_range",
            torch.tensor([
                float(0.0 if aoa_min is None else aoa_min),
                float(1.0 if aoa_max is None else aoa_max),
            ], dtype=torch.float32),
        )

    def _weighted_l1(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = self.channel_weights.to(dtype=pred.dtype, device=pred.device)
        return (torch.abs(pred - target) * weights).mean()

    @staticmethod
    def _gradients(field: torch.Tensor):
        dx = field[..., :, 1:] - field[..., :, :-1]
        dy = field[..., 1:, :] - field[..., :-1, :]
        return dx, dy

    def _cp_loss(self, pred: torch.Tensor, target: torch.Tensor,
                 inputs: Optional[torch.Tensor]) -> torch.Tensor:
        if self.cp_loss_weight <= 0:
            return pred.new_tensor(0.0)
        return self.surface_contract.torch_cp_loss(
            pred, target, inputs,
            mach_range=self.mach_range,
            aoa_range=self.aoa_range,
            pressure_range=self.pressure_range,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        flow_loss = self._weighted_l1(pred, target)
        pred_dx, pred_dy = self._gradients(pred)
        target_dx, target_dy = self._gradients(target)
        grad_loss = 0.5 * (
            self._weighted_l1(pred_dx, target_dx) +
            self._weighted_l1(pred_dy, target_dy)
        )
        cp_loss = self._cp_loss(pred, target, inputs)
        return flow_loss + self.gradient_weight * grad_loss + self.cp_loss_weight * cp_loss


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def _resolve_amp_dtype(dtype_name: str):
    name = (dtype_name or "float16").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported amp_dtype: {dtype_name}")


def _make_grad_scaler(device: str, enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler(device=device)
    except TypeError:  # Older PyTorch compatibility.
        return torch.cuda.amp.GradScaler(enabled=device == "cuda")


def _configure_accelerator(config, device: str):
    if device != "cuda":
        return
    if getattr(config, "cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    precision = getattr(config, "matmul_precision", "high")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(precision)


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
        self.config = config
        self.device = device
        _configure_accelerator(config, device)

        self.model = model.to(device)
        self.channels_last = bool(getattr(config, "use_channels_last", False))
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = FlowGradientLoss(
            gradient_weight=getattr(config, "gradient_loss_weight", 0.1),
            channel_weights=getattr(config, "flow_channel_weights", None),
            cp_loss_weight=getattr(config, "cp_loss_weight", 0.0),
            cp_surface_axis=getattr(config, "cp_surface_axis", "row"),
            cp_surface_index=getattr(config, "cp_surface_index", None),
            cp_surface_offset=getattr(config, "cp_surface_offset", 2),
            cp_gamma=getattr(config, "cp_gamma", 1.4),
            cp_p_inf=getattr(config, "cp_p_inf", None),
            cp_focus_weight=getattr(config, "cp_focus_weight", 1.0),
            cp_focus_leading_fraction=getattr(config, "cp_focus_leading_fraction", 0.08),
            cp_focus_mid_center=getattr(config, "cp_focus_mid_center", 0.50),
            cp_focus_mid_width=getattr(config, "cp_focus_mid_width", 0.08),
            cp_focus_trailing_fraction=getattr(config, "cp_focus_trailing_fraction", 0.08),
            cp_negative_aoa_lower_weight=getattr(config, "cp_negative_aoa_lower_weight", 1.0),
            mach_min=getattr(config, "mach_min", None),
            mach_max=getattr(config, "mach_max", None),
            aoa_min=getattr(config, "aoa_min", None),
            aoa_max=getattr(config, "aoa_max", None),
            pressure_min=getattr(config, "pressure_min", None),
            pressure_max=getattr(config, "pressure_max", None),
        ).to(device)
        self.monitor = ProgressMonitor(device)
        self.non_blocking = device == "cuda" and getattr(config, "pin_memory", False)

        if device == "mps":
            torch.set_num_threads(2)  # keep CPU threads minimal for Apple GPU

        self.use_amp = config.use_amp and device in ("mps", "cuda")
        self.amp_dtype = _resolve_amp_dtype(getattr(config, "amp_dtype", "float16"))
        if self.use_amp and device == "cuda":
            self.autocast = torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype)
            self.scaler = _make_grad_scaler(device, enabled=self.amp_dtype == torch.float16)
        elif self.use_amp and device == "mps":
            self.autocast = torch.amp.autocast(device_type="mps")
            self.scaler = None
        else:
            self.autocast = nullcontext()
            self.scaler = None

        self.best_val_loss = float("inf")
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

    def _prepare_batch(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device, non_blocking=self.non_blocking)
        y = y.to(self.device, non_blocking=self.non_blocking)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
            y = y.contiguous(memory_format=torch.channels_last)
        return x, y

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for x, y in loader:
            x, y = self._prepare_batch(x, y)
            self.optimizer.zero_grad(set_to_none=True)

            with self.autocast:
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y, x)

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
            x, y = self._prepare_batch(x, y)

            with self.autocast:
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y, x)

            total_loss += loss.item() * x.size(0)

        return total_loss / len(loader.dataset)

    def run(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = None) -> dict:
        epochs = epochs or self.config.epochs
        save_dir = self.config.save_dir
        os.makedirs(save_dir, exist_ok=True)

        device_tag = f"[{self.device.upper()}"
        if self.use_amp:
            dtype_name = str(self.amp_dtype).replace("torch.", "")
            device_tag += f" AMP:{dtype_name}"
        if self.channels_last:
            device_tag += " channels_last"
        device_tag += "]"

        print(f"Device: {device_tag}")
        if self.device == "cuda" and torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA runtime: {torch.version.cuda}")
        print(f"DataLoader workers: {train_loader.num_workers}, pin_memory={train_loader.pin_memory}")
        channel_weights = [
            float(x) for x in self.criterion.channel_weights.flatten().detach().cpu().tolist()
        ]
        print(
            f"Loss: weighted L1(flow) + {self.criterion.gradient_weight:g} "
            f"* weighted L1(grad(flow)); channel_weights={channel_weights}"
        )
        if self.criterion.cp_loss_weight > 0:
            contract = self.criterion.surface_contract
            print(
                f"Cp loss: weight={self.criterion.cp_loss_weight:g}, "
                f"axis={contract.axis}, "
                f"index={contract.index}, "
                f"offset={contract.offset}, "
                f"focus_weight={contract.focus_weight:g}, "
                f"neg_aoa_lower_weight={contract.negative_aoa_lower_weight:g}"
            )
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

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_path)

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
