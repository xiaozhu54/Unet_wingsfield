"""
Training configuration for UNet airfoil flow field prediction.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TrainConfig:
    # ── Data ──
    data_dir_2822: str = "train_data_2822"
    data_dir_hybrid: str = "train_data_hybridwings"
    grid_size: Tuple[int, int] = (128, 128)
    input_channels: int = 12       # 3 IC + 9 grid metrics
    output_channels: int = 4       # rho, p, u, v

    # channels: [0-2]=IC, [3-11]=grid metrics, [12-15]=flow field
    ic_channels: List[int] = field(default_factory=lambda: [0, 1, 2])
    grid_channels: List[int] = field(default_factory=lambda: list(range(3, 12)))
    flow_channels: List[int] = field(default_factory=lambda: [12, 13, 14, 15])

    # ── Dataset split ──
    val_ratio: float = 0.15
    test_ratio: float = 0.05
    random_seed: int = 42

    # ── Training ──
    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler_step: int = 50
    lr_scheduler_gamma: float = 0.5
    gradient_loss_weight: float = 0.1  # Loss = L1(flow) + w * L1(grad(flow))

    # ── UNet architecture ──
    base_filters: int = 32
    n_down: int = 6
    kernel_size: int = 3
    activation: str = "leaky_relu"
    use_batch_norm: bool = True
    dropout: float = 0.0

    # ── DataLoader ──
    num_workers: int = 0          # parallel data loading (auto-tuned later)
    prefetch_factor: int = 2

    # ── Mixed Precision ──
    amp_enabled: bool = True      # use float16 on MPS/CUDA

    # ── Checkpoint / Log ──
    save_dir: str = "results/models"
    log_interval: int = 10
    save_interval: int = 10

    # ── Device ──
    device: str = "auto"

    @property
    def device_str(self) -> str:
        """Auto-detect best available device: CUDA > MPS > CPU."""
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def use_amp(self) -> bool:
        """Enable AMP only on GPU devices for real speedup."""
        return self.amp_enabled and self.device_str in ("mps", "cuda")

    @property
    def pin_memory(self) -> bool:
        """Pin memory is NOT supported on MPS, disable for Apple Silicon."""
        return self.device_str == "cuda"

    def effective_num_workers(self) -> int:
        """Capped number of workers based on CPU cores and device."""
        import os
        max_cpus = os.cpu_count() or 4
        if self.device_str == "mps":
            # MPS benefits from some workers; too many causes contention
            return min(self.num_workers, max_cpus // 2)
        return min(self.num_workers, max_cpus)

    def __post_init__(self):
        assert len(self.flow_channels) == self.output_channels


@dataclass
class InferenceConfig:
    checkpoint_path: str = "results/models/best_model.pth"
    output_dir: str = "results"
    grid_size: Tuple[int, int] = (128, 128)


train_cfg = TrainConfig()
infer_cfg = InferenceConfig()
