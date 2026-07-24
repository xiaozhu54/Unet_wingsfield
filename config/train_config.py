"""
Training configuration for UNet airfoil flow field prediction.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TrainConfig:
    # Data
    data_dir_2822: str = "train_data_2822"
    data_dir_hybrid: str = "train_data_hybridwings"
    grid_size: Tuple[int, int] = (128, 128)
    input_channels: int = 12       # 3 IC + 9 grid metrics
    output_channels: int = 4       # rho, p, u, v

    # channels: [0-2]=IC, [3-11]=grid metrics, [12-15]=flow field
    ic_channels: List[int] = field(default_factory=lambda: [0, 1, 2])
    grid_channels: List[int] = field(default_factory=lambda: list(range(3, 12)))
    flow_channels: List[int] = field(default_factory=lambda: [12, 13, 14, 15])

    # Dataset split
    val_ratio: float = 0.15
    test_ratio: float = 0.05
    random_seed: int = 42

    # Training
    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler_step: int = 50
    lr_scheduler_gamma: float = 0.5
    gradient_loss_weight: float = 0.1  # Loss = L1(flow) + w * L1(grad(flow))
    flow_channel_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    cp_loss_weight: float = 0.0       # Extra near-wall Cp L1 loss. 0 disables it.
    cp_surface_axis: str = "row"      # row or col in transformed grid
    cp_surface_index: Optional[int] = None
    cp_surface_offset: int = 2
    cp_gamma: float = 1.4
    cp_p_inf: Optional[float] = None
    cp_focus_weight: float = 1.0
    cp_focus_leading_fraction: float = 0.08
    cp_focus_mid_center: float = 0.50
    cp_focus_mid_width: float = 0.08
    cp_focus_trailing_fraction: float = 0.08
    cp_negative_aoa_lower_weight: float = 1.0
    mach_min: Optional[float] = None  # filled from train normalizer for Cp loss
    mach_max: Optional[float] = None
    aoa_min: Optional[float] = None
    aoa_max: Optional[float] = None
    pressure_min: Optional[float] = None
    pressure_max: Optional[float] = None

    # UNet architecture
    base_filters: int = 32
    n_down: int = 6
    kernel_size: int = 3
    activation: str = "leaky_relu"
    use_batch_norm: bool = True
    dropout: float = 0.0
    upsample_mode: str = "transpose"  # transpose or bilinear
    attention_gates: bool = False      # attention-gated skip connections

    # DataLoader
    # None auto-tunes by device. Override from scripts with --num-workers when needed.
    num_workers: Optional[int] = None
    prefetch_factor: int = 2
    persistent_workers: bool = True
    cache_data: bool = False        # cache normalized tensors in RAM after startup

    # CUDA / mixed precision
    amp_enabled: bool = True       # mixed precision on CUDA/MPS
    amp_dtype: str = "float16"      # CUDA autocast dtype: float16 or bfloat16
    channels_last: bool = True     # faster Conv2d memory layout on NVIDIA GPUs
    cudnn_benchmark: bool = True   # optimize fixed 128x128 convolution kernels
    matmul_precision: str = "high" # enables TF32 matmul on RTX GPUs when supported

    # Checkpoint / Log
    save_dir: str = "results/models"
    log_interval: int = 10
    save_interval: int = 10

    # Device
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
        """Pinned host memory speeds CPU-to-CUDA transfers."""
        return self.device_str == "cuda"

    def effective_num_workers(self) -> int:
        """Capped number of workers based on CPU cores and device."""
        import os
        max_cpus = os.cpu_count() or 4
        if self.num_workers is not None:
            return min(max(0, self.num_workers), max_cpus)
        if self.device_str == "cuda":
            return min(8, max(2, max_cpus // 2))
        if self.device_str == "mps":
            return min(4, max(1, max_cpus // 2))
        return 0

    @property
    def use_channels_last(self) -> bool:
        """Use channels_last only where Conv2d kernels commonly benefit."""
        return self.channels_last and self.device_str == "cuda"

    def __post_init__(self):
        assert len(self.flow_channels) == self.output_channels
        assert len(self.flow_channel_weights) == self.output_channels
        assert self.upsample_mode in ("transpose", "bilinear")
        assert self.cp_surface_axis in ("row", "col")
        assert self.cp_surface_offset >= 1
        assert self.cp_focus_weight >= 1.0
        assert self.cp_negative_aoa_lower_weight >= 1.0


@dataclass
class InferenceConfig:
    checkpoint_path: str = "results/models/best_model.pth"
    output_dir: str = "results"
    grid_size: Tuple[int, int] = (128, 128)


train_cfg = TrainConfig()
infer_cfg = InferenceConfig()
