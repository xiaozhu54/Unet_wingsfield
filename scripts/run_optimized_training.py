#!/usr/bin/env python3
"""
Run the first-priority optimization plan for the RAE2822 UNet model.

Default workflow:
  1. Resume model weights from the current 200-epoch checkpoint.
  2. Disable AMP and continue with FP32 for better numerical stability.
  3. Fine-tune with a lower learning rate.
  4. Save checkpoints, loss curve, and normalizer.
  5. Run paper-style flow analysis on selected samples.

Example:
    python3 scripts/run_optimized_training.py --dry-run

    python3 scripts/run_optimized_training.py \
      --resume results_gen0/epoch_200.pth \
      --norm-path results_gen0/normalizer.npz \
      --epochs 200 \
      --lr 5e-5 \
      --batch-size 16 \
      --save-dir results_optimized_fp32
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.train_config import TrainConfig
from data.dataset import AirfoilDataset
from data.preprocess import Normalizer
from models.unet import build_unet
from scripts.analyze_flow import analyze_samples
from training.trainer import Trainer, build_optimizer
from utils.visualization import plot_training_history


def resolve_device(device_arg: str, allow_cpu_fallback: bool = False) -> str:
    if device_arg == "mps":
        try:
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("torch.backends.mps.is_available() is False")
            torch.ones(1).to("mps")
            return "mps"
        except Exception as exc:
            msg = f"Requested MPS is unavailable in this Python/Torch environment: {exc}"
            if allow_cpu_fallback:
                print(msg)
                print("Falling back to CPU. Progress will report GPU as n/a.")
                return "cpu"
            raise RuntimeError(
                msg + "\n"
                "CPU fallback is disabled to avoid accidentally starting a very slow run. "
                "Use --allow-cpu-fallback only if you intentionally want CPU training."
            )
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.ones(1).to("mps")
            return "mps"
        except Exception:
            pass
    return "cpu"


def build_finetune_loaders(data_dir: str, cfg: TrainConfig,
                           normalizer: Normalizer):
    """Create the same deterministic train/val/test split using a fixed normalizer."""
    full_dataset = AirfoilDataset(data_dir, normalizer=normalizer, config=cfg)

    n_total = len(full_dataset)
    n_test = int(n_total * cfg.test_ratio)
    n_val = int(n_total * cfg.val_ratio)
    n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg.random_seed),
    )

    nw = cfg.effective_num_workers()
    loader_kwargs = {
        "num_workers": nw,
        "pin_memory": cfg.pin_memory,
    }
    if nw > 0:
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        loader_kwargs["persistent_workers"] = cfg.persistent_workers

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs
    ) if n_test > 0 else None

    return train_loader, val_loader, test_loader


def load_checkpoint_weights(model, checkpoint_path: str) -> int:
    """Load model weights and return the source epoch when available."""
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
        return int(state.get("epoch", 0))
    model.load_state_dict(state)
    return 0


def build_cosine_scheduler(optimizer, epochs: int, min_lr: float):
    """Cosine scheduler for the low-LR fine-tuning stage."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=min_lr,
    )


def parse_extent(extent: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
    if extent is None:
        return None
    if len(extent) != 4:
        raise ValueError("--extent requires four values: xmin xmax ymin ymax")
    return tuple(float(v) for v in extent)


def write_plan(path: str, plan: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(plan, f, indent=2)


def print_plan(plan: dict):
    print("\nOptimization execution plan")
    print("-" * 32)
    for key, value in plan.items():
        print(f"{key}: {value}")
    print("-" * 32)


def run_analysis(args, checkpoint_path: str, norm_path: str, device: str):
    """Run paper-style analysis after fine-tuning."""
    analysis_args = SimpleNamespace(
        data_dir=args.data_dir,
        checkpoint=checkpoint_path,
        norm_path=norm_path,
        output_dir=args.analysis_dir,
        device=device,
        sample=args.analysis_sample,
        max_samples=args.analysis_samples,
        start_index=args.analysis_start_index,
        extent=parse_extent(args.extent),
        surface_axis=args.surface_axis,
        surface_index=args.surface_index,
        surface_offset=args.surface_offset,
        gamma=args.gamma,
    )
    analyze_samples(analysis_args)


def main():
    parser = argparse.ArgumentParser(
        description="Resume and fine-tune the UNet model using the recommended optimization order."
    )
    parser.add_argument("--data-dir", default="train_data_2822")
    parser.add_argument("--resume", default="results_gen0/epoch_200.pth",
                        help="Full checkpoint or state-dict to initialize from.")
    parser.add_argument("--norm-path", default="results_gen0/normalizer.npz",
                        help="Normalizer from the original training run.")
    parser.add_argument("--save-dir", default="results_optimized_fp32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu-fallback", action="store_true",
                        help="Allow CPU fallback when --device mps is unavailable.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-weight", type=float, default=0.1,
                        help="Weight for L1 spatial-gradient loss.")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--analysis-dir", default=None)
    parser.add_argument("--analysis-samples", type=int, default=3)
    parser.add_argument("--analysis-start-index", type=int, default=0)
    parser.add_argument("--analysis-sample", action="append",
                        help="Specific .npz sample to analyze. Can be repeated.")
    parser.add_argument("--extent", nargs=4, type=float,
                        default=[-0.25, 1.25, -0.4, 1.1],
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    parser.add_argument("--surface-axis", choices=("row", "col"), default="row")
    parser.add_argument("--surface-index", type=int, default=None)
    parser.add_argument("--surface-offset", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the execution plan without training.")
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.data_dir_2822 = args.data_dir
    cfg.save_dir = args.save_dir
    cfg.device = resolve_device(args.device, allow_cpu_fallback=args.allow_cpu_fallback)
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.gradient_loss_weight = args.gradient_weight
    cfg.save_interval = args.save_interval
    cfg.log_interval = args.log_interval
    cfg.num_workers = args.num_workers
    cfg.amp_enabled = False
    cfg.lr_scheduler_step = args.epochs + 1  # scheduler handled by cosine below

    analysis_dir = args.analysis_dir or os.path.join(args.save_dir, "flow_analysis")
    args.analysis_dir = analysis_dir

    plan = {
        "stage": "FP32 low-LR fine-tune from existing RAE2822 checkpoint",
        "data_dir": args.data_dir,
        "resume": args.resume,
        "normalizer": args.norm_path,
        "save_dir": args.save_dir,
        "device": cfg.device,
        "amp_enabled": cfg.amp_enabled,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "min_lr": args.min_lr,
        "loss": f"L1(flow) + {cfg.gradient_loss_weight:g} * L1(grad(flow))",
        "scheduler": "CosineAnnealingLR",
        "analysis_dir": None if args.skip_analysis else analysis_dir,
    }
    print_plan(plan)

    if args.dry_run:
        print("Dry run complete. No training was started.")
        return

    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
    if not os.path.exists(args.norm_path):
        raise FileNotFoundError(f"Normalizer not found: {args.norm_path}")

    os.makedirs(args.save_dir, exist_ok=True)
    write_plan(os.path.join(args.save_dir, "optimization_plan.json"), plan)

    normalizer = Normalizer.load(args.norm_path)
    normalizer.save(os.path.join(args.save_dir, "normalizer.npz"))
    train_loader, val_loader, _ = build_finetune_loaders(args.data_dir, cfg, normalizer)

    model = build_unet(cfg)
    source_epoch = load_checkpoint_weights(model, args.resume)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded checkpoint: {args.resume} (source epoch={source_epoch})")
    print(f"Model parameters: {n_params:,}")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    optimizer = build_optimizer(model, cfg)
    scheduler = build_cosine_scheduler(optimizer, cfg.epochs, args.min_lr)
    trainer = Trainer(model, optimizer, scheduler, cfg, cfg.device)
    history = trainer.run(train_loader, val_loader, epochs=cfg.epochs)

    plot_path = os.path.join(args.save_dir, "training_history.png")
    plot_training_history(history, save_path=plot_path)
    print(f"Training history saved to {plot_path}")

    best_checkpoint = os.path.join(args.save_dir, "best_model.pth")
    norm_path = os.path.join(args.save_dir, "normalizer.npz")
    if not args.skip_analysis:
        run_analysis(args, best_checkpoint, norm_path, cfg.device)


if __name__ == "__main__":
    main()

