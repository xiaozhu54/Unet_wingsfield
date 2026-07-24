"""
Training script for UNet airfoil flow field prediction.

Usage:
    python scripts/train.py --data-dir train_data_2822
    python scripts/train.py --data-dir train_data_2822 --save-dir results_gen0
"""
import argparse
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config.train_config import TrainConfig
from data.dataset import build_split_manifest, create_dataloaders, save_split_manifest
from models.unet import build_unet
from training.scheduler import build_scheduler
from training.trainer import Trainer, build_optimizer
from utils.visualization import plot_training_history


def main():
    parser = argparse.ArgumentParser(description="Train UNet airfoil flow field predictor")
    parser.add_argument("--data-dir", type=str, default="train_data_2822",
                        help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--base-filters", type=int, default=None)
    parser.add_argument("--upsample-mode", choices=("transpose", "bilinear"), default=None,
                        help="Decoder upsampling method")
    parser.add_argument("--attention-gates", action="store_true",
                        help="Enable Attention UNet skip gates")
    parser.add_argument("--gradient-weight", type=float, default=None,
                        help="Weight for L1 spatial-gradient loss")
    parser.add_argument("--channel-weights", type=str, default=None,
                        help="Comma-separated flow-channel weights: rho,p,u,v")
    parser.add_argument("--cp-loss-weight", type=float, default=None,
                        help="Weight for near-wall Cp L1 loss; 0 disables it")
    parser.add_argument("--cp-surface-axis", choices=("row", "col"), default=None,
                        help="Transformed-grid direction used for Cp sampling")
    parser.add_argument("--cp-surface-index", type=int, default=None,
                        help="Center row/column for Cp sampling")
    parser.add_argument("--cp-surface-offset", type=int, default=None,
                        help="Offset from center row/column for upper/lower Cp lines")
    parser.add_argument("--cp-focus-weight", type=float, default=None,
                        help="Weight multiplier for leading/mid/trailing Cp regions")
    parser.add_argument("--cp-negative-aoa-lower-weight", type=float, default=None,
                        help="Extra lower-surface Cp multiplier for negative AoA samples")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader worker count; default auto-tunes by device")
    parser.add_argument("--cache-data", action="store_true",
                        help="Cache normalized input/target tensors in RAM")
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default=None,
                        help="CUDA autocast dtype")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--no-channels-last", action="store_true",
                        help="Disable channels_last memory format on CUDA")
    parser.add_argument("--save-dir", type=str, default="results_gen0",
                        help="Directory to save models and outputs")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Optional model checkpoint used to initialize weights")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.base_filters is not None:
        cfg.base_filters = args.base_filters
    if args.upsample_mode is not None:
        cfg.upsample_mode = args.upsample_mode
    if args.attention_gates:
        cfg.attention_gates = True
    if args.gradient_weight is not None:
        cfg.gradient_loss_weight = args.gradient_weight
    if args.channel_weights is not None:
        weights = [float(x.strip()) for x in args.channel_weights.split(",")]
        if len(weights) != cfg.output_channels:
            raise ValueError(
                f"--channel-weights must contain {cfg.output_channels} values"
            )
        cfg.flow_channel_weights = weights
    if args.cp_loss_weight is not None:
        cfg.cp_loss_weight = args.cp_loss_weight
    if args.cp_surface_axis is not None:
        cfg.cp_surface_axis = args.cp_surface_axis
    if args.cp_surface_index is not None:
        cfg.cp_surface_index = args.cp_surface_index
    if args.cp_surface_offset is not None:
        cfg.cp_surface_offset = args.cp_surface_offset
    if args.cp_focus_weight is not None:
        cfg.cp_focus_weight = args.cp_focus_weight
    if args.cp_negative_aoa_lower_weight is not None:
        cfg.cp_negative_aoa_lower_weight = args.cp_negative_aoa_lower_weight
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.cache_data:
        cfg.cache_data = True
    if args.amp_dtype is not None:
        cfg.amp_dtype = args.amp_dtype
    if args.no_amp:
        cfg.amp_enabled = False
    if args.no_channels_last:
        cfg.channels_last = False
    cfg.save_dir = args.save_dir
    if args.device != "auto":
        cfg.device = args.device

    device = cfg.device_str
    print(f"Device: {device}")

    split_manifest = build_split_manifest(args.data_dir, cfg)
    train_loader, val_loader, test_loader, normalizer = create_dataloaders(
        args.data_dir, cfg, manifest=split_manifest
    )
    cfg.mach_min = float(normalizer.channel_mins[cfg.ic_channels[0]])
    cfg.mach_max = float(normalizer.channel_maxs[cfg.ic_channels[0]])
    cfg.aoa_min = float(normalizer.channel_mins[cfg.ic_channels[1]])
    cfg.aoa_max = float(normalizer.channel_maxs[cfg.ic_channels[1]])
    cfg.pressure_min = float(normalizer.channel_mins[cfg.flow_channels[1]])
    cfg.pressure_max = float(normalizer.channel_maxs[cfg.flow_channels[1]])
    print(f"Train samples: {len(train_loader.dataset)}, "
          f"Val samples: {len(val_loader.dataset)}, "
          f"Test samples: {len(test_loader.dataset) if test_loader else 0}")

    os.makedirs(cfg.save_dir, exist_ok=True)
    manifest_path = os.path.join(cfg.save_dir, "split_manifest.json")
    save_split_manifest(split_manifest, manifest_path)
    print(f"Split manifest saved to {manifest_path}")
    normalizer.save(os.path.join(cfg.save_dir, "normalizer.npz"))
    config_path = os.path.join(cfg.save_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"Training config saved to {config_path}")

    model = build_unet(cfg)
    if args.init_checkpoint is not None:
        state = torch.load(args.init_checkpoint, map_location="cpu")
        model_state = state["model_state"] if "model_state" in state else state
        model.load_state_dict(model_state)
        print(f"Initialized model weights from {args.init_checkpoint}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    trainer = Trainer(model, optimizer, scheduler, cfg, device)
    history = trainer.run(train_loader, val_loader)

    plot_path = os.path.join(cfg.save_dir, "training_history.png")
    plot_training_history(history, save_path=plot_path)
    print(f"Training history saved to {plot_path}")


if __name__ == "__main__":
    main()
