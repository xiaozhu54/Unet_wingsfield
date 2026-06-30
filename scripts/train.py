"""
Training script for UNet airfoil flow field prediction.

Usage:
    python scripts/train.py --data-dir train_data_2822
    python scripts/train.py --data-dir train_data_2822 --save-dir results_gen0
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

from config.train_config import TrainConfig
from data.dataset import create_dataloaders
from models.unet import build_unet
from training.trainer import Trainer, build_optimizer
from training.scheduler import build_scheduler
from utils.visualization import plot_training_history


def main():
    parser = argparse.ArgumentParser(description="Train UNet airfoil flow field predictor")
    parser.add_argument("--data-dir", type=str, default="train_data_2822",
                        help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--base-filters", type=int, default=None)
    parser.add_argument("--gradient-weight", type=float, default=None,
                        help="Weight for L1 spatial-gradient loss")
    parser.add_argument("--save-dir", type=str, default="results_gen0",
                        help="Directory to save models and outputs")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # ── Config ──
    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.base_filters is not None:
        cfg.base_filters = args.base_filters
    if args.gradient_weight is not None:
        cfg.gradient_loss_weight = args.gradient_weight
    cfg.save_dir = args.save_dir
    if args.device != "auto":
        cfg.device = args.device

    device = cfg.device_str
    print(f"Device: {device}")

    # ── Data ──
    train_loader, val_loader, _, normalizer = create_dataloaders(args.data_dir, cfg)
    print(f"Train samples: {len(train_loader.dataset)}, "
          f"Val samples: {len(val_loader.dataset)}")

    os.makedirs(cfg.save_dir, exist_ok=True)
    normalizer.save(os.path.join(cfg.save_dir, "normalizer.npz"))

    # ── Model ──
    model = build_unet(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── Optimizer & Scheduler ──
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # ── Train ──
    trainer = Trainer(model, optimizer, scheduler, cfg, device)
    history = trainer.run(train_loader, val_loader)

    # ── Plot ──
    plot_path = os.path.join(cfg.save_dir, "training_history.png")
    plot_training_history(history, save_path=plot_path)
    print(f"Training history saved to {plot_path}")


if __name__ == "__main__":
    main()
