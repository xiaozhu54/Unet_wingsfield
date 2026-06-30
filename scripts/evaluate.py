"""
Evaluation script for trained UNet model.

Usage:
    python scripts/evaluate.py --data-dir train_data_2822
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from config.train_config import TrainConfig
from data.dataset import AirfoilDataset, create_dataloaders
from data.preprocess import Normalizer
from models.unet import build_unet
from utils.metrics import evaluate_model
from utils.visualization import (
    plot_training_history, plot_flow_field_comparison
)


@torch.no_grad()
def visualize_predictions(model, loader, normalizer, device, cfg, save_dir="results"):
    """Generate flow field comparison plots for a few test samples."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)

        for i in range(x.size(0)):
            # Move to CPU and convert
            true_np = y[i].cpu().numpy()
            pred_np = y_pred[i].cpu().numpy()

            save_path = os.path.join(save_dir, f"pred_sample_{count:03d}.png")
            plot_flow_field_comparison(
                true_np, pred_np,
                title=f"Sample {count}",
                save_path=save_path,
            )
            count += 1
            if count >= 6:  # show first 6 samples
                break
        if count >= 6:
            break

    print(f"Saved {count} prediction comparison plots to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate UNet flow field predictor")
    parser.add_argument("--data-dir", type=str, default="train_data_2822",
                        help="Path to test data directory")
    parser.add_argument("--checkpoint", type=str,
                        default="results/models/best_model.pth")
    parser.add_argument("--norm-path", type=str,
                        default="results/models/normalizer.npz")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = TrainConfig()
    device = cfg.device_str if args.device == "auto" else args.device
    print(f"Device: {device}")

    # ── Load normalizer & create dataloader ──
    normalizer = Normalizer.load(args.norm_path)
    dataset = AirfoilDataset(args.data_dir, normalizer=normalizer, config=cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    # ── Load model ──
    model = build_unet(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model = model.to(device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # ── Evaluate ──
    results = evaluate_model(model, loader, device)
    print(f"Evaluation results: {results}")

    # ── Visualize ──
    visualize_predictions(model, loader, normalizer, device, cfg)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
