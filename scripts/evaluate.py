"""
Evaluation script for trained UNet models.

Formal metrics are computed on a split from split_manifest.json. By default,
the script evaluates the held-out test split from the checkpoint directory.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config.train_config import TrainConfig
from data.dataset import (
    AirfoilDataset,
    build_split_manifest,
    load_split_manifest,
    split_indices,
    validate_split_manifest,
)
from data.preprocess import Normalizer
from models.unet import build_unet
from utils.metrics import (
    CHANNEL_NAMES,
    per_channel_mae,
    per_channel_mse,
    per_channel_relative_l2,
)
from utils.visualization import plot_flow_field_comparison


@torch.no_grad()
def visualize_predictions(model, loader, device, save_dir="results"):
    """Generate normalized flow-field comparison plots for a few samples."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)

        for i in range(x.size(0)):
            true_np = y[i].cpu().numpy()
            pred_np = y_pred[i].cpu().numpy()
            save_path = os.path.join(save_dir, f"pred_sample_{count:03d}.png")
            plot_flow_field_comparison(
                true_np,
                pred_np,
                title=f"Sample {count}",
                save_path=save_path,
            )
            count += 1
            if count >= 6:
                break
        if count >= 6:
            break

    print(f"Saved {count} prediction comparison plots to {save_dir}/")


def _denormalize_flow(flow: np.ndarray, normalizer: Normalizer, cfg: TrainConfig) -> np.ndarray:
    dummy = np.zeros((flow.shape[0], 16, *flow.shape[2:]), dtype=np.float32)
    dummy[:, cfg.flow_channels] = flow
    denormed = normalizer.denormalize(dummy, cfg.flow_channels)
    return denormed[:, cfg.flow_channels]


def _channel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    mae = per_channel_mae(y_true, y_pred)
    mse = per_channel_mse(y_true, y_pred)
    rel_l2 = per_channel_relative_l2(y_true, y_pred)
    return [
        {
            "channel": name,
            "mae": float(mae[i]),
            "mse": float(mse[i]),
            "relative_l2": float(rel_l2[i]),
        }
        for i, name in enumerate(CHANNEL_NAMES)
    ]


@torch.no_grad()
def evaluate_flow_metrics(model, loader, normalizer, device, cfg):
    model.eval()
    criterion = torch.nn.L1Loss()
    total_loss = 0.0
    n_samples = 0
    y_true_batches = []
    y_pred_batches = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
        y_true_batches.append(y.cpu().numpy())
        y_pred_batches.append(y_pred.cpu().numpy())

    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)
    y_true_phys = _denormalize_flow(y_true, normalizer, cfg)
    y_pred_phys = _denormalize_flow(y_pred, normalizer, cfg)

    return {
        "sample_count": n_samples,
        "normalized_l1": total_loss / n_samples,
        "normalized": _channel_metrics(y_true, y_pred),
        "physical": _channel_metrics(y_true_phys, y_pred_phys),
    }


def _write_metrics(metrics: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    csv_path = os.path.join(output_dir, "evaluation_channel_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["scale", "channel", "mae", "mse", "relative_l2"]
        )
        writer.writeheader()
        for scale in ("normalized", "physical"):
            for row in metrics[scale]:
                writer.writerow({"scale": scale, **row})

    print(f"Saved metrics to {json_path} and {csv_path}")


def _print_metrics(metrics: dict) -> None:
    print(f"Test samples: {metrics['sample_count']}")
    print(f"Test normalized L1: {metrics['normalized_l1']:.7f}")
    for scale in ("normalized", "physical"):
        print(f"\nPer-channel metrics ({scale}):")
        print(f"{'channel':<8} {'mae':>12} {'mse':>12} {'relative_l2':>14}")
        for row in metrics[scale]:
            print(
                f"{row['channel']:<8} {row['mae']:12.6e} "
                f"{row['mse']:12.6e} {row['relative_l2']:14.6e}"
            )


def _load_or_build_manifest(args, cfg):
    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = os.path.join(os.path.dirname(args.checkpoint), "split_manifest.json")

    if os.path.exists(manifest_path):
        print(f"Loaded split manifest from {manifest_path}")
        return load_split_manifest(manifest_path)

    print(
        f"WARNING: split manifest not found at {manifest_path}; "
        "building a temporary deterministic manifest from current config."
    )
    return build_split_manifest(args.data_dir, cfg)


def _checkpoint_state(checkpoint_path: str) -> dict:
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" in state:
        return state["model_state"]
    return state


def _infer_upsample_mode(state: dict) -> str:
    if any(".up.1.weight" in key for key in state):
        return "bilinear"
    return "transpose"


def _infer_attention_gates(state: dict) -> bool:
    return any(".attention." in key for key in state)


def main():
    parser = argparse.ArgumentParser(description="Evaluate UNet flow field predictor")
    parser.add_argument("--data-dir", type=str, default="train_data_2822",
                        help="Path to data directory")
    parser.add_argument("--checkpoint", type=str,
                        default="results/models/best_model.pth")
    parser.add_argument("--norm-path", type=str,
                        default="results/models/normalizer.npz")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to split_manifest.json; defaults to checkpoint directory")
    parser.add_argument("--split", choices=("test", "val", "train"), default="test",
                        help="Split from the manifest to evaluate")
    parser.add_argument("--upsample-mode", choices=("transpose", "bilinear"), default=None,
                        help="Decoder upsampling method; inferred from checkpoint if omitted")
    parser.add_argument("--attention-gates", action="store_true",
                        help="Enable Attention UNet skip gates; inferred from checkpoint if omitted")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for metrics and prediction plots")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = TrainConfig()
    device = cfg.device_str if args.device == "auto" else args.device
    print(f"Device: {device}")
    model_state = _checkpoint_state(args.checkpoint)
    cfg.upsample_mode = args.upsample_mode or _infer_upsample_mode(model_state)
    cfg.attention_gates = args.attention_gates or _infer_attention_gates(model_state)
    print(f"Upsample mode: {cfg.upsample_mode}")
    print(f"Attention gates: {cfg.attention_gates}")

    manifest = _load_or_build_manifest(args, cfg)
    normalizer = Normalizer.load(args.norm_path)
    dataset = AirfoilDataset(args.data_dir, normalizer=normalizer, config=cfg)
    validate_split_manifest(dataset, manifest)

    indices = split_indices(manifest, args.split)
    if not indices:
        raise ValueError(f"Split '{args.split}' is empty")
    loader = DataLoader(
        Subset(dataset, indices), batch_size=cfg.batch_size, shuffle=False
    )
    print(f"Evaluating split: {args.split} ({len(indices)} samples)")

    model = build_unet(cfg)
    model.load_state_dict(model_state)
    model = model.to(device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    metrics = evaluate_flow_metrics(model, loader, normalizer, device, cfg)
    _print_metrics(metrics)
    _write_metrics(metrics, args.output_dir)
    visualize_predictions(model, loader, device, save_dir=args.output_dir)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
