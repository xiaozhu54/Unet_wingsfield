"""
Evaluate near-wall Cp metrics on a manifest split.

This is a Cp-first companion to scripts/evaluate.py. It uses the same
approximate near-wall sampling contract as scripts/analyze_flow.py.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.train_config import TrainConfig
from data.dataset import (
    AirfoilDataset,
    build_split_manifest,
    load_split_manifest,
    split_indices,
    validate_split_manifest,
)
from data.preprocess import Normalizer, parse_filename, split_input_target
from models.unet import build_unet
from utils.surface_sampling import sample_cp_lines, summarize_cp_error


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


def _resolve_device(device_arg: str, cfg: TrainConfig) -> str:
    if device_arg == "auto":
        return cfg.device_str
    return device_arg


def _denormalize_flow(flow_norm: np.ndarray, normalizer: Normalizer,
                      cfg: TrainConfig) -> np.ndarray:
    flow = flow_norm.copy().astype(np.float32)
    for out_idx, channel in enumerate(cfg.flow_channels):
        lo = normalizer.channel_mins[channel]
        hi = normalizer.channel_maxs[channel]
        flow[out_idx] = flow[out_idx] * (hi - lo) + lo
    return flow


def _predict_one(model, raw: np.ndarray, normalizer: Normalizer,
                 cfg: TrainConfig, device: str) -> np.ndarray:
    raw_norm = normalizer.normalize(raw[np.newaxis].astype(np.float32))
    inputs, _ = split_input_target(raw_norm, cfg)
    x = torch.from_numpy(inputs).to(device)
    pred_norm = model(x).detach().cpu().numpy()[0]
    return _denormalize_flow(pred_norm, normalizer, cfg)


def _load_or_build_manifest(args, cfg):
    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = os.path.join(os.path.dirname(args.checkpoint), "split_manifest.json")
    if os.path.exists(manifest_path):
        return load_split_manifest(manifest_path)
    return build_split_manifest(args.data_dir, cfg)


def _write_rows(path: str, rows: list[dict]) -> None:
    fieldnames = [
        "sample", "airfoil", "ma", "aoa", "re", "side", "line_index",
        "mae", "mse", "max_abs", "true_min", "true_max", "pred_min", "pred_max",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary(rows: list[dict]) -> dict:
    maes = np.array([row["mae"] for row in rows], dtype=np.float64)
    mses = np.array([row["mse"] for row in rows], dtype=np.float64)
    max_abs = np.array([row["max_abs"] for row in rows], dtype=np.float64)
    summary = {
        "line_count": int(len(rows)),
        "mean_cp_mae": float(maes.mean()),
        "mean_cp_mse": float(mses.mean()),
        "mean_cp_max_abs": float(max_abs.mean()),
        "worst_cp_mae": float(maes.max()),
        "worst_cp_max_abs": float(max_abs.max()),
    }
    for side in ("upper", "lower"):
        side_rows = [row for row in rows if row["side"] == side]
        side_maes = np.array([row["mae"] for row in side_rows], dtype=np.float64)
        summary[f"{side}_mean_cp_mae"] = float(side_maes.mean())
        summary[f"{side}_worst_cp_mae"] = float(side_maes.max())
    return summary


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate held-out near-wall Cp error")
    parser.add_argument("--data-dir", default="train_data_2822")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm-path", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--output-dir", default="results/cp_evaluation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--surface-axis", choices=("row", "col"), default="row")
    parser.add_argument("--surface-index", type=int, default=None)
    parser.add_argument("--surface-offset", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=1.4)
    args = parser.parse_args()

    cfg = TrainConfig()
    model_state = _checkpoint_state(args.checkpoint)
    cfg.upsample_mode = _infer_upsample_mode(model_state)
    cfg.attention_gates = _infer_attention_gates(model_state)
    device = _resolve_device(args.device, cfg)

    manifest = _load_or_build_manifest(args, cfg)
    normalizer = Normalizer.load(args.norm_path)
    dataset = AirfoilDataset(args.data_dir, normalizer=normalizer, config=cfg)
    validate_split_manifest(dataset, manifest)

    model = build_unet(cfg)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    rows = []
    for idx in split_indices(manifest, args.split):
        sample_name = dataset.files[idx]
        sample_path = os.path.join(args.data_dir, sample_name)
        raw = np.load(sample_path)["a"].astype(np.float32)
        airfoil, ma, aoa, re_val = parse_filename(sample_name)
        true_flow = raw[cfg.flow_channels]
        pred_flow = _predict_one(model, raw, normalizer, cfg, device)
        _, cp_true, cp_indices = sample_cp_lines(
            true_flow, ma=ma, axis=args.surface_axis,
            index=args.surface_index, offset=args.surface_offset,
            gamma=args.gamma,
        )
        _, cp_pred, _ = sample_cp_lines(
            pred_flow, ma=ma, axis=args.surface_axis,
            index=args.surface_index, offset=args.surface_offset,
            gamma=args.gamma,
        )
        rows.extend(summarize_cp_error(
            sample_name,
            {"airfoil": airfoil, "ma": ma, "aoa": aoa, "re": re_val},
            cp_true,
            cp_pred,
            cp_indices,
        ))

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "cp_metrics.csv")
    json_path = os.path.join(args.output_dir, "cp_summary.json")
    summary = _summary(rows)
    summary.update({
        "sample_count": len(split_indices(manifest, args.split)),
        "split": args.split,
        "surface_axis": args.surface_axis,
        "surface_index": args.surface_index,
        "surface_offset": args.surface_offset,
        "checkpoint": args.checkpoint,
    })
    _write_rows(csv_path, rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Samples: {summary['sample_count']}")
    print(f"Mean Cp MAE: {summary['mean_cp_mae']:.6e}")
    print(f"Worst Cp MAE: {summary['worst_cp_mae']:.6e}")
    print(f"Worst Cp max abs: {summary['worst_cp_max_abs']:.6e}")
    print(f"Saved Cp metrics to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
