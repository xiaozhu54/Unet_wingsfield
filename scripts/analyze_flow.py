"""
Paper-style flow-field analysis for trained UNet predictions.

This script compares CFD reference fields with UNet predictions and exports:
  - CFD / UNet / normalized e_map panels for rho, p, u, v
  - pressure e_map figure
  - near-wall Cp comparison
  - CSV metrics and a Markdown report
"""
import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.train_config import TrainConfig
from data.preprocess import Normalizer, parse_filename, split_input_target
from models.unet import build_unet
from utils.metrics import (
    CHANNEL_NAMES,
    compute_pressure_coefficient,
    summarize_flow_error,
)
from utils.visualization import (
    plot_paper_flow_comparison,
    plot_pressure_error_map,
    plot_surface_cp_comparison,
)


def resolve_device(device_arg: str) -> str:
    """Resolve auto/cuda/mps/cpu device choice."""
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path: str, cfg: TrainConfig, device: str):
    """Load a UNet checkpoint with the project's checkpoint conventions."""
    model = build_unet(cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def select_samples(data_dir: str, sample_args: Optional[Sequence[str]],
                   max_samples: int, start_index: int) -> List[str]:
    """Select explicit sample files or a deterministic slice from data_dir."""
    if sample_args:
        paths = []
        for sample in sample_args:
            path = sample
            if not os.path.isabs(path):
                path = os.path.join(data_dir, sample)
            if not path.endswith(".npz"):
                raise ValueError(f"Sample is not an .npz file: {sample}")
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            paths.append(path)
        return paths

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    selected = files[start_index:start_index + max_samples]
    return [os.path.join(data_dir, f) for f in selected]


def denormalize_flow(flow_norm: np.ndarray, normalizer: Normalizer,
                     cfg: TrainConfig) -> np.ndarray:
    """Denormalize a (4, H, W) normalized flow field."""
    flow = flow_norm.copy().astype(np.float32)
    for out_idx, channel in enumerate(cfg.flow_channels):
        lo = normalizer.channel_mins[channel]
        hi = normalizer.channel_maxs[channel]
        flow[out_idx] = flow[out_idx] * (hi - lo) + lo
    return flow


def parse_extent(extent_args: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
    if extent_args is None:
        return None
    if len(extent_args) != 4:
        raise ValueError("--extent requires four numbers: xmin xmax ymin ymax")
    return tuple(float(v) for v in extent_args)


def sample_cp_lines(flow: np.ndarray, ma: float, axis: str = "row",
                    index: Optional[int] = None, offset: int = 2,
                    pressure_channel: int = 1,
                    gamma: float = 1.4) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, int]]:
    """
    Extract upper/lower near-wall Cp lines from transformed-grid rows/columns.

    The dataset does not include a dedicated wall mask, so the default is a
    near-center pair of grid lines. Use --surface-index and --surface-offset
    to align this with a known wall index.
    """
    pressure = flow[pressure_channel]
    cp = compute_pressure_coefficient(pressure, ma=ma, gamma=gamma)
    h, w = cp.shape
    offset = max(1, int(offset))

    if axis == "row":
        center = h // 2 if index is None else int(index)
        lower_idx = max(0, center - offset)
        upper_idx = min(h - 1, center + offset)
        x = np.linspace(0.0, 1.0, w)
        lines = {
            "upper": cp[upper_idx, :],
            "lower": cp[lower_idx, :],
        }
    else:
        center = w // 2 if index is None else int(index)
        lower_idx = max(0, center - offset)
        upper_idx = min(w - 1, center + offset)
        x = np.linspace(0.0, 1.0, h)
        lines = {
            "upper": cp[:, upper_idx],
            "lower": cp[:, lower_idx],
        }

    return x, lines, {"upper": upper_idx, "lower": lower_idx}


@torch.no_grad()
def predict_one(model, raw: np.ndarray, normalizer: Normalizer,
                cfg: TrainConfig, device: str) -> np.ndarray:
    """Run one denormalized prediction for a raw (16, H, W) sample."""
    raw_norm = normalizer.normalize(raw[np.newaxis].astype(np.float32))
    inputs, _ = split_input_target(raw_norm, cfg)
    x = torch.from_numpy(inputs).to(device)
    pred_norm = model(x).detach().cpu().numpy()[0]
    return denormalize_flow(pred_norm, normalizer, cfg)


def write_metrics_csv(path: str, all_rows: List[Dict[str, object]]):
    fieldnames = [
        "sample", "airfoil", "ma", "aoa", "re", "channel",
        "mse", "mae", "relative_l2", "max_abs",
        "true_min", "true_max", "pred_min", "pred_max",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def markdown_metric_table(rows: List[Dict[str, object]]) -> str:
    lines = [
        "| Channel | MSE | MAE | Rel L2 | Max Abs | True Range | Pred Range |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        true_range = f"{row['true_min']:.4g}..{row['true_max']:.4g}"
        pred_range = f"{row['pred_min']:.4g}..{row['pred_max']:.4g}"
        lines.append(
            f"| {row['channel']} | {row['mse']:.4g} | {row['mae']:.4g} | "
            f"{row['relative_l2']:.4g} | {row['max_abs']:.4g} | "
            f"{true_range} | {pred_range} |"
        )
    return "\n".join(lines)


def write_report(path: str, summary: List[Dict[str, object]],
                 checkpoint: str, norm_path: str, device: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# UNet Flow Field Analysis",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Normalizer: `{norm_path}`",
        f"- Device: `{device}`",
        "",
        "The figures follow the paper's analysis pattern: CFD reference, UNet prediction, normalized error map (e_map), pressure e_map, and near-wall Cp curves.",
        "",
    ]

    for item in summary:
        meta = item["meta"]
        lines.extend([
            f"## {item['sample']}",
            "",
            f"- Airfoil: `{meta['airfoil']}`",
            f"- Freestream: Ma={meta['ma']:.3f}, AoA={meta['aoa']:.3f} deg, Re={meta['re']:.3g}",
            f"- Cp sampling: axis={item['cp_axis']}, lines={item['cp_indices']}",
            "",
            markdown_metric_table(item["metrics"]),
            "",
            f"![Flow panel]({item['flow_panel']})",
            "",
            f"![Pressure error]({item['pressure_error']})",
            "",
            f"![Cp]({item['cp_plot']})",
            "",
        ])

    with open(path, "w") as f:
        f.write("\n".join(lines))


def analyze_samples(args):
    cfg = TrainConfig()
    device = resolve_device(args.device)
    normalizer = Normalizer.load(args.norm_path)
    model = load_model(args.checkpoint, cfg, device)
    extent = parse_extent(args.extent)
    samples = select_samples(args.data_dir, args.sample, args.max_samples, args.start_index)

    os.makedirs(args.output_dir, exist_ok=True)
    metric_rows = []
    report_items = []

    for sample_path in samples:
        raw = np.load(sample_path)["a"].astype(np.float32)
        sample_name = os.path.basename(sample_path)
        airfoil, ma, aoa, re_val = parse_filename(sample_name)
        true_flow = raw[cfg.flow_channels]
        pred_flow = predict_one(model, raw, normalizer, cfg, device)
        metrics = summarize_flow_error(true_flow, pred_flow, CHANNEL_NAMES)

        stem = os.path.splitext(sample_name)[0]
        sample_dir = os.path.join(args.output_dir, stem)
        os.makedirs(sample_dir, exist_ok=True)

        flow_panel = os.path.join(sample_dir, "flow_panel.png")
        pressure_error = os.path.join(sample_dir, "pressure_error.png")
        cp_plot = os.path.join(sample_dir, "cp_comparison.png")

        title = f"{airfoil}: Ma={ma:.2f}, AoA={aoa:.2f} deg, Re={re_val:.3g}"
        plot_paper_flow_comparison(
            true_flow, pred_flow, metrics=metrics, title=title,
            save_path=flow_panel, extent=extent,
        )
        plot_pressure_error_map(
            true_flow, pred_flow, mse=metrics[1]["mse"],
            title=f"{airfoil} pressure e_map",
            save_path=pressure_error, extent=extent,
        )

        x_cp, cp_true, cp_indices = sample_cp_lines(
            true_flow, ma=ma, axis=args.surface_axis,
            index=args.surface_index, offset=args.surface_offset,
            gamma=args.gamma,
        )
        _, cp_pred, _ = sample_cp_lines(
            pred_flow, ma=ma, axis=args.surface_axis,
            index=args.surface_index, offset=args.surface_offset,
            gamma=args.gamma,
        )
        plot_surface_cp_comparison(
            x_cp, cp_true, cp_pred, title=f"{airfoil} near-wall Cp",
            save_path=cp_plot,
        )

        meta = {"airfoil": airfoil, "ma": ma, "aoa": aoa, "re": re_val}
        for row in metrics:
            metric_rows.append({
                "sample": sample_name,
                **meta,
                **row,
            })

        report_items.append({
            "sample": sample_name,
            "meta": meta,
            "metrics": metrics,
            "flow_panel": os.path.relpath(flow_panel, args.output_dir),
            "pressure_error": os.path.relpath(pressure_error, args.output_dir),
            "cp_plot": os.path.relpath(cp_plot, args.output_dir),
            "cp_axis": args.surface_axis,
            "cp_indices": cp_indices,
        })
        print(f"Analyzed {sample_name}: pressure MSE={metrics[1]['mse']:.6g}")

    metrics_csv = os.path.join(args.output_dir, "flow_metrics.csv")
    report_path = os.path.join(args.output_dir, "analysis_report.md")
    write_metrics_csv(metrics_csv, metric_rows)
    write_report(report_path, report_items, args.checkpoint, args.norm_path, device)
    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved report:  {report_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate paper-style UNet flow-field analysis figures."
    )
    parser.add_argument("--data-dir", default="train_data_2822")
    parser.add_argument("--checkpoint", default="results_gen0/best_model.pth")
    parser.add_argument("--norm-path", default="results_gen0/normalizer.npz")
    parser.add_argument("--output-dir", default="results_gen0/flow_analysis")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample", action="append",
                        help="Specific .npz sample filename/path. Can be repeated.")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--extent", nargs=4, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                        help="Optional imshow extent, e.g. -0.25 1.25 -0.4 1.1")
    parser.add_argument("--surface-axis", choices=("row", "col"), default="row")
    parser.add_argument("--surface-index", type=int, default=None)
    parser.add_argument("--surface-offset", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=1.4)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    analyze_samples(args)


if __name__ == "__main__":
    main()
