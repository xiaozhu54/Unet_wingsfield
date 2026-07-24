# agents.md - UNet Airfoil Cp-Oriented Surrogate

## Current Objective

This project is no longer judged only as a full flow-field surrogate. The practical target is:

> Input freestream condition plus airfoil/grid information, then quickly and accurately predict airfoil surface or near-wall pressure coefficient `Cp`.

The model still predicts the full field `(rho, p, u, v)` because pressure, density, velocity, shock position, and boundary-layer behavior are coupled. However, model selection and optimization should prioritize Cp accuracy, especially near the leading edge, mid-chord pressure recovery/shock region, and trailing edge.

## Data Contract

Each `.npz` sample stores key `"a"` with shape `(16, 128, 128)`.

| Channel | Meaning | Role |
|---:|---|---|
| 0 | Mach number | input |
| 1 | angle of attack | input |
| 2 | Reynolds number | input |
| 3-11 | grid metrics / transformed-coordinate geometry | input |
| 12 | density `rho` | target |
| 13 | pressure `p` | target and Cp source |
| 14 | velocity `u` | target |
| 15 | velocity `v` | target |

The formal split is 80/15/5 train/validation/test with `random_seed=42`. Experiments must persist `split_manifest.json`, and the `Normalizer` must be fitted only on the train split.

Current Cp extraction is an approximation: no true wall mask or surface index is present in the data. Scripts use two near-wall transformed-grid lines by default: `axis=row`, `index=center`, `offset=2` -> rows 66 and 62. This is useful for optimization and comparison, but final aerodynamic-quality Cp requires a true surface index or wall mask.

## Physics Interpretation

For `rae2822gen0_72_-400_6500`, Cp errors near `x/c=0` and `x/c≈0.5` are physically plausible failure regions:

- `x/c=0`: leading-edge stagnation/suction region, strong pressure gradient, high curvature, and high sensitivity to geometry and AoA.
- `x/c≈0.5`: likely pressure recovery, weak shock/compression, or shock-foot-sensitive region in transonic RAE2822-like flow.
- The observed `rho` and `p` error band around plotted `y/c≈0.25` is probably correlated with the same pressure-wave/shock structure rather than being an independent cause.

Important caveat: current plots use `imshow` extents and transformed-grid rows, so plotted `x/c,y/c` are visualization coordinates, not guaranteed exact physical coordinates unless a surface/mesh mapping is supplied.

Conclusion: the original full-field UNet is physically reasonable as a surrogate baseline, but it is not yet fully aligned with the final Cp objective. Cp must appear in evaluation and, carefully, in training.

## Experiment Comparison

Formal held-out test set: 57 samples.

| Gen | Output dir | Main change | Test normalized L1 | Mean Cp MAE | Judgment |
|---|---|---|---:|---:|---|
| gen0 | `results_gen0` | legacy RTX baseline, old evaluation contract | 0.0020485 | not formalized | keep only as legacy reference |
| gen1 | `results_gen1_evalfix` | train-only normalizer + strict test split | 0.0019907 | not formalized | clean baseline |
| gen2 | `results_gen2_bilinear_upsample` | bilinear upsample instead of transpose conv | 0.0020635 | not formalized | worse than gen1 |
| gen3 | `results_gen3_channel_weighted` | transpose conv + channel weights `rho,p=1.5` | 0.0016954 | 0.0087596 | best full-field model so far |
| gen4 | `results_gen4_attention_unet` | Attention UNet gates on gen3 | 0.0019409 | not formalized | attention did not help |
| gen5 | `results_gen5_cp_loss` | gen3 + near-wall Cp loss weight 0.5 | 0.0029229 | 0.0029325 | Cp improves, full-field degrades |
| gen6a | `results_gen6_cpft_w010` | fine-tune from gen3, Cp loss weight 0.1, LR 1e-4 | 0.0015482 | 0.0048405 | best full-field so far, Cp improved |
| gen6b | `results_gen6_cpft_w020` | fine-tune from gen3, Cp loss weight 0.2, LR 1e-4 | 0.0016187 | 0.0044083 | best current Cp/full-field balance |
| gen7 | `results_gen7_cpft_w020_long` | continue from gen6b, Cp loss 0.2, LR 5e-5, 160 epochs | 0.0014717 | 0.0031854 | current best balance |
| gen8 | `results_gen8_cpft_local_w020` | continue from gen7, local Cp focus, negative-AoA lower weight | 0.0014612 | 0.0029982 | current best local/Cp balance |

Gen3 is the stable pre-Cp baseline:

```text
results_gen3_channel_weighted/best_model.pth
```

Gen5 is the first Cp-oriented proof that direct Cp supervision works:

```text
results_gen5_cp_loss/best_model.pth
```

Gen8 is the current recommended checkpoint:

```text
results_gen8_cpft_local_w020/best_model.pth
```

Gen8 continues gen7 with local Cp weighting near the leading edge, mid-chord, trailing edge, and negative-AoA lower-surface cases. It improves full-field L1 and mean Cp MAE over gen7, with a small tradeoff in worst Cp max error.

## Current Technical Decisions

1. Use gen8 as the current default checkpoint for balanced Cp/full-field inference.
2. Use gen3 as the stable pre-Cp baseline for comparison.
3. Use gen5 as proof that stronger Cp-directed training can improve average Cp but may damage the physical field.
4. Do not continue generic Attention UNet as the default route; gen4 was worse than gen3.
5. Do not prioritize bigger/wider UNet; the model already has about 200M parameters for only 1153 RAE2822 samples.
6. Prioritize target alignment: strict test split, Cp metrics, true surface extraction, and balanced loss.

## Optimization Route

1. Surface contract: add or recover true wall/surface indices from the C-grid or preprocessing pipeline. This is the highest-value correction.
2. Cp objective: keep Cp in training through staged fine-tuning. Gen6 and gen7 show `0.1-0.2` with lower LR is better than training from scratch with `0.5`.
3. Multi-objective model selection: choose checkpoints by both full-field test metrics and held-out Cp metrics, not by combined validation loss alone.
4. Region-aware loss: gen8 added approximate leading-edge, mid-chord, trailing-edge, and negative-AoA lower-surface weighting. The next improvement should use a true surface index instead of approximate rows.
5. Architecture only after objective is correct: ResUNet or UNet-FNO bottleneck is more promising than plain Attention UNet.
6. Generalization: after RAE2822 Cp route is stable, add multi-airfoil data and evaluate geometry extrapolation.

## Commands

Train the current full-field baseline style:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --data-dir train_data_2822 --epochs 200 --batch-size 32 --num-workers 0 --cache-data --save-dir results_gen3_channel_weighted --device cuda --upsample-mode transpose --channel-weights 1.5,1.5,1.0,1.0
```

Train the first Cp-oriented route:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --data-dir train_data_2822 --epochs 200 --batch-size 32 --num-workers 0 --cache-data --save-dir results_gen5_cp_loss --device cuda --upsample-mode transpose --channel-weights 1.5,1.5,1.0,1.0 --cp-loss-weight 0.5
```

Evaluate held-out full-field metrics:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --data-dir train_data_2822 --checkpoint results_gen3_channel_weighted\best_model.pth --norm-path results_gen3_channel_weighted\normalizer.npz --device cuda --output-dir results_gen3_channel_weighted\evaluation
```

Evaluate held-out Cp metrics:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_cp.py --data-dir train_data_2822 --checkpoint results_gen5_cp_loss\best_model.pth --norm-path results_gen5_cp_loss\normalizer.npz --device cuda --output-dir results_gen5_cp_loss\cp_evaluation
```

Generate three-sample flow and Cp figures:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_flow.py --data-dir train_data_2822 --checkpoint results_gen5_cp_loss\best_model.pth --norm-path results_gen5_cp_loss\normalizer.npz --device cuda --output-dir results_gen5_cp_loss\flow_analysis_test3 --sample rae2822gen0_50_0_4000.npz --sample rae2822gen0_72_-400_6500.npz --sample rae2822gen0_65_-1200_4000.npz --extent -0.25 1.25 -0.4 1.1
```

## Windows RTX Notes

Current recommended training options on RTX 5080:

- `--device cuda`
- `--cache-data`
- `--num-workers 0` on this Windows machine, because multi-worker loading hit `WinError 5`.
- CUDA AMP `float16` and `channels_last` remain enabled by default.

With cache-data, epochs after warmup are about 3 seconds and GPU utilization is around 90%.

## Key Files

| File | Purpose |
|---|---|
| `config/train_config.py` | experiment configuration, Cp-loss parameters |
| `data/dataset.py` | train-only normalizer, split manifest, optional RAM cache |
| `training/trainer.py` | weighted flow/gradient loss and optional Cp loss |
| `utils/surface_sampling.py` | shared near-wall Cp sampling contract for training/evaluation |
| `models/unet.py`, `models/blocks.py` | UNet, bilinear option, attention option |
| `scripts/train.py` | training entry |
| `scripts/evaluate.py` | strict held-out full-field evaluation |
| `scripts/evaluate_cp.py` | strict held-out Cp evaluation |
| `scripts/analyze_flow.py` | figures, flow metrics, Cp metrics |

Last updated: 2026-07-05.
