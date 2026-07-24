# UNet Airfoil Cp-Oriented Surrogate

This repository trains UNet-style surrogate models for RAE2822 compressible flow data. The current engineering target is accurate near-wall pressure coefficient `Cp` from freestream condition plus airfoil/grid inputs.

The model still predicts `(rho, p, u, v)`, but experiments are now compared using both full-field held-out metrics and Cp metrics.

## Current Best Checkpoints

| Purpose | Checkpoint |
|---|---|
| Stable pre-Cp baseline | `results_gen3_channel_weighted/best_model.pth` |
| First strong Cp proof | `results_gen5_cp_loss/best_model.pth` |
| Current recommended balanced model | `results_gen8_cpft_local_w020/best_model.pth` |

Gen8 continues Cp fine-tuning from gen7 with local Cp weighting. It reaches test normalized L1 `0.0014612` and mean Cp MAE `0.0029982`, improving both over gen7 for the current Cp/full-field tradeoff.

## Setup

```powershell
cd C:\Users\xiaoz\Desktop\Unet_wingsfield
.\.venv\Scripts\python.exe scripts\check_gpu.py
```

Recommended Windows RTX settings:

- `--device cuda`
- `--cache-data`
- `--num-workers 0`

## Train

Full-field baseline route:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --data-dir train_data_2822 --epochs 200 --batch-size 32 --num-workers 0 --cache-data --save-dir results_gen3_channel_weighted --device cuda --upsample-mode transpose --channel-weights 1.5,1.5,1.0,1.0
```

Cp-oriented route:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --data-dir train_data_2822 --epochs 200 --batch-size 32 --num-workers 0 --cache-data --save-dir results_gen5_cp_loss --device cuda --upsample-mode transpose --channel-weights 1.5,1.5,1.0,1.0 --cp-loss-weight 0.5
```

Balanced Cp fine-tune route:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --data-dir train_data_2822 --epochs 80 --batch-size 32 --num-workers 0 --cache-data --save-dir results_gen6_cpft_w020 --device cuda --upsample-mode transpose --channel-weights 1.5,1.5,1.0,1.0 --cp-loss-weight 0.2 --lr 1e-4 --init-checkpoint results_gen3_channel_weighted\best_model.pth
```

Current extended fine-tune route:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --data-dir train_data_2822 --epochs 160 --batch-size 32 --num-workers 0 --cache-data --save-dir results_gen7_cpft_w020_long --device cuda --upsample-mode transpose --channel-weights 1.5,1.5,1.0,1.0 --cp-loss-weight 0.2 --lr 5e-5 --init-checkpoint results_gen6_cpft_w020\best_model.pth
```

Current local-focus fine-tune route:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --data-dir train_data_2822 --epochs 120 --batch-size 32 --num-workers 0 --cache-data --save-dir results_gen8_cpft_local_w020 --device cuda --upsample-mode transpose --channel-weights 1.5,1.5,1.0,1.0 --cp-loss-weight 0.2 --cp-focus-weight 2.0 --cp-negative-aoa-lower-weight 1.5 --lr 1e-5 --init-checkpoint results_gen7_cpft_w020_long\best_model.pth
```

## Evaluate

Full-field held-out test metrics:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --data-dir train_data_2822 --checkpoint results_gen3_channel_weighted\best_model.pth --norm-path results_gen3_channel_weighted\normalizer.npz --device cuda --output-dir results_gen3_channel_weighted\evaluation
```

Cp held-out test metrics:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_cp.py --data-dir train_data_2822 --checkpoint results_gen8_cpft_local_w020\best_model.pth --norm-path results_gen8_cpft_local_w020\normalizer.npz --device cuda --output-dir results_gen8_cpft_local_w020\cp_evaluation
```

Generate flow and Cp figures:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_flow.py --data-dir train_data_2822 --checkpoint results_gen5_cp_loss\best_model.pth --norm-path results_gen5_cp_loss\normalizer.npz --device cuda --output-dir results_gen5_cp_loss\flow_analysis_test3 --sample rae2822gen0_50_0_4000.npz --sample rae2822gen0_72_-400_6500.npz --sample rae2822gen0_65_-1200_4000.npz --extent -0.25 1.25 -0.4 1.1
```

## Documentation

- `agents.md`: current project state, experiment comparison, and optimization route.
- `CONTEXT.md`: domain vocabulary.
- `docs/adr/`: architecture and experiment decisions.
