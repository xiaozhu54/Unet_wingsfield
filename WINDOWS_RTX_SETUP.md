# Windows + NVIDIA RTX Training Setup

This project now defaults to CUDA first when PyTorch can see an NVIDIA GPU.
The code path remains compatible with MPS and CPU, but the recommended Windows
training path is CUDA.

## 1. Install Python

Install Python 3.10 or newer. PyTorch currently supports Python 3.10-3.14 on
Windows. After installing, reopen PowerShell and check:

```powershell
python --version
```

If this opens the Microsoft Store, install Python from python.org or disable the
Windows App Execution Alias for `python.exe`.

## 2. Install PyTorch for CUDA

For RTX 50-series GPUs, prefer a current PyTorch CUDA build. Example for CUDA
12.8 wheels:

```powershell
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy matplotlib
```

The project itself only needs `torch`, `torchvision`, `numpy`, and `matplotlib`.
`torchaudio` is included because the official PyTorch selector includes it.

## 3. Verify GPU Visibility

From the project root:

```powershell
python scripts/check_gpu.py
```

Expected result on the RTX machine:

```text
CUDA available: True
CUDA device count: 1
CUDA:0 name=NVIDIA GeForce RTX 5080, ...
CUDA tensor test: ok
Recommended training device: cuda
```

If CUDA is false, fix the NVIDIA driver / PyTorch CUDA wheel before training.

## 4. Train

Default CUDA-optimized run on this Windows machine:

```powershell
python scripts/train.py --data-dir train_data_2822 --epochs 200 --batch-size 32 --num-workers 0 --cache-data --device cuda --save-dir results_gen3_channel_weighted --upsample-mode transpose --channel-weights 1.5,1.5,1.0,1.0
```

This project originally tried multi-worker loading, but on this Windows setup it hit `WinError 5` in multiprocessing. Use `--num-workers 0 --cache-data` unless the local multiprocessing issue is fixed.

Cp-oriented training run:

```powershell
python scripts/train.py --data-dir train_data_2822 --epochs 200 --batch-size 32 --num-workers 0 --cache-data --device cuda --save-dir results_gen5_cp_loss --upsample-mode transpose --channel-weights 1.5,1.5,1.0,1.0 --cp-loss-weight 0.5
```

If you hit CUDA out-of-memory, lower `--batch-size` first.

For numerical comparison with an old FP32/Mac run:

```powershell
python scripts/train.py --data-dir train_data_2822 --epochs 200 --batch-size 16 --no-amp --no-channels-last --save-dir results_fp32_compare
```

## 5. Evaluate

```powershell
python scripts/evaluate.py --data-dir train_data_2822 --checkpoint results_gen3_channel_weighted/best_model.pth --norm-path results_gen3_channel_weighted/normalizer.npz --device cuda --output-dir results_gen3_channel_weighted/evaluation
```

Cp metrics:

```powershell
python scripts/evaluate_cp.py --data-dir train_data_2822 --checkpoint results_gen5_cp_loss/best_model.pth --norm-path results_gen5_cp_loss/normalizer.npz --device cuda --output-dir results_gen5_cp_loss/cp_evaluation
```

## CUDA-related defaults added

- CUDA is preferred over MPS and CPU when available.
- `pin_memory=True` on CUDA.
- DataLoader workers auto-tune to a capped value on CUDA, but this Windows machine should override to `--num-workers 0`.
- `--cache-data` preloads normalized tensors into RAM and gives about 3 second epochs after warmup.
- `persistent_workers=True` when workers are enabled.
- `torch.backends.cudnn.benchmark=True` for fixed 128x128 inputs.
- TF32 matmul/cuDNN is enabled for RTX GPUs.
- CUDA AMP defaults to `float16`.
- `channels_last` is enabled on CUDA Conv2d workloads.
