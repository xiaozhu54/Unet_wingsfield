#!/usr/bin/env python3
"""Check whether the current Python/Torch environment can use CUDA or MPS."""
import platform
import subprocess
import sys


def _run_optional(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=5).strip()
    except Exception as exc:
        return f"unavailable ({exc})"


def _print_cuda(torch):
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Torch CUDA runtime: {torch.version.cuda}")
    print(f"cuDNN available: {torch.backends.cudnn.is_available()}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print("nvidia-smi:")
    print(_run_optional(["nvidia-smi"]))

    if not torch.cuda.is_available():
        return False

    count = torch.cuda.device_count()
    print(f"CUDA device count: {count}")
    for idx in range(count):
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        print(
            f"CUDA:{idx} name={props.name}, "
            f"capability={props.major}.{props.minor}, memory={total_gb:.1f}GB"
        )

    device = torch.device("cuda:0")
    x = torch.ones((1024, 1024), device=device)
    y = x @ x
    torch.cuda.synchronize()
    print(f"CUDA tensor test: ok, device={y.device}, mean={y.mean().item():.1f}")
    return True


def _print_mps(torch):
    mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS built: {mps_built}")
    print(f"MPS available: {mps_available}")
    if not mps_available:
        return False
    x = torch.ones(4, device="mps")
    print(f"MPS tensor test: ok, device={x.device}, value={x.tolist()}")
    return True


def main():
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    try:
        import torch
    except Exception as exc:
        print(f"Torch import failed: {exc}")
        raise SystemExit(1)

    print(f"Torch: {torch.__version__}")
    cuda_ok = _print_cuda(torch)
    mps_ok = _print_mps(torch)

    if cuda_ok:
        print("Recommended training device: cuda")
        return
    if mps_ok:
        print("Recommended training device: mps")
        return

    print("No GPU backend is available to PyTorch. Training will fall back to CPU.")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
