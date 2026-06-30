#!/usr/bin/env python3
"""Check whether the current Python/Torch environment can use Apple MPS."""
import platform
import subprocess
import sys


def main():
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"platform.mac_ver: {platform.mac_ver()}")
    try:
        sw_vers = subprocess.check_output(["sw_vers"], text=True).strip()
        print("sw_vers:")
        print(sw_vers)
    except Exception as exc:
        print(f"sw_vers: unavailable ({exc})")

    try:
        import torch
    except Exception as exc:
        print(f"Torch import failed: {exc}")
        raise SystemExit(1)

    print(f"Torch: {torch.__version__}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.backends.mps.is_available():
        x = torch.ones(4, device="mps")
        print(f"MPS tensor test: ok, device={x.device}, value={x.tolist()}")
    else:
        print("MPS tensor test: skipped because MPS is not available.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
