"""Confirm torch is importable and that CPU (and CUDA, if present) tensors work."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError as exc:
        print(f"ERROR: torch is not installed: {exc}", file=sys.stderr)
        print("On the air-gapped target, install from the offline wheelhouse:",
              file=sys.stderr)
        print("  pip install --no-index --find-links wheelhouse torch torchvision",
              file=sys.stderr)
        return 1

    print(f"torch version : {torch.__version__}")
    cuda = bool(torch.cuda.is_available())
    print(f"cuda available: {cuda}")

    try:
        cpu_tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        print(f"cpu tensor    : shape={tuple(cpu_tensor.shape)} "
              f"sum={cpu_tensor.sum().item()}")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to create CPU tensor: {exc}", file=sys.stderr)
        return 2

    if cuda:
        try:
            print(f"gpu name      : {torch.cuda.get_device_name(0)}")
            print(f"torch.cuda    : built against CUDA "
                  f"{getattr(torch.version, 'cuda', '?')}")
            gpu_tensor = cpu_tensor.to("cuda")
            print(f"gpu tensor    : device={gpu_tensor.device} "
                  f"sum={gpu_tensor.sum().item()}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: failed to use CUDA: {exc}", file=sys.stderr)
            print("Likely cause: torch wheel's CUDA version does not match the "
                  "host driver (check `nvidia-smi`).", file=sys.stderr)
            return 3
    else:
        print("note          : no CUDA device — CPU-only smoke test passed")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
