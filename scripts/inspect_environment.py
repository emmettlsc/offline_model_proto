"""Print a snapshot of the runtime environment for triage on the air-gapped host."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _safe_pip_version() -> str:
    pip_path = shutil.which("pip") or shutil.which("pip3")
    if not pip_path:
        return "<not found>"
    try:
        out = subprocess.run(
            [pip_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or out.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<error: {exc}>"


def _try_torch() -> dict:
    info: dict = {"installed": False}
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        info["import_error"] = str(exc)
        return info
    info["installed"] = True
    info["version"] = torch.__version__
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name_0"] = torch.cuda.get_device_name(0)
            info["cuda_build_version"] = getattr(torch.version, "cuda", None)
    except Exception as exc:  # noqa: BLE001
        info["cuda_probe_error"] = str(exc)
    return info


def _try_torchvision() -> dict:
    try:
        import torchvision  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"installed": False, "import_error": str(exc)}
    return {"installed": True, "version": torchvision.__version__}


def _try_cv2() -> dict:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"installed": False, "import_error": str(exc)}
    return {"installed": True, "version": cv2.__version__}


def main() -> int:
    print("=== Environment ===")
    print(f"Python version : {sys.version.replace(chr(10), ' ')}")
    print(f"Executable     : {sys.executable}")
    print(f"Platform       : {platform.platform()}")
    print(f"Machine        : {platform.machine()}")
    print(f"System         : {platform.system()} {platform.release()}")
    print(f"CWD            : {Path.cwd()}")
    print(f"pip            : {_safe_pip_version()}")

    print("\n=== torch ===")
    for k, v in _try_torch().items():
        print(f"  {k:22s}: {v}")

    print("\n=== torchvision ===")
    for k, v in _try_torchvision().items():
        print(f"  {k:22s}: {v}")

    print("\n=== opencv ===")
    for k, v in _try_cv2().items():
        print(f"  {k:22s}: {v}")

    print("\n=== Selected env vars ===")
    for key in ("LD_LIBRARY_PATH", "PATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES",
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        print(f"  {key:22s}: {os.environ.get(key, '<unset>')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
