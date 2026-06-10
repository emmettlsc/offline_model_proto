import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def pip_version():
    p = shutil.which("pip") or shutil.which("pip3")
    if not p:
        return "not found"
    try:
        r = subprocess.run([p, "--version"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or r.stderr.strip()
    except Exception as e:
        return f"error: {e}"


def torch_info():
    try:
        import torch
    except Exception as e:
        return {"installed": False, "import_error": str(e)}
    info = {"installed": True, "version": torch.__version__}
    try:
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["devices"] = torch.cuda.device_count()
            info["device_0"] = torch.cuda.get_device_name(0)
            info["cuda_build"] = getattr(torch.version, "cuda", None)
    except Exception as e:
        info["cuda_error"] = str(e)
    return info


def mod_version(name):
    try:
        m = __import__(name)
        return {"installed": True, "version": getattr(m, "__version__", "?")}
    except Exception as e:
        return {"installed": False, "import_error": str(e)}


def main():
    print(f"python      {sys.version.replace(chr(10), ' ')}")
    print(f"executable  {sys.executable}")
    print(f"platform    {platform.platform()}")
    print(f"machine     {platform.machine()}")
    print(f"system      {platform.system()} {platform.release()}")
    print(f"cwd         {Path.cwd()}")
    print(f"pip         {pip_version()}")

    for label, info in [
        ("torch", torch_info()),
        ("torchvision", mod_version("torchvision")),
        ("opencv", mod_version("cv2")),
        ("onnxruntime", mod_version("onnxruntime")),
        ("transformers", mod_version("transformers")),
    ]:
        print(f"\n{label}")
        for k, v in info.items():
            print(f"  {k}: {v}")

    print("\nenv")
    for k in ("LD_LIBRARY_PATH", "PATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES",
              "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        print(f"  {k}: {os.environ.get(k, '<unset>')}")


if __name__ == "__main__":
    main()
