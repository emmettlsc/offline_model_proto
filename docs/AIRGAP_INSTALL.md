# Air-Gapped Install

Steps for the target Linux machine (RHEL 8.10) that has no internet access.

## Prerequisites — target environment

The target is RHEL 8.10 with NVIDIA driver 580.105.08 and an RTX PRO 5000
Blackwell GPU. The system Python (`/usr/bin/python3` = 3.6.8) and system pip
(9.0.3) are **too old to host this stack** and must be ignored. Every command
below uses `python3.11` explicitly — calling `python3` would pick up 3.6.8
and fail in confusing ways.

## Step 0 — Install Python 3.11 from RPMs

Do this once per target. It is independent of the offline bundle and only
needs to be redone if the box is reimaged.

On a connected RHEL 8.10 box (matching the minor exactly):

    sudo dnf install --downloadonly --downloaddir=./py311_rpms \
        python3.11 python3.11-pip python3.11-devel

Transfer `py311_rpms/` through the bastion alongside the offline bundle.
On the target:

    sudo dnf install ./py311_rpms/*.rpm
    /usr/bin/python3.11 --version          # expect 3.11.x
    /usr/bin/python3.11 -m pip --version   # expect pip >= 22

If `python3.11 -m pip` is missing or ancient, `python3.11 -m ensurepip
--upgrade` will bootstrap it from the stdlib copy.

## 1. Transfer the offline bundle

Move `offline_model_proto_bundle.tar.gz` onto the target via your bastion or
approved transfer mechanism.

## 2. Unpack

    tar -xzf offline_model_proto_bundle.tar.gz
    cd offline_model_proto

## 3. Create a virtual environment with Python 3.11

    python3.11 -m venv venv
    source venv/bin/activate
    python --version                       # confirm 3.11.x — must match wheelhouse build

If the minor version differs from the wheelhouse build (e.g. 3.11 vs 3.12),
the install will fail with "Could not find a version that satisfies".

## 4. Install everything from the wheelhouse (no internet)

    pip install --upgrade --no-index --find-links wheelhouse pip setuptools wheel
    pip install --no-index --find-links wheelhouse -r requirements.txt

`--no-index` forces pip to ignore PyPI entirely and only use the local
wheelhouse.

## 5. Sanity checks

    python scripts/inspect_environment.py
    python scripts/smoke_test_python.py
    python scripts/smoke_test_torch.py     # expect cuda_available=True
    python scripts/smoke_test_cv.py

`smoke_test_torch.py` is the moment of truth for the Blackwell + cu128
pairing. If it prints `cuda available: True` and the GPU tensor sums work,
the wheelhouse is good. If it crashes with
`no kernel image is available for execution on the device`, the torch wheel
is the wrong CUDA tag — rebuild the wheelhouse against `cu128` or newer.

## 6. Place model weights and run

Put weights under `models/` (see `docs/MODEL_WEIGHTS.md`). Then:

    python scripts/run_pytorch_pipeline.py \
        --video data/clip.mp4 \
        --rfdetr-weights models/rfdetr/ \
        --sam3-weights models/sam3/

or the ONNX equivalent:

    python scripts/run_onnx_pipeline.py \
        --video data/clip.mp4 \
        --rfdetr-onnx models/rfdetr.onnx \
        --sam3-encoder-onnx models/sam3_encoder.onnx \
        --sam3-decoder-onnx models/sam3_decoder.onnx

## Common failure modes

- **Missing wheel** — `pip` reports
  `ERROR: Could not find a version that satisfies the requirement ...`.
  The wheelhouse was built against a different Python/arch/glibc than the
  target. Rebuild on a machine that matches the target.

- **Python version mismatch** — wheels are tagged for specific minor versions
  (`cp310`, `cp311`, `cp312`, ...). The target's `python3 --version` must
  match the wheelhouse build environment.

- **CUDA mismatch** — torch silently falls back to CPU, or fails at runtime
  with CUDA driver errors. The torch wheel's CUDA version (e.g. `cu128`)
  must be compatible with the host's installed driver. Check with
  `nvidia-smi`.

- **`no kernel image is available for execution on the device`** — the torch
  wheel was built without kernels for this GPU's compute capability. On the
  target's RTX PRO 5000 Blackwell (sm_120), this means the wheel is `cu121`
  or `cu124` — both predate Blackwell kernels. Rebuild the wheelhouse against
  `cu128` (or newer) from `https://download.pytorch.org/whl/cu128`.

- **`python3` runs 3.6.8** — you skipped Step 0 or are not inside the venv.
  All commands must run inside `source venv/bin/activate` where `python`
  resolves to the Python 3.11 interpreter.

- **OpenCV missing system libraries** — `import cv2` fails with errors about
  `libGL.so.1`, `libgthread-2.0.so.0`, `libSM.so.6`, etc. Either:
    - install the OS packages providing those shared libraries
      (`mesa-libGL`, `libglib2.0-0`, `libsm6`), or
    - switch the requirement from `opencv-python` to
      `opencv-python-headless`, rebuild the wheelhouse, and re-bundle.

- **Code trying to download checkpoints/configs** — a model library tries to
  call out to `huggingface.co`, `github.com`, S3, etc. There is no internet
  on the target. Pre-stage all checkpoints and configs under `models/` and
  pass them explicitly. Some libraries also need offline env vars:

        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1

- **RF-DETR package not available offline** — the `rfdetr` wheel was not in
  the wheelhouse, or its dependencies were not resolved. Vendor the wheel (or
  vendored source) on the build machine and rebuild the bundle.

- **`pip install` succeeds but `import torch` segfaults** — usually a glibc
  mismatch between build host and target, or a CUDA runtime mismatch. Verify
  `ldd --version` and `nvidia-smi` on both hosts.
