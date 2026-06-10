# Building the Linux Wheelhouse

The wheelhouse is the set of `.whl` files that `pip` will install from on the
air-gapped target. It must be built on a Linux machine — not on macOS — and
should match the target as closely as possible.

## Target profile (this deployment)

These are the values observed on the air-gapped target. Match them on the
build host:

| Property                | Target value                                   | Build-host requirement                                    |
|-------------------------|------------------------------------------------|-----------------------------------------------------------|
| Distro                  | RHEL 8.10                                      | RHEL 8.x / Rocky 8 / AlmaLinux 8, or `manylinux_2_28` Docker image |
| Kernel / arch           | `4.18` / `x86_64`                              | `x86_64`                                                  |
| glibc                   | 2.28                                           | ≤ 2.28 (RHEL 8 family or `manylinux_2_28`)                |
| **System Python**       | **3.6.8 — UNUSABLE, ignore it**                | n/a                                                       |
| **Installed Python**    | **`python3.11` (install separately, see below)** | **`python3.11` exact minor match**                      |
| GPU                     | NVIDIA RTX PRO 5000 Blackwell (sm_120)         | n/a                                                       |
| NVIDIA driver           | 580.105.08 (max CUDA 13.0)                     | n/a                                                       |
| **PyTorch CUDA wheel**  | **`cu128` (Blackwell needs CUDA ≥ 12.8)**      | `--index-url https://download.pytorch.org/whl/cu128`      |
| OpenCV system libs      | `libGL`, `libSM`, `libXrender`, `libglib` present | `opencv-python` (not headless) is OK                   |
| ffmpeg                  | 7.1.1                                          | n/a                                                       |

**Why the GPU constraint matters:** Blackwell (sm_120) kernels were added to
PyTorch in 2.7 / CUDA 12.8 wheels. Older `cu121` / `cu124` builds install
cleanly and then fail at first GPU op with
`no kernel image is available for execution on the device`.

### Recommended build hosts, in order

1. **Rocky Linux 8 or AlmaLinux 8 VM** — binary-compatible with RHEL 8.10. Cleanest match.
2. **`quay.io/pypa/manylinux_2_28_x86_64` container** on any Linux host — purpose-built for this exact tag.
3. **RHEL 8.10 itself** if you have a license.

Do **not** build on Ubuntu 22.04+ (glibc 2.35), Fedora 38+, or macOS. Wheels
tagged `manylinux_2_31` or `manylinux_2_34` will not install on glibc 2.28.

## Installing Python 3.11 on the air-gapped target

The system Python on RHEL 8.10 is 3.6.8, which is too old for this stack
(PyTorch 2.x, `rfdetr`, `supervision` all require ≥3.9). Install Python 3.11
from RPMs alongside the system Python — do not replace `/usr/bin/python3`.

On a connected RHEL 8.10 box (matching the minor exactly):

    sudo dnf install --downloadonly --downloaddir=./py311_rpms \
        python3.11 python3.11-pip python3.11-devel

Transfer the `py311_rpms/` directory through the bastion, then on the target:

    sudo dnf install ./py311_rpms/*.rpm
    /usr/bin/python3.11 --version    # confirm 3.11.x
    /usr/bin/python3.11 -m pip --version

From this point forward on the target, every command in `AIRGAP_INSTALL.md`
uses `python3.11` explicitly. Calling `python3` would pick up 3.6.8 and
fail in confusing ways.

## Building the wheelhouse via a bastion (cross-platform `pip download`)

If your build environment is a bastion that isn't RHEL 8 — a Mac, an Ubuntu
box, or a fetch proxy like `python_downloader` — you can still produce a
target-compatible wheelhouse by giving `pip download` explicit
platform/Python/ABI flags. The host running the download does not have to
match the target, but it must have a recent pip (≥ 22).

First, resolve the lock. Run this somewhere with a real interpreter (Linux
preferred — some packages resolve to slightly different sub-deps on Mac):

    pip install pip-tools
    pip-compile requirements.in -o requirements.txt

Then fetch wheels targeting the air-gapped box explicitly:

    pip download \
        --platform manylinux_2_28_x86_64 \
        --platform manylinux2014_x86_64 \
        --python-version 3.11 \
        --implementation cp \
        --abi cp311 \
        --only-binary=:all: \
        -r requirements.txt \
        -d wheelhouse

    pip download \
        --platform manylinux_2_28_x86_64 \
        --python-version 3.11 \
        --implementation cp \
        --abi cp311 \
        --only-binary=:all: \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision \
        -d wheelhouse

Also pre-stage pip itself so the target can bootstrap from the offline
mirror (the target's system pip 9 cannot install from `manylinux_2_28`):

    pip download pip setuptools wheel \
        --platform manylinux_2_28_x86_64 \
        --python-version 3.11 --abi cp311 \
        --implementation cp --only-binary=:all: \
        -d wheelhouse

Notes:

- **`--only-binary=:all:` is required for cross-platform downloads.** sdists
  cannot be cross-resolved — if pip fell back to source distribution, the
  build would happen on the bastion (wrong platform) instead of the target.
- **If a package is sdist-only**, the command above will fail with
  "Could not find a version that satisfies". You then need to build that
  one wheel on a real RHEL 8 / `manylinux_2_28` host and drop the wheel
  into `wheelhouse/` manually:

        pip wheel <package> -w wheelhouse/

  `rfdetr` is the most likely candidate to need this.

### If the bastion has a wrapper like `python_downloader`

If your bastion exposes package fetching through a wrapper rather than raw
`pip`, the same flags should pass through. Look for an option to forward
arguments to `pip download`, then invoke the wrapper with the flag set
above. The wheelhouse layout on disk is identical regardless of how the
wheels got there: a flat directory of `.whl` files. Validate with the
"Validating the wheelhouse" section below once everything is staged.

## Why Mac wheels cannot be used on Linux

Wheel filenames encode the platform tag, e.g.:

    numpy-1.26.4-cp311-cp311-macosx_11_0_arm64.whl
    numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.whl

`pip` on Linux refuses to install the `macosx_*` wheel. Many libraries
(torch, opencv, numpy) ship compiled extensions, so a Mac wheel cannot be made
to work on Linux by renaming or repacking. Always build the wheelhouse on the
target OS family.

## Why Python minor version must match

Wheels are also tagged by Python ABI (`cp310`, `cp311`, `cp312`, ...). A wheel
built for `cp311` will not install under `cp312`. Use the same
`python3 --version` on the build machine as on the air-gapped target.

## How to inspect the target machine

Run these on the air-gapped target (or read their output from a prior visit):

    python3 --version
    uname -a
    cat /etc/os-release
    nvidia-smi          # CUDA driver and GPU info, if applicable
    ldd --version       # glibc version

Bring those values back to the build machine and match them. Pay particular
attention to:

- Python minor version (e.g. `3.11.6`)
- Architecture (`x86_64` vs `aarch64`)
- glibc version (manylinux tag picks a minimum)
- CUDA driver version (constrains which `cu*` torch build works)

## CPU-only wheelhouse

Use `python3.11` explicitly — match the target's installed Python.

    python3.11 -m venv venv
    source venv/bin/activate
    python -m pip install --upgrade pip setuptools wheel pip-tools
    pip-compile requirements.in -o requirements.txt
    mkdir -p wheelhouse
    pip download -r requirements.txt -d wheelhouse

## CUDA PyTorch wheelhouse (Blackwell — use cu128)

PyTorch CUDA builds live on a separate index URL. For the target's Blackwell
GPU (RTX PRO 5000, sm_120) under driver 580 you need **`cu128` or newer**.
`cu121` and `cu124` install but crash with
`no kernel image is available for execution on the device` at first GPU op.

    pip download torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128 \
        -d wheelhouse

Then download the rest of the dependencies into the same directory:

    pip download -r requirements.txt -d wheelhouse

Tag selection rules of thumb:
- Blackwell (sm_100, sm_120) → `cu128` minimum (`cu129`, `cu130` also work if available)
- Hopper (sm_90) → `cu121`+
- Ampere (sm_80, sm_86) → `cu118`+
- No GPU → `cpu`

Check `https://pytorch.org/get-started/locally/` from a connected machine for
the current matrix and the latest cu-tag PyTorch ships.

## Validating the wheelhouse

Before tar-bundling, confirm the wheelhouse resolves fully offline on the
build host:

    python3.11 -m venv /tmp/validate_venv
    source /tmp/validate_venv/bin/activate
    pip install --no-index --find-links wheelhouse -r requirements.txt
    python -c "import torch, torchvision, cv2; print(torch.__version__, cv2.__version__)"

If that succeeds on the build machine with `--no-index`, the wheelhouse should
also work on the target — provided the target's Python minor version and CUDA
driver match.

## Vendoring source-only dependencies

For packages that are not on PyPI (notably SAM, sometimes specific RF-DETR
forks), build a wheel from source on the connected build machine and drop it
into `wheelhouse/`:

    pip wheel git+https://github.com/facebookresearch/segment-anything.git \
        -w wheelhouse/

If even cloning is not possible from the build host, transfer the source tree
manually and run:

    pip wheel ./segment-anything -w wheelhouse/
