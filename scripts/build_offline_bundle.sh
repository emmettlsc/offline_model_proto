#!/usr/bin/env bash
# Build the offline bundle on a LINUX machine that matches the air-gapped target.
#
# Warnings:
#   - This script refuses to run on macOS. Wheelhouses built on Darwin are NOT
#     usable on Linux: wheel tags (e.g. macosx_*_arm64) are platform-locked.
#   - Match the target's Python minor version, glibc, architecture, and CUDA
#     toolkit when building.
#   - PyTorch CUDA wheels need a separate index URL. For the current target
#     (RTX PRO 5000 Blackwell, sm_120, driver 580) you MUST use cu128 or
#     newer — older tags install fine and then crash with
#     "no kernel image is available for execution on the device":
#       pip download torch torchvision \
#         --index-url https://download.pytorch.org/whl/cu128 -d wheelhouse
#   - Model weights are NOT downloaded by this script. Place them under models/
#     before bundling; the directory will be included if it exists.
#
# Usage:
#   ./scripts/build_offline_bundle.sh

set -euo pipefail

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "ERROR: this script must be run on Linux, not macOS." >&2
  echo "Wheels built on Darwin will not install on the Linux target." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

WHEELHOUSE="wheelhouse"
BUNDLE="offline_model_proto_bundle.tar.gz"

if [[ ! -f requirements.txt ]] || grep -q "WARNING: This file is a placeholder" requirements.txt; then
  echo "ERROR: requirements.txt is missing or is still the placeholder." >&2
  echo "Generate a real lock on this Linux host first:" >&2
  echo "  pip install pip-tools && pip-compile requirements.in -o requirements.txt" >&2
  exit 2
fi

mkdir -p "${WHEELHOUSE}"

# Optional venv setup (the script does not require an existing venv):
#   python3 -m venv venv
#   source venv/bin/activate
#   pip install --upgrade pip setuptools wheel pip-tools

echo ">> downloading wheels into ${WHEELHOUSE}/"
python3 -m pip download -r requirements.txt -d "${WHEELHOUSE}"

# Optional: also stage a CUDA-specific torch build into the same wheelhouse.
# For the current Blackwell target use cu128 (or newer):
#
#   python3 -m pip download torch torchvision \
#       --index-url https://download.pytorch.org/whl/cu128 \
#       -d "${WHEELHOUSE}"

INCLUDE=(
  README.md
  requirements.in
  requirements.txt
  sample_config.yaml
  scripts
  docs
  "${WHEELHOUSE}"
)

if [[ -d models ]]; then
  echo ">> including models/ directory in bundle"
  INCLUDE+=(models)
else
  echo "NOTE: models/ not present — bundle will not contain model weights."
  echo "      Stage checkpoints under models/ before running this script if you"
  echo "      want them in the tarball. See docs/MODEL_WEIGHTS.md."
fi

echo ">> creating ${BUNDLE}"
tar -czf "${BUNDLE}" "${INCLUDE[@]}"

echo
echo "OK: ${BUNDLE}"
ls -lh "${BUNDLE}"
