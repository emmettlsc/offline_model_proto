#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "linux only" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

WHEELHOUSE="wheelhouse"
BUNDLE="bundle.tar.gz"

if [[ ! -f requirements.txt ]]; then
  echo "missing requirements.txt — run: pip-compile requirements.in -o requirements.txt" >&2
  exit 2
fi

mkdir -p "${WHEELHOUSE}"

python3 -m pip download -r requirements.txt -d "${WHEELHOUSE}"

INCLUDE=(requirements.in requirements.txt scripts "${WHEELHOUSE}")
[[ -d models ]] && INCLUDE+=(models)

tar -czf "${BUNDLE}" "${INCLUDE[@]}"
ls -lh "${BUNDLE}"
