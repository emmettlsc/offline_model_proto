# offline_model_proto

Prototype repo for running object detection + segmentation models
(RF-DETR + SAM-style) on video frames in an **air-gapped Linux** environment.

## Constraints

- Target machine is Linux. **No internet, no Docker, no containers.**
- Files reach the target through a bastion as a single tarball.
- Model weights, videos, virtualenvs, wheelhouses, and outputs are **never
  committed to git**.
- Dev happens on a MacBook; the dependency wheelhouse must be built on a
  Linux machine that matches the target.

## Workflow overview

1. Develop and write scripts on the MacBook.
2. On a Linux build host (matching the target), generate a locked
   `requirements.txt` and a `wheelhouse/` of `.whl` files.
3. Drop model weights into `models/` on the Linux build host.
4. Bundle the repo + wheelhouse + models into
   `offline_model_proto_bundle.tar.gz` via `scripts/build_offline_bundle.sh`.
5. Transfer to the air-gapped target via the bastion.
6. On the target, unpack, create a venv, install from the wheelhouse, run.

## What can be done on the MacBook

- Edit and run all of the scripts on synthetic data and small examples
  (smoke tests, frame extraction).
- Edit `requirements.in` and the documentation.
- Test the RF-DETR script skeleton up to (but not through) the real model
  load — full model integration is best done on the target's environment.

## What must be done on Linux

- Generating `requirements.txt` (the lock file).
- Building `wheelhouse/`.
- Running `scripts/build_offline_bundle.sh` to produce the tarball.

## Repo structure

    offline_model_proto/
      README.md
      .gitignore
      requirements.in
      requirements.txt              # placeholder; regenerate on Linux
      sample_config.yaml
      scripts/
        smoke_test_python.py
        smoke_test_torch.py
        smoke_test_cv.py
        inspect_environment.py
        run_rfdetr_image.py
        run_video_frames.py
        build_offline_bundle.sh
      docs/
        AIRGAP_INSTALL.md
        LINUX_WHEELHOUSE_BUILD.md
        MODEL_WEIGHTS.md

Directories created at runtime (all gitignored, never committed):

    venv/           # local virtualenv
    wheelhouse/     # downloaded .whl files for offline install
    models/         # model checkpoints
    data/           # input images and videos
    outputs/        # script outputs

## Quickstart — MacBook dev

    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.in
    python scripts/smoke_test_python.py
    python scripts/smoke_test_cv.py
    python scripts/inspect_environment.py

The torch wheel installed on macOS here is NOT the one you will ship.
`scripts/smoke_test_torch.py` will work, but you cannot copy this venv to
the target.

## Quickstart — Linux wheelhouse build

Build on RHEL 8 / Rocky 8 / AlmaLinux 8, or in a `manylinux_2_28_x86_64`
container — the target is RHEL 8.10 (glibc 2.28). Build host Python minor
must equal the target's Python minor (`3.11.x`).

If your build host is a non-RHEL bastion (Mac, Ubuntu, a `python_downloader`
proxy), use the cross-platform `pip download` flag set in
`docs/LINUX_WHEELHOUSE_BUILD.md` ("Building the wheelhouse via a bastion")
instead of the recipe below.

    python3.11 -m venv venv
    source venv/bin/activate
    python -m pip install --upgrade pip setuptools wheel pip-tools
    pip-compile requirements.in -o requirements.txt
    mkdir -p wheelhouse
    pip download torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128 \
        -d wheelhouse
    pip download -r requirements.txt -d wheelhouse

The `cu128` index is required for the target's Blackwell GPU (sm_120) —
older `cu121` / `cu124` builds install but crash at first GPU op. Full
detail in `docs/LINUX_WHEELHOUSE_BUILD.md`.

Then place any weights under `models/` and run:

    ./scripts/build_offline_bundle.sh

This produces `offline_model_proto_bundle.tar.gz` ready for transfer.

## Quickstart — air-gapped install

The system Python on the target (RHEL 8.10) is 3.6.8, which is too old.
Install Python 3.11 from RPMs first — see Step 0 in `docs/AIRGAP_INSTALL.md`.

    tar -xzf offline_model_proto_bundle.tar.gz
    cd offline_model_proto
    python3.11 -m venv venv
    source venv/bin/activate
    pip install --no-index --find-links wheelhouse -r requirements.txt
    python scripts/inspect_environment.py
    python scripts/smoke_test_torch.py        # confirms Blackwell + cu128 pairing
    python scripts/smoke_test_cv.py

Full troubleshooting in `docs/AIRGAP_INSTALL.md`.

## Do NOT commit

- model weights (`*.pth`, `*.pt`, `*.safetensors`, `*.onnx`, `*.ckpt`, `*.bin`)
- videos (`*.mp4`, `*.mov`, `*.avi`, `*.mkv`)
- the wheelhouse (`wheelhouse/`, `*.whl`)
- virtualenvs (`venv/`)
- the bundle tarball (`offline_model_proto_bundle.tar.gz`)
- `outputs/`, `data/`, `models/`
