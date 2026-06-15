# Model Weights

## Where to put them

All model checkpoints belong under `models/` at the repo root:

    models/
      rfdetr_weights.pth
      sam_vit_h_4b8939.pth
      ...

`models/` is in `.gitignore`. **Do not commit weights to git.**

## Usage

Each model has its own self-contained directory under `scripts/` with a
`run.py` and a `requirements.txt`. All take `--video` and `--weights` (a
directory path for HF transformers weights):

    python scripts/rfdetr-pytorch/run.py \
        --video data/clip.mp4 \
        --weights models/rfdetr/

    python scripts/sam2-pytorch/run.py \
        --video data/clip.mp4 \
        --weights models/sam2/

    python scripts/sam3-pytorch/run.py \
        --video data/clip.mp4 \
        --weights models/sam3/ \
        --text "person"

Weights from HF are a directory containing `config.json`, `model.safetensors`,
and `preprocessor_config.json` (SAM3 additionally has tokenizer files for
its CLIP text encoder). Pass the directory path, not a single file.

## Air-gapped constraints

Scripts in this repo must not auto-download from Hugging Face, GitHub, or any
other network location. The target has no internet.

If you adopt a library that tries to fetch checkpoints on import or on first
call, you must:

  - pre-stage the checkpoint into the library's expected cache location, **or**
  - pass the local path explicitly via the library's API, **and**
  - set the library's offline mode env vars where applicable:

        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1

## Transferring checkpoints and source

If a model's package or source repo is not pip-installable from a wheel, you
will need to transfer all of:

  - the checkpoint file (e.g. `.pth`, `.safetensors`)
  - any required config files (e.g. `config.json`, `model_config.yaml`)
  - the source repository archive, if the model code lives in a GitHub repo
    rather than a PyPI package (true for Meta's `segment-anything` at time of
    writing)

Stage these alongside `models/` and bundle them into
`offline_model_proto_bundle.tar.gz` before transferring.

## RF-DETR specifics

The PyTorch pipeline loads RF-DETR via Hugging Face transformers
(`AutoModelForObjectDetection` + `AutoImageProcessor`) from the downloaded
HF weights directory. No `rfdetr` PyPI package is required — `transformers`
is the only Python dependency for detection.

## SAM specifics

Meta's Segment Anything (`segment-anything`) is typically installed from
GitHub source rather than PyPI. To use SAM offline:

  1. On a connected host, build a wheel:

         pip wheel git+https://github.com/facebookresearch/segment-anything.git \
             -w wheelhouse/

     Or clone, then:

         pip wheel ./segment-anything -w wheelhouse/

  2. Transfer SAM checkpoints (e.g. `sam_vit_h_4b8939.pth`) into `models/`.
  3. Add `segment-anything` (or whatever the wheel's distribution name turns
     out to be) to `requirements.in` and regenerate `requirements.txt`.
