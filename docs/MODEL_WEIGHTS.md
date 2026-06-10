# Model Weights

## Where to put them

All model checkpoints belong under `models/` at the repo root:

    models/
      rfdetr_weights.pth
      sam_vit_h_4b8939.pth
      ...

`models/` is in `.gitignore`. **Do not commit weights to git.**

## Usage

Scripts take weight paths as explicit arguments. Example:

    python scripts/run_pytorch_pipeline.py \
        --video data/clip.mp4 \
        --rfdetr-weights models/rfdetr.pth \
        --sam3-weights models/sam3/

    python scripts/run_onnx_pipeline.py \
        --video data/clip.mp4 \
        --rfdetr-onnx models/rfdetr.onnx \
        --sam3-encoder-onnx models/sam3_encoder.onnx \
        --sam3-decoder-onnx models/sam3_decoder.onnx

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

RF-DETR is published by Roboflow as the `rfdetr` PyPI package. Confirm that
the wheelhouse contains the `rfdetr` wheel and its dependencies before the
offline install. If the package fails to resolve, vendor the upstream source
as a sibling directory and `pip install -e ./rfdetr_src`, or build a wheel on
the connected host with `pip wheel ./rfdetr_src -w wheelhouse/`.

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
