# sam2-cpp

C++ SAM2 ONNX inference over a video / stream. Counterpart to
`scripts/sam2-pytorch/run.py`.

## Inputs

The two ONNX files from
[onnx-community/sam2.1-hiera-base-plus-ONNX](https://huggingface.co/onnx-community/sam2.1-hiera-base-plus-ONNX):

  * `vision_encoder.onnx` (+ `vision_encoder.onnx_data`, ~292 MB)
  * `prompt_encoder_mask_decoder.onnx` (+ `_data`, ~20 MB)

Both `.onnx` + `.onnx_data` pairs must stay together in the same directory.

## Build (Mac / dev)

```
brew install onnxruntime opencv cmake
cmake -B build -S .
cmake --build build -j
```

## Build (RHEL 8.10 / target)

```
sudo dnf install opencv-devel cmake gcc-c++
export ONNXRUNTIME_ROOT=/opt/onnxruntime    # where libonnxruntime.so + headers live
cmake -B build -S .
cmake --build build -j
```

For the Blackwell GPU target supply an `onnxruntime-gpu` build:

```
cmake -B build -S . -DUSE_CUDA=ON
cmake --build build -j
./build/sam2_cpp --cuda ...
```

## Run — file mode (`sam2_cpp`)

```
./build/sam2_cpp \
    --video data/clip.mp4 \
    --encoder models/sam2_onnx/vision_encoder.onnx \
    --decoder models/sam2_onnx/prompt_encoder_mask_decoder.onnx \
    --output out.mp4 \
    --box 100 150 500 450      # optional, default whole frame
    --stride 5
```

Writes an annotated video (mask overlaid in green, prompt box drawn in blue)
and a JSON sidecar with per-frame `iou_score` and `mask_area_px`.

## Run — stream mode (`sam2_stream`)

```
./build/sam2_stream \
    --input "udp://0.0.0.0:1234" \
    --encoder vision_encoder.onnx \
    --decoder prompt_encoder_mask_decoder.onnx \
    [--save out.mp4]            # also record annotated frames
    [--stream-out URL]          # also re-broadcast
    [--no-print-json]           # silence stdout ndjson
    [--display]                 # local preview
    [--box X1 Y1 X2 Y2]         # default whole frame
    [--cuda]
```

Default sink is one JSON object per processed frame to stdout:

    {"frame":N,"box_xyxy":[...],"iou_score":F,"mask_area_px":I}

SIGINT (Ctrl-C) exits cleanly and flushes any `--save` output.
