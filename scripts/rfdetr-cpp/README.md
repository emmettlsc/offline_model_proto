# rfdetr-cpp

C++ RF-DETR ONNX inference over a video. Counterpart to `scripts/rfdetr-pytorch/run.py`.

## Inputs

The ONNX file produced by `scripts/rfdetr-onnx-convert/convert.py`. Layout: input
`pixel_values [1, 3, 576, 576]`, three outputs `(boxes, scores, labels)` at
576-input scale. The `.onnx` file and its `.onnx.data` external-weights
sibling must live in the same directory.

## Build (Mac / dev)

```
brew install onnxruntime opencv cmake
cmake -B build -S .
cmake --build build -j
```

## Build (RHEL 8.10 / target)

```
sudo dnf install opencv-devel cmake gcc-c++   # OpenCV from EPEL or AppStream
# ONNX Runtime: place libonnxruntime.so + headers somewhere; point ONNXRUNTIME_ROOT
export ONNXRUNTIME_ROOT=/opt/onnxruntime
cmake -B build -S .
cmake --build build -j
```

For the Blackwell GPU target, supply an `onnxruntime-gpu` build (linked against
CUDA 12.8+) and pass `--cuda` at runtime:

```
cmake -B build -S . -DUSE_CUDA=ON
cmake --build build -j
./build/rfdetr_cpp --cuda --video clip.mov --model rfdetr.onnx
```

## Run

```
./build/rfdetr_cpp \
    --video data/clip.mp4 \
    --model models/rfdetr.onnx \
    --output out.mp4 \
    --conf 0.4 \
    --stride 5
```

Writes the annotated video and a JSON sidecar (same path with `.json`
extension). Accepts `.mp4`, `.mov`, `.ts`, any container OpenCV can decode.

## Streaming variant — `rfdetr_stream`

For consuming a network video stream and emitting detections in real time.
Takes any URL OpenCV's FFmpeg backend can open: `rtsp://`, `rtmp://`,
`http://`, `udp://`, `tcp://`, or a plain file path.

```
./build/rfdetr_stream \
    --input "udp://0.0.0.0:1234" \
    --model models/rfdetr.onnx \
    [--save out.mp4]        # also record annotated frames
    [--stream-out rtsp://]  # also re-broadcast (OpenCV FFmpeg backend)
    [--no-print-json]       # silence per-frame ndjson on stdout
    [--display]             # local preview window (dev only)
    [--cuda]
```

Default behaviour: one JSON object per processed frame on stdout (ndjson),
suitable for piping to another process. Per-line shape:

    {"frame":N,"detections":[{"box_xyxy":[...],"score":F,"label_id":I,"label":"S"},...]}

Ctrl-C (SIGINT) exits cleanly so partially-recorded files are flushed.

### Quick local test against a UDP broadcast

```
# producer: re-encode + broadcast a local file to UDP at real-time pace
ffmpeg -re -stream_loop -1 -i clip.mp4 \
    -c:v libx264 -preset ultrafast -tune zerolatency -g 15 -keyint_min 15 \
    -an -f mpegts udp://127.0.0.1:1234 &

# consumer: detection + JSON stdout
./build/rfdetr_stream --input udp://127.0.0.1:1234 --model rfdetr.onnx \
    --stride 10 --conf 0.4
```

Short GOP (`-g 15`) is important for clean joins — without it the first
frames hit H.264 PPS warnings until the next keyframe.
