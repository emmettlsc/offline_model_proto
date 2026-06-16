# rfdetr-cpp

C++ RF-DETR ONNX inference over a video. Counterpart to `scripts/rfdetr-pytorch/run.py`.

## Inputs

The ONNX file produced by `tools/export_rfdetr_onnx.py`. Layout: input
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
