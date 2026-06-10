"""Run RF-DETR + SAM3 over a video using PyTorch weights. Bare pipeline.

For each sampled frame: detect with RF-DETR, prompt SAM3 with each box,
overlay masks and boxes, write the annotated frame to an output video.

Adjust load_detector / load_segmenter if your weight files have a different
loader signature than the defaults below.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RF-DETR + SAM3 video pipeline (PyTorch)")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--rfdetr-weights", type=Path, required=True,
                   help="Path to RF-DETR checkpoint (.pth)")
    p.add_argument("--sam3-weights", type=Path, required=True,
                   help="Path to HF-format SAM3 weights directory (config.json + "
                        "safetensors), or single checkpoint depending on your loader")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "outputs" / "pytorch_pipeline.mp4")
    p.add_argument("--device", default="cuda")
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Process every Nth frame (1 = every frame)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after this many processed frames (0 = no limit)")
    return p.parse_args()


# --- model loaders -----------------------------------------------------------

def load_detector(weights: Path):
    # ADJUST IF YOUR FORK DIFFERS: standard Roboflow rfdetr loader.
    from rfdetr import RFDETRBase
    return RFDETRBase(pretrain_weights=str(weights))


def load_segmenter(weights: Path, device: str):
    # ADJUST IF YOUR WEIGHTS ARE NOT HF-FORMAT: for a .pt from Meta's native
    # sam3 source repo, swap to that repo's build_sam3() + predictor instead.
    import torch
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(str(weights))
    model = AutoModel.from_pretrained(str(weights), torch_dtype=torch.float16)
    model = model.to(device).eval()
    return model, processor


# --- model runners -----------------------------------------------------------

def detect(model, frame_bgr: np.ndarray, conf: float):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    dets = model.predict(rgb, threshold=conf)  # supervision.Detections
    boxes = np.asarray(dets.xyxy, dtype=np.float32)
    class_ids = np.asarray(
        getattr(dets, "class_id", np.zeros(len(boxes))), dtype=int
    )
    return boxes, class_ids


def segment(seg, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray, device: str) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if len(boxes_xyxy) == 0:
        return np.zeros((0, h, w), dtype=bool)
    import torch
    model, processor = seg
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # HF SAM-family processors expect input_boxes shape (batch, n_boxes, 4) in xyxy.
    input_boxes = [[b.tolist() for b in boxes_xyxy]]
    inputs = processor(images=rgb, input_boxes=input_boxes,
                       return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]  # batch=1 → first (and only) entry
    return masks.squeeze(1).numpy().astype(bool)  # (n_boxes, H, W)


# --- visualization -----------------------------------------------------------

def overlay(frame_bgr: np.ndarray, boxes: np.ndarray, masks: np.ndarray,
            class_ids: np.ndarray) -> np.ndarray:
    out = frame_bgr.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)
    for m in masks:
        blend = (out * 0.5 + color * 0.5).astype(np.uint8)
        out = np.where(m[..., None], blend, out)
    for (x1, y1, x2, y2), c in zip(boxes.astype(int), class_ids):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(out, str(int(c)), (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return out


# --- main loop ---------------------------------------------------------------

def main() -> int:
    args = parse_args()
    for path in (args.video, args.rfdetr_weights, args.sam3_weights):
        if not path.exists():
            print(f"ERROR: not found: {path}", file=sys.stderr)
            return 2

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {args.video}", file=sys.stderr)
        return 3
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / max(1, args.frame_stride),
        (w, h),
    )

    print("loading models ...")
    detector = load_detector(args.rfdetr_weights)
    segmenter = load_segmenter(args.sam3_weights, args.device)
    print("running ...")

    frame_idx = 0
    processed = 0
    limit = args.max_frames or None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.frame_stride == 0:
                boxes, class_ids = detect(detector, frame, args.confidence_threshold)
                masks = segment(segmenter, frame, boxes, args.device)
                writer.write(overlay(frame, boxes, masks, class_ids))
                processed += 1
                if processed % 20 == 0:
                    print(f"  processed {processed} frames")
                if limit and processed >= limit:
                    break
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    print(f"OK: wrote {args.output} ({processed} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
