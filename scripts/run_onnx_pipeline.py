"""Run RF-DETR + SAM3 over a video using ONNX weights. Bare pipeline.

For each sampled frame: detect with RF-DETR, prompt SAM3 with each box,
overlay masks and boxes, write the annotated frame to an output video.

Adjust the preprocessing constants and the run_* functions if your specific
exports use different input sizes, output orderings, or decoder input names.
The session() helper prints input/output names on load so you can spot
mismatches quickly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parent.parent

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ADJUST IF YOUR EXPORTS DIFFER:
RFDETR_INPUT_SIZE = 560   # Common Roboflow RF-DETR export size
SAM_INPUT_SIZE    = 1024  # Standard SAM image encoder canonical input


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RF-DETR + SAM3 video pipeline (ONNX)")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--rfdetr-onnx", type=Path, required=True)
    p.add_argument("--sam3-encoder-onnx", type=Path, required=True,
                   help="SAM3 image encoder ONNX")
    p.add_argument("--sam3-decoder-onnx", type=Path, required=True,
                   help="SAM3 prompt+mask decoder ONNX")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "outputs" / "onnx_pipeline.mp4")
    p.add_argument("--providers", nargs="+",
                   default=["CUDAExecutionProvider", "CPUExecutionProvider"])
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


# --- session loader ---------------------------------------------------------

def session(path: Path, providers) -> ort.InferenceSession:
    sess = ort.InferenceSession(str(path), providers=providers)
    inputs = [(i.name, i.shape) for i in sess.get_inputs()]
    outputs = [(o.name, o.shape) for o in sess.get_outputs()]
    print(f"loaded {path.name}")
    print(f"  inputs : {inputs}")
    print(f"  outputs: {outputs}")
    return sess


# --- RF-DETR ----------------------------------------------------------------

def preprocess_rfdetr(frame_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(frame_bgr, (RFDETR_INPUT_SIZE, RFDETR_INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[None]  # (1, 3, H, W)


def run_rfdetr(sess: ort.InferenceSession, frame_bgr: np.ndarray, conf: float):
    h, w = frame_bgr.shape[:2]
    x = preprocess_rfdetr(frame_bgr)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: x})
    # ADJUST IF YOUR EXPORT DIFFERS: typical Roboflow ONNX export emits
    # 3 outputs in this order: boxes (xyxy in input scale), scores, labels.
    # Re-order or postprocess (e.g. softmax + cxcywh→xyxy) if yours differs.
    boxes  = np.atleast_2d(np.squeeze(outputs[0])).astype(np.float32)
    scores = np.squeeze(outputs[1]).reshape(-1).astype(np.float32)
    labels = np.squeeze(outputs[2]).reshape(-1).astype(int)
    keep = scores >= conf
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if boxes.size:
        sx, sy = w / RFDETR_INPUT_SIZE, h / RFDETR_INPUT_SIZE
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
    return boxes, scores, labels


# --- SAM3 -------------------------------------------------------------------

def preprocess_sam_image(frame_bgr: np.ndarray):
    # Letterbox-resize so longest side == SAM_INPUT_SIZE, pad to square.
    h, w = frame_bgr.shape[:2]
    scale = SAM_INPUT_SIZE / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(frame_bgr, (nw, nh))
    canvas = np.zeros((SAM_INPUT_SIZE, SAM_INPUT_SIZE, 3), dtype=np.uint8)
    canvas[:nh, :nw] = img
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return rgb.transpose(2, 0, 1)[None], scale


def run_sam(encoder: ort.InferenceSession, decoder: ort.InferenceSession,
            frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if len(boxes_xyxy) == 0:
        return np.zeros((0, h, w), dtype=bool)

    x, scale = preprocess_sam_image(frame_bgr)
    enc_in = encoder.get_inputs()[0].name
    embeddings = encoder.run(None, {enc_in: x})[0]

    masks = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        # Encode a box as two point prompts: top-left (label 2) + bottom-right (label 3).
        # Standard SAM ONNX decoder convention; SAM3 may differ — check input names below.
        coords = np.array([[[x1 * scale, y1 * scale],
                            [x2 * scale, y2 * scale]]], dtype=np.float32)
        labels = np.array([[2, 3]], dtype=np.float32)
        feed = {
            "image_embeddings": embeddings,
            "point_coords":     coords,
            "point_labels":     labels,
            "mask_input":       np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input":   np.zeros((1,), dtype=np.float32),
            "orig_im_size":     np.array([h, w], dtype=np.float32),
        }
        # ADJUST IF YOUR SAM3 DECODER INPUT NAMES DIFFER: the print from session()
        # at load time shows the actual names; rename the keys above to match.
        out = decoder.run(None, feed)
        mask_logits = out[0]  # typically (1, n_masks, H, W); we asked for 1
        mask = mask_logits[0, 0] > 0.0
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        masks.append(mask)
    return np.stack(masks)


# --- visualization & results ------------------------------------------------

def overlay(frame_bgr: np.ndarray, boxes: np.ndarray, masks: np.ndarray,
            scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = frame_bgr.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)
    for m in masks:
        blend = (out * 0.5 + color * 0.5).astype(np.uint8)
        out = np.where(m[..., None], blend, out)
    for (x1, y1, x2, y2), s, c in zip(boxes.astype(int), scores, labels):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(out, f"{int(c)}:{s:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def mask_summary(mask: np.ndarray):
    """Return (area_px, bbox_xyxy) for a boolean mask. Compact, verifiable."""
    if mask.size == 0 or not mask.any():
        return 0, [0, 0, 0, 0]
    ys, xs = np.where(mask)
    return int(mask.sum()), [int(xs.min()), int(ys.min()),
                             int(xs.max()), int(ys.max())]


# --- main loop --------------------------------------------------------------

def main() -> int:
    args = parse_args()
    for path in (args.video, args.rfdetr_onnx,
                 args.sam3_encoder_onnx, args.sam3_decoder_onnx):
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
    detector = session(args.rfdetr_onnx, args.providers)
    enc      = session(args.sam3_encoder_onnx, args.providers)
    dec      = session(args.sam3_decoder_onnx, args.providers)
    print("running ...")

    frame_records: list[dict] = []
    detect_seconds = 0.0
    segment_seconds = 0.0
    frame_idx = 0
    processed = 0
    total_detections = 0
    limit = args.max_frames or None
    run_started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.frame_stride == 0:
                t0 = time.perf_counter()
                boxes, scores, labels = run_rfdetr(
                    detector, frame, args.confidence_threshold)
                t1 = time.perf_counter()
                masks = run_sam(enc, dec, frame, boxes)
                t2 = time.perf_counter()
                detect_seconds += (t1 - t0)
                segment_seconds += (t2 - t1)

                writer.write(overlay(frame, boxes, masks, scores, labels))

                dets = []
                for i, (box, score, cls) in enumerate(zip(boxes, scores, labels)):
                    mask = masks[i] if i < len(masks) else np.zeros(frame.shape[:2], bool)
                    area, mbbox = mask_summary(mask)
                    dets.append({
                        "box_xyxy": [float(v) for v in box],
                        "score": float(score),
                        "class_id": int(cls),
                        "mask_area_px": area,
                        "mask_bbox_xyxy": mbbox,
                    })
                frame_records.append({
                    "video_frame_index": frame_idx,
                    "detections": dets,
                })
                total_detections += len(dets)

                processed += 1
                if processed % 20 == 0:
                    print(f"  processed {processed} frames")
                if limit and processed >= limit:
                    break
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - run_started

    sidecar = {
        "schema": "offline_model_proto.detections.v1",
        "backend": "onnx",
        "video": {
            "path": str(args.video),
            "fps": float(fps),
            "width": w,
            "height": h,
        },
        "models": {
            "detector": str(args.rfdetr_onnx),
            "segmenter_encoder": str(args.sam3_encoder_onnx),
            "segmenter_decoder": str(args.sam3_decoder_onnx),
        },
        "params": {
            "confidence_threshold": args.confidence_threshold,
            "frame_stride": args.frame_stride,
            "max_frames": args.max_frames,
            "providers": args.providers,
        },
        "summary": {
            "frames_processed": processed,
            "total_detections": total_detections,
            "elapsed_seconds": round(elapsed, 3),
            "detect_seconds_total": round(detect_seconds, 3),
            "segment_seconds_total": round(segment_seconds, 3),
            "throughput_fps": round(processed / elapsed, 3) if elapsed > 0 else None,
        },
        "frames": frame_records,
    }
    sidecar_path = args.output.with_name(args.output.stem + "_detections.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    print(f"OK: wrote {args.output} ({processed} frames, {total_detections} detections)")
    print(f"    sidecar: {sidecar_path}")
    if elapsed > 0:
        print(f"    timing : detect={detect_seconds:.2f}s segment={segment_seconds:.2f}s "
              f"total={elapsed:.2f}s ({processed / elapsed:.2f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
