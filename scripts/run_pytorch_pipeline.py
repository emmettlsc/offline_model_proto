import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--rfdetr-weights", type=Path, default=None,
                   help="HF weights dir. Omit to skip detection and prompt SAM2 with the whole frame.")
    p.add_argument("--sam2-weights", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "pytorch_pipeline.mp4")
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def load_detector(weights, device):
    from transformers import AutoModelForObjectDetection, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(str(weights))
    model = AutoModelForObjectDetection.from_pretrained(str(weights)).to(device).eval()
    return model, proc


def load_segmenter(weights, device):
    from transformers import Sam2Model, Sam2Processor
    proc = Sam2Processor.from_pretrained(str(weights))
    model = Sam2Model.from_pretrained(str(weights)).to(device).eval()
    return model, proc


def detect(det, frame, conf, device):
    import torch
    model, proc = det
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    inputs = proc(images=rgb, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs)
    results = proc.post_process_object_detection(
        out, threshold=conf, target_sizes=torch.tensor([[h, w]]))[0]
    return (
        results["boxes"].cpu().numpy().astype(np.float32),
        results["scores"].cpu().numpy().astype(np.float32),
        results["labels"].cpu().numpy().astype(int),
    )


def segment(seg, frame, boxes, device):
    h, w = frame.shape[:2]
    if len(boxes) == 0:
        return np.zeros((0, h, w), dtype=bool)
    import torch
    model, proc = seg
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = proc(images=rgb, input_boxes=[[b.tolist() for b in boxes]],
                  return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs, multimask_output=False)
    masks = proc.post_process_masks(
        out.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]
    return masks.squeeze(1).numpy().astype(bool)


def draw(frame, boxes, masks, scores, classes):
    out = frame.copy()
    green = np.array([0, 255, 0], dtype=np.uint8)
    for m in masks:
        blend = (out * 0.5 + green * 0.5).astype(np.uint8)
        out = np.where(m[..., None], blend, out)
    for (x1, y1, x2, y2), s, c in zip(boxes.astype(int), scores, classes):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(out, f"{int(c)}:{s:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def mask_stats(m):
    if not m.any():
        return 0, [0, 0, 0, 0]
    ys, xs = np.where(m)
    return int(m.sum()), [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def main():
    args = parse_args()
    must_exist = [args.video, args.sam2_weights]
    if args.rfdetr_weights is not None:
        must_exist.append(args.rfdetr_weights)
    for p in must_exist:
        if not p.exists():
            sys.exit(f"not found: {p}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / max(1, args.stride), (w, h))

    print("loading models")
    detector = load_detector(args.rfdetr_weights, args.device) if args.rfdetr_weights else None
    segmenter = load_segmenter(args.sam2_weights, args.device)
    if detector is None:
        print("no detector — prompting SAM2 with the whole frame per sampled frame")

    frames = []
    t_det = t_seg = 0.0
    n = total = 0
    limit = args.max_frames or None
    start = time.perf_counter()
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % args.stride == 0:
                t0 = time.perf_counter()
                if detector is not None:
                    boxes, scores, classes = detect(detector, frame, args.conf, args.device)
                else:
                    fh, fw = frame.shape[:2]
                    boxes = np.array([[0, 0, fw, fh]], dtype=np.float32)
                    scores = np.array([1.0], dtype=np.float32)
                    classes = np.array([0], dtype=int)
                t1 = time.perf_counter()
                masks = segment(segmenter, frame, boxes, args.device)
                t2 = time.perf_counter()
                t_det += t1 - t0
                t_seg += t2 - t1

                writer.write(draw(frame, boxes, masks, scores, classes))

                dets = []
                for j, (b, s, c) in enumerate(zip(boxes, scores, classes)):
                    m = masks[j] if j < len(masks) else np.zeros(frame.shape[:2], bool)
                    area, bbox = mask_stats(m)
                    dets.append({
                        "box_xyxy": [float(v) for v in b],
                        "score": float(s),
                        "class_id": int(c),
                        "mask_area_px": area,
                        "mask_bbox_xyxy": bbox,
                    })
                frames.append({"video_frame_index": i, "detections": dets})
                total += len(dets)
                n += 1
                if n % 20 == 0:
                    print(f"  {n} frames")
                if limit and n >= limit:
                    break
            i += 1
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - start
    sidecar = args.output.with_name(args.output.stem + "_detections.json")
    sidecar.write_text(json.dumps({
        "schema": "offline_model_proto.detections.v1",
        "backend": "pytorch",
        "video": {"path": str(args.video), "fps": float(fps), "width": w, "height": h},
        "models": {
            "detector": str(args.rfdetr_weights) if args.rfdetr_weights else None,
            "segmenter": str(args.sam2_weights),
        },
        "params": {"conf": args.conf, "stride": args.stride,
                   "max_frames": args.max_frames, "device": args.device},
        "summary": {
            "frames_processed": n,
            "total_detections": total,
            "elapsed_seconds": round(elapsed, 3),
            "detect_seconds_total": round(t_det, 3),
            "segment_seconds_total": round(t_seg, 3),
            "throughput_fps": round(n / elapsed, 3) if elapsed else None,
        },
        "frames": frames,
    }, indent=2))

    print(f"wrote {args.output} ({n} frames, {total} dets)")
    print(f"sidecar {sidecar}")
    if elapsed:
        print(f"detect {t_det:.2f}s segment {t_seg:.2f}s "
              f"total {elapsed:.2f}s ({n / elapsed:.2f} fps)")


if __name__ == "__main__":
    main()
