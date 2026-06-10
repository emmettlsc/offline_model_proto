import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent.parent

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RFDETR_SIZE = 560
SAM_SIZE = 1024


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--rfdetr-onnx", type=Path, required=True)
    p.add_argument("--sam3-encoder-onnx", type=Path, required=True)
    p.add_argument("--sam3-decoder-onnx", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "onnx_pipeline.mp4")
    p.add_argument("--providers", nargs="+",
                   default=["CUDAExecutionProvider", "CPUExecutionProvider"])
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def session(path, providers):
    s = ort.InferenceSession(str(path), providers=providers)
    print(f"loaded {path.name}")
    print(f"  in:  {[(i.name, i.shape) for i in s.get_inputs()]}")
    print(f"  out: {[(o.name, o.shape) for o in s.get_outputs()]}")
    return s


def prep_rfdetr(frame):
    img = cv2.resize(frame, (RFDETR_SIZE, RFDETR_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((img - MEAN) / STD).transpose(2, 0, 1)[None]


def detect(sess, frame, conf):
    h, w = frame.shape[:2]
    out = sess.run(None, {sess.get_inputs()[0].name: prep_rfdetr(frame)})
    boxes = np.atleast_2d(np.squeeze(out[0])).astype(np.float32)
    scores = np.squeeze(out[1]).reshape(-1).astype(np.float32)
    classes = np.squeeze(out[2]).reshape(-1).astype(int)
    keep = scores >= conf
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    if boxes.size:
        boxes[:, [0, 2]] *= w / RFDETR_SIZE
        boxes[:, [1, 3]] *= h / RFDETR_SIZE
    return boxes, scores, classes


def prep_sam(frame):
    h, w = frame.shape[:2]
    scale = SAM_SIZE / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((SAM_SIZE, SAM_SIZE, 3), dtype=np.uint8)
    canvas[:nh, :nw] = img
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None], scale


def segment(enc, dec, frame, boxes):
    h, w = frame.shape[:2]
    if len(boxes) == 0:
        return np.zeros((0, h, w), dtype=bool)
    x, scale = prep_sam(frame)
    emb = enc.run(None, {enc.get_inputs()[0].name: x})[0]

    masks = []
    for x1, y1, x2, y2 in boxes:
        feed = {
            "image_embeddings": emb,
            "point_coords": np.array([[[x1 * scale, y1 * scale],
                                       [x2 * scale, y2 * scale]]], dtype=np.float32),
            "point_labels": np.array([[2, 3]], dtype=np.float32),
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros((1,), dtype=np.float32),
            "orig_im_size": np.array([h, w], dtype=np.float32),
        }
        logits = dec.run(None, feed)[0]
        m = logits[0, 0] > 0.0
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        masks.append(m)
    return np.stack(masks)


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
    for p in (args.video, args.rfdetr_onnx, args.sam3_encoder_onnx, args.sam3_decoder_onnx):
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
    det = session(args.rfdetr_onnx, args.providers)
    enc = session(args.sam3_encoder_onnx, args.providers)
    dec = session(args.sam3_decoder_onnx, args.providers)

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
                boxes, scores, classes = detect(det, frame, args.conf)
                t1 = time.perf_counter()
                masks = segment(enc, dec, frame, boxes)
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
        "backend": "onnx",
        "video": {"path": str(args.video), "fps": float(fps), "width": w, "height": h},
        "models": {
            "detector": str(args.rfdetr_onnx),
            "segmenter_encoder": str(args.sam3_encoder_onnx),
            "segmenter_decoder": str(args.sam3_decoder_onnx),
        },
        "params": {"conf": args.conf, "stride": args.stride,
                   "max_frames": args.max_frames, "providers": args.providers},
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
