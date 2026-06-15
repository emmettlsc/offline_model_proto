import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True,
                   help="HF weights dir (e.g. ./models/rfdetr-medium/)")
    p.add_argument("--output", type=Path, default=Path("rfdetr_out.mp4"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()

    if not args.video.exists():
        sys.exit(f"not found: {args.video}")
    if not args.weights.exists():
        sys.exit(f"not found: {args.weights}")

    proc = AutoImageProcessor.from_pretrained(str(args.weights))
    model = AutoModelForObjectDetection.from_pretrained(str(args.weights)).to(args.device).eval()
    id2label = model.config.id2label

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / max(1, args.stride), (w, h))

    frames_out = []
    i = n = 0
    limit = args.max_frames or None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = proc(images=rgb, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                out = model(**inputs)
            r = proc.post_process_object_detection(
                out, threshold=args.conf, target_sizes=torch.tensor([[h, w]]))[0]
            boxes = r["boxes"].cpu().numpy()
            scores = r["scores"].cpu().numpy()
            labels = r["labels"].cpu().numpy()

            for box, s, lbl in zip(boxes.astype(int), scores, labels):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"{id2label[int(lbl)]}:{s:.2f}",
                            (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)

            frames_out.append({
                "frame": i,
                "detections": [
                    {"box_xyxy": [float(v) for v in b],
                     "score": float(s),
                     "label": id2label[int(lbl)]}
                    for b, s, lbl in zip(boxes, scores, labels)
                ],
            })
            n += 1
            if n % 10 == 0:
                print(f"  {n} frames")
            if limit and n >= limit:
                break
        i += 1
    cap.release()
    writer.release()

    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "video": str(args.video),
        "weights": str(args.weights),
        "n_frames_processed": n,
        "frames": frames_out,
    }, indent=2))
    print(f"wrote {args.output} ({n} frames)")
    print(f"sidecar {sidecar}")


if __name__ == "__main__":
    main()
