import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import Sam3Model, Sam3Processor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True,
                   help="HF weights dir (e.g. ./models/sam3/)")
    p.add_argument("--text", required=True,
                   help="Concept to segment, e.g. 'person', 'car', 'yellow school bus'")
    p.add_argument("--output", type=Path, default=Path("sam3_out.mp4"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()

    if not args.video.exists():
        sys.exit(f"not found: {args.video}")
    if not args.weights.exists():
        sys.exit(f"not found: {args.weights}")

    proc = Sam3Processor.from_pretrained(str(args.weights))
    model = Sam3Model.from_pretrained(str(args.weights)).to(args.device).eval()

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
            inputs = proc(images=rgb, text=args.text,
                          return_tensors="pt").to(args.device)
            with torch.inference_mode():
                out = model(**inputs)
            r = proc.post_process_instance_segmentation(
                out, threshold=args.threshold, mask_threshold=0.5,
                target_sizes=inputs["original_sizes"].tolist())[0]

            annotated = frame.copy()
            color = np.array([0, 255, 0], dtype=np.uint8)
            scores = r["scores"].cpu().numpy() if len(r["masks"]) else np.array([])
            for mask in r["masks"]:
                m = mask.cpu().numpy().astype(bool)
                blend = (annotated * 0.5 + color * 0.5).astype(np.uint8)
                annotated = np.where(m[..., None], blend, annotated)
            cv2.putText(annotated, f'"{args.text}" x{len(r["masks"])}',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(annotated)

            frames_out.append({
                "frame": i,
                "concept": args.text,
                "num_instances": len(r["masks"]),
                "scores": [float(s) for s in scores],
            })
            n += 1
            if n % 5 == 0:
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
        "concept": args.text,
        "n_frames_processed": n,
        "frames": frames_out,
    }, indent=2))
    print(f"wrote {args.output} ({n} frames)")
    print(f"sidecar {sidecar}")


if __name__ == "__main__":
    main()
