import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor


def segment(proc, model, frame, boxes, device):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = proc(images=rgb, input_boxes=[[list(b) for b in boxes]],
                  return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs, multimask_output=False)
    masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
    return [masks[k, 0].cpu().numpy().astype(bool) for k in range(masks.shape[0])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("sam2_out.mp4"))
    p.add_argument("--detections", type=Path, default=None,
                   help="detector JSON sidecar; segment each box per frame")
    p.add_argument("--device", default="cpu")
    p.add_argument("--box", type=float, nargs=4, default=None,
                   metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()

    if not args.video.exists():
        sys.exit(f"not found: {args.video}")
    if not args.weights.exists():
        sys.exit(f"not found: {args.weights}")

    det_map = None
    if args.detections:
        if not args.detections.exists():
            sys.exit(f"not found: {args.detections}")
        d = json.loads(args.detections.read_text())
        det_map = {f["frame"]: [x["box_xyxy"] for x in f.get("detections", [])]
                   for f in d.get("frames", [])}

    proc = Sam2Processor.from_pretrained(str(args.weights))
    model = Sam2Model.from_pretrained(str(args.weights)).to(args.device).eval()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / max(1, args.stride), (w, h))

    color = np.array([0, 255, 0], dtype=np.uint8)
    frames_out = []
    i = n = 0
    limit = args.max_frames or None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if det_map is not None:
            if i not in det_map:
                i += 1
                continue
            boxes = det_map[i]
        elif i % args.stride == 0:
            boxes = [args.box if args.box else [0.0, 0.0, float(w), float(h)]]
        else:
            i += 1
            continue

        recs = []
        annotated = frame
        if boxes:
            masks = segment(proc, model, frame, boxes, args.device)
            for box, mask in zip(boxes, masks):
                blend = (annotated * 0.5 + color * 0.5).astype(np.uint8)
                annotated = np.where(mask[..., None], blend, annotated)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 1)
                recs.append({"box_xyxy": [float(v) for v in box],
                             "mask_area_px": int(mask.sum())})
        writer.write(annotated)
        frames_out.append({"frame": i, "masks": recs})

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
        "detections": str(args.detections) if args.detections else None,
        "n_frames_processed": n,
        "frames": frames_out,
    }, indent=2))
    print(f"wrote {args.output} ({n} frames)")
    print(f"sidecar {sidecar}")


if __name__ == "__main__":
    main()
