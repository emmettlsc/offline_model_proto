import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


SIZE = 1024
LOW_MASK = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(frame):
    img = cv2.resize(frame, (SIZE, SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--output", default="sam2_out.mp4")
    ap.add_argument("--box", type=float, nargs=4, default=None,
                    metavar=("X1", "Y1", "X2", "Y2"))
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    enc = ort.InferenceSession(args.encoder, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(args.decoder, providers=["CPUExecutionProvider"])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    box = list(args.box) if args.box else [0.0, 0.0, float(w), float(h)]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / max(1, args.stride), (w, h))

    sx, sy = SIZE / w, SIZE / h

    frames = []
    i = n = 0
    limit = args.max_frames or None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.stride == 0:
            x = preprocess(frame)
            emb0, emb1, emb2 = enc.run(None, {"pixel_values": x})

            scaled_box = np.array([[[box[0] * sx, box[1] * sy,
                                     box[2] * sx, box[3] * sy]]], dtype=np.float32)
            iou, masks, _obj = dec.run(None, {
                "input_points": np.zeros((1, 1, 0, 2), dtype=np.float32),
                "input_labels": np.zeros((1, 1, 0), dtype=np.int64),
                "input_boxes": scaled_box,
                "image_embeddings.0": emb0,
                "image_embeddings.1": emb1,
                "image_embeddings.2": emb2,
            })

            best = int(np.argmax(iou[0, 0]))
            mask_low = masks[0, 0, best]              # (256, 256)
            mask_orig = cv2.resize(mask_low, (w, h))
            mask_bin = (mask_orig > 0).astype(np.uint8) * 255

            green = np.full_like(frame, (0, 255, 0))
            blended = (frame * 0.5 + green * 0.5).astype(np.uint8)
            np.copyto(frame, blended, where=mask_bin[..., None].astype(bool))
            cv2.rectangle(frame,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (255, 0, 0), 1)
            writer.write(frame)

            mask_area = int((mask_bin > 0).sum())
            frames.append({
                "frame": i,
                "box_xyxy": [float(v) for v in box],
                "iou_score": float(iou[0, 0, best]),
                "mask_area_px": mask_area,
            })
            n += 1
            if n % 5 == 0:
                print(f"  {n} frames")
            if limit and n >= limit:
                break
        i += 1
    cap.release()
    writer.release()

    sidecar = Path(args.output).with_suffix(".json")
    sidecar.write_text(json.dumps(
        {"video": args.video, "encoder": args.encoder, "decoder": args.decoder,
         "frames": frames}, indent=2))
    print(f"wrote {args.output} ({n} frames)")
    print(f"sidecar {sidecar}")


if __name__ == "__main__":
    main()
