import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


SIZE = 640

COCO = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
    "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def preprocess(frame):
    img = cv2.resize(frame, (SIZE, SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default="rtdetr_out.mp4")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / max(1, args.stride), (w, h))

    frames = []
    i = n = 0
    limit = args.max_frames or None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.stride == 0:
            x = preprocess(frame)
            boxes, scores, labels = sess.run(None, {"pixel_values": x})
            boxes, scores, labels = boxes[0], scores[0], labels[0]
            sx, sy = w / SIZE, h / SIZE
            dets = []
            for j in range(len(scores)):
                if scores[j] < args.conf:
                    continue
                x1, y1, x2, y2 = boxes[j] * np.array([sx, sy, sx, sy])
                lbl_id = int(labels[j])
                lbl = COCO[lbl_id] if 0 <= lbl_id < len(COCO) else "?"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 255), 2)
                cv2.putText(frame, f"{lbl}:{scores[j]:.2f}",
                            (int(x1), max(0, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                            cv2.LINE_AA)
                dets.append({
                    "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(scores[j]),
                    "label_id": lbl_id,
                    "label": lbl,
                })
            frames.append({"frame": i, "detections": dets})
            writer.write(frame)
            n += 1
            if n % 20 == 0:
                print(f"  {n} frames")
            if limit and n >= limit:
                break
        i += 1
    cap.release()
    writer.release()

    sidecar = Path(args.output).with_suffix(".json")
    sidecar.write_text(json.dumps(
        {"video": args.video, "model": args.model, "frames": frames}, indent=2))
    print(f"wrote {args.output} ({n} frames)")
    print(f"sidecar {sidecar}")


if __name__ == "__main__":
    main()
