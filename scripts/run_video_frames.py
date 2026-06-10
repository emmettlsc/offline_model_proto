import argparse
import json
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "frames")
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=100)
    args = p.parse_args()

    if not args.video.exists():
        sys.exit(f"not found: {args.video}")
    if args.stride < 1:
        sys.exit("stride must be >= 1")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"cannot open: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    i = 0
    limit = args.max_frames or None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % args.stride == 0:
                out = args.output_dir / f"frame_{saved + 1:06d}.jpg"
                if cv2.imwrite(str(out), frame):
                    saved += 1
                if limit and saved >= limit:
                    break
            i += 1
    finally:
        cap.release()

    (args.output_dir / "frames_metadata.json").write_text(json.dumps({
        "video": str(args.video),
        "fps": fps,
        "frame_count": total,
        "width": w,
        "height": h,
        "stride": args.stride,
        "max_frames": args.max_frames,
        "extracted": saved,
        "output_dir": str(args.output_dir),
    }, indent=2))
    print(f"extracted {saved} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
