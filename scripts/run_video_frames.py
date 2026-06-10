"""Extract frames from a video at a fixed stride.

Stepping stone before full model integration. Writes JPEGs and a metadata JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract frames from a video at a fixed stride"
    )
    p.add_argument("--video", type=Path, required=True, help="Path to source video")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "outputs" / "frames",
                   help="Where to write extracted frames")
    p.add_argument("--frame-stride", type=int, default=30,
                   help="Save every Nth frame")
    p.add_argument("--max-frames", type=int, default=100,
                   help="Stop after this many frames (0 = no limit)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video.exists():
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        return 2
    if args.frame_stride < 1:
        print("ERROR: --frame-stride must be >= 1", file=sys.stderr)
        return 3

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        print(f"ERROR: opencv-python is not installed: {exc}", file=sys.stderr)
        return 4

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: failed to open video: {args.video}", file=sys.stderr)
        return 5

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    frame_idx = 0
    limit = args.max_frames if args.max_frames > 0 else None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.frame_stride == 0:
                out_path = args.output_dir / f"frame_{extracted + 1:06d}.jpg"
                if not cv2.imwrite(str(out_path), frame):
                    print(f"WARNING: failed to write {out_path}", file=sys.stderr)
                else:
                    extracted += 1
                if limit is not None and extracted >= limit:
                    break
            frame_idx += 1
    finally:
        cap.release()

    meta = {
        "video": str(args.video),
        "fps": fps,
        "reported_frame_count": total_frames,
        "width": width,
        "height": height,
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "extracted": extracted,
        "output_dir": str(args.output_dir),
    }
    meta_path = args.output_dir / "frames_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"extracted {extracted} frames -> {args.output_dir}")
    print(f"metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
