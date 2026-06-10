"""Confirm OpenCV can read an image (or generate one) and write a JPEG."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenCV smoke test")
    p.add_argument("--image", type=Path, default=None,
                   help="Optional input image path. If omitted, a synthetic image is used.")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "outputs" / "smoke_test_cv.jpg",
                   help="Output image path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import cv2  # type: ignore
        import numpy as np
    except ImportError as exc:
        print(f"ERROR: required import failed: {exc}", file=sys.stderr)
        return 1

    print(f"opencv version: {cv2.__version__}")

    if args.image is not None:
        if not args.image.exists():
            print(f"ERROR: image not found: {args.image}", file=sys.stderr)
            return 2
        img = cv2.imread(str(args.image))
        if img is None:
            print(f"ERROR: cv2.imread returned None for {args.image}",
                  file=sys.stderr)
            return 3
        print(f"loaded image  : {args.image} shape={img.shape}")
    else:
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (300, 220), (0, 255, 0), 2)
        cv2.putText(img, "smoke test", (40, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        print("loaded image  : <synthetic 320x240>")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), img):
        print(f"ERROR: cv2.imwrite failed for {args.output}", file=sys.stderr)
        return 4
    print(f"wrote         : {args.output}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
