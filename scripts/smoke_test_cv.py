import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "smoke_test_cv.jpg")
    args = p.parse_args()

    print(f"opencv {cv2.__version__}")

    if args.image:
        if not args.image.exists():
            sys.exit(f"not found: {args.image}")
        img = cv2.imread(str(args.image))
        if img is None:
            sys.exit(f"imread returned None: {args.image}")
        print(f"loaded {args.image} {img.shape}")
    else:
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (300, 220), (0, 255, 0), 2)
        cv2.putText(img, "smoke test", (40, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        print("synthetic 320x240")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), img):
        sys.exit(f"imwrite failed: {args.output}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
