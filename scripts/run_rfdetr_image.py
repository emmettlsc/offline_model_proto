"""Skeleton for running RF-DETR on a single image.

The script validates inputs, isolates the model-loading and inference calls in
named functions, and prints clear TODOs where the real integration goes.

Nothing in this script reaches the network. Weights must already be on disk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RF-DETR on a single image (skeleton)")
    p.add_argument("--image", type=Path, required=True, help="Path to input image")
    p.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "sample_config.yaml",
                   help="Path to YAML config (informational, not required)")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "outputs" / "rfdetr",
                   help="Where annotated image + JSON go")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cpu | cuda")
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    return p.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_rfdetr_model(weights_path: Path, device: str) -> Any:
    """Integration TODO: load and return an RF-DETR model.

    Implementation notes:
      - import the rfdetr package (or vendored module)
      - construct the model class with the right architecture key
      - load weights from `weights_path` with map_location=device
      - move to device and set .eval()
    """
    try:
        import rfdetr  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "RF-DETR is not installed. The `rfdetr` package/wheel must be "
            "included in the offline wheelhouse, or vendored as source and "
            "installed with `pip install -e ./rfdetr_src`. "
            f"Underlying error: {exc}"
        ) from exc
    raise NotImplementedError("load_rfdetr_model: integration TODO")


def run_detection(model: Any, image: Any, confidence_threshold: float) -> List[dict]:
    """Integration TODO: forward `image` through `model` and return detections.

    Expected return shape:
      [{"bbox": [x1, y1, x2, y2], "score": float, "label": int|str}, ...]
    """
    raise NotImplementedError("run_detection: integration TODO")


def draw_detections(image: Any, detections: List[dict]) -> Any:
    """Integration TODO: draw boxes/labels onto a copy of `image` and return it."""
    raise NotImplementedError("draw_detections: integration TODO")


def save_results(
    output_dir: Path,
    image_annotated: Any,
    detections: List[dict],
    source_image: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "source_image": str(source_image),
        "num_detections": len(detections),
        "detections": detections,
    }
    json_path = output_dir / f"{source_image.stem}.json"
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # TODO: when draw_detections is implemented, also write the annotated image:
    #     import cv2
    #     cv2.imwrite(str(output_dir / f"{source_image.stem}_annot.jpg"), image_annotated)


def main() -> int:
    args = parse_args()

    if not args.image.exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 2
    if not args.weights.exists():
        print(f"ERROR: weights not found: {args.weights}", file=sys.stderr)
        print("Place weights under models/ (not committed) and pass --weights. "
              "See docs/MODEL_WEIGHTS.md.", file=sys.stderr)
        return 3
    if not args.config.exists():
        print(f"WARNING: config not found: {args.config}", file=sys.stderr)

    device = resolve_device(args.device)
    print(f"device           : {device}")
    print(f"image            : {args.image}")
    print(f"weights          : {args.weights}")
    print(f"output_dir       : {args.output_dir}")
    print(f"conf_threshold   : {args.confidence_threshold}")

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        print(f"ERROR: opencv-python is not installed: {exc}", file=sys.stderr)
        return 4

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"ERROR: cv2.imread returned None for {args.image}", file=sys.stderr)
        return 5

    try:
        model = load_rfdetr_model(args.weights, device)
        detections = run_detection(model, image, args.confidence_threshold)
        annotated = draw_detections(image, detections)
        save_results(args.output_dir, annotated, detections, args.image)
    except NotImplementedError as exc:
        print(f"TODO: {exc}", file=sys.stderr)
        print("This is a skeleton. Fill in load_rfdetr_model / run_detection / "
              "draw_detections to complete integration.", file=sys.stderr)
        return 10
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 11

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
