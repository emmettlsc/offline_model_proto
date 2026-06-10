"""Compare two detections.json sidecars produced by the pipeline scripts.

Use this to verify the ONNX export against the PyTorch reference (or any two
runs against each other). Reports per-frame detection-count deltas, score
distribution per backend, and mask-area distribution per backend.

    python scripts/compare_runs.py \\
        --a outputs/pytorch_pipeline_detections.json \\
        --b outputs/onnx_pipeline_detections.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diff two detections.json sidecars")
    p.add_argument("--a", type=Path, required=True, help="First detections.json")
    p.add_argument("--b", type=Path, required=True, help="Second detections.json")
    return p.parse_args()


def load(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: not found: {path}", file=sys.stderr)
        sys.exit(2)
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values) -> str:
    if not values:
        return "(none)"
    return (f"n={len(values)} mean={statistics.fmean(values):.3f} "
            f"min={min(values):.3f} max={max(values):.3f}")


def header(label: str, run: dict, path: Path) -> None:
    s = run.get("summary", {})
    print(f"{label}: {path.name}")
    print(f"   backend            = {run.get('backend')}")
    print(f"   frames_processed   = {s.get('frames_processed')}")
    print(f"   total_detections   = {s.get('total_detections')}")
    print(f"   elapsed_seconds    = {s.get('elapsed_seconds')}")
    print(f"   throughput_fps     = {s.get('throughput_fps')}")


def main() -> int:
    args = parse_args()
    a, b = load(args.a), load(args.b)

    header("A", a, args.a)
    print()
    header("B", b, args.b)
    print()

    # Frame index alignment.
    a_by = {f["video_frame_index"]: f for f in a.get("frames", [])}
    b_by = {f["video_frame_index"]: f for f in b.get("frames", [])}
    common = sorted(set(a_by) & set(b_by))
    a_only = sorted(set(a_by) - set(b_by))
    b_only = sorted(set(b_by) - set(a_by))

    print(f"frame alignment: common={len(common)} a_only={len(a_only)} b_only={len(b_only)}")
    if a_only:
        print(f"   first a_only frames: {a_only[:8]}")
    if b_only:
        print(f"   first b_only frames: {b_only[:8]}")
    print()

    # Per-frame detection-count agreement on shared frames.
    deltas = [len(a_by[i]["detections"]) - len(b_by[i]["detections"]) for i in common]
    abs_deltas = [abs(d) for d in deltas]
    exact = sum(1 for d in deltas if d == 0)
    print(f"per-frame detection count: |A - B| {stats(abs_deltas)}")
    print(f"   frames with exact count match: {exact}/{len(common)}")
    print(f"   sum(A) - sum(B) = {sum(deltas)}")
    print()

    # Score distribution per backend.
    a_scores = [d["score"] for f in a["frames"] for d in f["detections"]]
    b_scores = [d["score"] for f in b["frames"] for d in f["detections"]]
    print(f"score distribution")
    print(f"   A: {stats(a_scores)}")
    print(f"   B: {stats(b_scores)}")
    print()

    # Mask area distribution per backend.
    a_areas = [d.get("mask_area_px", 0) for f in a["frames"] for d in f["detections"]]
    b_areas = [d.get("mask_area_px", 0) for f in b["frames"] for d in f["detections"]]
    print(f"mask area (px)")
    print(f"   A: {stats(a_areas)}")
    print(f"   B: {stats(b_areas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
