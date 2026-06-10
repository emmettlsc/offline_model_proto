import argparse
import json
import statistics
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    return p.parse_args()


def load(path):
    if not path.exists():
        sys.exit(f"not found: {path}")
    return json.loads(path.read_text())


def stats(xs):
    if not xs:
        return "n=0"
    return f"n={len(xs)} mean={statistics.fmean(xs):.3f} min={min(xs):.3f} max={max(xs):.3f}"


def head(label, run, path):
    s = run.get("summary", {})
    print(f"{label} {path.name}")
    print(f"   backend          {run.get('backend')}")
    print(f"   frames_processed {s.get('frames_processed')}")
    print(f"   total_detections {s.get('total_detections')}")
    print(f"   elapsed_seconds  {s.get('elapsed_seconds')}")
    print(f"   throughput_fps   {s.get('throughput_fps')}")


def main():
    args = parse_args()
    a, b = load(args.a), load(args.b)

    head("A", a, args.a)
    print()
    head("B", b, args.b)
    print()

    by_a = {f["video_frame_index"]: f for f in a.get("frames", [])}
    by_b = {f["video_frame_index"]: f for f in b.get("frames", [])}
    common = sorted(set(by_a) & set(by_b))
    a_only = sorted(set(by_a) - set(by_b))
    b_only = sorted(set(by_b) - set(by_a))

    print(f"frame alignment: common={len(common)} a_only={len(a_only)} b_only={len(b_only)}")
    if a_only:
        print(f"   first a_only: {a_only[:8]}")
    if b_only:
        print(f"   first b_only: {b_only[:8]}")
    print()

    deltas = [len(by_a[i]["detections"]) - len(by_b[i]["detections"]) for i in common]
    abs_deltas = [abs(d) for d in deltas]
    exact = sum(1 for d in deltas if d == 0)
    print(f"per-frame |A - B|: {stats(abs_deltas)}")
    print(f"   exact match on {exact}/{len(common)} frames")
    print(f"   sum(A) - sum(B) = {sum(deltas)}")
    print()

    a_scores = [d["score"] for f in a["frames"] for d in f["detections"]]
    b_scores = [d["score"] for f in b["frames"] for d in f["detections"]]
    print(f"scores  A: {stats(a_scores)}")
    print(f"scores  B: {stats(b_scores)}")
    print()

    a_areas = [d.get("mask_area_px", 0) for f in a["frames"] for d in f["detections"]]
    b_areas = [d.get("mask_area_px", 0) for f in b["frames"] for d in f["detections"]]
    print(f"areas   A: {stats(a_areas)}")
    print(f"areas   B: {stats(b_areas)}")


if __name__ == "__main__":
    main()
