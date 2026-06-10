import argparse
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def parse_args():
    p = argparse.ArgumentParser(description="Export RF-DETR HF weights to ONNX")
    p.add_argument("--weights", type=Path, required=True,
                   help="HF weights directory: config.json + model.safetensors + preprocessor_config.json")
    p.add_argument("--output", type=Path, required=True, help=".onnx output path")
    p.add_argument("--size", type=int, default=560,
                   help="Square input H = W. RF-DETR base/medium use 560; large uses 728.")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", default="cpu",
                   help="cpu is safest for export; cuda works if you have the VRAM")
    p.add_argument("--no-check", action="store_true",
                   help="Skip onnx.checker.check_model + ORT parity check")
    return p.parse_args()


class PostprocessWrapper(torch.nn.Module):
    """Decode HF DETR-style outputs to (boxes_xyxy, scores, labels) at input scale,
    so the consumer in scripts/run_onnx_pipeline.py can use them directly."""

    def __init__(self, model, size):
        super().__init__()
        self.model = model
        self.size = float(size)

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        logits = out.logits             # [B, N, C]
        cxcywh = out.pred_boxes         # [B, N, 4] normalized [0, 1]

        probs = logits.sigmoid()
        scores, labels = probs.max(dim=-1)

        cx, cy, bw, bh = cxcywh.unbind(-1)
        x1 = (cx - bw / 2) * self.size
        y1 = (cy - bh / 2) * self.size
        x2 = (cx + bw / 2) * self.size
        y2 = (cy + bh / 2) * self.size
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return boxes, scores, labels


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.weights}")
    proc = AutoImageProcessor.from_pretrained(str(args.weights), trust_remote_code=True)
    model = AutoModelForObjectDetection.from_pretrained(
        str(args.weights), trust_remote_code=True
    ).to(args.device).eval()

    cfg = model.config
    print("model config:")
    for k in ("model_type", "num_queries", "num_labels", "hidden_size",
              "encoder_layers", "decoder_layers", "image_size"):
        if hasattr(cfg, k):
            print(f"  {k}: {getattr(cfg, k)}")
    print(f"preprocessor: mean={getattr(proc, 'image_mean', '?')} "
          f"std={getattr(proc, 'image_std', '?')} size={getattr(proc, 'size', '?')}")

    wrapped = PostprocessWrapper(model, args.size).to(args.device).eval()
    dummy = torch.randn(1, 3, args.size, args.size, device=args.device)

    print(f"exporting to {args.output} (opset {args.opset}, size {args.size})")
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(args.output),
        input_names=["pixel_values"],
        output_names=["boxes", "scores", "labels"],
        opset_version=args.opset,
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "boxes":  {0: "batch"},
            "scores": {0: "batch"},
            "labels": {0: "batch"},
        },
        do_constant_folding=True,
    )
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")

    if args.no_check:
        return

    import onnx
    m = onnx.load(str(args.output))
    onnx.checker.check_model(m)
    print(f"onnx ok: ir_version={m.ir_version}, opset={m.opset_import[0].version}")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipping parity check")
        return
    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"pixel_values": dummy.cpu().numpy()})
    with torch.inference_mode():
        pt_out = wrapped(dummy.cpu())
    for name, p_t, o in zip(["boxes", "scores", "labels"], pt_out, ort_out):
        diff = (torch.from_numpy(o) - p_t).abs().max().item()
        print(f"  parity {name}: max |pt - ort| = {diff:.6f}")


if __name__ == "__main__":
    main()
