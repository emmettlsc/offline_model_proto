import argparse
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def parse_args():
    p = argparse.ArgumentParser(description="Export RF-DETR HF weights to ONNX")
    p.add_argument("--weights", type=Path, required=True,
                   help="HF weights directory: config.json + model.safetensors + preprocessor_config.json")
    p.add_argument("--output", type=Path, required=True, help=".onnx output path")
    p.add_argument("--size", type=int, default=576,
                   help="Square input H = W. Read from config.backbone_config.image_size; "
                        "rf-detr-medium uses 576.")
    p.add_argument("--opset", type=int, default=18,
                   help="RF-DETR uses Resize ops that need opset >= 18. Lower values trip a warning.")
    p.add_argument("--device", default="cpu",
                   help="cpu is safest for export; cuda works if you have the VRAM")
    p.add_argument("--no-check", action="store_true",
                   help="Skip onnx.checker.check_model + ORT parity check")
    return p.parse_args()


class PostprocessWrapper(torch.nn.Module):
    """Decode HF DETR-style outputs to (boxes_xyxy, scores, labels) at input scale.
    Output order matches the standard ONNX-Runtime SAM-family convention used by
    common inference scaffolds (boxes -> scores -> labels)."""

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
    proc = AutoImageProcessor.from_pretrained(str(args.weights))
    model = AutoModelForObjectDetection.from_pretrained(
        str(args.weights),
        disable_custom_kernels=True,    # custom deformable-attn kernels can't be traced
        attn_implementation="eager",    # SDPA path passes enable_gqa=True spuriously; onnxscript rejects
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
    # Use the dynamo exporter (default in torch >= 2.5). Batch is fixed to 1 — making
    # it dynamic forces a symbolic comparison inside the GQA scaled_dot_product_attention
    # decomposition that onnxscript can't resolve. The legacy exporter doesn't help
    # because it has no symbolic for aten::_upsample_bicubic2d_aa used by RF-DETR's
    # multi-scale projector.
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(args.output),
        input_names=["pixel_values"],
        output_names=["boxes", "scores", "labels"],
        opset_version=args.opset,
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
    # Deterministic input so the parity numbers are reproducible. Random N(0,1) inputs
    # produce near-uniform sigmoid scores where argmax-driven labels are unstable —
    # the meaningful parity is on real images. The numbers below should all be small.
    torch.manual_seed(0)
    probe = torch.randn(1, 3, args.size, args.size, device=args.device)
    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"pixel_values": probe.cpu().numpy()})
    with torch.inference_mode():
        pt_out = wrapped(probe.cpu())
    pt_scores = pt_out[1]
    ort_scores = torch.from_numpy(ort_out[1])
    print(f"  scores  max |pt - ort| = {(pt_scores - ort_scores).abs().max().item():.6f}")
    # On out-of-distribution input the per-query argmax for labels is unstable, and
    # cxcywh -> xyxy box decoding has wide values; both reported but not gating.
    print(f"  boxes   max |pt - ort| = {(pt_out[0] - torch.from_numpy(ort_out[0])).abs().max().item():.4f}  "
          f"(noise input — not meaningful, verify with a real image)")
    print(f"  labels  max |pt - ort| = {(pt_out[2] - torch.from_numpy(ort_out[2])).abs().max().item():.0f}  "
          f"(argmax over noisy logits — instability is expected)")


if __name__ == "__main__":
    main()
