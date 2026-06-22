import argparse
from pathlib import Path

import onnx
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection


class Wrapper(torch.nn.Module):
    def __init__(self, model, size):
        super().__init__()
        self.model = model
        self.size = float(size)

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        scores, labels = out.logits.sigmoid().max(dim=-1)
        cx, cy, bw, bh = out.pred_boxes.unbind(-1)
        x1 = (cx - bw / 2) * self.size
        y1 = (cy - bh / 2) * self.size
        x2 = (cx + bw / 2) * self.size
        y2 = (cy + bh / 2) * self.size
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return boxes, scores, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    AutoImageProcessor.from_pretrained(str(args.weights))
    model = AutoModelForObjectDetection.from_pretrained(str(args.weights)).eval()

    wrapped = Wrapper(model, args.size).eval()
    dummy = torch.randn(1, 3, args.size, args.size)

    torch.onnx.export(
        wrapped, (dummy,), str(args.output),
        input_names=["pixel_values"],
        output_names=["boxes", "scores", "labels"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    m = onnx.load(str(args.output), load_external_data=False)
    if m.ir_version > 9:
        m.ir_version = 9
        onnx.save(m, str(args.output))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
