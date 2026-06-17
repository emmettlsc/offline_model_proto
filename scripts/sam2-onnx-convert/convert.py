import argparse
from pathlib import Path

import torch
from transformers import Sam2Model


SIZE = 1024


class EncoderWrapper(torch.nn.Module):
    def __init__(self, sam2):
        super().__init__()
        self.sam2 = sam2

    def forward(self, pixel_values):
        emb = self.sam2.get_image_embeddings(pixel_values)
        return emb[0], emb[1], emb[2]


class DecoderWrapper(torch.nn.Module):
    def __init__(self, sam2):
        super().__init__()
        self.sam2 = sam2

    def forward(self, input_points, input_labels, input_boxes, emb0, emb1, emb2):
        out = self.sam2(
            image_embeddings=[emb0, emb1, emb2],
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            multimask_output=True,
        )
        return out.iou_scores, out.pred_masks, out.object_score_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True,
                    help="HF Sam2 weights dir (e.g. facebook/sam2.1-hiera-base-plus)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = Sam2Model.from_pretrained(str(args.weights)).eval()

    enc = EncoderWrapper(model).eval()
    dummy_pix = torch.randn(1, 3, SIZE, SIZE)
    enc_path = args.output_dir / "vision_encoder.onnx"
    print(f"exporting encoder ...")
    # dynamo=True forces the new exporter — the legacy path has no symbolic for
    # several ops used in SAM2's Hiera backbone / FPN projector.
    torch.onnx.export(
        enc, (dummy_pix,), str(enc_path),
        input_names=["pixel_values"],
        output_names=["image_embeddings.0", "image_embeddings.1", "image_embeddings.2"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=True,
    )
    print(f"wrote {enc_path}")

    with torch.no_grad():
        emb0, emb1, emb2 = enc(dummy_pix)

    dec = DecoderWrapper(model).eval()
    points = torch.zeros(1, 1, 0, 2, dtype=torch.float32)
    labels = torch.zeros(1, 1, 0, dtype=torch.int64)
    box = torch.tensor([[[10.0, 10.0, 500.0, 500.0]]], dtype=torch.float32)

    dec_path = args.output_dir / "prompt_encoder_mask_decoder.onnx"
    print(f"exporting decoder ...")
    torch.onnx.export(
        dec, (points, labels, box, emb0, emb1, emb2), str(dec_path),
        input_names=["input_points", "input_labels", "input_boxes",
                     "image_embeddings.0", "image_embeddings.1", "image_embeddings.2"],
        output_names=["iou_scores", "pred_masks", "object_score_logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=True,
    )
    print(f"wrote {dec_path}")


if __name__ == "__main__":
    main()
