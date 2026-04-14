import argparse
import json
import os
from pathlib import Path

import torch
from torchvision import transforms

# ── Project imports ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data.spair import SPairDataset
from models.extractors import build_extractor, get_transform
from models.matcher import predict_argmax
from evaluation.pck import PCKAccumulator


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 – Training-Free Correspondence")
    p.add_argument("--backbone",        type=str, default="dinov2",
                   choices=["dinov2", "dinov3", "sam"],
                   help="Which pretrained backbone to use")
    p.add_argument("--dinov2_variant",  type=str, default="vitb",
                   choices=["vits", "vitb", "vitl", "vitg"],
                   help="DINOv2 model size")
    p.add_argument("--sam_checkpoint",  type=str, default=None,
                   help="Path to SAM checkpoint (required if backbone=sam)")
    p.add_argument("--sam_model_type",  type=str, default="vit_b",
                   choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--spair_root",      type=str, required=True,
                   help="Path to the SPair-71k root directory")
    p.add_argument("--split",           type=str, default="test",
                   choices=["trn", "val", "test"])
    p.add_argument("--category",        type=str, default=None,
                   help="Restrict evaluation to one category")
    p.add_argument("--image_size",      type=int, default=840,
                   help="Images are resized to this square size")
    p.add_argument("--device",          type=str, default="cuda",
                   help="'cuda' or 'cpu'")
    p.add_argument("--output_dir",      type=str, default="results",
                   help="Directory to save JSON result files")
    p.add_argument("--max_pairs",       type=int, default=None,
                   help="Limit evaluation to first N pairs (for quick testing)")
    return p.parse_args()


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Build backbone ───────────────────────────────────────────────────────
    extractor = build_extractor(
        backbone=args.backbone,
        device=device,
        dinov2_variant=args.dinov2_variant,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
    )
    patch_size = extractor.patch_size

    # ── Preprocessing transform ──────────────────────────────────────────────
    transform = get_transform(image_size=args.image_size)

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = SPairDataset(
        root=args.spair_root,
        split=args.split,
        category=args.category,
        image_size=args.image_size,
        transform=transform,
    )
    print(f"Evaluating on {len(dataset)} pairs  (split={args.split})")

    # ── PCK accumulator ──────────────────────────────────────────────────────
    acc = PCKAccumulator(thresholds=(0.05, 0.10, 0.20))

    # ── Evaluation loop ───────────────────────────────────────────────────────
    extractor.eval()
    for i, sample in enumerate(dataset):
        if args.max_pairs is not None and i >= args.max_pairs:
            break

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(dataset)}]")

        # Add batch dimension
        src_tensor = sample["src_image"].unsqueeze(0).to(device)   # (1, 3, H, W)
        trg_tensor = sample["trg_image"].unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():
            src_feat = extractor.extract(src_tensor)[0]   # (h, w, C)
            trg_feat = extractor.extract(trg_tensor)[0]

        # Source keypoints as tensor
        src_kps = torch.tensor(sample["src_kps"], dtype=torch.float32)  # (N, 2)
        gt_kps  = torch.tensor(sample["trg_kps"], dtype=torch.float32)  # (N, 2)

        # Predict target keypoints
        pred_kps = predict_argmax(
            src_feat=src_feat,
            trg_feat=trg_feat,
            src_kps=src_kps,
            image_size=args.image_size,
            patch_size=patch_size,
        )

        # Accumulate PCK
        acc.update(
            pred_kps=pred_kps,
            gt_kps=gt_kps,
            image_size=args.image_size,
            category=sample["category"],
        )

    # ── Print and save results ────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Backbone: {args.backbone} ({args.dinov2_variant if args.backbone=='dinov2' else ''})")
    acc.print_summary(backbone_name=args.backbone)

    # Save to JSON
    os.makedirs(args.output_dir, exist_ok=True)
    result_name = f"{args.backbone}"
    if args.backbone == "dinov2":
        result_name += f"_{args.dinov2_variant}"
    if args.category:
        result_name += f"_{args.category}"
    result_name += f"_{args.split}.json"

    out_path = os.path.join(args.output_dir, result_name)
    results = acc.summarise()
    results["config"] = vars(args)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
