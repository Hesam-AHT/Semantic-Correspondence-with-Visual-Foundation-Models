"""
SPair-71k Dataset Loader
========================
SPair-71k provides pairs of images with annotated keypoints for semantic correspondence.

Dataset structure expected:
  SPair-71k/
    ImageAnnotation/
      <category>/
        <pair_id>.json    ← per-pair annotation (keypoints, image names, etc.)
    JPEGImages/
      <category>/
        <image_name>.jpg
    PairAnnotation/
      test/
        <pair_id>.json    ← split-level annotation
      val/
        <pair_id>.json
      trn/
        <pair_id>.json

Download:
  wget http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SPairDataset(Dataset):
    """
    PyTorch Dataset for SPair-71k semantic correspondence benchmark.

    Each item contains:
      - src_image: PIL Image (source)
      - trg_image: PIL Image (target)
      - src_kps: (N, 2) array of (x, y) keypoints in the source image
      - trg_kps: (N, 2) array of (x, y) keypoints in the target image
      - src_imsize: (W, H) original source image size
      - trg_imsize: (W, H) original target image size
      - category: object category string
      - pair_id: unique identifier for this pair
    """

    def __init__(
        self,
        root: str,
        split: str = "test",
        category: Optional[str] = None,
        image_size: int = 840,
        transform=None,
    ):
        """
        Args:
            root:       path to the SPair-71k root directory
            split:      one of 'trn', 'val', 'test'
            category:   if given, restrict to a single category (e.g. 'cat', 'dog')
            image_size: resize images to this size (square) before feature extraction
            transform:  optional torchvision transform applied to both images
        """
        assert split in ("trn", "val", "test"), f"Unknown split: {split}"
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transform = transform

        # Collect all pair annotation files for the requested split
        pair_ann_dir = self.root / "PairAnnotation" / split
        if not pair_ann_dir.exists():
            raise FileNotFoundError(
                f"PairAnnotation directory not found: {pair_ann_dir}\n"
                "Please download SPair-71k from:\n"
                "  http://cvlab.postech.ac.kr/research/SPair-71k/"
            )

        self.pairs: List[Dict] = []
        for ann_path in sorted(pair_ann_dir.glob("*.json")):
            with open(ann_path) as f:
                ann = json.load(f)
            if category is not None and ann["category"] != category:
                continue
            self.pairs.append(ann)

        if len(self.pairs) == 0:
            raise ValueError(
                f"No pairs found for split='{split}', category='{category}'"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        ann = self.pairs[idx]
        category = ann["category"]

        # ── Load images ──────────────────────────────────────────────────────
        src_path = self.root / "JPEGImages" / category / ann["src_imname"]
        trg_path = self.root / "JPEGImages" / category / ann["trg_imname"]
        src_img = Image.open(src_path).convert("RGB")
        trg_img = Image.open(trg_path).convert("RGB")

        src_w, src_h = src_img.size
        trg_w, trg_h = trg_img.size

        # ── Keypoints ─────────────────────────────────────────────────────────
        # SPair stores keypoints as a flat list: [x0, y0, x1, y1, ...]
        # and a visibility flag per keypoint
        src_kps = np.array(ann["src_kps"], dtype=np.float32).reshape(-1, 2)  # (N, 2)
        trg_kps = np.array(ann["trg_kps"], dtype=np.float32).reshape(-1, 2)  # (N, 2)
        # visibility: 1 = visible, 0 = occluded/absent
        kp_mask = np.array(ann.get("kps_used", [1] * len(src_kps)), dtype=bool)

        # Keep only visible keypoints
        src_kps = src_kps[kp_mask]
        trg_kps = trg_kps[kp_mask]

        # ── Resize images and rescale keypoints ───────────────────────────────
        src_img_resized = src_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        trg_img_resized = trg_img.resize((self.image_size, self.image_size), Image.BILINEAR)

        src_kps_scaled = src_kps * np.array(
            [self.image_size / src_w, self.image_size / src_h], dtype=np.float32
        )
        trg_kps_scaled = trg_kps * np.array(
            [self.image_size / trg_w, self.image_size / trg_h], dtype=np.float32
        )

        # ── Optional transform (e.g. ToTensor + Normalize) ────────────────────
        if self.transform is not None:
            src_img_resized = self.transform(src_img_resized)
            trg_img_resized = self.transform(trg_img_resized)

        return {
            "src_image": src_img_resized,
            "trg_image": trg_img_resized,
            "src_kps": src_kps_scaled,       # (N, 2) in resized image coords
            "trg_kps": trg_kps_scaled,       # (N, 2) in resized image coords
            "src_imsize": (src_w, src_h),    # original size (for reference)
            "trg_imsize": (trg_w, trg_h),
            "category": category,
            "pair_id": ann.get("pair_id", str(idx)),
            "n_kps": len(src_kps),
        }


def get_dataloader(
    root: str,
    split: str = "test",
    category: Optional[str] = None,
    image_size: int = 840,
    transform=None,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = False,
):
    """Convenience function to build a DataLoader for SPair-71k."""
    from torch.utils.data import DataLoader

    dataset = SPairDataset(
        root=root,
        split=split,
        category=category,
        image_size=image_size,
        transform=transform,
    )
    # batch_size=1 is typical: keypoint counts differ per pair
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_variable_kp_collate if batch_size == 1 else None,
    )
    return loader, dataset


def _variable_kp_collate(batch):
    """Pass-through collate: keeps each item as a dict (no stacking across pairs)."""
    assert len(batch) == 1, "Use batch_size=1 with variable keypoint counts"
    return batch[0]
