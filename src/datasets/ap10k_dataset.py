"""AP-10K semantic correspondence dataset.

AP-10K is a large-scale animal pose dataset covering 54 animal species with
COCO-style keypoint annotations.  This module wraps it as a PyTorch Dataset
whose sample dict is interface-compatible with SPairDataset so that the
existing evaluate_* functions work without modification.

Normalisation convention
------------------------
Images are loaded as float32 tensors in [C, H, W] format with pixel values in
[0, 255].  Before each sample is returned it is normalised with
``Normalize(['src_img', 'trg_img'])``, which divides by 255 and applies the
standard ImageNet channel statistics (mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]).

Expected directory structure
-----------------------------
AP-10K/
    images/
        000000001.jpg
        000000002.jpg
        ...
    annotations/
        ap10k-train-split1.json
        ap10k-val-split1.json
        ap10k-test-split1.json

Annotation format
-----------------
COCO-style JSON with top-level keys ``images``, ``annotations``, and
``categories``.  Each annotation entry carries a flat ``keypoints`` list of
the form [x1, y1, v1, x2, y2, v2, ...] where v=0 means not labelled,
v=1 means labelled but not visible, and v=2 means visible.  Only v==2
keypoints are used for matching.

Pair construction
-----------------
Images in the same animal category are sorted by their COCO image-id and then
paired consecutively in non-overlapping pairs (index 0↔1, 2↔3, …).  A pair is
kept only when both images share at least ``min_visible_kps`` keypoint indices
that are visible (v==2) in both images.
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.spair_dataset import Normalize, read_img


class AP10KDataset(Dataset):
    """PyTorch Dataset for AP-10K animal keypoint correspondences.

    Constructs image pairs within each animal category and exposes per-pair
    shared visible keypoints.  The returned sample dict mirrors the keys used
    by SPairDataset so that all existing evaluate_* helpers can consume it.

    Args:
        root: Path to the AP-10K root directory (must contain ``images/`` and
            ``annotations/`` sub-directories).
        split: Dataset split — one of ``'train'``, ``'val'``, or ``'test'``.
        split_num: Annotation split number (default 1, giving split1 files).
        min_visible_kps: Minimum number of shared visible keypoints required
            for a pair to be included.  Pairs with fewer shared keypoints are
            silently discarded.
    """

    def __init__(self, root: str, split: str = 'test',
                 split_num: int = 1, min_visible_kps: int = 4):
        self.root = root
        self.pairs = []

        ann_path = os.path.join(
            root, 'annotations', f'ap10k-{split}-split{split_num}.json'
        )
        with open(ann_path, 'r') as f:
            data = json.load(f)

        # Image id → image info dict
        id_to_img = {img['id']: img for img in data['images']}

        # Category id → category name
        id_to_cat = {cat['id']: cat['name'] for cat in data['categories']}

        # Image id → annotation (keep only annotations with enough visible kps)
        id_to_ann = {}
        for ann in data['annotations']:
            raw = ann['keypoints']
            n_visible = sum(1 for k in range(2, len(raw), 3) if raw[k] == 2)
            if n_visible >= min_visible_kps:
                id_to_ann[ann['image_id']] = ann

        # Group image infos by category_id (images that have a valid annotation)
        category_to_imgs = defaultdict(list)
        for image_id, ann in id_to_ann.items():
            if image_id in id_to_img:
                category_to_imgs[ann['category_id']].append(id_to_img[image_id])

        # Sort images within each category by id for reproducibility
        for cat_id in category_to_imgs:
            category_to_imgs[cat_id].sort(key=lambda img: img['id'])

        # Build non-overlapping consecutive pairs within each category
        for cat_id, img_infos in category_to_imgs.items():
            for i in range(0, len(img_infos) - 1, 2):
                src_info = img_infos[i]
                tgt_info = img_infos[i + 1]

                src_ann = id_to_ann[src_info['id']]
                tgt_ann = id_to_ann[tgt_info['id']]

                src_kps_all, src_ids = self._parse_visible_kps(src_ann['keypoints'])
                tgt_kps_all, tgt_ids = self._parse_visible_kps(tgt_ann['keypoints'])

                # Shared keypoint indices visible in both images
                shared_ids = sorted(set(src_ids) & set(tgt_ids))
                if len(shared_ids) < min_visible_kps:
                    continue

                # Select only the shared coordinates
                src_idx_map = {kid: pos for pos, kid in enumerate(src_ids)}
                tgt_idx_map = {kid: pos for pos, kid in enumerate(tgt_ids)}
                src_kps_shared = [src_kps_all[src_idx_map[k]] for k in shared_ids]
                tgt_kps_shared = [tgt_kps_all[tgt_idx_map[k]] for k in shared_ids]

                self.pairs.append({
                    'src_path': os.path.join(root, 'images', src_info['file_name']),
                    'tgt_path': os.path.join(root, 'images', tgt_info['file_name']),
                    'src_kps': src_kps_shared,
                    'tgt_kps': tgt_kps_shared,
                    'kps_ids': shared_ids,
                    'category': id_to_cat[cat_id],
                })

        self.normalize = Normalize(['src_img', 'trg_img'])
        print(f"AP-10K {split} split{split_num}: {len(self.pairs)} pairs")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_visible_kps(self, raw_kps: list):
        """Extract keypoints with visibility flag v==2 from a flat COCO list.

        Args:
            raw_kps: Flat list [x1, y1, v1, x2, y2, v2, ...] as stored in
                the COCO annotation ``keypoints`` field.

        Returns:
            Tuple ``(kps_list, ids_list)`` where ``kps_list`` is a list of
            ``[float(x), float(y)]`` coordinates and ``ids_list`` contains
            the corresponding zero-based keypoint indices.
        """
        kps_list = []
        ids_list = []
        for i in range(0, len(raw_kps), 3):
            x, y, v = raw_kps[i], raw_kps[i + 1], raw_kps[i + 2]
            if v == 2:
                kps_list.append([float(x), float(y)])
                ids_list.append(i // 3)
        return kps_list, ids_list

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        """Return a normalised sample dict for pair at position *idx*.

        The returned dict is interface-compatible with SPairDataset so that
        existing evaluate_* functions can consume AP-10K samples without
        modification.

        Keys
        ----
        src_img      : float32 tensor [C, H, W], ImageNet-normalised
        trg_img      : float32 tensor [C, H, W], ImageNet-normalised
        src_kps      : float tensor [N, 2] — shared visible keypoints in src
        trg_kps      : float tensor [N, 2] — shared visible keypoints in tgt
        src_imsize   : torch.Size([C, H, W]) of the source image
        trg_imsize   : torch.Size([C, H, W]) of the target image
        category     : animal category name (string)
        kps_ids      : list of shared keypoint indices (ints)
        src_imname   : absolute path to the source image (string)
        trg_imname   : absolute path to the target image (string)
        """
        pair = self.pairs[idx]

        src_img = read_img(pair['src_path'])
        tgt_img = read_img(pair['tgt_path'])

        sample = {
            'src_img':    src_img,
            'trg_img':    tgt_img,
            'src_kps':    torch.tensor(pair['src_kps']).float(),
            'trg_kps':    torch.tensor(pair['tgt_kps']).float(),
            'src_imsize': src_img.size(),
            'trg_imsize': tgt_img.size(),
            'category':   pair['category'],
            'kps_ids':    pair['kps_ids'],
            'src_imname': pair['src_path'],
            'trg_imname': pair['tgt_path'],
        }

        sample = self.normalize(sample)
        return sample
