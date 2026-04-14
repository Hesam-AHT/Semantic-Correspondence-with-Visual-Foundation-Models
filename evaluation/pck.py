"""
Evaluation: PCK (Percentage of Correct Keypoints)
==================================================
PCK@T measures what fraction of predicted keypoints fall within a distance
threshold T * max(image_height, image_width) from the ground-truth keypoint.

We follow the standard protocol from DIFT [Tang et al., NeurIPS 2023]:
  - Threshold is normalised by the longer side of the image
  - Report at T = 0.05, 0.10, 0.20

Results can be aggregated:
  - per-keypoint:  average PCK across all individual keypoint predictions
  - per-image:     average PCK per image (each image contributes equally)
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ── Core PCK computation ──────────────────────────────────────────────────────

def compute_pck(
    pred_kps: torch.Tensor,
    gt_kps: torch.Tensor,
    image_size: int,
    thresholds: Tuple[float, ...] = (0.05, 0.10, 0.20),
) -> Dict[float, torch.Tensor]:
    """
    Compute per-keypoint correct/incorrect at each threshold.

    Args:
        pred_kps:   (N, 2)  predicted keypoints  [x, y]
        gt_kps:     (N, 2)  ground-truth keypoints  [x, y]
        image_size: side length of the (square) evaluation image
        thresholds: normalised distance thresholds

    Returns:
        dict mapping threshold → (N,) bool tensor (True = correct prediction)
    """
    # Euclidean distance between prediction and ground truth
    dist = torch.norm(pred_kps.float() - gt_kps.float(), dim=-1)  # (N,)

    results = {}
    for t in thresholds:
        threshold_px = t * image_size      # convert to pixels
        results[t] = dist <= threshold_px  # (N,) bool
    return results


# ── Accumulator ──────────────────────────────────────────────────────────────

class PCKAccumulator:
    """
    Accumulates per-keypoint and per-image PCK across a dataset.

    Usage:
        acc = PCKAccumulator()
        for batch in dataloader:
            pred_kps = ...
            acc.update(pred_kps, batch['trg_kps'], image_size, batch['category'])
        results = acc.summarise()
    """

    def __init__(self, thresholds: Tuple[float, ...] = (0.05, 0.10, 0.20)):
        self.thresholds = thresholds
        self.reset()

    def reset(self):
        # Per-keypoint accumulators
        self._correct: Dict[float, List[bool]] = {t: [] for t in self.thresholds}
        # Per-image accumulators
        self._img_correct: Dict[float, List[float]] = {t: [] for t in self.thresholds}
        # Per-category accumulators
        self._cat_correct: Dict[str, Dict[float, List[bool]]] = defaultdict(
            lambda: {t: [] for t in self.thresholds}
        )
        self._n_pairs = 0

    def update(
        self,
        pred_kps: torch.Tensor,
        gt_kps: torch.Tensor,
        image_size: int,
        category: Optional[str] = None,
    ):
        """
        Register predictions for one image pair.

        Args:
            pred_kps:   (N, 2)  predictions
            gt_kps:     (N, 2)  ground truth
            image_size: side length of the evaluation image
            category:   optional category string for per-category breakdown
        """
        correct = compute_pck(pred_kps, gt_kps, image_size, self.thresholds)
        self._n_pairs += 1

        for t in self.thresholds:
            c = correct[t].cpu().tolist()          # list of booleans
            self._correct[t].extend(c)             # per-keypoint
            self._img_correct[t].append(float(np.mean(c)))  # per-image mean

            if category is not None:
                self._cat_correct[category][t].extend(c)

    def summarise(self) -> Dict:
        """
        Compute final PCK numbers.

        Returns a dict with keys:
          'per_keypoint':  {threshold: pck_value}
          'per_image':     {threshold: pck_value}
          'per_category':  {category: {threshold: pck_value}}
          'n_pairs':       int
          'n_keypoints':   int
        """
        per_kp = {}
        per_img = {}
        for t in self.thresholds:
            per_kp[t]  = float(np.mean(self._correct[t])) * 100.0   # as %
            per_img[t] = float(np.mean(self._img_correct[t])) * 100.0

        per_cat = {}
        for cat, cat_data in self._cat_correct.items():
            per_cat[cat] = {
                t: float(np.mean(cat_data[t])) * 100.0
                for t in self.thresholds
            }

        return {
            "per_keypoint":  per_kp,
            "per_image":     per_img,
            "per_category":  per_cat,
            "n_pairs":       self._n_pairs,
            "n_keypoints":   len(self._correct[self.thresholds[0]]),
        }

    def print_summary(self, backbone_name: str = ""):
        """Print a formatted summary table."""
        results = self.summarise()
        header = f"{'Backbone':>20}  " + "  ".join(
            f"PCK@{t:.2f}" for t in self.thresholds
        )
        sep = "-" * len(header)

        print(sep)
        print(header)
        print(sep)

        # Per-keypoint row
        kp_row = f"{'Per-keypoint':>20}  " + "  ".join(
            f"{results['per_keypoint'][t]:>8.2f}%" for t in self.thresholds
        )
        print(kp_row)

        # Per-image row
        img_row = f"{'Per-image':>20}  " + "  ".join(
            f"{results['per_image'][t]:>8.2f}%" for t in self.thresholds
        )
        print(img_row)

        print(sep)

        # Per-category breakdown (sorted)
        if results["per_category"]:
            print("\nPer-category PCK@0.10:")
            t_ref = 0.10
            for cat in sorted(results["per_category"].keys()):
                v = results["per_category"][cat].get(t_ref, 0.0)
                print(f"  {cat:>20}: {v:.2f}%")

        print(f"\nTotal pairs evaluated: {results['n_pairs']}")
        print(f"Total keypoints:       {results['n_keypoints']}")
