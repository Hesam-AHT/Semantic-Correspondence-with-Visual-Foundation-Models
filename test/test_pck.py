import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from src.metrics.pck import (
        compute_pck_spair71k,
        compute_pck_pfpascal,
        compute_pck_ap10k,
    )
    try:
        from src.metrics.pck import compute_pck
    except ImportError:
        compute_pck = None
except ImportError:
    from pck import compute_pck_spair71k, compute_pck_pfpascal
    try:
        from pck import compute_pck_ap10k
    except ImportError:
        compute_pck_ap10k = None
    try:
        from pck import compute_pck
    except ImportError:
        compute_pck = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_points(n=4):
    """Ground truth keypoints in a 100x100 image."""
    return [[10.0, 20.0], [50.0, 50.0], [80.0, 30.0], [40.0, 70.0]][:n]


# ── compute_pck_spair71k ──────────────────────────────────────────────────────

class TestComputePCKSPair71k:

    def test_perfect_prediction_gives_100(self):
        gt = make_points(4)
        pred = make_points(4)  # identical
        bbox = [0, 0, 200, 100]  # w=200, h=100 → norm=200
        pck, mask, dists = compute_pck_spair71k(pred, gt, bbox, threshold=0.1)
        assert pck == 100.0
        assert all(mask)
        assert all(d == 0.0 for d in dists)

    def test_zero_prediction_all_wrong(self):
        gt = [[50.0, 50.0], [100.0, 100.0]]
        pred = [[0.0, 0.0], [0.0, 0.0]]
        bbox = [0, 0, 10, 10]  # tiny bbox → norm=10, distances will be huge
        pck, mask, _ = compute_pck_spair71k(pred, gt, bbox, threshold=0.1)
        assert pck == 0.0
        assert not any(mask)

    def test_normalisation_uses_max_bbox_dimension(self):
        """norm = max(bbox_width, bbox_height)"""
        gt = [[0.0, 0.0]]
        pred = [[10.0, 0.0]]   # distance = 10
        bbox = [0, 0, 100, 50]  # w=100, h=50 → norm=100
        _, _, dists = compute_pck_spair71k(pred, gt, bbox, threshold=0.1)
        assert abs(dists[0] - 0.1) < 1e-6  # 10/100 = 0.1

    def test_wide_bbox_vs_tall_bbox(self):
        """max(w,h) must be used, not w or h alone."""
        gt = [[0.0, 0.0]]
        pred = [[30.0, 0.0]]   # distance = 30
        # wide: max(200, 50) = 200 → norm_dist = 0.15
        bbox_wide = [0, 0, 200, 50]
        _, _, d_wide = compute_pck_spair71k(pred, gt, bbox_wide, 0.1)
        # tall: max(50, 200) = 200 → same
        bbox_tall = [0, 0, 50, 200]
        _, _, d_tall = compute_pck_spair71k(pred, gt, bbox_tall, 0.1)
        assert abs(d_wide[0] - d_tall[0]) < 1e-6

    def test_half_correct(self):
        gt   = [[0.0, 0.0], [0.0, 0.0]]
        pred = [[0.0, 0.0], [999.0, 999.0]]
        bbox = [0, 0, 100, 100]  # norm=100
        pck, mask, _ = compute_pck_spair71k(pred, gt, bbox, threshold=0.1)
        assert pck == 50.0
        assert mask[0] and not mask[1]

    def test_returns_three_values(self):
        result = compute_pck_spair71k([[0,0]], [[0,0]], [0,0,100,100], 0.1)
        assert len(result) == 3

    def test_threshold_boundary(self):
        """Point exactly at threshold distance should be correct."""
        gt   = [[0.0, 0.0]]
        pred = [[10.0, 0.0]]  # distance=10
        bbox = [0, 0, 100, 100]  # norm=100 → normalised=0.1
        pck, mask, _ = compute_pck_spair71k(pred, gt, bbox, threshold=0.1)
        assert mask[0]  # exactly at boundary → correct

    def test_single_keypoint(self):
        pck, mask, dists = compute_pck_spair71k([[5.0, 5.0]], [[5.0, 5.0]],
                                                 [0, 0, 100, 100], 0.1)
        assert pck == 100.0
        assert len(mask) == 1
        assert len(dists) == 1


# ── compute_pck_pfpascal ──────────────────────────────────────────────────────

class TestComputePCKPFPascal:

    def test_perfect_prediction_gives_100(self):
        gt = make_points(3)
        pred = make_points(3)
        pck, mask, _ = compute_pck_pfpascal(pred, gt, img_size=(640, 480), threshold=0.1)
        assert pck == 100.0
        assert all(mask)

    def test_normalisation_uses_max_hw(self):
        """norm = max(width, height)"""
        gt   = [[0.0, 0.0]]
        pred = [[64.0, 0.0]]   # distance = 64
        # img_size = (640, 480): max = 640 → norm_dist = 0.1
        _, _, dists = compute_pck_pfpascal(pred, gt, img_size=(640, 480), threshold=0.1)
        assert abs(dists[0] - 0.1) < 1e-6

    def test_different_from_spair71k_normalisation(self):
        """PF-Pascal uses max(H,W) of image; SPair uses max(bbox_w, bbox_h)."""
        gt   = [[0.0, 0.0]]
        pred = [[20.0, 0.0]]
        img_size = (200, 100)   # max = 200
        bbox     = [0, 0, 50, 50]  # max = 50
        _, _, d_pfpascal  = compute_pck_pfpascal(pred, gt, img_size, 0.1)
        _, _, d_spair     = compute_pck_spair71k(pred, gt, bbox, 0.1)
        # pfpascal: 20/200=0.1, spair: 20/50=0.4
        assert abs(d_pfpascal[0] - 0.1) < 1e-6
        assert abs(d_spair[0]    - 0.4) < 1e-6

    def test_returns_three_values(self):
        result = compute_pck_pfpascal([[0,0]], [[0,0]], (100, 100), 0.1)
        assert len(result) == 3

    def test_tall_image(self):
        """max(w, h) when h > w."""
        gt   = [[0.0, 0.0]]
        pred = [[0.0, 50.0]]   # distance = 50
        # img (200 wide, 500 tall) → max = 500 → norm = 0.1
        _, _, dists = compute_pck_pfpascal(pred, gt, img_size=(200, 500), threshold=0.1)
        assert abs(dists[0] - 0.1) < 1e-6


# ── compute_pck_ap10k ─────────────────────────────────────────────────────────

@pytest.mark.skipif(compute_pck_ap10k is None,
                    reason="compute_pck_ap10k not yet implemented in this repo version")
class TestComputePCKAP10K:

    def test_perfect_prediction_gives_100(self):
        gt = make_points(4)
        pred = make_points(4)
        pck, mask, _ = compute_pck_ap10k(pred, gt, img_size=(640, 480), threshold=0.1)
        assert pck == 100.0
        assert all(mask)

    def test_normalisation_uses_diagonal(self):
        """norm = sqrt(W^2 + H^2)"""
        gt   = [[0.0, 0.0]]
        pred = [[30.0, 40.0]]  # distance = 50
        # img (300, 400) → diag = sqrt(300^2 + 400^2) = 500 → norm = 0.1
        _, _, dists = compute_pck_ap10k(pred, gt, img_size=(300, 400), threshold=0.1)
        assert abs(dists[0] - 0.1) < 1e-6

    def test_different_from_pfpascal(self):
        """AP-10K uses diagonal; PF-Pascal uses max(H,W)."""
        gt   = [[0.0, 0.0]]
        pred = [[30.0, 40.0]]   # dist=50
        img_size = (300, 400)   # max=400, diag=500
        _, _, d_ap10k   = compute_pck_ap10k(pred, gt, img_size, 0.1)
        _, _, d_pfpascal = compute_pck_pfpascal(pred, gt, img_size, 0.1)
        assert abs(d_ap10k[0]   - 50/500) < 1e-6
        assert abs(d_pfpascal[0] - 50/400) < 1e-6

    def test_returns_three_values(self):
        result = compute_pck_ap10k([[0,0]], [[0,0]], (100, 100), 0.1)
        assert len(result) == 3

    def test_zero_distance(self):
        pck, mask, dists = compute_pck_ap10k([[10.0, 20.0]], [[10.0, 20.0]],
                                              (640, 480), 0.05)
        assert pck == 100.0
        assert dists[0] == 0.0


# ── compute_pck (diagonal, generic) ──────────────────────────────────────────

@pytest.mark.skipif(compute_pck is None,
                    reason="compute_pck not present in this repo version")
class TestComputePCKGeneric:

    def test_perfect_prediction(self):
        gt = make_points(2)
        pred = make_points(2)
        pck, mask, _ = compute_pck(pred, gt, img_size=(100, 100), threshold=0.1)
        assert pck == 100.0

    def test_normalisation_is_diagonal(self):
        gt   = [[0.0, 0.0]]
        pred = [[30.0, 40.0]]  # dist=50
        # (300, 400) → diag=500 → norm=0.1
        _, _, dists = compute_pck(pred, gt, img_size=(300, 400), threshold=0.1)
        assert abs(dists[0] - 0.1) < 1e-6