import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import with fallback
try:
    from src.features.extractor import (
        extract_dense_features,
        pixel_to_patch_coord,
        patch_to_pixel_coord,
    )
    from src.matching.strategies import (
        find_best_match_argmax,
        find_best_match_window_softargmax,
    )
    from src.metrics.pck import compute_pck_spair71k, compute_pck_pfpascal
    try:
        from src.features.extractor import (
            extract_dense_features_multilayer,
            apply_pca_whitening,
        )
        HAS_MULTILAYER = True
    except ImportError:
        HAS_MULTILAYER = False
    try:
        from src.metrics.pck import compute_pck_ap10k
        HAS_AP10K = True
    except ImportError:
        HAS_AP10K = False
except ImportError:
    from helper_functions import (
        extract_dense_features,
        pixel_to_patch_coord,
        patch_to_pixel_coord,
    )
    from matching_strategies import (
        find_best_match_argmax,
        find_best_match_window_softargmax,
    )
    from pck import compute_pck_spair71k, compute_pck_pfpascal
    try:
        from helper_functions import (
            extract_dense_features_multilayer,
            apply_pca_whitening,
        )
        HAS_MULTILAYER = True
    except ImportError:
        HAS_MULTILAYER = False
    try:
        from pck import compute_pck_ap10k
        HAS_AP10K = True
    except ImportError:
        HAS_AP10K = False


# ── Mock model ────────────────────────────────────────────────────────────────

class IdentityViT(nn.Module):
    """
    ViT mock whose patch tokens encode position explicitly.
    Patch (row, col) gets feature vector = one-hot at index (row*W + col).
    This makes the correct match always deterministic and verifiable.
    """
    def __init__(self, H=4, W=4, embed_dim=None):
        super().__init__()
        self.H = H
        self.W = W
        self.N = H * W
        self.embed_dim = embed_dim or self.N   # default: one-hot size = N

    def _make_tokens(self, B):
        tokens = torch.zeros(B, self.N, self.embed_dim)
        for i in range(self.N):
            tokens[:, i, i % self.embed_dim] = 1.0
        return tokens

    def forward_features(self, x):
        B = x.shape[0]
        return {'x_norm_patchtokens': self._make_tokens(B)}

    def get_intermediate_layers(self, x, n=3, return_class_token=False, norm=True):
        B = x.shape[0]
        return [self._make_tokens(B) for _ in range(n)]


# ── Helpers ───────────────────────────────────────────────────────────────────

PATCH_SIZE   = 14
RESIZED_SIZE = 56    # 56/14 = 4 patches → 4x4 grid
IMG_SIZE     = (56, 56)   # width, height


def run_pipeline(model, src_pixel, tgt_pixel, use_softargmax=False, K=3, temperature=0.1):
    """
    Full pipeline for one keypoint pair.
    Returns (predicted_pixel_x, predicted_pixel_y).
    """
    img = torch.randn(1, 3, RESIZED_SIZE, RESIZED_SIZE)

    # Extract features
    src_feats = extract_dense_features(model, img)  # [1, H, W, D]
    tgt_feats = extract_dense_features(model, img)
    _, H, W, D = tgt_feats.shape

    # Pixel → patch
    src_px, src_py = pixel_to_patch_coord(
        src_pixel[0], src_pixel[1], IMG_SIZE, PATCH_SIZE, RESIZED_SIZE)

    # Source feature
    src_feat = src_feats[0, src_py, src_px, :]   # [D]
    tgt_flat = tgt_feats.reshape(H * W, D)

    # Cosine similarity
    sims = F.cosine_similarity(src_feat.unsqueeze(0), tgt_flat, dim=1)

    # Match
    if use_softargmax:
        match_px, match_py = find_best_match_window_softargmax(sims, W, H, K, temperature)
    else:
        match_px, match_py = find_best_match_argmax(sims, width=W)

    # Patch → pixel
    pred_x, pred_y = patch_to_pixel_coord(match_px, match_py, IMG_SIZE, PATCH_SIZE, RESIZED_SIZE)
    return pred_x, pred_y


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFullPipeline:

    def setup_method(self):
        self.model = IdentityViT(H=4, W=4)

    def test_pipeline_runs_without_error(self):
        pred_x, pred_y = run_pipeline(self.model, (14, 14), (28, 28))
        assert isinstance(pred_x, float)
        assert isinstance(pred_y, float)

    def test_prediction_within_image_bounds(self):
        for src_pixel in [(0, 0), (28, 28), (55, 55), (14, 42)]:
            pred_x, pred_y = run_pipeline(self.model, src_pixel, src_pixel)
            assert 0 <= pred_x <= IMG_SIZE[0], f"pred_x={pred_x} out of bounds"
            assert 0 <= pred_y <= IMG_SIZE[1], f"pred_y={pred_y} out of bounds"

    def test_softargmax_pipeline_runs(self):
        pred_x, pred_y = run_pipeline(
            self.model, (14, 14), (14, 14),
            use_softargmax=True, K=3, temperature=0.5)
        assert isinstance(pred_x, float)
        assert isinstance(pred_y, float)

    def test_pck_100_when_prediction_correct(self):
        """If prediction equals ground truth, PCK must be 100."""
        gt = [[20.0, 20.0], [35.0, 35.0]]
        pred = [[20.0, 20.0], [35.0, 35.0]]
        bbox = [0, 0, 56, 56]
        pck, _, _ = compute_pck_spair71k(pred, gt, bbox, threshold=0.1)
        assert pck == 100.0

    def test_pck_0_when_prediction_wrong(self):
        gt   = [[5.0,  5.0]]
        pred = [[55.0, 55.0]]   # far away
        bbox = [0, 0, 10, 10]   # tiny bbox → large normalised distance
        pck, _, _ = compute_pck_spair71k(pred, gt, bbox, threshold=0.1)
        assert pck == 0.0

    def test_multiple_keypoints_partial_correct(self):
        gt   = [[10.0, 10.0], [50.0, 50.0]]
        pred = [[10.0, 10.0], [0.0,   0.0]]   # first correct, second wrong
        bbox = [0, 0, 100, 100]
        pck, mask, _ = compute_pck_spair71k(pred, gt, bbox, threshold=0.1)
        assert pck == 50.0
        assert mask[0] and not mask[1]

    def test_coord_conversion_roundtrip(self):
        """pixel → patch → pixel stays within one patch width of origin."""
        for orig_x in [0, 7, 14, 21, 28, 35, 42, 49, 55]:
            px, py = pixel_to_patch_coord(orig_x, orig_x, IMG_SIZE, PATCH_SIZE, RESIZED_SIZE)
            rx, ry = patch_to_pixel_coord(px, py, IMG_SIZE, PATCH_SIZE, RESIZED_SIZE)
            assert abs(rx - orig_x) <= PATCH_SIZE + 1
            assert abs(ry - orig_x) <= PATCH_SIZE + 1

    def test_similarity_map_shape(self):
        img = torch.randn(1, 3, RESIZED_SIZE, RESIZED_SIZE)
        feats = extract_dense_features(self.model, img)
        _, H, W, D = feats.shape
        src_feat = feats[0, 0, 0, :]
        tgt_flat = feats.reshape(H * W, D)
        sims = F.cosine_similarity(src_feat.unsqueeze(0), tgt_flat, dim=1)
        assert sims.shape == (H * W,)

    def test_similarity_values_in_valid_range(self):
        """Cosine similarity must be in [-1, 1]."""
        img = torch.randn(1, 3, RESIZED_SIZE, RESIZED_SIZE)
        feats = extract_dense_features(self.model, img)
        _, H, W, D = feats.shape
        src_feat = F.normalize(feats[0, 0, 0, :].unsqueeze(0), dim=1)
        tgt_flat = F.normalize(feats.reshape(H * W, D), dim=1)
        sims = F.cosine_similarity(src_feat, tgt_flat, dim=1)
        assert sims.min().item() >= -1.01
        assert sims.max().item() <= 1.01


@pytest.mark.skipif(not HAS_MULTILAYER,
                    reason="multilayer extraction not in this repo version")
class TestMultilayerPipeline:

    def setup_method(self):
        self.model = IdentityViT(H=4, W=4)
        self.img   = torch.randn(1, 3, RESIZED_SIZE, RESIZED_SIZE)

    def test_multilayer_pipeline_runs(self):
        src_feats = extract_dense_features_multilayer(self.model, self.img, n_last_layers=3)
        tgt_feats = extract_dense_features_multilayer(self.model, self.img, n_last_layers=3)
        _, H, W, D = tgt_feats.shape
        src_feat = src_feats[0, 1, 1, :]
        tgt_flat = tgt_feats.reshape(H * W, D)
        sims = F.cosine_similarity(src_feat.unsqueeze(0), tgt_flat, dim=1)
        x, y = find_best_match_argmax(sims, width=W)
        assert 0 <= x < W and 0 <= y < H

    def test_pca_pipeline_runs(self):
        src_feats = extract_dense_features_multilayer(self.model, self.img, n_last_layers=3)
        tgt_feats = extract_dense_features_multilayer(self.model, self.img, n_last_layers=3)
        src_pca, tgt_pca = apply_pca_whitening(src_feats, tgt_feats, n_components=8)
        _, H, W, D = tgt_pca.shape
        src_feat = src_pca[0, 1, 1, :]
        tgt_flat = tgt_pca.reshape(H * W, D)
        sims = F.cosine_similarity(src_feat.unsqueeze(0), tgt_flat, dim=1)
        x, y = find_best_match_argmax(sims, width=W)
        assert 0 <= x < W and 0 <= y < H

    def test_pck_after_multilayer_pipeline(self):
        src_feats = extract_dense_features_multilayer(self.model, self.img)
        tgt_feats = extract_dense_features_multilayer(self.model, self.img)
        _, H, W, D = tgt_feats.shape
        src_feat = src_feats[0, 2, 2, :]
        tgt_flat = tgt_feats.reshape(H * W, D)
        sims = F.cosine_similarity(src_feat.unsqueeze(0), tgt_flat, dim=1)
        match_px, match_py = find_best_match_argmax(sims, W)
        pred_x, pred_y = patch_to_pixel_coord(
            match_px, match_py, IMG_SIZE, PATCH_SIZE, RESIZED_SIZE)
        gt_x, gt_y = patch_to_pixel_coord(2, 2, IMG_SIZE, PATCH_SIZE, RESIZED_SIZE)
        pck, _, _ = compute_pck_spair71k(
            [[pred_x, pred_y]], [[gt_x, gt_y]],
            bbox=[0, 0, 56, 56], threshold=0.5)
        assert isinstance(pck, float)
        assert 0.0 <= pck <= 100.0