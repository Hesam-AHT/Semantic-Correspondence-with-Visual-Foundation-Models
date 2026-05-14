import sys
import os
import torch
import torch.nn.functional as F
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from src.matching.strategies import (
        find_best_match_argmax,
        find_best_match_window_softargmax,
    )
    try:
        from src.matching.strategies import find_best_match_mnn, apply_mnn_filter
        HAS_MNN = True
    except ImportError:
        HAS_MNN = False
except ImportError:
    from matching_strategies import (
        find_best_match_argmax,
        find_best_match_window_softargmax,
    )
    try:
        from matching_strategies import find_best_match_mnn, apply_mnn_filter
        HAS_MNN = True
    except ImportError:
        HAS_MNN = False


def make_sim_map(H, W, peak_y, peak_x, peak_val=10.0, bg_val=0.0):
    """Create a flat similarity map with one dominant peak."""
    s = torch.full((H * W,), bg_val)
    s[peak_y * W + peak_x] = peak_val
    return s


# ── find_best_match_argmax ────────────────────────────────────────────────────

class TestArgmax:

    def test_single_peak_top_left(self):
        s = make_sim_map(4, 4, peak_y=0, peak_x=0)
        x, y = find_best_match_argmax(s, width=4)
        assert (x, y) == (0, 0)

    def test_single_peak_bottom_right(self):
        s = make_sim_map(4, 4, peak_y=3, peak_x=3)
        x, y = find_best_match_argmax(s, width=4)
        assert (x, y) == (3, 3)

    def test_single_peak_middle(self):
        s = make_sim_map(6, 8, peak_y=3, peak_x=5)
        x, y = find_best_match_argmax(s, width=8)
        assert (x, y) == (5, 3)

    def test_returns_x_y_not_y_x(self):
        """x is column, y is row — check order is correct."""
        H, W = 4, 8
        s = make_sim_map(H, W, peak_y=1, peak_x=6)
        x, y = find_best_match_argmax(s, width=W)
        assert x == 6  # column
        assert y == 1  # row

    def test_all_equal_returns_valid_index(self):
        s = torch.ones(16)
        x, y = find_best_match_argmax(s, width=4)
        assert 0 <= x < 4
        assert 0 <= y < 4

    def test_1x1_map(self):
        s = torch.tensor([5.0])
        x, y = find_best_match_argmax(s, width=1)
        assert (x, y) == (0, 0)

    def test_1d_map(self):
        s = make_sim_map(1, 8, peak_y=0, peak_x=5)
        x, y = find_best_match_argmax(s, width=8)
        assert x == 5 and y == 0

    def test_negative_similarities(self):
        s = torch.full((16,), -5.0)
        s[7] = -1.0   # least negative = best
        x, y = find_best_match_argmax(s, width=4)
        assert y * 4 + x == 7


# ── find_best_match_window_softargmax ─────────────────────────────────────────

class TestWindowSoftArgmax:

    def test_returns_floats(self):
        s = make_sim_map(8, 8, peak_y=4, peak_x=4)
        x, y = find_best_match_window_softargmax(s, width=8, height=8, K=3, temperature=1.0)
        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_strong_peak_close_to_argmax(self):
        """With a very strong peak and small temperature, soft-argmax ≈ argmax."""
        s = make_sim_map(8, 8, peak_y=4, peak_x=4, peak_val=100.0)
        x, y = find_best_match_window_softargmax(s, width=8, height=8, K=3, temperature=0.01)
        assert abs(x - 4.0) < 0.1
        assert abs(y - 4.0) < 0.1

    def test_output_within_map_bounds(self):
        """Result must be within [0, W-1] x [0, H-1]."""
        for peak_y, peak_x in [(0, 0), (0, 7), (7, 0), (7, 7), (3, 5)]:
            s = make_sim_map(8, 8, peak_y=peak_y, peak_x=peak_x, peak_val=10.0)
            x, y = find_best_match_window_softargmax(s, 8, 8, K=5, temperature=1.0)
            assert 0.0 <= x <= 7.0, f"x={x} out of bounds for peak ({peak_x},{peak_y})"
            assert 0.0 <= y <= 7.0, f"y={y} out of bounds for peak ({peak_x},{peak_y})"

    def test_corner_peak_clamped(self):
        """Window at corners must not go out of bounds."""
        s = make_sim_map(8, 8, peak_y=0, peak_x=0, peak_val=10.0)
        x, y = find_best_match_window_softargmax(s, 8, 8, K=7, temperature=1.0)
        assert x >= 0.0 and y >= 0.0

    def test_k_must_be_odd(self):
        s = make_sim_map(8, 8, peak_y=4, peak_x=4)
        with pytest.raises(AssertionError):
            find_best_match_window_softargmax(s, 8, 8, K=4, temperature=1.0)

    def test_high_temperature_pulls_toward_window_center(self):
        """Very high temperature → uniform weights → result near window center."""
        s = make_sim_map(9, 9, peak_y=4, peak_x=4, peak_val=1.0, bg_val=0.99)
        x_hot, y_hot = find_best_match_window_softargmax(
            s, 9, 9, K=5, temperature=1000.0)
        x_cold, y_cold = find_best_match_window_softargmax(
            s, 9, 9, K=5, temperature=0.001)
        # Cold (peaked) should be closer to argmax (4,4)
        dist_cold = abs(x_cold - 4) + abs(y_cold - 4)
        dist_hot  = abs(x_hot  - 4) + abs(y_hot  - 4)
        assert dist_cold <= dist_hot + 0.5

    def test_k1_equals_argmax(self):
        """K=1 window contains only the peak patch → identical to argmax."""
        s = make_sim_map(8, 8, peak_y=3, peak_x=5, peak_val=10.0)
        x_soft, y_soft = find_best_match_window_softargmax(
            s, 8, 8, K=1, temperature=1.0)
        x_hard, y_hard = find_best_match_argmax(s, width=8)
        assert abs(x_soft - x_hard) < 1e-4
        assert abs(y_soft - y_hard) < 1e-4

    def test_symmetric_map_symmetric_result(self):
        """Symmetric similarity map → result at center of symmetry."""
        H, W = 7, 7
        s = torch.zeros(H * W)
        # Two equal peaks symmetric around (3,3)
        s[3 * W + 2] = 5.0   # (x=2, y=3)
        s[3 * W + 4] = 5.0   # (x=4, y=3)
        x, y = find_best_match_window_softargmax(s, W, H, K=5, temperature=1.0)
        # argmax will pick one of them; soft-argmax should pull toward center x=3
        assert abs(y - 3.0) < 0.5   # y stays at 3


# ── find_best_match_mnn + apply_mnn_filter ────────────────────────────────────

@pytest.mark.skipif(not HAS_MNN, reason="MNN functions not yet implemented")
class TestMNN:

    def _make_feature_maps(self, H, W, D=8):
        """Random normalised feature maps."""
        src = F.normalize(torch.randn(1, H, W, D), dim=-1)
        tgt = F.normalize(torch.randn(1, H, W, D), dim=-1)
        return src, tgt

    def test_mnn_returns_four_values(self):
        H, W, D = 4, 4, 8
        src, tgt = self._make_feature_maps(H, W, D)
        tgt_flat = tgt.reshape(H * W, D)
        src_feat = src[0, 2, 2, :]
        sims = F.cosine_similarity(src_feat.unsqueeze(0), tgt_flat, dim=1)
        result = find_best_match_mnn(sims, src, tgt, W, H, K=3, temperature=1.0)
        assert len(result) == 4

    def test_mnn_fwd_within_bounds(self):
        H, W, D = 6, 6, 8
        src, tgt = self._make_feature_maps(H, W, D)
        tgt_flat = tgt.reshape(H * W, D)
        src_feat = src[0, 3, 3, :]
        sims = F.cosine_similarity(src_feat.unsqueeze(0), tgt_flat, dim=1)
        fwd_x, fwd_y, back_x, back_y = find_best_match_mnn(
            sims, src, tgt, W, H, K=3, temperature=1.0)
        assert 0.0 <= fwd_x <= W - 1
        assert 0.0 <= fwd_y <= H - 1
        assert 0 <= back_x <= W - 1
        assert 0 <= back_y <= H - 1

    def test_apply_mnn_accepts_good_match(self):
        """When backward match lands on source patch, MNN accepts forward result."""
        H, W, D = 4, 4, 8
        # Make src and tgt identical so forward=backward always matches
        feat = F.normalize(torch.randn(1, H, W, D), dim=-1)
        src = feat.clone()
        tgt = feat.clone()
        tgt_flat = tgt.reshape(H * W, D)
        src_feat = src[0, 2, 2, :]
        sims = F.cosine_similarity(src_feat.unsqueeze(0), tgt_flat, dim=1)
        fwd_x, fwd_y, back_x, back_y = find_best_match_mnn(
            sims, src, tgt, W, H, K=3, temperature=1.0)
        result_x, result_y = apply_mnn_filter(
            fwd_x, fwd_y, back_x, back_y,
            src_patch_x=2, src_patch_y=2,
            similarities=sims, width=W, height=H,
            K=3, temperature=1.0, max_patch_dist=1)
        # Should accept — result equals forward
        assert abs(result_x - fwd_x) < 1e-4
        assert abs(result_y - fwd_y) < 1e-4

    def test_apply_mnn_rejects_bad_match(self):
        """When backward match is far from source, MNN falls back to second best."""
        H, W, D = 8, 8, 8
        sims = torch.zeros(H * W)
        # Best match at (7,7) — far from source (0,0)
        sims[7 * W + 7] = 10.0
        # Second best at (1,1)
        sims[1 * W + 1] = 5.0
        src = F.normalize(torch.randn(1, H, W, D), dim=-1)
        tgt = F.normalize(torch.randn(1, H, W, D), dim=-1)
        fwd_x, fwd_y, back_x, back_y = find_best_match_mnn(
            sims, src, tgt, W, H, K=3, temperature=1.0)
        result_x, result_y = apply_mnn_filter(
            fwd_x, fwd_y, back_x, back_y,
            src_patch_x=0, src_patch_y=0,
            similarities=sims, width=W, height=H,
            K=3, temperature=1.0, max_patch_dist=1)
        # Forward was at (7,7); if MNN rejects it, fallback should NOT be (7,7)
        # (it falls back to second best near (1,1))
        assert not (abs(result_x - 7) < 0.5 and abs(result_y - 7) < 0.5), \
            "MNN should have rejected the bad match and fallen back"

    def test_apply_mnn_returns_floats(self):
        H, W, D = 4, 4, 8
        src, tgt = self._make_feature_maps(H, W, D)
        tgt_flat = tgt.reshape(H * W, D)
        src_feat = src[0, 1, 1, :]
        sims = F.cosine_similarity(src_feat.unsqueeze(0), tgt_flat, dim=1)
        fwd_x, fwd_y, back_x, back_y = find_best_match_mnn(
            sims, src, tgt, W, H, K=3, temperature=1.0)
        rx, ry = apply_mnn_filter(fwd_x, fwd_y, back_x, back_y,
                                   1, 1, sims, W, H, K=3, temperature=1.0)
        assert isinstance(rx, float)
        assert isinstance(ry, float)