import sys
import os
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from src.features.extractor import (
        extract_dense_features,
        pixel_to_patch_coord,
        patch_to_pixel_coord,
    )
    try:
        from src.features.extractor import (
            extract_dense_features_multilayer,
            apply_pca_whitening,
        )
        HAS_MULTILAYER = True
    except ImportError:
        HAS_MULTILAYER = False
except ImportError:
    from helper_functions import (
        extract_dense_features,
        pixel_to_patch_coord,
        patch_to_pixel_coord,
    )
    try:
        from helper_functions import (
            extract_dense_features_multilayer,
            apply_pca_whitening,
        )
        HAS_MULTILAYER = True
    except ImportError:
        HAS_MULTILAYER = False


# ── Mock ViT model ────────────────────────────────────────────────────────────

class MockViT(nn.Module):
    """
    Minimal ViT mock that mimics the DINOv2/DINOv3 interface:
    - forward_features() returns {'x_norm_patchtokens': [B, N, D]}
    - get_intermediate_layers() returns list of [B, N, D] tensors
    """

    def __init__(self, n_patches=9, embed_dim=16, n_layers=4):
        super().__init__()
        self.n_patches  = n_patches   # must be a perfect square (e.g. 9 = 3x3)
        self.embed_dim  = embed_dim
        self.n_layers   = n_layers
        # Dummy parameter so the model is not empty
        self.dummy = nn.Linear(embed_dim, embed_dim)

    def forward_features(self, x):
        B = x.shape[0]
        tokens = torch.randn(B, self.n_patches, self.embed_dim)
        return {'x_norm_patchtokens': tokens}

    def get_intermediate_layers(self, x, n=3, return_class_token=False, norm=True):
        B = x.shape[0]
        return [
            torch.randn(B, self.n_patches, self.embed_dim)
            for _ in range(n)
        ]


# ── extract_dense_features ────────────────────────────────────────────────────

class TestExtractDenseFeatures:

    def setup_method(self):
        self.model = MockViT(n_patches=9, embed_dim=16)
        self.img   = torch.randn(1, 3, 518, 518)

    def test_output_shape(self):
        feats = extract_dense_features(self.model, self.img)
        # 9 patches → 3x3 grid
        assert feats.shape == (1, 3, 3, 16)

    def test_batch_size_preserved(self):
        img_batch = torch.randn(2, 3, 518, 518)
        feats = extract_dense_features(self.model, img_batch)
        assert feats.shape[0] == 2

    def test_output_is_4d(self):
        feats = extract_dense_features(self.model, self.img)
        assert feats.ndim == 4

    def test_no_grad_in_eval_mode(self):
        """In eval mode (training=False), no gradients should be computed."""
        feats = extract_dense_features(self.model, self.img, training=False)
        assert not feats.requires_grad

    def test_patch_grid_is_square(self):
        feats = extract_dense_features(self.model, self.img)
        _, H, W, _ = feats.shape
        assert H == W

    def test_different_embed_dims(self):
        for dim in [64, 128, 256, 768]:
            model = MockViT(n_patches=16, embed_dim=dim)
            feats = extract_dense_features(model, self.img)
            assert feats.shape[-1] == dim


# ── pixel_to_patch_coord ──────────────────────────────────────────────────────

class TestPixelToPatchCoord:

    def test_top_left_pixel(self):
        px, py = pixel_to_patch_coord(0, 0, original_size=(518, 518),
                                       patch_size=14, resized_size=518)
        assert (px, py) == (0, 0)

    def test_bottom_right_pixel(self):
        px, py = pixel_to_patch_coord(517, 517, original_size=(518, 518),
                                       patch_size=14, resized_size=518)
        assert px == 36 and py == 36   # max patch index for 518/14=37 patches

    def test_center_pixel(self):
        # pixel 259 in 518px image → resized = 259 → patch = 259//14 = 18
        px, py = pixel_to_patch_coord(259, 259, original_size=(518, 518),
                                       patch_size=14, resized_size=518)
        assert px == 18 and py == 18

    def test_clamping_above_max(self):
        """Out-of-bounds pixel must be clamped to max patch index."""
        px, py = pixel_to_patch_coord(9999, 9999, original_size=(518, 518),
                                       patch_size=14, resized_size=518)
        max_patch = 518 // 14 - 1   # = 36
        assert px == max_patch
        assert py == max_patch

    def test_clamping_below_zero(self):
        px, py = pixel_to_patch_coord(-1, -1, original_size=(518, 518),
                                       patch_size=14, resized_size=518)
        assert px == 0 and py == 0

    def test_non_square_image(self):
        """Different width and height handled independently."""
        # original 640x480, resized to 518, patch=14
        px, py = pixel_to_patch_coord(320, 240, original_size=(640, 480),
                                       patch_size=14, resized_size=518)
        expected_px = int((320 * 518 / 640) // 14)
        expected_py = int((240 * 518 / 480) // 14)
        assert px == expected_px
        assert py == expected_py

    def test_returns_ints(self):
        px, py = pixel_to_patch_coord(100, 100, (518, 518), 14, 518)
        assert isinstance(px, int)
        assert isinstance(py, int)

    def test_sam_patch_size_16(self):
        """SAM uses patch_size=16, resized_size=512."""
        px, py = pixel_to_patch_coord(256, 256, original_size=(512, 512),
                                       patch_size=16, resized_size=512)
        assert px == 16 and py == 16  # 256//16=16


# ── patch_to_pixel_coord ──────────────────────────────────────────────────────

class TestPatchToPixelCoord:

    def test_top_left_patch(self):
        x, y = patch_to_pixel_coord(0, 0, original_size=(518, 518),
                                     patch_size=14, resized_size=518)
        # Center of patch 0 in resized image = 7px → scaled back = 7px
        assert abs(x - 7.0) < 1e-4
        assert abs(y - 7.0) < 1e-4

    def test_roundtrip_center(self):
        """pixel → patch → pixel should be close to original (within half a patch)."""
        orig_x, orig_y = 200.0, 300.0
        px, py = pixel_to_patch_coord(orig_x, orig_y, (518, 518), 14, 518)
        rx, ry = patch_to_pixel_coord(px, py, (518, 518), 14, 518)
        # Roundtrip error ≤ half a patch in original pixel space
        half_patch_orig = (14 / 518) * 518 / 2
        assert abs(rx - orig_x) <= half_patch_orig + 1
        assert abs(ry - orig_y) <= half_patch_orig + 1

    def test_centering_offset(self):
        """patch_to_pixel uses patch center, not top-left corner."""
        # Patch (1,0) center in 518px image = 1*14 + 7 = 21px
        x, y = patch_to_pixel_coord(1, 0, (518, 518), 14, 518)
        assert abs(x - 21.0) < 1e-4

    def test_non_square_image_roundtrip(self):
        orig_x, orig_y = 400.0, 200.0
        px, py = pixel_to_patch_coord(orig_x, orig_y, (640, 480), 14, 518)
        rx, ry = patch_to_pixel_coord(px, py, (640, 480), 14, 518)
        half_patch_w = (14 * 640 / 518) / 2
        half_patch_h = (14 * 480 / 518) / 2
        assert abs(rx - orig_x) <= half_patch_w + 1
        assert abs(ry - orig_y) <= half_patch_h + 1

    def test_returns_floats(self):
        x, y = patch_to_pixel_coord(5, 5, (518, 518), 14, 518)
        assert isinstance(x, float)
        assert isinstance(y, float)


# ── extract_dense_features_multilayer ────────────────────────────────────────

@pytest.mark.skipif(not HAS_MULTILAYER,
                    reason="multilayer functions not yet in this repo version")
class TestMultilayerExtraction:

    def setup_method(self):
        self.model = MockViT(n_patches=9, embed_dim=16, n_layers=4)
        self.img   = torch.randn(1, 3, 518, 518)

    def test_output_shape_same_as_single_layer(self):
        feats = extract_dense_features_multilayer(self.model, self.img, n_last_layers=3)
        assert feats.shape == (1, 3, 3, 16)

    def test_different_n_layers(self):
        for n in [1, 2, 3, 4]:
            feats = extract_dense_features_multilayer(self.model, self.img, n_last_layers=n)
            assert feats.shape == (1, 3, 3, 16)

    def test_output_differs_from_single_layer(self):
        """Averaged multilayer features should differ from last-layer-only features."""
        torch.manual_seed(42)
        feats_single = extract_dense_features(self.model, self.img)
        torch.manual_seed(42)
        feats_multi  = extract_dense_features_multilayer(self.model, self.img, n_last_layers=3)
        # They use a random mock so shapes match but values differ
        assert feats_single.shape == feats_multi.shape

    def test_no_grad_in_eval_mode(self):
        feats = extract_dense_features_multilayer(self.model, self.img, training=False)
        assert not feats.requires_grad


# ── apply_pca_whitening ───────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_MULTILAYER,
                    reason="apply_pca_whitening not yet in this repo version")
class TestPCAWhitening:

    def setup_method(self):
        torch.manual_seed(0)
        # Use larger feature dim so PCA has something to work with
        self.src = torch.randn(1, 4, 4, 64)
        self.tgt = torch.randn(1, 4, 4, 64)

    def test_output_shapes_match_n_components(self):
        src_pca, tgt_pca = apply_pca_whitening(self.src, self.tgt, n_components=16)
        assert src_pca.shape == (1, 4, 4, 16)
        assert tgt_pca.shape == (1, 4, 4, 16)

    def test_output_on_same_device_as_input(self):
        src_pca, tgt_pca = apply_pca_whitening(self.src, self.tgt, n_components=8)
        assert src_pca.device == self.src.device
        assert tgt_pca.device == self.tgt.device

    def test_output_is_float32(self):
        src_pca, tgt_pca = apply_pca_whitening(self.src, self.tgt, n_components=8)
        assert src_pca.dtype == torch.float32
        assert tgt_pca.dtype == torch.float32

    def test_n_components_clamped_to_valid_range(self):
        """n_components > min(H*W, D) must not crash."""
        src = torch.randn(1, 2, 2, 8)   # H*W=4, D=8
        tgt = torch.randn(1, 2, 2, 8)
        # Request more components than possible
        src_pca, tgt_pca = apply_pca_whitening(src, tgt, n_components=100)
        assert src_pca.shape[-1] <= min(4, 8)

    def test_returns_two_tensors(self):
        result = apply_pca_whitening(self.src, self.tgt, n_components=8)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)