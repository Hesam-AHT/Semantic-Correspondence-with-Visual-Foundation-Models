import sys
import os
import math
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from src.lora.lora import LoRALinear, inject_lora, remove_lora, count_trainable_params
    HAS_LORA = True
except ImportError:
    try:
        from lora.lora import LoRALinear, inject_lora, remove_lora, count_trainable_params
        HAS_LORA = True
    except ImportError:
        HAS_LORA = False


pytestmark = pytest.mark.skipif(not HAS_LORA, reason="lora module not found")


# ── Minimal ViT-like model for testing ───────────────────────────────────────

class FakeAttention(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.qkv  = nn.Linear(d, d * 3)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        return self.proj(self.qkv(x)[..., :16])


class FakeBlock(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.attn = FakeAttention(d)
        self.mlp  = nn.Linear(d, d)

    def forward(self, x):
        return self.mlp(self.attn(x))


class FakeMiniViT(nn.Module):
    """Two-block ViT-like model with qkv and proj layers — matches inject_lora targets."""
    def __init__(self, d=16):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock(d), FakeBlock(d)])
        self.norm   = nn.LayerNorm(d)
        self.d = d

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ── LoRALinear ────────────────────────────────────────────────────────────────

class TestLoRALinear:

    def setup_method(self):
        self.original = nn.Linear(16, 32)
        self.lora = LoRALinear(self.original, r=4, alpha=4.0)

    def test_original_weight_frozen(self):
        assert not self.lora.original.weight.requires_grad

    def test_lora_params_trainable(self):
        assert self.lora.lora_A.requires_grad
        assert self.lora.lora_B.requires_grad

    def test_lora_B_initialized_to_zero(self):
        """lora_B = 0 → initial LoRA delta = 0 → output same as original."""
        assert torch.all(self.lora.lora_B == 0)

    def test_initial_output_equals_original(self):
        """At init (lora_B=0), LoRALinear must produce identical output to original."""
        x = torch.randn(2, 16)
        with torch.no_grad():
            out_orig = self.original(x)
            out_lora = self.lora(x)
        assert torch.allclose(out_orig, out_lora, atol=1e-5)

    def test_output_shape(self):
        x = torch.randn(3, 16)
        out = self.lora(x)
        assert out.shape == (3, 32)

    def test_scale_is_alpha_over_r(self):
        assert abs(self.lora.scale - 4.0 / 4) < 1e-6

    def test_lora_A_shape(self):
        assert self.lora.lora_A.shape == (4, 16)   # (r, d_in)

    def test_lora_B_shape(self):
        assert self.lora.lora_B.shape == (32, 4)   # (d_out, r)

    def test_different_ranks(self):
        for r in [1, 2, 4, 8, 16]:
            lin  = nn.Linear(32, 64)
            lora = LoRALinear(lin, r=r, alpha=float(r))
            assert lora.lora_A.shape == (r, 32)
            assert lora.lora_B.shape == (64, r)

    def test_forward_after_lora_update(self):
        """After modifying lora_B, output should differ from original."""
        x = torch.randn(2, 16)
        with torch.no_grad():
            self.lora.lora_B.fill_(0.1)
            out_orig = self.original(x)
            out_lora = self.lora(x)
        assert not torch.allclose(out_orig, out_lora, atol=1e-5)


# ── inject_lora ───────────────────────────────────────────────────────────────

class TestInjectLoRA:

    def setup_method(self):
        self.model = FakeMiniViT(d=16)

    def test_qkv_layers_replaced(self):
        inject_lora(self.model, r=4, alpha=4.0, target_modules=('qkv',))
        for block in self.model.blocks:
            assert isinstance(block.attn.qkv, LoRALinear)

    def test_proj_layers_replaced(self):
        inject_lora(self.model, r=4, alpha=4.0, target_modules=('proj',))
        for block in self.model.blocks:
            assert isinstance(block.attn.proj, LoRALinear)

    def test_non_target_layers_unchanged(self):
        inject_lora(self.model, r=4, alpha=4.0, target_modules=('qkv',))
        for block in self.model.blocks:
            assert isinstance(block.attn.proj, nn.Linear)  # not replaced
            assert isinstance(block.mlp, nn.Linear)

    def test_all_non_lora_params_frozen(self):
        inject_lora(self.model, r=4, alpha=4.0, target_modules=('qkv', 'proj'))
        for name, p in self.model.named_parameters():
            if 'lora_' not in name:
                assert not p.requires_grad, f"{name} should be frozen"

    def test_lora_params_trainable(self):
        inject_lora(self.model, r=4, alpha=4.0, target_modules=('qkv', 'proj'))
        for name, p in self.model.named_parameters():
            if 'lora_' in name:
                assert p.requires_grad, f"{name} should be trainable"

    def test_fewer_trainable_params_than_original(self):
        total_before = sum(p.numel() for p in self.model.parameters())
        inject_lora(self.model, r=2, alpha=2.0, target_modules=('qkv', 'proj'))
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        assert trainable < total_before

    def test_returns_model(self):
        result = inject_lora(self.model, r=4, alpha=4.0)
        assert result is self.model

    def test_forward_still_works_after_injection(self):
        inject_lora(self.model, r=4, alpha=4.0, target_modules=('qkv', 'proj'))
        x = torch.randn(2, 5, 16)
        out = self.model(x)
        assert out.shape == (2, 5, 16)

    def test_output_same_as_before_injection(self):
        """At init (lora_B=0), injected model must produce same output as original."""
        x = torch.randn(1, 3, 16)
        model_orig = FakeMiniViT(d=16)
        model_lora = FakeMiniViT(d=16)
        # Copy identical weights
        model_lora.load_state_dict(model_orig.state_dict())
        with torch.no_grad():
            out_orig = model_orig(x)
        inject_lora(model_lora, r=4, alpha=4.0, target_modules=('qkv', 'proj'))
        with torch.no_grad():
            out_lora = model_lora(x)
        assert torch.allclose(out_orig, out_lora, atol=1e-5)


# ── count_trainable_params ────────────────────────────────────────────────────

class TestCountTrainableParams:

    def test_all_trainable_before_injection(self):
        model = FakeMiniViT(d=16)
        info = count_trainable_params(model)
        assert info['trainable'] == info['total']
        assert info['percentage'] == 100.0

    def test_fewer_trainable_after_injection(self):
        model = FakeMiniViT(d=16)
        inject_lora(model, r=2, alpha=2.0, target_modules=('qkv', 'proj'))
        info = count_trainable_params(model)
        assert info['trainable'] < info['total']
        assert info['percentage'] < 100.0

    def test_returns_dict_with_correct_keys(self):
        model = FakeMiniViT(d=16)
        info = count_trainable_params(model)
        assert 'trainable'  in info
        assert 'total'      in info
        assert 'percentage' in info

    def test_percentage_matches_counts(self):
        model = FakeMiniViT(d=16)
        inject_lora(model, r=4, alpha=4.0, target_modules=('qkv',))
        info = count_trainable_params(model)
        expected_pct = round(100 * info['trainable'] / info['total'], 4)
        assert abs(info['percentage'] - expected_pct) < 1e-3

    def test_zero_trainable_fully_frozen(self):
        model = FakeMiniViT(d=16)
        for p in model.parameters():
            p.requires_grad = False
        info = count_trainable_params(model)
        assert info['trainable'] == 0
        assert info['percentage'] == 0.0


# ── remove_lora ───────────────────────────────────────────────────────────────

class TestRemoveLoRA:

    def test_lora_layers_replaced_back_to_linear(self):
        model = FakeMiniViT(d=16)
        inject_lora(model, r=4, alpha=4.0, target_modules=('qkv',))
        remove_lora(model)
        for block in model.blocks:
            assert isinstance(block.attn.qkv, nn.Linear)

    def test_weights_unfrozen_after_remove(self):
        model = FakeMiniViT(d=16)
        inject_lora(model, r=4, alpha=4.0, target_modules=('qkv', 'proj'))
        remove_lora(model)
        for name, p in model.named_parameters():
            assert p.requires_grad, f"{name} should be trainable after remove_lora"

    def test_delta_merged_into_weights(self):
        """After remove_lora, the merged weights should incorporate the LoRA delta."""
        model_orig = FakeMiniViT(d=16)
        model_lora = FakeMiniViT(d=16)
        model_lora.load_state_dict(model_orig.state_dict())

        inject_lora(model_lora, r=4, alpha=4.0, target_modules=('qkv',))
        # Manually set lora_B to non-zero so delta is non-trivial
        with torch.no_grad():
            for m in model_lora.modules():
                if isinstance(m, LoRALinear):
                    m.lora_B.fill_(0.01)

        x = torch.randn(1, 3, 16)
        with torch.no_grad():
            out_before_remove = model_lora(x)

        remove_lora(model_lora)
        with torch.no_grad():
            out_after_remove = model_lora(x)

        # Output should be identical before and after remove (delta is merged)
        assert torch.allclose(out_before_remove, out_after_remove, atol=1e-4)