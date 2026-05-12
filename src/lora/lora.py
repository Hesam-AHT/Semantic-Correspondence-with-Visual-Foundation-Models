"""Low-Rank Adaptation (LoRA) utilities for DINOv2 / ViT models.

This module provides:
  - LoRALinear   : drop-in replacement for nn.Linear that adds a low-rank
                   side branch (W' = W + scale * B @ A) while keeping the
                   original weights frozen.
  - inject_lora  : freeze a model then swap every target nn.Linear for a
                   LoRALinear in-place.
  - remove_lora  : merge the learned deltas back into the frozen weights and
                   restore plain nn.Linear modules.
  - count_trainable_params : report trainable / total parameter counts.

References:
    Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021.
    https://arxiv.org/abs/2106.09685
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A low-rank adapter that wraps a frozen nn.Linear layer.

    The forward pass computes:

        out = W x + b  +  scale * (B A x)

    where W (and optional b) are kept frozen, and only A ∈ R^{r × d_in}
    and B ∈ R^{d_out × r} are trained.  This keeps the parameter count low
    while allowing expressive weight updates via the outer product B @ A.

    Initialisation follows the original LoRA paper: A is initialised with
    Kaiming-uniform (so the initial delta is non-zero), B is initialised to
    zero (so the adapter has no effect at the start of training).

    Args:
        original_linear: The nn.Linear layer to wrap.  Its weight and bias
            are frozen (requires_grad set to False).
        r: Rank of the low-rank decomposition.  Smaller values use fewer
            parameters; typical values are 4, 8, 16.
        alpha: Scaling factor for the LoRA output.  The effective scale
            applied to B @ A is alpha / r.  Setting alpha == r gives a
            scale of 1.0.
    """

    def __init__(self, original_linear: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()

        d_in = original_linear.in_features
        d_out = original_linear.out_features

        self.original = original_linear

        # Freeze the original weights so only A and B are trained.
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        # A: initialised with Kaiming-uniform so the initial product B@A is
        # non-trivial once B moves away from zero during training.
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: zero-initialised so the adapter is a no-op at step 0.
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))

        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the frozen linear output plus the scaled LoRA delta.

        Args:
            x: Input tensor of any shape [..., d_in].

        Returns:
            Output tensor of shape [..., d_out].
        """
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def inject_lora(model: nn.Module, r: int = 4, alpha: float = 4.0,
                target_modules: tuple = ('qkv', 'proj')) -> nn.Module:
    """Freeze a model and inject LoRA adapters into target linear layers.

    All model parameters are frozen first.  Then every nn.Linear whose
    dotted module name contains at least one string from *target_modules* is
    replaced in-place with a LoRALinear.  Only the LoRA parameters A and B
    will have requires_grad=True after this call.

    Args:
        model: The nn.Module to modify (modified in-place and returned).
        r: LoRA rank passed to LoRALinear.
        alpha: LoRA alpha scaling passed to LoRALinear.
        target_modules: Tuple of name substrings to match.  A linear layer
            is replaced if *any* of these strings appears in its full dotted
            name (e.g. 'qkv' matches 'blocks.3.attn.qkv').

    Returns:
        The modified model with LoRA layers injected.
    """
    # Freeze every parameter unconditionally; LoRALinear will un-freeze A/B.
    for param in model.parameters():
        param.requires_grad = False

    # Collect targets before iterating to avoid mutating the module tree
    # while the named_modules() generator is active.
    to_replace = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules)
    ]

    replaced = 0
    for name, module in to_replace:
        parts = name.rsplit('.', 1)
        if len(parts) == 1:
            parent = model
            child_name = parts[0]
        else:
            parent_name, child_name = parts
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)

        setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha))
        replaced += 1

    print(f"Injected LoRA (r={r}, alpha={alpha}) into {replaced} linear layers")
    return model


def remove_lora(model: nn.Module) -> nn.Module:
    """Merge LoRA deltas into the original weights and remove the adapters.

    For each LoRALinear in the model the learned delta
    (lora_B @ lora_A) * scale is added to the frozen weight matrix,
    requires_grad is restored on the original weight (and bias if present),
    and the LoRALinear is replaced back with the underlying nn.Linear.

    After this call the model has the same structure as before inject_lora
    was called, but with weights that incorporate the LoRA fine-tuning.

    Args:
        model: The nn.Module containing LoRALinear layers (modified in-place
            and returned).

    Returns:
        The model with all LoRA adapters merged and removed.
    """
    to_restore = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]

    for name, module in to_restore:
        # Merge: W_new = W + scale * B @ A  (shapes: [d_out, d_in])
        delta = (module.lora_B @ module.lora_A) * module.scale
        module.original.weight.data += delta

        # Restore gradient tracking on the original layer.
        module.original.weight.requires_grad = True
        if module.original.bias is not None:
            module.original.bias.requires_grad = True

        # Swap LoRALinear back to the plain nn.Linear in the parent module.
        parts = name.rsplit('.', 1)
        if len(parts) == 1:
            parent = model
            child_name = parts[0]
        else:
            parent_name, child_name = parts
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)

        setattr(parent, child_name, module.original)

    return model


def count_trainable_params(model: nn.Module) -> dict:
    """Count trainable and total parameters in a model.

    Args:
        model: Any nn.Module.

    Returns:
        A dict with keys:
            'trainable'  (int)   — number of parameters with requires_grad=True,
            'total'      (int)   — total number of parameters,
            'percentage' (float) — trainable / total * 100, rounded to 4 d.p.
                                   Returns 0.0 for a model with no parameters.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = round(100.0 * trainable / total, 4) if total > 0 else 0.0
    return {'trainable': trainable, 'total': total, 'percentage': percentage}
