from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

#Imagenet normalisation
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_transform(image_size: int = 840) -> transforms.Compose:
    """Standard preprocessing: resize → tensor → normalize."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


#Base

class FeatureExtractor(ABC, nn.Module):
    """
    Abstract base. Subclasses must implement `extract`.

    Input:  image tensor  (B, 3, H, W)  — already normalised
    Output: feature map   (B, h, w, C)  — one vector per patch
                where h = H // patch_size, w = W // patch_size
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

    @abstractmethod
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract(x)

    @property
    @abstractmethod
    def patch_size(self) -> int:
        ...

    @property
    @abstractmethod
    def feat_dim(self) -> int:
        ...


# DINOv2

class DINOv2Extractor(FeatureExtractor):
    """
    DINOv2 (ViT-B/14 by default).
    Loads from torch.hub: facebookresearch/dinov2.

    Model options: 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'
    """

    MODELS = {
        "vits": ("dinov2_vits14", 384),
        "vitb": ("dinov2_vitb14", 768),
        "vitl": ("dinov2_vitl14", 1024),
        "vitg": ("dinov2_vitg14", 1536),
    }

    def __init__(self, variant: str = "vitb", device: str = "cuda"):
        super().__init__(device)
        hub_name, dim = self.MODELS[variant]
        self._patch_size = 14
        self._feat_dim = dim

        print(f"[DINOv2] Loading {hub_name} from torch.hub ...")
        self.model = torch.hub.load(
            "facebookresearch/dinov2", hub_name, pretrained=True
        )
        self.model.eval().to(device)
        # Freeze params(training-free baseline)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)  — H and W must be divisible by 14
        Returns:
            (B, h, w, C) patch features (excluding [CLS] token)
        """
        x = x.to(self.device)
        B, _, H, W = x.shape
        h, w = H // self._patch_size, W // self._patch_size

        with torch.no_grad():
            # get_intermediate_layers returns a list of layer outputs
            # We take the last layer; each output: (B, 1 + h*w, C)
            out = self.model.get_intermediate_layers(x, n=1)[0]  # (B, 1+h*w, C)

        patch_tokens = out[:, 1:, :]          # drop [CLS]  → (B, h*w, C)
        feat_map = patch_tokens.reshape(B, h, w, self._feat_dim)
        return feat_map                        # (B, h, w, C)

    def unfreeze_last_n_layers(self, n: int):
        """Unfreeze the last n transformer blocks for fine-tuning (Stage 2)."""
        blocks = list(self.model.blocks)
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad_(True)
        print(f"[DINOv2] Unfroze last {n} transformer block(s).")


#DINOv3

class DINOv3Extractor(FeatureExtractor):
    """
    DINOv3 (Simeoni et al., 2025).
    Loads from HuggingFace: 'naver-ai/dinov3-vitb14' (update hub_id as released).
    Falls back gracefully if the model is not yet publicly available.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._patch_size = 14
        self._feat_dim = 768  # ViT-B default

        try:
            from transformers import AutoModel
            hub_id = "naver-ai/dinov3-vitb14"
            print(f"[DINOv3] Loading {hub_id} from HuggingFace ...")
            self.model = AutoModel.from_pretrained(hub_id)
            self.model.eval().to(device)
            for p in self.model.parameters():
                p.requires_grad_(False)
            self._loaded = True
        except Exception as e:
            print(f"[DINOv3] WARNING: Could not load DINOv3 ({e}).")
            print("[DINOv3] Falling back to DINOv2 as a placeholder.")
            fallback = DINOv2Extractor(variant="vitb", device=device)
            self.model = fallback.model
            self._loaded = False

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        B, _, H, W = x.shape
        h, w = H // self._patch_size, W // self._patch_size

        with torch.no_grad():
            if self._loaded:
                # HuggingFace ViT interface
                outputs = self.model(pixel_values=x, output_hidden_states=True)
                patch_tokens = outputs.last_hidden_state[:, 1:, :]  # drop CLS
            else:
                out = self.model.get_intermediate_layers(x, n=1)[0]
                patch_tokens = out[:, 1:, :]

        feat_map = patch_tokens.reshape(B, h, w, self._feat_dim)
        return feat_map


#SAM - Segment Anything Model

class SAMExtractor(FeatureExtractor):
    """
    SAM image encoder (ViT-B by default).
    Uses the official segment_anything package.

    Checkpoint download:
      wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    """

    VARIANTS = {
        "vit_b": ("sam_vit_b_01ec64.pth", 256),
        "vit_l": ("sam_vit_l_0b3195.pth", 256),
        "vit_h": ("sam_vit_h_4b8939.pth", 256),
    }

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_b",
        device: str = "cuda",
    ):
        """
        Args:
            checkpoint: path to the downloaded SAM .pth checkpoint
            model_type: one of 'vit_b', 'vit_l', 'vit_h'
        """
        super().__init__(device)
        # SAM uses a fixed 16-pixel patch stride from its ViT
        self._patch_size = 16
        # The image encoder outputs a (256, 64, 64) feature map for 1024×1024 input
        self._feat_dim = 256

        try:
            from segment_anything import sam_model_registry
            print(f"[SAM] Loading {model_type} from {checkpoint} ...")
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.eval().to(device)
            self.encoder = sam.image_encoder
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        except ImportError:
            raise ImportError(
                "Please install segment_anything:\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        SAM's image encoder expects 1024×1024 input and returns (B, C, 64, 64).
        We resize internally to 1024 if needed.
        """
        x = x.to(self.device)
        if x.shape[-1] != 1024 or x.shape[-2] != 1024:
            x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)

        with torch.no_grad():
            feat = self.encoder(x)   # (B, 256, 64, 64)

        # Rearrange to (B, h, w, C) to match DINOv2 convention
        feat_map = feat.permute(0, 2, 3, 1)  # (B, 64, 64, 256)
        return feat_map

    def unfreeze_last_n_layers(self, n: int):
        """Unfreeze last n transformer blocks of SAM's ViT encoder."""
        blocks = list(self.encoder.blocks)
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad_(True)
        print(f"[SAM] Unfroze last {n} transformer block(s).")


# ── Factory ───────────────────────────────────────────────────────────────────

def build_extractor(
    backbone: Literal["dinov2", "dinov3", "sam"],
    device: str = "cuda",
    dinov2_variant: str = "vitb",
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "vit_b",
) -> FeatureExtractor:
    """
    Build a feature extractor by name.

    Args:
        backbone:        'dinov2' | 'dinov3' | 'sam'
        device:          'cuda' or 'cpu'
        dinov2_variant:  for DINOv2: 'vits' | 'vitb' | 'vitl' | 'vitg'
        sam_checkpoint:  path to SAM .pth file (required for 'sam')
        sam_model_type:  'vit_b' | 'vit_l' | 'vit_h'
    """
    backbone = backbone.lower()
    if backbone == "dinov2":
        return DINOv2Extractor(variant=dinov2_variant, device=device)
    elif backbone == "dinov3":
        return DINOv3Extractor(device=device)
    elif backbone == "sam":
        if sam_checkpoint is None:
            raise ValueError("sam_checkpoint path is required for SAM extractor.")
        return SAMExtractor(
            checkpoint=sam_checkpoint,
            model_type=sam_model_type,
            device=device,
        )
    else:
        raise ValueError(f"Unknown backbone: '{backbone}'. Choose from dinov2, dinov3, sam.")
