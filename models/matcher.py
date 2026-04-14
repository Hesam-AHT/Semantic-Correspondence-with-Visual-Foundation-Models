import torch
import torch.nn.functional as F


# Cosine similarity map

def compute_similarity_map(
    src_feat: torch.Tensor,
    trg_feat: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity between every source patch and every target patch.

    Args:
        src_feat: (h_s, w_s, C)  source feature map  (single image, no batch dim)
        trg_feat: (h_t, w_t, C)  target feature map

    Returns:
        sim: (h_s * w_s, h_t * w_t)  pairwise cosine similarities
    """
    h_s, w_s, C = src_feat.shape
    h_t, w_t, _  = trg_feat.shape

    # Flatten spatial dims
    src_flat = src_feat.reshape(-1, C)   # (h_s*w_s, C)
    trg_flat = trg_feat.reshape(-1, C)   # (h_t*w_t, C)

    # L2-normalise along the feature dimension
    src_norm = F.normalize(src_flat, dim=-1)   # (h_s*w_s, C)
    trg_norm = F.normalize(trg_flat, dim=-1)   # (h_t*w_t, C)

    # Cosine similarity matrix
    sim = src_norm @ trg_norm.T            # (h_s*w_s, h_t*w_t)
    return sim


#Keypoint to patch index helper

def kp_to_patch(kp_xy: torch.Tensor, image_size: int, patch_size: int) -> torch.Tensor:
    """
    Convert keypoint pixel coordinates (x, y) to patch grid indices (row, col).

    Args:
        kp_xy:      (N, 2)  keypoints in pixel space  [x, y]
        image_size: integer side length of the (square) image
        patch_size: side length of each patch in pixels

    Returns:
        (N, 2)  [row, col] patch indices (integers, clamped to valid range)
    """
    num_patches = image_size // patch_size
    # (x, y) to (col, row), then clamp to [0, num_patches-1]
    col = (kp_xy[:, 0] / patch_size).long().clamp(0, num_patches - 1)
    row = (kp_xy[:, 1] / patch_size).long().clamp(0, num_patches - 1)
    return torch.stack([row, col], dim=1)   # (N, 2)


def patch_to_kp(patch_rc: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert patch (row, col) indices back to pixel (x, y) — centre of patch.

    Args:
        patch_rc:  (N, 2)  [row, col]
        patch_size: patch side length in pixels

    Returns:
        (N, 2)  [x, y] pixel coordinates at patch centres
    """
    row = patch_rc[:, 0].float()
    col = patch_rc[:, 1].float()
    x = col * patch_size + patch_size / 2.0
    y = row * patch_size + patch_size / 2.0
    return torch.stack([x, y], dim=1)   # (N, 2)


#  Stage 1: argmax matcher

def predict_argmax(
    src_feat: torch.Tensor,
    trg_feat: torch.Tensor,
    src_kps: torch.Tensor,
    image_size: int,
    patch_size: int,
) -> torch.Tensor:
    """
    Stage 1 prediction: find the target patch most similar (cosine) to each
    source keypoint's patch, then convert back to pixel coordinates.

    Args:
        src_feat:   (h, w, C)  source feature map  (single image)
        trg_feat:   (h, w, C)  target feature map
        src_kps:    (N, 2)     source keypoints in pixel space  [x, y]
        image_size: side length of the (square) resized image
        patch_size: model's patch size in pixels

    Returns:
        pred_kps:   (N, 2)  predicted keypoints in the target image  [x, y]
    """
    h, w, C = src_feat.shape
    device = src_feat.device

    # 1. Map source keypoints to patch indices
    src_patch_rc = kp_to_patch(src_kps.to(device), image_size, patch_size)  # (N, 2)

    # 2. Look up source feature vectors for each keypoint
    src_vecs = src_feat[src_patch_rc[:, 0], src_patch_rc[:, 1], :]  # (N, C)

    # 3. Flatten target feature map
    trg_flat = trg_feat.reshape(-1, C)                               # (h*w, C)

    # 4. Cosine similarity: (N, h*w)
    src_norm = F.normalize(src_vecs, dim=-1)
    trg_norm = F.normalize(trg_flat, dim=-1)
    sim = src_norm @ trg_norm.T                                       # (N, h*w)

    # 5. Argmax → best matching patch index
    best_idx = sim.argmax(dim=-1)                                     # (N,)

    # 6. Convert flat index to (row, col)
    best_row = best_idx // w
    best_col = best_idx % w
    best_rc = torch.stack([best_row, best_col], dim=1)               # (N, 2)

    # 7. Convert patch (row, col) → pixel (x, y) at patch centre
    pred_kps = patch_to_kp(best_rc, patch_size)                      # (N, 2)
    return pred_kps


# ── Stage 3: window soft-argmax (placeholder, implemented in stage3.py) ──────

def predict_window_soft_argmax(
    src_feat: torch.Tensor,
    trg_feat: torch.Tensor,
    src_kps: torch.Tensor,
    image_size: int,
    patch_size: int,
    window_size: int = 5,
) -> torch.Tensor:
    """
    Stage 3 prediction: argmax to find the peak, then soft-argmax within
    a local window for sub-pixel refinement.

    Args:
        window_size: side length of the local window (in patches)

    Returns:
        pred_kps: (N, 2) predicted keypoints [x, y] in pixel space
    """
    h, w, C = src_feat.shape
    device = src_feat.device
    half = window_size // 2

    # Steps 1-5 same as argmax
    src_patch_rc = kp_to_patch(src_kps.to(device), image_size, patch_size)
    src_vecs = src_feat[src_patch_rc[:, 0], src_patch_rc[:, 1], :]
    trg_flat = trg_feat.reshape(-1, C)

    src_norm = F.normalize(src_vecs, dim=-1)
    trg_norm = F.normalize(trg_flat, dim=-1)
    sim = src_norm @ trg_norm.T                # (N, h*w)

    # Reshape to spatial
    sim_map = sim.reshape(-1, h, w)            # (N, h, w)

    pred_kps = []
    for i in range(sim_map.shape[0]):
        s = sim_map[i]                         # (h, w)

        # Find peak
        peak_flat = s.argmax()
        peak_r = (peak_flat // w).item()
        peak_c = (peak_flat % w).item()

        # Clamp window to valid range
        r0 = max(0, peak_r - half)
        r1 = min(h, peak_r + half + 1)
        c0 = max(0, peak_c - half)
        c1 = min(w, peak_c + half + 1)

        # Extract window and compute soft-argmax
        window = s[r0:r1, c0:c1]              # (wr, wc)
        weights = F.softmax(window.reshape(-1), dim=0).reshape(window.shape)

        # Weighted average of row/col coordinates within the window
        wr, wc = weights.shape
        rows = torch.arange(r0, r0 + wr, device=device).float()
        cols = torch.arange(c0, c0 + wc, device=device).float()

        soft_r = (weights.sum(dim=1) * rows).sum()
        soft_c = (weights.sum(dim=0) * cols).sum()

        # Convert to pixel centre
        x = soft_c * patch_size + patch_size / 2.0
        y = soft_r * patch_size + patch_size / 2.0
        pred_kps.append(torch.stack([x, y]))

    return torch.stack(pred_kps)               # (N, 2)
